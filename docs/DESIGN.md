# quarry design

## Goals

1. Answer natural-language questions about a SQL database with an agent whose every step
   (schema choice, plan, tool calls, SQL attempts, checks, execution, verification) is
   visible while it runs and inspectable afterwards.
2. Make it impossible for the agent, or anything it reads, to change the database.
3. Recover from its own mistakes within a bounded budget, using the same feedback a human
   would get: guard refusals, unresolved columns, engine errors, suspicious results.
4. Hold a conversation: follow-ups resolve against earlier turns, and conversations survive
   restarts.
5. Be measurable without an LLM, and measurable with one through a single command.
6. Run anywhere: one Python package, SQLite or DuckDB files, no GPU, no services.

## Non-goals

- Writing data, migrations or DDL. quarry is read-only by design, not by configuration.
- Server databases (PostgreSQL, MySQL) in this version; the guard is dialect-aware and the
  engine protocol is small, so this is a roadmap item rather than a redesign.
- Fine-tuning a text-to-SQL model. quarry is the system around any tool-calling model.
- Authentication and multi-tenant isolation for the web UI, which binds to 127.0.0.1.

## Invariants

These hold for every input and are tested (several with Hypothesis):

1. **The only path from SQL text to a database is `SafeExecutor.run`**: guard, then a fresh
   read-only connection. Tools, the agent, the MCP server and the benchmark's prediction
   scoring all use it. (Trusted internal code, such as catalog introspection and gold-SQL
   execution in the benchmark, talks to the engine directly; the benchmark first runs every
   gold query through `SafeExecutor` and leaves out, and lists, any that cannot run there.)
2. **No statement outside the allowlist reaches an engine.** Property test, on SQLite and
   DuckDB: a spy engine records every statement the guard lets through, and an oracle that
   shares no code with the guard judges each one with the engine's own parser and planner.
   SQLite compiles it (`EXPLAIN`, never run) under a recording authorizer, which may see
   only `SELECT`, reads of user tables or of CTEs, and a few pure functions; DuckDB must
   find exactly one `SELECT` (`extract_statements`), its `EXPLAIN (FORMAT JSON)` plan may
   scan only user tables (no catalog views, table functions or files), and its own parse
   tree may call only pure functions. The generator targets the guard's scoping logic:
   CTEs inside subqueries, names that shadow catalog views or look like files, table
   functions, qualified names, recursion, unions, plus stacked statements, comment tricks
   and arbitrary text. Run against the guard as it was before the CTE-scope fix (below), it
   finds the bypass on both engines.
3. **The database file is byte-identical after any run.** Checked by hashing in the
   property tests, the red-team suite and the injection scenario.
4. **The engine refuses writes even without the guard.** SQLite: `mode=ro` URI,
   `PRAGMA query_only`, extension loading off, an authorizer that allows only `SELECT`,
   column reads (not of `sqlite_*` tables), recursive CTEs and allowlisted functions, plus
   `SQLITE_LIMIT_LENGTH` (10 MB), `SQLITE_LIMIT_SQL_LENGTH` and no attached databases.
   DuckDB: `read_only`, `enable_external_access=false`, no extension autoload or install,
   a 512 MB memory cap, two threads, and `lock_configuration` so `SET` cannot undo any of it.
5. **Every statement is bounded**: a row cap at fetch time and a wall-clock timeout
   (progress handler on SQLite, `interrupt()` from a timer on DuckDB).
6. **Database text reaches any model only through the sanitiser**: sample values, table
   and column names and descriptions in the rendered schema and `list_tables`,
   `describe_table`, `sample_rows` and `run_sql` previews (column names and aliases
   included), the rows the answer step sees, and the rows MCP `ask` returns to the calling
   app are scanned by the injection detector; flagged text is replaced by a placeholder,
   long values clipped. The full, unscanned result goes only to the human-facing outputs
   (the CLI table, the web UI, `--json`).
7. **Every question terminates**: SQL attempts are bounded by `max_repairs`, schema tool
   calls by `max_explore_calls`, LLM + tool calls by `max_steps`, tokens by `max_tokens`,
   and LangGraph's recursion limit is the backstop.
8. **What runs is what is shown**: the guard never rewrites SQL.
9. **A CTE name vouches for a table reference only where the CTE is in scope**: in the body
   of the query that defines it (and that body's subqueries), in later CTEs of the same
   `WITH`, and for itself only in a non-anchor branch of a `WITH RECURSIVE` union. A
   qualified name (`main.x`) is never a CTE. An earlier version collected CTE names for the
   whole statement, so a CTE inside one subquery let the same name elsewhere skip the
   catalog and file checks (`... FROM (WITH sqlite_master AS (...) ...) AS a, sqlite_master
   AS b`); the SQLite authorizer still refused it, DuckDB returned catalog rows. Found in
   review, fixed, and now in the red-team set and the property tests.
10. **Thread ids are plain ids** (`[A-Za-z0-9_-]{1,64}`), checked where traces are written,
   so no caller (CLI `--thread`, MCP `ask`, the web API) can steer a trace file outside
   `runs/traces`.

## Data model

**Graph state** (`agent/state.py`, a `TypedDict` the SQLite checkpointer stores):

| Field | Lifetime | Meaning |
|---|---|---|
| `question` | turn | the input |
| `history` | conversation | `Turn` records appended by `finish` (question, accepted SQL, answer, status, row count) |
| `linked`, `schema_text` | turn | linking result and the schema text the model sees |
| `plan` | turn | the planning step's text |
| `work` | turn | the tool-calling transcript (assistant tool calls and tool results) |
| `pending_sql`, `pending_call_id` | turn | the SQL submission in flight and its tool-call id |
| `attempts` | turn | one `Attempt` per SQL try: SQL, stage reached (`check`, `execute`, `verify`, `accepted`), feedback, rows, latency |
| `checks_used` | turn | suspicion kinds already raised (each fires once) |
| `result` | turn | columns, rows (capped), sanitised preview, truncation, latency |
| `answer`, `chart`, `warnings`, `status` | turn | the outcome |
| `steps`, `tokens`, `explore_calls`, `can_retry`, `refused` | turn | budgets and routing flags |

`start_turn` resets every turn field; only `history` accumulates (an `operator.add`
reducer). Routing functions read only the state, so the graph can be inspected and tested
without dependencies; the dependencies (model, database, executor, linker, detector,
config) and the run's trace arrive as LangGraph run context.

**Catalog** (`db/catalog.py`): tables with columns (type, description, primary key,
sampled distinct values), foreign keys and row counts. Descriptions come from a
`<database>.docs.json` sidecar (the demo warehouse ships one; Spider's natural-language
column names are used for Spider).

**Trace** (`tracing/tracer.py`): one JSONL file per question; one line per span with
`trace_id`, `span_id`, `parent_id`, `name`, `kind` (`run`, `node`, `llm`, `tool`, `event`),
start time, latency, inputs and outputs (summarised), token counts (marked as estimated for
the scripted model), attempt number, status and error.

**Demo warehouse** (`demo/warehouse.py`): `campaigns` (3) ← `campaign_sends` (64,000) →
`customers` (64,000), and `outcomes` (64,000, one per send), from the Hillstrom e-mail
experiment. Foreign keys are declared so linking can bridge through `campaign_sends`.

## Failure modes and how they surface

| Failure | Where it is caught | What the user sees |
|---|---|---|
| Write, DDL, `PRAGMA`, `ATTACH`, stacked statements, file functions | guard (`check` node), then the engine | refused attempt with reasons; status `refused` if the model gives up; database unchanged |
| Misspelt column or table | static check (sqlglot `qualify`), or the engine | the error goes back to the model as its tool result; repair attempt. A double-quoted unknown name counts as a string (SQLite's rule) only as the value side of a comparison with a column, so `SELECT "nmae"` is still reported |
| Runtime error (types, sub-select shape) | engine | same as above |
| Runaway query | timeout | `query exceeded N ms` fed back; repair |
| Empty, all-NULL or raw-rows-for-a-number result | `verify` | one second look per kind; resubmitting the same SQL accepts it |
| Too many rows | fetch cap | result marked truncated; a warning in the answer |
| Injected instructions in rows | detector | values withheld; warning in UI and trace; no effect on safety either way |
| Model unreachable / rate limited | `call_llm` | status `failed`, an `llm_unavailable` warning; the benchmark does not record the question and stops after five in a row with a one-line error (exit 2) so the same command resumes |
| Gold SQL that cannot run in the sandbox | benchmark preflight | the question is left out of the denominator and listed in the report, not scored as a model error |
| Text that is not valid UTF-8 | SQLite `text_factory` | bad bytes become U+FFFD instead of failing the query |
| Budget exhausted | every LLM node | best answer from the data so far without another LLM call, or a clear message |
| Model answers in prose with a fenced SQL block | `generate` | the block is treated as a `run_sql` submission |
| Model replies without SQL | `generate` | its text is the answer; status `unanswerable` (or `refused` after a guard refusal) |
| Missing database / bad config / missing key | CLI | one `error:` line with the fix, exit code 2 |
| Benchmark download blocked (Google Drive quota) | `datasets.fetch` | an explicit error naming the manual download page; schemas-only mode for the offline metrics |

## Alternatives considered

- **Regex or keyword blocklists for SQL.** Defeated by comments, CTEs, `SELECT ... INTO`,
  dialect quirks and unknown functions. An AST allowlist fails closed.
- **Relying only on a read-only connection.** Necessary but not sufficient: DuckDB still
  runs temp tables, `VACUUM`, catalog reads and stacked statements on a read-only file (the
  red-team report lists them), and a read-only connection does nothing about file-reading
  functions on engines without an external-access switch.
- **Rewriting SQL to inject `LIMIT`.** Changes semantics for `ORDER BY` inside subqueries
  and makes "what ran" differ from "what was shown"; fetch caps give the same bound.
- **LangGraph's prebuilt ReAct agent.** Shorter, but a single loop cannot express "static
  check before execution", "verify after", per-kind retry policy or explicit budgets, and
  its traces would not line up with those stages.
- **An LLM-chosen chart.** Costs a call per answer and is hard to test; result shape
  determines a sensible chart in almost every case.
- **An ML injection classifier.** Higher recall on jailbreak text, but a model download,
  latency per cell and false positives on ordinary data; kept on the roadmap because the
  safety boundary does not depend on it.
- **Dense embeddings by default.** Better semantic matching, but a download and a heavier
  dependency. BM25 over names already links Spider's small schemas well (names-only
  measurement: all gold tables in the top 5 for 98.7% of dev questions, 98.9% with the
  hashing embedder fused in); the sampled-values
  variant is not measured yet because the Spider databases did not download. The
  `Embedder` protocol makes a dense model a drop-in.
- **BIRD mini-dev instead of Spider.** Harder and more realistic, but 801 MB to download
  and about 1.5 GB of databases, over this project's data budget.

## Security notes

- The web UI escapes all model and database text (`textContent`, an HTML escaper before
  the bold-only Markdown) and binds to 127.0.0.1 by default (so do `compose.yaml` and the
  `docker run` example). Thread and trace ids are validated against `[A-Za-z0-9_-]{1,64}`
  by the tracer itself, so the web API, the CLI and MCP `ask` share one rule.
- The CLI reads `.env` from the working directory only, and the test suite removes
  `OPENROUTER_API_KEY` and disables `.env` loading, so a real key never turns a test into
  an API call.
- The guard resolves its forbidden node types from sqlglot at import and refuses to load if
  one is missing, and `pyproject.toml` pins the sqlglot and MCP major versions the code was
  written against.
- The LLM response cache revives only generation and message classes
  (`allowed_objects=[...]`), never arbitrary LangChain objects.
- Secrets live in the environment or a git-ignored `.env`; the config file holds none.
- Langfuse and LangSmith export need both a config flag and credentials.
