"""The LLM benchmark: execution accuracy of the full agent on Spider dev.

One command, once ``OPENROUTER_API_KEY`` is set::

    uv run quarry eval benchmark            # all 1,034 dev questions
    uv run quarry eval benchmark --limit 50 # a quick slice

* **Resumable.** Every finished question is appended to
  ``runs/bench/<run>/predictions.jsonl``; a rerun skips what is already there.
* **Cached.** Every LLM response is stored in ``runs/bench/llm_cache.sqlite`` (keyed by the
  exact prompt and model parameters), so reruns and resumed runs cost nothing.
* **Rate-limit aware.** A client-side token bucket (``[llm] requests_per_second``) plus the
  OpenAI client's retries with backoff. Questions that still fail because the API is
  unavailable are *not* recorded, and after several in a row the run stops so the same
  command can resume it later.
* **Scored like Spider's test-suite**: execution match of predicted vs gold results
  (multiset unless the gold query orders, column order free), per difficulty level.
* **Gold checked first.** Before any model call, every gold query is run through the same
  path a prediction takes (SQL guard, then the read-only engine). A question whose gold
  cannot run there is unanswerable under quarry's sandbox: it is left out of the
  denominator and listed in the report, never silently scored as a model error.

Writes ``docs/results/spider_dev_<model>.json`` and ``.md``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from rich.console import Console

from quarry.agent import AgentDeps, Outcome, QuarryAgent
from quarry.agent.nodes import LLM_UNAVAILABLE
from quarry.config import QuarryConfig, api_key
from quarry.db import Database, SafeExecutor
from quarry.errors import BenchmarkInterrupted, QueryFailed
from quarry.eval.hardness import hardness
from quarry.eval.matching import has_top_level_order, results_match
from quarry.eval.spider import Example, SpiderLayout, prepare_spider
from quarry.eval.stats import fmt_pct, hardware, pct
from quarry.llm import SQLiteLLMCache, build_openrouter_model

LEVELS = ("easy", "medium", "hard", "extra")
MAX_CONSECUTIVE_API_FAILURES = 5
COMPARE_ROW_LIMIT = 100_000
COMPARE_TIMEOUT_MS = 30_000


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Score:
    """One question's verdict. ``gold_failed`` questions are excluded, not counted wrong."""

    correct: bool
    note: str
    gold_failed: bool = False


def score_prediction(database: Database, gold_sql: str, pred_sql: str | None) -> Score:
    """Execution match of a prediction against gold on one database."""
    try:
        gold = database.engine.run(
            gold_sql, row_limit=COMPARE_ROW_LIMIT, timeout_ms=COMPARE_TIMEOUT_MS
        )
    except QueryFailed as exc:
        return Score(False, f"gold failed: {exc}", gold_failed=True)
    if not pred_sql:
        return Score(False, "no SQL")
    executor = SafeExecutor(database, row_limit=COMPARE_ROW_LIMIT, timeout_ms=COMPARE_TIMEOUT_MS)
    run = executor.run(pred_sql)
    if run.result is None:
        return Score(False, f"prediction failed: {run.error}")
    ok = results_match(gold.rows, run.result.rows, order_matters=has_top_level_order(gold_sql))
    return Score(ok, "" if ok else "results differ")


def gold_preflight(layout: SpiderLayout, examples: Sequence[Example]) -> dict[str, str]:
    """Run every gold query the way a prediction runs (guard, then the read-only engine).

    Returns ``{example id: error}`` for the gold queries that cannot run there.
    """
    failures: dict[str, str] = {}
    executors: dict[str, SafeExecutor] = {}
    for example in examples:
        if example.db_id not in executors:
            database = Database(layout.db_path(example.db_id), sample_values=0)
            executors[example.db_id] = SafeExecutor(
                database, row_limit=COMPARE_ROW_LIMIT, timeout_ms=COMPARE_TIMEOUT_MS
            )
        run = executors[example.db_id].run(example.gold_sql)
        if run.result is None:
            failures[example.id] = run.error or "unknown error"
    return failures


def _record(example: Example, outcome: Outcome, score: Score, ms: float) -> dict[str, Any]:
    first_failed = bool(outcome.attempts) and not outcome.attempts[0]["ok"]
    return {
        "id": example.id,
        "db_id": example.db_id,
        "hardness": hardness(example.gold_sql),
        "question": example.question,
        "gold": example.gold_sql,
        "pred": outcome.sql,
        "status": outcome.status,
        "correct": score.correct,
        "gold_failed": score.gold_failed,
        "note": score.note,
        "attempts": len(outcome.attempts),
        "first_attempt_failed": first_failed,
        "steps": outcome.steps,
        "tokens": outcome.tokens,
        "latency_ms": round(ms, 1),
    }


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[str(row["id"])] = row
    return done


def evaluate(
    examples: Sequence[Example],
    layout: SpiderLayout,
    llm: BaseChatModel,
    cfg: QuarryConfig,
    predictions: Path,
    *,
    trace_dir: Path | None,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the agent on every example not yet in ``predictions``; return all records."""
    predictions.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(predictions)
    todo = [e for e in examples if e.id not in done]
    by_db: dict[str, list[Example]] = defaultdict(list)
    for example in todo:
        by_db[example.db_id].append(example)
    failures_in_a_row = 0
    finished = len(examples) - len(todo)
    with predictions.open("a", encoding="utf-8") as sink:
        for db_id, group in by_db.items():
            database = Database(
                layout.db_path(db_id),
                sample_values=cfg.linking.sample_values,
                docs=layout.schemas[db_id].docs(),
            )
            agent = QuarryAgent(
                AgentDeps(llm=llm, database=database, config=cfg), trace_dir=trace_dir
            )
            for example in group:
                start = time.perf_counter()
                outcome = agent.ask(example.question)
                elapsed = (time.perf_counter() - start) * 1000
                if any(w.startswith(LLM_UNAVAILABLE) for w in outcome.warnings):
                    failures_in_a_row += 1
                    if failures_in_a_row >= MAX_CONSECUTIVE_API_FAILURES:
                        raise BenchmarkInterrupted(
                            f"{failures_in_a_row} questions in a row failed to reach the model "
                            f"({outcome.warnings[-1]}). Progress is saved in {predictions}; "
                            "rerun the same command to resume."
                        )
                    continue
                failures_in_a_row = 0
                score = score_prediction(database, example.gold_sql, outcome.sql)
                record = _record(example, outcome, score, elapsed)
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                done[example.id] = record
                finished += 1
                if on_progress is not None:
                    on_progress(finished, len(examples), record)
    return [done[e.id] for e in examples if e.id in done]


def summarize(
    records: Iterable[dict[str, Any]], gold_failures: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Accuracy over the questions whose gold runs; gold failures are reported separately.

    ``gold_failures`` are the questions the preflight excluded; a record whose gold failed
    at scoring time (``gold_failed``) is excluded the same way.
    """
    everything = list(records)
    rows = [r for r in everything if not r.get("gold_failed")]
    excluded = dict(gold_failures or {})
    excluded.update({str(r["id"]): str(r["note"]) for r in everything if r.get("gold_failed")})
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[row["hardness"]].append(row)
    repaired = [r for r in rows if r["first_attempt_failed"]]
    return {
        "questions": len(rows),
        "excluded_gold_failures": len(excluded),
        "gold_failures": excluded,
        "execution_accuracy": pct(sum(r["correct"] for r in rows), len(rows)),
        "by_hardness": {
            level: {
                "questions": len(by_level[level]),
                "execution_accuracy": pct(
                    sum(r["correct"] for r in by_level[level]), len(by_level[level])
                ),
            }
            for level in LEVELS
        },
        "status": dict(Counter(r["status"] for r in rows)),
        "mean_attempts": round(sum(r["attempts"] for r in rows) / len(rows), 3) if rows else None,
        "first_attempt_failed": len(repaired),
        "correct_after_repair": sum(r["correct"] for r in repaired),
        "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows), 1) if rows else None,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 1)
        if rows
        else None,
    }


def render_markdown(model: str, summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [
        f"# Spider dev execution accuracy: `{model}`",
        "",
        f"{summary['questions']} questions, run {meta['finished_at']} "
        f"on {meta['hardware']['cpu']}.",
        "",
        "| Difficulty | Questions | Execution accuracy |",
        "|---|---:|---:|",
    ]
    for level in LEVELS:
        entry = summary["by_hardness"][level]
        lines.append(f"| {level} | {entry['questions']} | {fmt_pct(entry['execution_accuracy'])} |")
    excluded = sorted(summary["gold_failures"])
    lines += [
        f"| **all** | **{summary['questions']}** | **{fmt_pct(summary['execution_accuracy'])}** |",
        "",
        "- Left out because their gold SQL does not run in quarry's sandbox (guard + read-only "
        f"engine): {len(excluded)}" + (f" ({', '.join(excluded)})." if excluded else "."),
        f"- Mean SQL attempts per question: {summary['mean_attempts']}; the first attempt failed "
        f"on {summary['first_attempt_failed']} questions and {summary['correct_after_repair']} "
        "of those ended correct after self-correction.",
        f"- Mean tokens per question: {summary['mean_tokens']}; mean latency "
        f"{summary['mean_latency_ms']} ms (includes the client-side rate limiter).",
        f"- Final status counts: {summary['status']}.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    cfg: QuarryConfig,
    *,
    root: Path,
    out: Path,
    limit: int | None,
    run_name: str | None,
    console: Console,
) -> Path:
    """Download Spider if needed, run (or resume) the benchmark, write the report."""
    api_key("required")  # fail before creating any files when the key is missing
    llm_cache = SQLiteLLMCache(Path("runs/bench/llm_cache.sqlite"))
    llm = build_openrouter_model(cfg.llm, cache=llm_cache)  # raises ConfigError without a key
    layout = prepare_spider(root)
    selected = layout.examples("dev")[:limit] if limit else layout.examples("dev")
    console.print(f"checking {len(selected)} gold queries in the sandbox before any model call")
    gold_failures = gold_preflight(layout, selected)
    for example_id, error in gold_failures.items():
        console.print(f"[yellow]left out[/] {example_id}: gold SQL fails in the sandbox ({error})")
    examples = [e for e in selected if e.id not in gold_failures]
    name = run_name or slug(cfg.llm.model)
    run_dir = Path("runs/bench") / name
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    def progress(done: int, total: int, record: dict[str, Any]) -> None:
        mark = "ok " if record["correct"] else "x  "
        console.print(f"[dim]{done:>5}/{total}[/] {mark}{record['id']} {record['question'][:70]}")

    records = evaluate(
        examples, layout, llm, cfg, run_dir / "predictions.jsonl",
        trace_dir=run_dir / "traces", on_progress=progress,
    )  # fmt: skip
    summary = summarize(records, gold_failures)
    meta = {
        "model": cfg.llm.model,
        "dataset": "Spider 1.0 dev",
        "limit": limit,
        "started_at": started,
        "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
        "llm_cache": {"hits": llm_cache.hits, "misses": llm_cache.misses},
        "config": cfg.model_dump(mode="json"),
    }
    out.mkdir(parents=True, exist_ok=True)
    stem = f"spider_dev_{name}" + (f"_first{limit}" if limit else "")
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(render_markdown(cfg.llm.model, summary, meta), encoding="utf-8")
    llm_cache.close()
    return json_path
