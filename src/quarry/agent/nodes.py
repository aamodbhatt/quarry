"""Graph nodes. Each takes the state and the run context and returns a partial update.

Flow (see ``graph.py`` for the edges)::

    start_turn -> link_schema -> plan <-> explore -> generate <-> explore
        -> check -> execute -> verify -> answer -> chart -> finish
    with check/execute/verify looping back to generate while repairs remain.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import openai
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from quarry.agent import prompts
from quarry.agent.chart import chart_spec
from quarry.agent.checks import static_errors, suspicions
from quarry.agent.context import AgentDeps, RunContext
from quarry.agent.state import AgentState, Attempt, Stage, Turn
from quarry.db import render_schema
from quarry.tools import EXPLORE_TOOLS, RUN_SQL, sanitize_table

Update = dict[str, Any]

_SQL_FENCE = re.compile(r"```(?:sql|sqlite|duckdb)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMUnavailable(RuntimeError):
    """The model API failed after the client's own retries."""


# --------------------------------------------------------------------------- helpers


def budget_left(state: AgentState, deps: AgentDeps) -> bool:
    cfg = deps.config.agent
    return state.get("steps", 0) < cfg.max_steps and state.get("tokens", 0) < cfg.max_tokens


def repairs_left(state: AgentState, deps: AgentDeps) -> bool:
    return len(state.get("attempts", [])) < 1 + deps.config.agent.max_repairs


def _can_retry(state: AgentState, deps: AgentDeps, attempts: list[Attempt]) -> bool:
    """After recording ``attempts``: is there a repair and budget left for another try?"""
    return len(attempts) < 1 + deps.config.agent.max_repairs and budget_left(state, deps)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
    return "".join(parts).strip()


def call_llm(
    ctx: RunContext, phase: str, messages: Sequence[BaseMessage], *, with_tools: bool
) -> tuple[AIMessage, int]:
    """Invoke the model inside an ``llm`` span; return the reply and tokens used."""
    deps = ctx.deps
    runnable = deps.llm.bind_tools(list(deps.tools.values())) if with_tools else deps.llm
    with ctx.trace.span(
        f"llm:{phase}", "llm", {"model": deps.model_name, "messages": len(messages)}
    ) as span:
        try:
            reply = runnable.invoke(list(messages))
        except openai.OpenAIError as exc:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(reply, AIMessage):
            raise LLMUnavailable(f"model returned {type(reply).__name__}, not an AIMessage")
        usage = reply.usage_metadata
        tokens_in = usage["input_tokens"] if usage else 0
        tokens_out = usage["output_tokens"] if usage else 0
        span.tokens_in, span.tokens_out = tokens_in, tokens_out
        span.tokens_estimated = deps.tokens_estimated
        span.outputs = {
            "tool_calls": [c["name"] for c in reply.tool_calls],
            "text": _text(reply)[:300],
        }
    return reply, tokens_in + tokens_out


LLM_UNAVAILABLE = "llm_unavailable"


def _llm_failed(state: AgentState, exc: LLMUnavailable) -> Update:
    return {
        "status": "failed",
        "answer": f"The language model could not be reached ({exc}).",
        "pending_sql": None,
        "steps": state.get("steps", 0) + 1,
        # A machine-readable marker: the benchmark retries such questions on resume.
        "warnings": [*state.get("warnings", []), f"{LLM_UNAVAILABLE}: {exc}"],
    }


def _absorb_reply(state: AgentState, reply: AIMessage) -> Update:
    """Fold a tool-calling reply into the transcript; pick out a run_sql submission."""
    work = [*state.get("work", []), reply]
    run_calls = [c for c in reply.tool_calls if c["name"] == RUN_SQL]
    if not run_calls:
        return {"work": work}
    chosen = run_calls[0]
    skipped = [
        ToolMessage("Skipped: submit run_sql on its own.", tool_call_id=c["id"] or "")
        for c in reply.tool_calls
        if c is not chosen
    ]
    return {
        "work": work + skipped,
        "pending_sql": str(chosen["args"].get("sql", "")).strip(),
        "pending_call_id": chosen["id"] or f"call_{len(state.get('attempts', []))}",
    }


def _finding_warnings(ctx: RunContext, artifact: dict[str, Any], source: str) -> list[str]:
    findings = artifact.get("injection_findings") or []
    for finding in findings:
        ctx.trace.event("injection_withheld", {"source": source, **finding})
    if not findings:
        return []
    return [
        f"{len(findings)} value(s) from {source} looked like prompt injection and were "
        "withheld from the model."
    ]


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> ToolCall:
    return ToolCall(name=name, args=args, id=call_id, type="tool_call")


# --------------------------------------------------------------------------- nodes


def start_turn(state: AgentState, ctx: RunContext) -> Update:
    """Reset per-question fields; conversation history is left alone."""
    return {
        "phase": "plan",
        "linked": {},
        "schema_text": "",
        "plan": "",
        "work": [],
        "pending_sql": None,
        "pending_call_id": None,
        "attempts": [],
        "checks_used": [],
        "result": None,
        "answer": "",
        "chart": None,
        "warnings": [],
        "refused": False,
        "can_retry": True,
        "steps": 0,
        "tokens": 0,
        "explore_calls": 0,
        "status": "running",
    }


def link_schema(state: AgentState, ctx: RunContext) -> Update:
    """Rank tables/columns for the question (plus the previous question, for follow-ups)."""
    deps = ctx.deps
    history = state.get("history", [])
    previous = history[-1] if history else None
    link = deps.linker.link(
        state["question"],
        context=previous["question"] if previous else "",
        top_tables=deps.config.linking.top_tables,
    )
    selected = list(link.selected)
    if previous and previous["sql"]:
        catalog = deps.database.catalog()
        for name in sorted(deps.executor.guard.check(previous["sql"]).tables):
            table = catalog.table(name)
            if table is not None and table.name not in selected:
                selected.append(table.name)
    schema_text = render_schema(
        deps.database.catalog(),
        selected,
        columns=link.columns_by_table(deps.config.linking.top_columns),
        detector=deps.detector,
    )
    return {
        "schema_text": schema_text,
        "linked": {
            "tables": [{"name": t.name, "score": round(t.score, 5)} for t in link.tables[:8]],
            "columns": [[c.table, c.column] for c in link.columns[:10]],
            "selected": selected,
            "value_hits": [list(v) for v in link.value_hits],
        },
    }


def plan(state: AgentState, ctx: RunContext) -> Update:
    """Explore with schema tools if needed, then write a short plan."""
    deps = ctx.deps
    if not budget_left(state, deps):
        return {"status": "budget_exhausted"}
    history = prompts.render_history(state.get("history", []), deps.config.agent.history_turns)
    messages: list[BaseMessage] = [
        SystemMessage(
            prompts.plan_system(deps.database.dialect, deps.config.agent.max_explore_calls)
        ),
        HumanMessage(prompts.plan_request(state["question"], state["schema_text"], history)),
        *state.get("work", []),
    ]
    try:
        reply, used = call_llm(ctx, "plan", messages, with_tools=True)
    except LLMUnavailable as exc:
        return _llm_failed(state, exc)
    update: Update = {
        "phase": "plan",
        "steps": state.get("steps", 0) + 1,
        "tokens": state.get("tokens", 0) + used,
    }
    if reply.tool_calls:
        update.update(_absorb_reply(state, reply))
    else:
        update["plan"] = _text(reply)
        update["phase"] = "generate"
    return update


def explore(state: AgentState, ctx: RunContext) -> Update:
    """Run the schema tools the model asked for (bounded by ``max_explore_calls``)."""
    deps = ctx.deps
    last = state["work"][-1]
    assert isinstance(last, AIMessage)  # noqa: S101 - routing only sends tool-calling replies here
    answered = {m.tool_call_id for m in state["work"] if isinstance(m, ToolMessage)}
    used = state.get("explore_calls", 0)
    replies: list[BaseMessage] = []
    warnings = list(state.get("warnings", []))
    executed = 0
    for call in last.tool_calls:
        call_id = call["id"] or ""
        if call_id in answered or call["name"] == RUN_SQL:
            continue
        if call["name"] not in EXPLORE_TOOLS:
            replies.append(ToolMessage(f"Unknown tool {call['name']!r}.", tool_call_id=call_id))
            continue
        if used >= deps.config.agent.max_explore_calls:
            replies.append(
                ToolMessage(
                    "Tool budget used up: write the query now and submit it with run_sql.",
                    tool_call_id=call_id,
                )
            )
            continue
        with ctx.trace.span(f"tool:{call['name']}", "tool", dict(call["args"])) as span:
            message = deps.tools[call["name"]].invoke(
                _tool_call(call["name"], call["args"], call_id)
            )
            span.outputs = {"chars": len(str(message.content))}
        assert isinstance(message, ToolMessage)  # noqa: S101 - StructuredTool contract
        warnings += _finding_warnings(ctx, message.artifact or {}, call["name"])
        replies.append(message)
        used += 1
        executed += 1
    return {
        "work": [*state["work"], *replies],
        "explore_calls": used,
        "steps": state.get("steps", 0) + executed,
        "warnings": warnings,
    }


def generate(state: AgentState, ctx: RunContext) -> Update:
    """Ask the model for SQL (a run_sql call); repairs see every earlier error in ``work``."""
    deps = ctx.deps
    if not budget_left(state, deps):
        return {"status": "budget_exhausted"}
    history = prompts.render_history(state.get("history", []), deps.config.agent.history_turns)
    messages: list[BaseMessage] = [
        SystemMessage(prompts.generate_system(deps.database.dialect)),
        HumanMessage(
            prompts.generate_request(
                state["question"], state["schema_text"], state.get("plan", ""), history
            )
        ),
        *state.get("work", []),
    ]
    try:
        reply, used = call_llm(ctx, "generate", messages, with_tools=True)
    except LLMUnavailable as exc:
        return _llm_failed(state, exc)
    update: Update = {
        "phase": "generate",
        "steps": state.get("steps", 0) + 1,
        "tokens": state.get("tokens", 0) + used,
    }
    if reply.tool_calls:
        update.update(_absorb_reply(state, reply))
        return update
    text = _text(reply)
    fenced = _SQL_FENCE.search(text)
    if fenced and fenced.group(1).strip():
        # Some models answer in prose with a ```sql block instead of calling the tool.
        call_id = f"text_{len(state.get('attempts', []))}"
        sql = fenced.group(1).strip()
        as_call = AIMessage(content=text, tool_calls=[_tool_call(RUN_SQL, {"sql": sql}, call_id)])
        update.update({"work": [*state.get("work", []), as_call], "pending_sql": sql})
        update["pending_call_id"] = call_id
        return update
    update.update(
        {
            "work": [*state.get("work", []), reply],
            "answer": text or "The model returned an empty reply.",
            "status": "refused" if state.get("refused") else "unanswerable",
        }
    )
    return update


def check(state: AgentState, ctx: RunContext) -> Update:
    """Guard (security) and static check (correctness) before anything runs."""
    deps = ctx.deps
    sql = state.get("pending_sql") or ""
    verdict = deps.executor.guard.check(sql)
    if not verdict.allowed:
        reasons = "; ".join(str(v) for v in verdict.violations)
        ctx.trace.event("guard_refused", {"sql": sql, "violations": reasons})
        feedback = (
            f"ERROR: refused by quarry's read-only SQL guard ({reasons}). Only one SELECT "
            "can run. If the user asked to change data, explain in plain text that quarry "
            "is read-only."
        )
        return _attempt_failed(state, deps, sql, "check", feedback, refused=True)
    errors = static_errors(sql, deps.database.catalog())
    if errors:
        feedback = (
            f"ERROR: static check failed: {'; '.join(errors)}. Check names with describe_table."
        )
        return _attempt_failed(state, deps, sql, "check", feedback)
    return {"pending_sql": sql}


def _attempt_failed(
    state: AgentState,
    deps: AgentDeps,
    sql: str,
    stage: Stage,
    feedback: str,
    *,
    refused: bool = False,
) -> Update:
    attempt = Attempt(
        sql=sql, stage=stage, ok=False, feedback=feedback, row_count=None, elapsed_ms=None
    )
    attempts = [*state.get("attempts", []), attempt]
    message = ToolMessage(feedback, tool_call_id=state.get("pending_call_id") or "")
    update: Update = {
        "attempts": attempts,
        "work": [*state.get("work", []), message],
        "pending_sql": None,
        "can_retry": _can_retry(state, deps, attempts),
    }
    if refused:
        update["refused"] = True
    return update


def execute(state: AgentState, ctx: RunContext) -> Update:
    """Run the statement through the run_sql tool (guard again, then the read-only engine)."""
    deps = ctx.deps
    sql = state.get("pending_sql") or ""
    call_id = state.get("pending_call_id") or "run_sql"
    with ctx.trace.span("tool:run_sql", "tool", {"sql": sql}) as span:
        message = deps.tools[RUN_SQL].invoke(_tool_call(RUN_SQL, {"sql": sql}, call_id))
        assert isinstance(message, ToolMessage)  # noqa: S101 - StructuredTool contract
        artifact: dict[str, Any] = message.artifact or {}
        span.outputs = {
            "row_count": artifact.get("row_count"),
            "error": artifact.get("error"),
            "elapsed_ms": artifact.get("elapsed_ms"),
        }
    steps = state.get("steps", 0) + 1
    if "error" in artifact:
        attempt = Attempt(
            sql=sql,
            stage="execute",
            ok=False,
            feedback=str(artifact["error"]),
            row_count=None,
            elapsed_ms=None,
        )
        attempts = [*state.get("attempts", []), attempt]
        return {
            "attempts": attempts,
            "work": [*state.get("work", []), message],
            "pending_sql": None,
            "steps": steps,
            "can_retry": _can_retry({**state, "steps": steps}, deps, attempts),
            "refused": bool(state.get("refused")) or bool(artifact.get("blocked")),
        }
    warnings = list(state.get("warnings", []))
    warnings += _finding_warnings(ctx, artifact, "the query result")
    if artifact.get("truncated"):
        warnings.append(
            f"The result was cut at {deps.config.sql.row_limit} rows (the sandbox row limit)."
        )
    preview = sanitize_table(
        artifact["columns"],
        artifact["rows"][: deps.config.agent.preview_rows],
        deps.detector,
        deps.config.sql.max_value_chars,
    )
    result = {
        "sql": sql,
        "columns": artifact["columns"],
        "rows": artifact["rows"],
        "preview_columns": preview.columns,
        "preview": preview.rows,
        "row_count": artifact["row_count"],
        "truncated": artifact["truncated"],
        "elapsed_ms": artifact["elapsed_ms"],
    }
    return {
        "result": result,
        "work": [*state.get("work", []), message],
        "steps": steps,
        "warnings": warnings,
    }


def verify(state: AgentState, ctx: RunContext) -> Update:
    """Accept the result, or send it back once per suspicious pattern."""
    deps = ctx.deps
    result = state["result"]
    assert result is not None  # noqa: S101 - routing only sends successful runs here
    sql = state.get("pending_sql") or result["sql"]
    flagged_before = {a["sql"] for a in state.get("attempts", []) if a["stage"] == "verify"}
    found = []
    if sql not in flagged_before and repairs_left(state, deps):
        found = suspicions(
            state["question"],
            sql,
            deps.database.dialect,
            result["columns"],
            result["rows"],
            state.get("checks_used", []),
        )
    if found:
        issue = found[0]
        ctx.trace.event("result_suspicious", {"kind": issue.kind, "sql": sql})
        attempt = Attempt(
            sql=sql,
            stage="verify",
            ok=False,
            feedback=issue.message,
            row_count=result["row_count"],
            elapsed_ms=result["elapsed_ms"],
        )
        attempts = [*state.get("attempts", []), attempt]
        return {
            "attempts": attempts,
            "checks_used": [*state.get("checks_used", []), issue.kind],
            "work": [
                *state.get("work", []),
                HumanMessage(f"Check before answering: {issue.message}"),
            ],
            "pending_sql": None,
            "can_retry": _can_retry(state, deps, attempts),
        }
    attempt = Attempt(
        sql=sql,
        stage="accepted",
        ok=True,
        feedback=None,
        row_count=result["row_count"],
        elapsed_ms=result["elapsed_ms"],
    )
    return {"attempts": [*state.get("attempts", []), attempt], "pending_sql": None}


def answer(state: AgentState, ctx: RunContext) -> Update:
    """Write the final answer from the accepted result, or explain what went wrong."""
    deps = ctx.deps
    result = state.get("result")
    status = state.get("status", "running")
    if status in {"unanswerable", "refused", "failed"} and state.get("answer"):
        return {"status": status}
    if result is None:
        if status == "budget_exhausted":
            final = "budget_exhausted"
        elif state.get("refused"):
            final = "refused"
        else:
            final = "failed"
        return {"status": final, "answer": _failure_text(state)}
    warnings = list(state.get("warnings", []))
    if not any(a["stage"] == "accepted" for a in state.get("attempts", [])):
        warnings.append("quarry ran out of repair attempts; this result was flagged as suspicious.")
    if status == "budget_exhausted" or not budget_left(state, deps):
        return {"status": "ok", "answer": _fallback_answer(result), "warnings": warnings}
    messages = [
        SystemMessage(prompts.answer_system()),
        HumanMessage(prompts.answer_request(state["question"], result["sql"], result, warnings)),
    ]
    try:
        reply, used = call_llm(ctx, "answer", messages, with_tools=False)
    except LLMUnavailable:
        return {"status": "ok", "answer": _fallback_answer(result), "warnings": warnings}
    return {
        "status": "ok",
        "answer": _text(reply) or _fallback_answer(result),
        "steps": state.get("steps", 0) + 1,
        "tokens": state.get("tokens", 0) + used,
        "warnings": warnings,
    }


def chart(state: AgentState, ctx: RunContext) -> Update:
    result = state.get("result")
    if state.get("status") != "ok" or result is None:
        return {"chart": None}
    return {"chart": chart_spec(result["columns"], result["rows"], title=state["question"])}


def finish(state: AgentState, ctx: RunContext) -> Update:
    """Append this question to the conversation memory."""
    result = state.get("result")
    accepted = [a for a in state.get("attempts", []) if a["stage"] == "accepted"]
    sql = accepted[-1]["sql"] if accepted else (result["sql"] if result else None)
    turn = Turn(
        question=state["question"],
        sql=sql if state.get("status") == "ok" else None,
        answer=state.get("answer", ""),
        status=state.get("status", "failed"),
        row_count=result["row_count"] if result else None,
    )
    return {"history": [turn]}


# --------------------------------------------------------------------------- fallbacks


def _fallback_answer(result: dict[str, Any]) -> str:
    """A plain summary used when the token/step budget leaves no room for an LLM answer."""
    rows = result["preview"]
    if not rows:
        return "The query returned no rows."
    columns = result["preview_columns"]
    head = ", ".join(f"{c} = {v}" for c, v in zip(columns, rows[0], strict=False))
    more = f" ({result['row_count']} rows in total)" if result["row_count"] > 1 else ""
    return f"First row: {head}{more}."


def _failure_text(state: AgentState) -> str:
    attempts = state.get("attempts", [])
    if state.get("status") == "budget_exhausted":
        return "quarry stopped because this question used up its step or token budget."
    if state.get("refused"):
        return (
            "quarry only runs read-only queries, so this request was refused by the SQL "
            "guard and nothing in the database was changed."
        )
    if not attempts:
        return "quarry could not produce a query for this question."
    last = attempts[-1]
    return (
        f"quarry could not get a working query after {len(attempts)} attempt(s). "
        f"Last problem: {last['feedback']}"
    )
