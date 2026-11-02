"""Run the graph: stream step events, return a final outcome, keep conversation memory.

``QuarryAgent`` is the one entry point the CLI, the web server, the MCP server and the
benchmark share. Memory is a LangGraph checkpointer keyed by ``thread_id``: the same id
continues a conversation (follow-ups see earlier turns), a new id starts fresh.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from quarry.agent.context import AgentDeps, RunContext
from quarry.agent.graph import build_graph
from quarry.config import QuarryConfig
from quarry.db import Database
from quarry.llm import Responder, build_chat_model
from quarry.tracing import Trace, check_id, export_callbacks, new_id

RECURSION_LIMIT = 80
MAX_ROWS_RETURNED = 200


@dataclass(frozen=True)
class AgentEvent:
    """One step of a run, as shown by the CLI and streamed to the web UI."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, **self.data}, default=str)


@dataclass
class Outcome:
    """The final, JSON-ready result of one question."""

    thread_id: str
    question: str
    status: str
    answer: str
    sql: str | None
    plan: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    chart: dict[str, Any] | None
    attempts: list[dict[str, Any]]
    warnings: list[str]
    linked: dict[str, Any]
    steps: int
    tokens: int
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuarryAgent:
    """A compiled graph plus its dependencies, memory and trace settings."""

    def __init__(
        self,
        deps: AgentDeps,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        trace_dir: Path | None = None,
    ) -> None:
        self.deps = deps
        self.callbacks = callbacks or []
        self.trace_dir = trace_dir
        self.graph = build_graph().compile(checkpointer=checkpointer or InMemorySaver())

    def stream(self, question: str, *, thread_id: str | None = None) -> Iterator[AgentEvent]:
        """Run one question, yielding an event per graph step and a final ``done`` event."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        thread = new_id() if thread_id is None else check_id(thread_id)
        trace = Trace(self.trace_dir, thread_id=thread)
        config: RunnableConfig = {
            "configurable": {"thread_id": thread},
            "callbacks": self.callbacks,
            "recursion_limit": RECURSION_LIMIT,
        }
        context = RunContext(self.deps, trace)
        seen_work = 0
        with trace.span("run", "run", {"question": question, "thread_id": thread}) as run_span:
            for chunk in self.graph.stream(
                {"question": question}, config, context=context, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    if node == "start_turn":
                        seen_work = 0
                    event = _event(node, update, seen_work)
                    if "work" in update:
                        seen_work = len(update["work"])
                    yield event
            values = self.graph.get_state(config).values
            outcome = _outcome(thread, question, values, trace)
            run_span.outputs = {"status": outcome.status, "row_count": outcome.row_count}
        outcome.trace = trace.totals()
        yield AgentEvent("done", outcome.to_dict())

    def ask(self, question: str, *, thread_id: str | None = None) -> Outcome:
        """Run one question to completion and return its outcome."""
        final: dict[str, Any] | None = None
        for event in self.stream(question, thread_id=thread_id):
            if event.kind == "done":
                final = event.data
        if final is None:
            raise RuntimeError("the graph finished without a final state")
        return Outcome(**final)

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """Earlier turns of a conversation (empty for an unknown thread)."""
        config: RunnableConfig = {"configurable": {"thread_id": check_id(thread_id)}}
        values = self.graph.get_state(config).values
        return [dict(t) for t in values.get("history", [])]


def _new_messages(update: dict[str, Any], seen: int) -> list[BaseMessage]:
    work = update.get("work")
    return list(work[seen:]) if isinstance(work, list) else []


def _event(node: str, update: dict[str, Any], seen_work: int) -> AgentEvent:
    """Turn a node's state update into a compact, UI-friendly event."""
    data: dict[str, Any] = {}
    fresh = _new_messages(update, seen_work)
    if node == "link_schema":
        data = dict(update.get("linked", {}))
    elif node in {"plan", "generate"}:
        calls = [
            {"name": c["name"], "args": c["args"]}
            for m in fresh
            if isinstance(m, AIMessage)
            for c in m.tool_calls
        ]
        data = {"tool_calls": calls}
        if update.get("plan"):
            data["plan"] = update["plan"]
        if update.get("pending_sql") is not None:
            data["sql"] = update["pending_sql"]
        if update.get("answer"):
            data["answer"] = update["answer"]
        if update.get("status"):
            data["status"] = update["status"]
    elif node == "explore":
        data = {
            "tools": [
                {"tool_call_id": m.tool_call_id, "content": str(m.content)[:1200]}
                for m in fresh
                if isinstance(m, ToolMessage)
            ]
        }
    elif node == "check":
        attempts = update.get("attempts")
        data = {"ok": update.get("pending_sql") is not None}
        if attempts:
            data["feedback"] = attempts[-1]["feedback"]
    elif node == "execute":
        result = update.get("result")
        if result:
            data = {
                "ok": True,
                "columns": result["columns"],
                "rows": result["rows"][:50],
                "row_count": result["row_count"],
                "truncated": result["truncated"],
                "elapsed_ms": result["elapsed_ms"],
            }
        else:
            attempts = update.get("attempts") or [{}]
            data = {"ok": False, "error": attempts[-1].get("feedback")}
    elif node == "verify":
        attempts = update.get("attempts") or [{}]
        last = attempts[-1]
        data = {"accepted": last.get("stage") == "accepted", "feedback": last.get("feedback")}
    elif node == "answer":
        data = {"answer": update.get("answer", ""), "status": update.get("status")}
    elif node == "chart":
        data = {"chart": update.get("chart")}
    return AgentEvent(node, data)


def _outcome(thread: str, question: str, values: dict[str, Any], trace: Trace) -> Outcome:
    result = values.get("result") or {}
    attempts = list(values.get("attempts", []))
    accepted = [a for a in attempts if a["stage"] == "accepted"]
    sql = accepted[-1]["sql"] if accepted else result.get("sql")
    return Outcome(
        thread_id=thread,
        question=question,
        status=str(values.get("status", "failed")),
        answer=str(values.get("answer", "")),
        sql=sql,
        plan=str(values.get("plan", "")),
        columns=list(result.get("columns", [])),
        rows=[list(r) for r in result.get("rows", [])[:MAX_ROWS_RETURNED]],
        row_count=int(result.get("row_count", 0)),
        truncated=bool(result.get("truncated", False)),
        chart=values.get("chart"),
        attempts=[dict(a) for a in attempts],
        warnings=list(values.get("warnings", [])),
        linked=dict(values.get("linked", {})),
        steps=int(values.get("steps", 0)),
        tokens=int(values.get("tokens", 0)),
        trace=trace.totals(),
    )


@contextmanager
def open_agent(
    config: QuarryConfig,
    *,
    db_path: Path | None = None,
    offline: bool = False,
    responder: Responder | None = None,
    memory_path: Path | None = None,
    cache: BaseCache | None = None,
    trace_dir: Path | None = None,
) -> Iterator[QuarryAgent]:
    """Build an agent for one database; ``memory_path`` persists conversations in SQLite."""
    database = Database(db_path or config.database, sample_values=config.linking.sample_values)
    if offline and responder is None:
        from quarry.demo.script import DemoResponder

        responder = DemoResponder.load()
    llm = build_chat_model(config.llm, offline=offline, responder=responder, cache=cache)
    deps = AgentDeps(llm=llm, database=database, config=config)
    connection: sqlite3.Connection | None = None
    checkpointer: BaseCheckpointSaver[Any]
    if memory_path is not None:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(memory_path), check_same_thread=False)
        checkpointer = SqliteSaver(connection)
    else:
        checkpointer = InMemorySaver()
    try:
        yield QuarryAgent(
            deps,
            checkpointer,
            callbacks=export_callbacks(config.tracing),
            trace_dir=trace_dir if trace_dir is not None else config.tracing.dir,
        )
    finally:
        if connection is not None:
            connection.close()
