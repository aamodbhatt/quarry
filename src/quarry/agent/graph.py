"""The LangGraph ``StateGraph``: nodes, conditional edges and per-node tracing.

Routing functions are pure functions of the state. Nodes that can fail record whether a
retry is still allowed (``can_retry``: repairs and budget left), so the edges never need
the dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from quarry.agent import nodes
from quarry.agent.context import RunContext
from quarry.agent.state import AgentState
from quarry.tools import RUN_SQL

NodeFn = Callable[[AgentState, RunContext], dict[str, Any]]
Graph = StateGraph[AgentState, RunContext, AgentState, AgentState]

NODES: dict[str, NodeFn] = {
    "start_turn": nodes.start_turn,
    "link_schema": nodes.link_schema,
    "plan": nodes.plan,
    "explore": nodes.explore,
    "generate": nodes.generate,
    "check": nodes.check,
    "execute": nodes.execute,
    "verify": nodes.verify,
    "answer": nodes.answer,
    "chart": nodes.chart,
    "finish": nodes.finish,
}

_TERMINAL = {"budget_exhausted", "failed", "unanswerable", "refused"}


def _summarize(value: Any) -> Any:
    """Keep traces small: clip strings, count lists, describe results."""
    if isinstance(value, str):
        return value if len(value) <= 400 else value[:399] + "…"
    if isinstance(value, list):
        return {"items": len(value)}
    if isinstance(value, dict):
        if "row_count" in value:
            return {"row_count": value["row_count"], "columns": value.get("columns")}
        return {k: _summarize(v) for k, v in list(value.items())[:8]}
    return value


class TracedNode(Protocol):
    """A node LangGraph calls with the run context injected as ``runtime``."""

    def __call__(self, state: AgentState, *, runtime: Runtime[RunContext]) -> dict[str, Any]: ...


def traced(name: str, fn: NodeFn) -> TracedNode:
    """Wrap a node so it runs inside a ``node`` span of the run's trace."""

    def run(state: AgentState, *, runtime: Runtime[RunContext]) -> dict[str, Any]:
        ctx = runtime.context
        inputs = {"question": state.get("question")} if name == "start_turn" else {}
        attempt = len(state.get("attempts", [])) + 1
        with ctx.trace.span(name, "node", inputs, attempt=attempt) as span:
            update = fn(state, ctx)
            span.outputs = {k: _summarize(v) for k, v in update.items() if k != "work"}
        return update

    run.__name__ = name
    return run


def has_open_tool_calls(state: AgentState) -> bool:
    """Does the latest assistant message have schema-tool calls that are not answered yet?"""
    work = state.get("work", [])
    last_ai = next((m for m in reversed(work) if isinstance(m, AIMessage)), None)
    if last_ai is None or not last_ai.tool_calls:
        return False
    answered = {m.tool_call_id for m in work if isinstance(m, ToolMessage)}
    return any(c["id"] not in answered and c["name"] != RUN_SQL for c in last_ai.tool_calls)


def route_after_llm(state: AgentState) -> Literal["explore", "check", "generate", "answer"]:
    """After plan or generate: run tools, check a submitted query, or move on."""
    if state.get("status") in _TERMINAL:
        return "answer"
    if state.get("pending_sql") is not None:
        return "check"
    if has_open_tool_calls(state):
        return "explore"
    return "generate"


def route_after_explore(state: AgentState) -> Literal["plan", "generate"]:
    return "generate" if state.get("phase") == "generate" else "plan"


def route_after_check(state: AgentState) -> Literal["execute", "generate", "answer"]:
    if state.get("pending_sql") is not None:
        return "execute"
    return "generate" if state.get("can_retry") else "answer"


def route_after_execute(state: AgentState) -> Literal["verify", "generate", "answer"]:
    if state.get("pending_sql") is not None:
        return "verify"
    return "generate" if state.get("can_retry") else "answer"


def route_after_verify(state: AgentState) -> Literal["generate", "answer"]:
    attempts = state.get("attempts", [])
    if attempts and attempts[-1]["stage"] == "accepted":
        return "answer"
    return "generate" if state.get("can_retry") else "answer"


def build_graph() -> Graph:
    """Assemble the (uncompiled) graph. Dependencies arrive at run time as context."""
    graph: Graph = StateGraph(AgentState, context_schema=RunContext)
    for name, fn in NODES.items():
        graph.add_node(name, traced(name, fn))
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "link_schema")
    graph.add_edge("link_schema", "plan")
    graph.add_conditional_edges("plan", route_after_llm)
    graph.add_conditional_edges("explore", route_after_explore)
    graph.add_conditional_edges("generate", route_after_llm)
    graph.add_conditional_edges("check", route_after_check)
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("answer", "chart")
    graph.add_edge("chart", "finish")
    graph.add_edge("finish", END)
    return graph
