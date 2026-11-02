"""Graph state. Plain dicts and LangChain messages only, so the SQLite checkpointer can store it."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage

Status = Literal["running", "ok", "failed", "refused", "unanswerable", "budget_exhausted"]
Phase = Literal["plan", "generate"]
Stage = Literal["check", "execute", "verify", "accepted"]


class Turn(TypedDict):
    """One finished question in the conversation (what follow-ups get to see)."""

    question: str
    sql: str | None
    answer: str
    status: str
    row_count: int | None


class Attempt(TypedDict):
    """One SQL attempt and how far it got."""

    sql: str
    stage: Stage
    ok: bool
    feedback: str | None
    row_count: int | None
    elapsed_ms: float | None


class AgentState(TypedDict, total=False):
    # input
    question: str
    # memory across turns (appended by the reducer; persisted by the checkpointer)
    history: Annotated[list[Turn], operator.add]
    # per-turn working state (reset by start_turn)
    phase: Phase
    linked: dict[str, Any]
    schema_text: str
    plan: str
    work: list[BaseMessage]
    pending_sql: str | None
    pending_call_id: str | None
    attempts: list[Attempt]
    checks_used: list[str]
    result: dict[str, Any] | None
    answer: str
    chart: dict[str, Any] | None
    warnings: list[str]
    refused: bool
    can_retry: bool
    steps: int
    tokens: int
    explore_calls: int
    status: Status
