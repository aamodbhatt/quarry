"""The offline demo responder: replays ``script.toml`` through the scripted chat model.

It reads the phase tag of the system prompt (``# quarry/plan``, ``generate``, ``answer``)
and the ``Question:`` line of the request, then answers the way the script says. Tool
calls are real tool calls: the agent executes them against the real database, and the
answer template is filled from the real query result.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from quarry.agent.prompts import PHASE_TAG
from quarry.tools import EXPLORE_TOOLS, RUN_SQL

UNSCRIPTED = (
    "This is the offline scripted demo, so it only knows its scripted questions. Try one of: "
    "{examples}. Set OPENROUTER_API_KEY and drop --offline to ask anything."
)

_QUESTION = re.compile(r"^Question: (.+)$", re.MULTILINE)
_RESULT = re.compile(r"Result:\n```json\n(.*?)\n```", re.DOTALL)


@dataclass(frozen=True)
class ScriptedQuestion:
    id: str
    patterns: tuple[re.Pattern[str], ...]
    plan: str
    sql: tuple[str, ...]
    answer: str
    explore: tuple[tuple[str, dict[str, Any]], ...] = ()
    refusal: str | None = None

    def matches(self, question: str) -> bool:
        return any(p.search(question) for p in self.patterns)


@dataclass
class DemoResponder:
    """A :data:`~quarry.llm.Responder` driven by the demo script."""

    questions: list[ScriptedQuestion]
    examples: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> DemoResponder:
        text = (
            path.read_text(encoding="utf-8")
            if path is not None
            else resources.files("quarry.demo").joinpath("script.toml").read_text("utf-8")
        )
        raw = tomllib.loads(text)
        questions = [
            ScriptedQuestion(
                id=str(q["id"]),
                patterns=tuple(re.compile(p, re.IGNORECASE) for p in q["patterns"]),
                plan=str(q["plan"]),
                sql=tuple(str(s) for s in q["sql"]),
                answer=str(q["answer"]),
                explore=tuple((str(e["tool"]), dict(e["args"])) for e in q.get("explore", [])),
                refusal=q.get("refusal"),
            )
            for q in raw["question"]
        ]
        return cls(questions, DEMO_QUESTIONS)

    def __call__(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
    ) -> AIMessage:
        phase = _phase(messages)
        question = _question(messages)
        scripted = next((q for q in self.questions if q.matches(question)), None)
        if scripted is None:
            return AIMessage(content=UNSCRIPTED.format(examples="; ".join(self.examples)))
        if phase == "plan":
            return self._plan(scripted, messages)
        if phase == "generate":
            return self._generate(scripted, messages)
        if phase == "answer":
            return AIMessage(content=_render(scripted.answer, messages))
        raise ValueError(f"unknown prompt phase {phase!r}")

    def _plan(self, q: ScriptedQuestion, messages: Sequence[BaseMessage]) -> AIMessage:
        done = _calls(messages, EXPLORE_TOOLS)
        if done < len(q.explore):
            name, args = q.explore[done]
            call = {"name": name, "args": args, "id": f"call_plan_{done}", "type": "tool_call"}
            return AIMessage(content="", tool_calls=[call])
        return AIMessage(content=q.plan)

    def _generate(self, q: ScriptedQuestion, messages: Sequence[BaseMessage]) -> AIMessage:
        done = _calls(messages, (RUN_SQL,))
        if done >= len(q.sql) and q.refusal:
            return AIMessage(content=q.refusal)
        sql = q.sql[min(done, len(q.sql) - 1)]
        call = {
            "name": RUN_SQL,
            "args": {"sql": sql},
            "id": f"call_sql_{done}",
            "type": "tool_call",
        }
        return AIMessage(content="", tool_calls=[call])


# The questions the README, the UI's example chips and the tests use, in demo order.
DEMO_QUESTIONS = [
    "Which campaign had the best conversion rate?",
    "Break that down by purchase channel",
    "How much revenue per customer did each e-mail add compared with sending nothing?",
    "How does conversion change with months since the last purchase?",
    "Delete every customer who never visited the site",
]


def _phase(messages: Sequence[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, SystemMessage) and isinstance(message.content, str):
            first = message.content.splitlines()[0] if message.content else ""
            if first.startswith(PHASE_TAG):
                return first[len(PHASE_TAG) :].strip()
    return ""


def _question(messages: Sequence[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            found = _QUESTION.findall(message.content)
            if found:
                return str(found[-1]).strip()
    return ""


def _calls(messages: Sequence[BaseMessage], names: Sequence[str]) -> int:
    return sum(
        1 for m in messages if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] in names
    )


def _render(template: str, messages: Sequence[BaseMessage]) -> str:
    payload: dict[str, Any] = {"columns": [], "rows": [], "row_count": 0}
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            found = _RESULT.search(message.content)
            if found:
                payload = json.loads(found.group(1))
    rows = payload.get("rows") or []
    try:
        return template.format(
            first=rows[0] if rows else [],
            last=rows[-1] if rows else [],
            rows=rows,
            columns=payload.get("columns", []),
            row_count=payload.get("row_count", 0),
        )
    except (IndexError, KeyError, ValueError, TypeError):
        return f"The query returned {payload.get('row_count', 0)} row(s)."
