"""Structured run traces: one JSONL file per question, one line per span.

A span is a graph node, an LLM call, a tool call or a point event (for example an
injection finding). Spans nest through ``parent_id``. Every line is self-contained, so a
trace can be tailed while the run is still going and parsed with nothing but ``json``::

    {"trace_id": "...", "span_id": "...", "parent_id": "...", "name": "generate",
     "kind": "node", "start": "2026-11-02T10:00:00.123+00:00", "latency_ms": 812.4,
     "inputs": {...}, "outputs": {...}, "tokens_in": 1840, "tokens_out": 96,
     "tokens_estimated": false, "attempt": 2, "status": "ok", "error": null}
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SpanKind = Literal["run", "node", "llm", "tool", "event"]


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# Thread and trace ids become path components under the trace directory, so only plain
# ids are accepted: no separators, no dots, nothing absolute.
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def check_id(value: str, what: str = "thread id") -> str:
    """Return ``value`` if it is a safe id, else raise ``ValueError``."""
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {what} {value[:80]!r}: use 1-64 letters, digits, '_' or '-'")
    return value


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start: str
    latency_ms: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_estimated: bool = False
    attempt: int | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None


class Trace:
    """Collects the spans of one run and appends them to ``<dir>/<thread>/<trace>.jsonl``."""

    def __init__(
        self, directory: Path | None, *, thread_id: str, trace_id: str | None = None
    ) -> None:
        self.trace_id = check_id(trace_id, "trace id") if trace_id else new_id()
        self.thread_id = check_id(thread_id)
        self.path = (directory / thread_id / f"{self.trace_id}.jsonl") if directory else None
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind,
        inputs: dict[str, Any] | None = None,
        *,
        attempt: int | None = None,
    ) -> Iterator[Span]:
        span = Span(
            trace_id=self.trace_id,
            span_id=new_id(),
            parent_id=self._stack[-1] if self._stack else None,
            name=name,
            kind=kind,
            start=_now_iso(),
            inputs=inputs or {},
            attempt=attempt,
        )
        self._stack.append(span.span_id)
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            self._stack.pop()
            self._write(span)

    def event(self, name: str, attributes: dict[str, Any]) -> None:
        """Record a zero-duration span under the current span."""
        self._write(
            Span(
                trace_id=self.trace_id,
                span_id=new_id(),
                parent_id=self._stack[-1] if self._stack else None,
                name=name,
                kind="event",
                start=_now_iso(),
                outputs=attributes,
            )
        )

    def _write(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(span), default=str, ensure_ascii=False) + "\n")

    def totals(self) -> dict[str, Any]:
        """Aggregate numbers for the run (LLM calls, tool calls, tokens, latency)."""
        llm = [s for s in self.spans if s.kind == "llm"]
        tools = [s for s in self.spans if s.kind == "tool"]
        runs = [s for s in self.spans if s.kind == "run"]
        return {
            "trace_id": self.trace_id,
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tokens_in": sum(s.tokens_in for s in llm),
            "tokens_out": sum(s.tokens_out for s in llm),
            "tokens_estimated": any(s.tokens_estimated for s in llm),
            "latency_ms": runs[-1].latency_ms if runs else None,
            "path": str(self.path) if self.path else None,
        }


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL trace written by :class:`Trace`."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
