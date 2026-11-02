"""FastAPI app: a chat UI that streams every agent step over Server-Sent Events.

Endpoints
---------
``GET  /``                         the single-page UI (static files, no build step)
``GET  /api/info``                 database, dialect, tables, mode and example questions
``GET  /api/ask/stream?q=&thread=`` SSE: one ``step`` event per graph node, then ``done``
``POST /api/ask``                  the same run, returned as one JSON outcome
``GET  /api/trace/{thread}/{id}``  the JSONL trace of a finished run, as JSON
``GET  /api/history/{thread}``     earlier turns of a conversation
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quarry.agent import QuarryAgent
from quarry.demo.script import DEMO_QUESTIONS
from quarry.errors import QuarryError
from quarry.tracing import SAFE_ID, read_trace


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


def _sse(event: str, payload: str) -> str:
    return f"event: {event}\ndata: {payload}\n\n"


def create_app(agent: QuarryAgent, *, offline: bool) -> FastAPI:
    app = FastAPI(title="quarry", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    static_dir = Path(str(resources.files("quarry.web").joinpath("static")))
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        catalog = agent.deps.database.catalog()
        return {
            "database": agent.deps.database.name,
            "dialect": agent.deps.database.dialect,
            "tables": [
                {"name": t.name, "rows": t.row_count, "columns": [c.name for c in t.columns]}
                for t in catalog.tables
            ],
            "offline": offline,
            "model": "scripted demo (no LLM)" if offline else agent.deps.model_name,
            "examples": DEMO_QUESTIONS[:4],
        }

    @app.get("/api/ask/stream")
    def ask_stream(
        q: str = Query(min_length=1, max_length=2000),
        thread: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    ) -> StreamingResponse:
        def events() -> Iterator[str]:
            try:
                for event in agent.stream(q, thread_id=thread):
                    yield _sse("done" if event.kind == "done" else "step", event.to_json())
            except QuarryError as exc:
                yield _sse("error", json.dumps({"message": str(exc)}))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        try:
            return agent.ask(body.question, thread_id=body.thread_id).to_dict()
        except QuarryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/trace/{thread}/{trace_id}")
    def trace(thread: str, trace_id: str) -> list[dict[str, Any]]:
        if agent.trace_dir is None or not (
            SAFE_ID.fullmatch(thread) and SAFE_ID.fullmatch(trace_id)
        ):
            raise HTTPException(status_code=404, detail="no such trace")
        path = agent.trace_dir / thread / f"{trace_id}.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no such trace")
        return read_trace(path)

    @app.get("/api/history/{thread}")
    def history(thread: str) -> list[dict[str, Any]]:
        if not SAFE_ID.fullmatch(thread):
            raise HTTPException(status_code=404, detail="no such thread")
        return agent.history(thread)

    return app
