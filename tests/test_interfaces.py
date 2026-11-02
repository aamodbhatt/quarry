"""Web UI (FastAPI + SSE), MCP server and CLI, all against the offline scripted model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from mcp import Client
from typer.testing import CliRunner

import quarry.cli
from quarry.agent import AgentDeps, QuarryAgent, open_agent
from quarry.cli import app as cli_app
from quarry.config import QuarryConfig
from quarry.db import Database
from quarry.demo import DEMO_QUESTIONS
from quarry.guard import WITHHELD
from quarry.llm import QueueResponder, ScriptedChatModel, tool_call_message
from quarry.mcp_server import build_mcp_server
from quarry.web import create_app


@pytest.fixture
def agent(warehouse: Path, config: QuarryConfig) -> Iterator[QuarryAgent]:
    with open_agent(config, db_path=warehouse, offline=True) as built:
        yield built


def _sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_web_streams_steps_and_serves_traces(agent: QuarryAgent) -> None:
    client = TestClient(create_app(agent, offline=True))
    info = client.get("/api/info").json()
    assert info["offline"] is True and info["dialect"] == "sqlite"
    assert info["examples"] == DEMO_QUESTIONS[:4]
    assert "quarry" in client.get("/").text

    response = client.get("/api/ask/stream", params={"q": DEMO_QUESTIONS[0], "thread": "web1"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse(response.text)
    kinds = [data["kind"] for _, data in events]
    assert events[-1][0] == "done" and kinds.count("generate") == 2
    done = events[-1][1]
    assert done["status"] == "ok"

    trace = done["trace"]
    assert isinstance(trace, dict)
    spans = client.get(f"/api/trace/web1/{trace['trace_id']}").json()
    assert any(s["kind"] == "llm" for s in spans)
    assert client.get("/api/trace/web1/..%2F..%2Fsecret").status_code == 404
    assert client.get("/api/history/web1").json()[0]["question"] == DEMO_QUESTIONS[0]


def test_web_post_and_validation(agent: QuarryAgent) -> None:
    client = TestClient(create_app(agent, offline=True))
    body = client.post("/api/ask", json={"question": DEMO_QUESTIONS[4]}).json()
    assert body["status"] == "refused"
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.get("/api/ask/stream", params={"q": "x", "thread": "../x"}).status_code == 422


def test_mcp_server_exposes_read_only_tools(agent: QuarryAgent, tmp_path: Path) -> None:
    server = build_mcp_server(agent)
    outside = tmp_path / "outside"

    async def scenario() -> None:
        async with Client(server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {"list_tables", "describe_table", "sample_rows", "run_sql", "ask"}
            annotations = tools["run_sql"].annotations
            assert annotations is not None and annotations.read_only_hint is True
            blocked = await client.call_tool("run_sql", {"sql": "DROP TABLE customers"})
            assert "refused by the SQL guard" in blocked.content[0].text  # type: ignore[union-attr]
            rows = await client.call_tool("run_sql", {"sql": "SELECT COUNT(*) AS n FROM customers"})
            assert "600" in rows.content[0].text  # type: ignore[union-attr]
            answer = await client.call_tool("ask", {"question": DEMO_QUESTIONS[0]})
            payload = answer.structured_content or json.loads(answer.content[0].text)  # type: ignore[union-attr]
            assert payload["status"] == "ok"
            # thread ids become trace paths: nothing may point outside the trace directory
            for thread in ("../escaped", str(outside), "a/b", "x\n"):
                bad = await client.call_tool("ask", {"question": "hi", "thread_id": thread})
                assert bad.is_error, thread

    anyio.run(scenario)
    assert not (tmp_path / "escaped").exists() and not outside.exists()


def test_mcp_ask_sanitises_rows_for_the_calling_model(shop_db: Path, config: QuarryConfig) -> None:
    run_sql = tool_call_message("run_sql", {"sql": "SELECT body FROM reviews"}, "c1")
    replies: list[AIMessage | str] = ["plan", run_sql, "Here are the reviews."]
    deps = AgentDeps(
        llm=ScriptedChatModel(responder=QueueResponder(replies)),
        database=Database(shop_db),
        config=config,
    )
    server = build_mcp_server(QuarryAgent(deps, trace_dir=None))

    async def scenario() -> dict[str, object]:
        async with Client(server) as client:
            answer = await client.call_tool("ask", {"question": "List every review body."})
            payload = answer.structured_content or json.loads(answer.content[0].text)  # type: ignore[union-attr]
            assert isinstance(payload, dict)
            return payload

    payload = anyio.run(scenario)
    rows = payload["rows"]
    assert rows == [["Great grip on wet rocks."], [WITHHELD]]
    assert any("withheld" in str(w) for w in payload["warnings"])  # type: ignore[attr-defined]


def test_cli_ask_json_and_errors(
    warehouse: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_app, ["ask", DEMO_QUESTIONS[2], "--db", str(warehouse), "--offline", "--json"]
    )
    assert result.exit_code == 0, result.output
    outcome = json.loads(result.stdout)
    assert outcome["status"] == "ok" and outcome["columns"][0] == "campaign"

    pretty = runner.invoke(cli_app, ["ask", DEMO_QUESTIONS[0], "--db", str(warehouse), "--offline"])
    assert pretty.exit_code == 0 and "converted best" in pretty.stdout

    missing = runner.invoke(
        cli_app, ["ask", "hi", "--db", str(tmp_path / "none.sqlite"), "--offline"]
    )
    assert missing.exit_code == 2

    no_key = runner.invoke(cli_app, ["ask", "hi", "--db", str(warehouse)])
    assert no_key.exit_code == 2 and "OPENROUTER_API_KEY" in no_key.output

    escape = runner.invoke(
        cli_app, ["ask", "hi", "--db", str(warehouse), "--offline", "--thread", "../x"]
    )
    assert escape.exit_code == 2 and "invalid thread id" in escape.output


def test_cli_reads_dotenv_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded: list[str] = []

    def record(path: str) -> bool:
        loaded.append(path)
        return True

    monkeypatch.setattr(quarry.cli, "load_dotenv", record)
    quarry.cli._config(None)
    assert loaded == [str(tmp_path / ".env")]


def test_cli_chat_session(warehouse: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    script = "\n".join([DEMO_QUESTIONS[0], DEMO_QUESTIONS[1], "/quit"]) + "\n"
    result = CliRunner().invoke(
        cli_app, ["chat", "--db", str(warehouse), "--offline"], input=script
    )
    assert result.exit_code == 0, result.output
    assert "converted best" in result.stdout and "Broken down by purchase channel" in result.stdout

    bad = CliRunner().invoke(
        cli_app, ["chat", "--db", str(warehouse), "--offline", "--thread", "/abs"], input="/quit\n"
    )
    assert bad.exit_code == 2 and "invalid thread id" in bad.output
