"""Config, LLM factory and cache, chart picking, tracing."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from quarry.agent.chart import chart_spec
from quarry.config import API_KEY_ENV, QuarryConfig, api_key, load_config
from quarry.errors import ConfigError
from quarry.llm import (
    QueueResponder,
    ScriptedChatModel,
    SQLiteLLMCache,
    build_chat_model,
    build_openrouter_model,
)
from quarry.tracing import Trace, read_trace


def test_shipped_config_parses() -> None:
    cfg = load_config(Path("configs/quarry.toml"))
    assert cfg.llm.base_url == "https://openrouter.ai/api/v1"
    assert cfg.sql.row_limit > 0 and cfg.agent.max_repairs >= 1


def test_config_errors_are_explicit(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[llm]\nmodle = 'typo'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        load_config(bad)
    broken = tmp_path / "broken.toml"
    broken.write_text("[llm\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(broken)
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.toml")


def test_api_key_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert api_key("optional") is None
    with pytest.raises(ConfigError, match="--offline"):
        build_openrouter_model(QuarryConfig().llm)
    monkeypatch.setenv(API_KEY_ENV, "sk-or-test-not-a-real-key")
    model = build_openrouter_model(QuarryConfig().llm)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.model_name == QuarryConfig().llm.model
    with pytest.raises(ValueError, match="responder"):
        build_chat_model(QuarryConfig().llm, offline=True)


def test_scripted_model_binds_tools_and_reports_usage() -> None:
    responder = QueueResponder(["hello"])
    model = ScriptedChatModel(responder=responder)
    bound = model.bind_tools([{"type": "function", "function": {"name": "f", "parameters": {}}}])
    reply = bound.invoke([HumanMessage("hi")])
    assert isinstance(reply, AIMessage) and reply.content == "hello"
    assert reply.usage_metadata is not None and reply.usage_metadata["total_tokens"] > 0
    with pytest.raises(RuntimeError, match="ran out"):
        model.invoke([HumanMessage("again")])


def test_llm_cache_serves_repeats(tmp_path: Path) -> None:
    cache = SQLiteLLMCache(tmp_path / "cache.sqlite")
    responder = QueueResponder(["first"])
    model = ScriptedChatModel(responder=responder, cache=cache)
    assert model.invoke([HumanMessage("q")]).content == "first"
    assert model.invoke([HumanMessage("q")]).content == "first"
    assert cache.hits == 1 and len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    cache.close()


def test_chart_choices() -> None:
    bar = chart_spec(["campaign", "rate"], [["A", 1.0], ["B", 2.0]])
    assert bar is not None and bar["mark"]["type"] == "bar"
    line = chart_spec(["recency_months", "rate"], [[1, 0.5], [2, 0.4], [3, 0.3]])
    assert line is not None and line["mark"]["type"] == "line"
    dates = chart_spec(["day", "n"], [["2024-01-01", 1], ["2024-01-02", 3]])
    assert dates is not None and dates["encoding"]["x"]["type"] == "temporal"
    grouped = chart_spec(["a", "b", "n"], [["x", "p", 1], ["y", "q", 2]])
    assert grouped is not None and "color" in grouped["encoding"]
    scatter = chart_spec(["x", "y"], [[1.5, 2.5], [2.5, 3.5]])
    assert scatter is not None and scatter["mark"]["type"] == "point"
    assert chart_spec(["n"], [[1], [2]]) is None
    assert chart_spec(["a", "b"], [["x", 1]]) is None
    dotted = chart_spec(["t.name", "n"], [["x", 1], ["y", 2]])
    assert dotted is not None and dotted["encoding"]["y"]["field"] == "t\\.name"


def test_trace_spans_nest_and_record_errors(tmp_path: Path) -> None:
    trace = Trace(tmp_path, thread_id="th")
    with trace.span("run", "run"), trace.span("node", "node") as span:
        span.tokens_in = 3
        trace.event("flag", {"x": 1})
    with pytest.raises(ValueError), trace.span("bad", "node"):
        raise ValueError("boom")
    spans = read_trace(trace.path) if trace.path else []
    by_name = {s["name"]: s for s in spans}
    assert by_name["node"]["parent_id"] == by_name["run"]["span_id"]
    assert by_name["flag"]["parent_id"] == by_name["node"]["span_id"]
    assert by_name["bad"]["status"] == "error" and "boom" in by_name["bad"]["error"]
    assert trace.totals()["llm_calls"] == 0
