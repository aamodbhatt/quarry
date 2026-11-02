from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import openai
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from quarry.agent import AgentDeps, QuarryAgent, open_agent
from quarry.agent.nodes import LLM_UNAVAILABLE
from quarry.config import QuarryConfig
from quarry.db import Database
from quarry.guard import WITHHELD
from quarry.llm import QueueResponder, ScriptedChatModel, tool_call_message
from quarry.tracing import read_trace

GOOD_SQL = "SELECT country, SUM(quantity) AS units FROM orders GROUP BY country ORDER BY units DESC"


def make_agent(
    db: Path, config: QuarryConfig, replies: Sequence[AIMessage | str]
) -> tuple[QuarryAgent, QueueResponder]:
    responder = QueueResponder(replies)
    deps = AgentDeps(
        llm=ScriptedChatModel(responder=responder), database=Database(db), config=config
    )
    return QuarryAgent(deps, trace_dir=config.tracing.dir), responder


def sql_call(sql: str, call_id: str = "c1") -> AIMessage:
    return tool_call_message("run_sql", {"sql": sql}, call_id)


def tool_texts(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages if isinstance(m, ToolMessage))


def test_happy_path_plans_runs_answers_and_traces(shop_db: Path, config: QuarryConfig) -> None:
    agent, responder = make_agent(
        shop_db, config, ["1. sum quantity by country", sql_call(GOOD_SQL), "France leads."]
    )
    outcome = agent.ask("Which country ordered the most units?")
    assert outcome.status == "ok"
    assert outcome.answer == "France leads."
    assert outcome.sql == GOOD_SQL
    assert outcome.rows[0] == ["Germany", 6]
    assert [a["stage"] for a in outcome.attempts] == ["accepted"]
    assert outcome.chart is not None and outcome.chart["mark"]["type"] == "bar"
    assert outcome.trace["llm_calls"] == 3 and outcome.trace["tool_calls"] == 1
    assert outcome.trace["tokens_estimated"] is True
    spans = read_trace(Path(outcome.trace["path"]))
    names = [s["name"] for s in spans]
    for node in ("start_turn", "link_schema", "plan", "generate", "check", "execute", "verify",
                 "answer", "chart", "finish", "llm:plan", "tool:run_sql", "run"):  # fmt: skip
        assert node in names
    assert all(s["latency_ms"] >= 0 for s in spans)
    assert responder.remaining == 0


def test_static_check_failure_is_repaired(shop_db: Path, config: QuarryConfig) -> None:
    bad = "SELECT country, SUM(qty) FROM orders GROUP BY country"
    agent, responder = make_agent(
        shop_db, config, ["plan", sql_call(bad, "c1"), sql_call(GOOD_SQL, "c2"), "done"]
    )
    outcome = agent.ask("Units per country?")
    assert outcome.status == "ok"
    assert [a["stage"] for a in outcome.attempts] == ["check", "accepted"]
    assert "static check failed" in tool_texts(responder.calls[2])


def test_execution_error_is_fed_back(shop_db: Path, config: QuarryConfig) -> None:
    bad = "SELECT name FROM products WHERE price > (SELECT price, name FROM products)"
    agent, responder = make_agent(
        shop_db, config, ["plan", sql_call(bad), sql_call(GOOD_SQL, "c2"), "ok"]
    )
    outcome = agent.ask("Units per country?")
    assert [a["stage"] for a in outcome.attempts] == ["execute", "accepted"]
    feedback = tool_texts(responder.calls[2])
    assert "ERROR: " in feedback and outcome.attempts[0]["feedback"] in feedback


def test_writes_are_refused_and_nothing_changes(shop_db: Path, config: QuarryConfig) -> None:
    before = hashlib.sha256(shop_db.read_bytes()).hexdigest()
    agent, responder = make_agent(
        shop_db, config, ["plan", sql_call("DROP TABLE orders"), "quarry is read-only."]
    )
    outcome = agent.ask("Delete all orders")
    assert outcome.status == "refused"
    assert outcome.answer == "quarry is read-only."
    assert "read-only SQL guard" in tool_texts(responder.calls[2])
    assert hashlib.sha256(shop_db.read_bytes()).hexdigest() == before


def test_refusal_after_exhausting_repairs(shop_db: Path, config: QuarryConfig) -> None:
    cfg = config.model_copy(update={"agent": config.agent.model_copy(update={"max_repairs": 1})})
    agent, _ = make_agent(
        shop_db, cfg, ["plan", sql_call("DELETE FROM orders"), sql_call("DROP TABLE orders", "c2")]
    )
    outcome = agent.ask("Remove the orders")
    assert outcome.status == "refused"
    assert "nothing in the database was changed" in outcome.answer


def test_empty_result_gets_one_second_look(shop_db: Path, config: QuarryConfig) -> None:
    sql = "SELECT order_id FROM orders WHERE country = 'Italy'"
    agent, responder = make_agent(
        shop_db, config, ["plan", sql_call(sql), sql_call(sql, "c2"), "No orders went to Italy."]
    )
    outcome = agent.ask("Which orders went to Italy?")
    assert outcome.status == "ok"
    assert [a["stage"] for a in outcome.attempts] == ["verify", "accepted"]
    assert any("no rows" in str(m.content) for m in responder.calls[2])


def test_budget_exhaustion_skips_the_answer_llm(shop_db: Path, config: QuarryConfig) -> None:
    cfg = config.model_copy(update={"agent": config.agent.model_copy(update={"max_steps": 2})})
    agent, responder = make_agent(shop_db, cfg, ["plan", sql_call(GOOD_SQL)])
    outcome = agent.ask("Units per country?")
    assert outcome.status == "ok"
    assert outcome.answer.startswith("First row: country = Germany")
    assert responder.remaining == 0


def test_schema_tools_and_injected_rows(shop_db: Path, config: QuarryConfig) -> None:
    replies: list[AIMessage | str] = [
        tool_call_message("sample_rows", {"table": "reviews", "limit": 5}, "t1"),
        tool_call_message("describe_table", {"table": "products"}, "t2"),
        "plan",
        sql_call("SELECT COUNT(*) AS n FROM reviews"),
        "There are 2 reviews.",
    ]
    agent, responder = make_agent(shop_db, config, replies)
    outcome = agent.ask("How many reviews are there?")
    assert outcome.status == "ok"
    seen = tool_texts(responder.calls[1])
    assert WITHHELD in seen and "DROP TABLE orders" not in seen
    assert "Trail Shoe" not in tool_texts(responder.calls[1]) or "reviews" in seen
    assert any("prompt injection" in w for w in outcome.warnings)


def test_explore_budget_is_enforced(shop_db: Path, config: QuarryConfig) -> None:
    cfg = config.model_copy(
        update={"agent": config.agent.model_copy(update={"max_explore_calls": 1})}
    )
    replies: list[AIMessage | str] = [
        tool_call_message("list_tables", {}, "t1"),
        tool_call_message("describe_table", {"table": "orders"}, "t2"),
        "plan",
        sql_call(GOOD_SQL),
        "done",
    ]
    agent, responder = make_agent(shop_db, cfg, replies)
    assert agent.ask("Units per country?").status == "ok"
    assert "Tool budget used up" in tool_texts(responder.calls[2])


def test_sql_in_a_fenced_block_is_accepted(shop_db: Path, config: QuarryConfig) -> None:
    agent, _ = make_agent(
        shop_db, config, ["plan", f"Here you go:\n```sql\n{GOOD_SQL}\n```", "France."]
    )
    outcome = agent.ask("Units per country?")
    assert outcome.status == "ok" and outcome.sql == GOOD_SQL


def test_plain_text_reply_means_unanswerable(shop_db: Path, config: QuarryConfig) -> None:
    agent, _ = make_agent(shop_db, config, ["plan", "This database has no weather data."])
    outcome = agent.ask("Will it rain tomorrow?")
    assert outcome.status == "unanswerable"
    assert outcome.answer == "This database has no weather data."


def test_api_failures_are_reported_not_raised(shop_db: Path, config: QuarryConfig) -> None:
    def broken(messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]) -> AIMessage:
        raise openai.OpenAIError("connection refused")

    deps = AgentDeps(
        llm=ScriptedChatModel(responder=broken), database=Database(shop_db), config=config
    )
    outcome = QuarryAgent(deps, trace_dir=None).ask("Units per country?")
    assert outcome.status == "failed"
    assert any(w.startswith(LLM_UNAVAILABLE) for w in outcome.warnings)


def test_memory_survives_a_restart(shop_db: Path, config: QuarryConfig) -> None:
    first = QueueResponder(["plan", sql_call(GOOD_SQL), "France leads."])
    with open_agent(config, db_path=shop_db, offline=True, responder=first,
                    memory_path=config.memory_path) as agent:  # fmt: skip
        agent.ask("Units per country?", thread_id="conv1")
    follow_sql = "SELECT country, COUNT(*) AS orders FROM orders GROUP BY country"
    second = QueueResponder(["plan", sql_call(follow_sql), "Two orders from France."])
    with open_agent(config, db_path=shop_db, offline=True, responder=second,
                    memory_path=config.memory_path) as agent:  # fmt: skip
        outcome = agent.ask("And how many orders is that?", thread_id="conv1")
        history = agent.history("conv1")
    prompt = str(second.calls[0][1].content)
    assert "Earlier in this conversation" in prompt
    assert "Units per country?" in prompt and GOOD_SQL in prompt
    assert outcome.status == "ok"
    assert [t["question"] for t in history] == [
        "Units per country?",
        "And how many orders is that?",
    ]


def test_events_stream_in_graph_order(shop_db: Path, config: QuarryConfig) -> None:
    agent, _ = make_agent(shop_db, config, ["plan", sql_call(GOOD_SQL), "France."])
    kinds = [e.kind for e in agent.stream("Units per country?")]
    assert kinds[0] == "start_turn" and kinds[-1] == "done"
    assert kinds.index("generate") < kinds.index("execute") < kinds.index("answer")
