from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import openai
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from quarry.config import QuarryConfig
from quarry.db import Database
from quarry.errors import BenchmarkInterrupted
from quarry.eval import benchmark
from quarry.eval.benchmark import Score, evaluate, gold_preflight, score_prediction, summarize
from quarry.eval.hardness import hardness
from quarry.eval.matching import gold_schema_items, has_top_level_order, results_match
from quarry.eval.offline import catalog_from_schema, render_markdown
from quarry.eval.redteam import ATTACKS, run_attacks, run_injection_scenario
from quarry.eval.spider import Example, SpiderLayout
from quarry.eval.stats import latency_summary, percentile
from quarry.llm import ScriptedChatModel, SQLiteLLMCache


def test_hardness_follows_spider_rules() -> None:
    assert hardness("SELECT count(*) FROM singer") == "easy"
    assert hardness("SELECT name FROM singer WHERE age > 20 ORDER BY age") == "medium"
    assert (
        hardness("SELECT name FROM singer WHERE id NOT IN (SELECT singer_id FROM concert)")
        == "hard"
    )
    assert (
        hardness(
            "SELECT T1.name, count(*) FROM a AS T1 JOIN b AS T2 ON T1.id = T2.aid "
            "WHERE T2.x = 1 OR T2.y LIKE '%z%' GROUP BY T1.name ORDER BY count(*) DESC LIMIT 1"
        )
        == "extra"
    )
    assert hardness("not sql at all ((") == "extra"


def test_results_match_rules() -> None:
    gold = [(1, "a"), (2, "b")]
    assert results_match(gold, [(2, "b"), (1, "a")], order_matters=False)
    assert not results_match(gold, [(2, "b"), (1, "a")], order_matters=True)
    assert results_match(gold, [("a", 1), ("b", 2)], order_matters=False)
    assert results_match([(1.0,)], [(1,)], order_matters=False)
    assert not results_match(gold, [(1, "a")], order_matters=False)
    assert results_match([], [], order_matters=True)
    assert has_top_level_order("SELECT a FROM t ORDER BY a")
    assert not has_top_level_order("SELECT a FROM (SELECT a FROM t ORDER BY a)")


@settings(max_examples=100, deadline=None)
@given(rows=st.lists(st.tuples(st.integers(-5, 5), st.sampled_from(["x", "y", None])), max_size=8))
def test_results_match_is_reflexive_and_order_free(rows: list[tuple[int, str | None]]) -> None:
    assert results_match(rows, rows, order_matters=True)
    assert results_match(rows, list(reversed(rows)), order_matters=False)


def test_gold_schema_items_resolve_aliases() -> None:
    schema: dict[str, object] = {
        "singer": {"id": "int", "name": "text"},
        "concert": {"singer_id": "int", "year": "int"},
    }
    tables, columns = gold_schema_items(
        "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id "
        "WHERE T2.year = 2014",
        schema,
    )
    assert tables == {"singer", "concert"}
    assert columns == {
        ("singer", "name"),
        ("singer", "id"),
        ("concert", "singer_id"),
        ("concert", "year"),
    }


def test_latency_helpers() -> None:
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4 and summary["p95_ms"] == 4.0
    assert latency_summary([]) == {"n": 0}


def test_redteam_subset_and_scenarios() -> None:
    subset = [a for a in ATTACKS if a.category in {"destructive", "benign"}][::3]
    results = run_attacks(subset, timeout_ms=500)
    assert results and all(r.passed for r in results)
    assert all(r.database_unchanged for r in results)
    on, off = run_injection_scenario(detector_on=True), run_injection_scenario(detector_on=False)
    assert on.withheld_values > 0 and on.writes_executed == 0 and on.database_unchanged
    assert off.writes_attempted > 0 and off.writes_executed == 0 and off.database_unchanged


# -------------------------------------------------------------- the benchmark harness


def _fixture_spider(root: Path, db_source: Path) -> SpiderLayout:
    """A two-question Spider-shaped dataset over the shop database."""
    spider = root / "spider"
    (spider / "database" / "shop").mkdir(parents=True)
    shutil.copy(db_source, spider / "database" / "shop" / "shop.sqlite")
    dev = [
        {
            "db_id": "shop",
            "question": "How many products are there?",
            "query": "SELECT count(*) FROM products",
        },
        {
            "db_id": "shop",
            "question": "Which countries ordered?",
            "query": "SELECT DISTINCT country FROM orders",
        },
    ]
    conn = sqlite3.connect(db_source)
    tables = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]
    columns: list[list[Any]] = [[-1, "*"]]
    for i, table in enumerate(tables):
        columns += [[i, r[1]] for r in conn.execute(f"PRAGMA table_info({table})")]
    conn.close()
    schema = {
        "db_id": "shop",
        "table_names_original": tables,
        "table_names": tables,
        "column_names_original": columns,
        "column_names": [[t, str(n).replace("_", " ")] for t, n in columns],
    }
    for name, payload in {
        "dev.json": dev,
        "tables.json": [schema],
        "train_spider.json": [],
        "train_others.json": [],
    }.items():
        (spider / name).write_text(json.dumps(payload), encoding="utf-8")
    return SpiderLayout(spider)


class OracleResponder:
    """Answers each question with its gold SQL (checks the harness, not a model)."""

    def __init__(self, gold: dict[str, str]) -> None:
        self.gold = gold
        self.calls = 0

    def __call__(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
    ) -> AIMessage:
        self.calls += 1
        system = str(messages[0].content)
        question = next(
            str(m.content).splitlines()[0].removeprefix("Question: ")
            for m in messages
            if isinstance(m, HumanMessage)
        )
        if system.startswith("# quarry/generate"):
            call = {
                "name": "run_sql",
                "args": {"sql": self.gold[question]},
                "id": f"g{self.calls}",
                "type": "tool_call",
            }
            return AIMessage(content="", tool_calls=[call])
        return AIMessage(content="plan" if system.startswith("# quarry/plan") else "answer")


def test_benchmark_harness_scores_and_resumes(shop_db: Path, tmp_path: Path) -> None:
    layout = _fixture_spider(tmp_path, shop_db)
    examples = layout.examples("dev")
    oracle = OracleResponder({e.question: e.gold_sql for e in examples})
    cache = SQLiteLLMCache(tmp_path / "cache.sqlite")
    llm = ScriptedChatModel(responder=oracle, cache=cache)
    preds = tmp_path / "run" / "predictions.jsonl"
    cfg = QuarryConfig()

    records = evaluate(examples, layout, llm, cfg, preds, trace_dir=None)
    assert [r["correct"] for r in records] == [True, True]
    summary = summarize(records)
    assert summary["execution_accuracy"] == 100.0 and summary["questions"] == 2
    calls_after_first_run = oracle.calls

    again = evaluate(examples, layout, llm, cfg, preds, trace_dir=None)
    assert len(again) == 2 and oracle.calls == calls_after_first_run
    assert len(preds.read_text(encoding="utf-8").splitlines()) == 2

    preds.unlink()
    evaluate(examples, layout, llm, cfg, preds, trace_dir=None)
    assert oracle.calls == calls_after_first_run, "second pass should be served by the LLM cache"
    assert cache.hits > 0
    cache.close()


def test_score_prediction_cases(shop_db: Path) -> None:
    db = Database(shop_db)
    assert score_prediction(
        db, "SELECT count(*) FROM products", "SELECT COUNT(product_id) FROM products"
    ).correct
    assert score_prediction(db, "SELECT 1", None) == Score(False, "no SQL")
    refused = score_prediction(db, "SELECT count(*) FROM products", "DROP TABLE products")
    assert not refused.correct and "refused" in refused.note and not refused.gold_failed
    broken_gold = score_prediction(db, "SELECT nope FROM products", "SELECT 1")
    assert broken_gold.gold_failed and not broken_gold.correct


def test_gold_that_cannot_run_in_the_sandbox_is_left_out_not_scored(
    shop_db: Path, tmp_path: Path
) -> None:
    layout = _fixture_spider(tmp_path, shop_db)
    runnable = layout.examples("dev")
    blocked = Example("dev-x", "dev", "shop", "Which tables exist?", "SELECT * FROM sqlite_master")
    failures = gold_preflight(layout, [*runnable, blocked])
    assert list(failures) == ["dev-x"] and "refused by the SQL guard" in failures["dev-x"]

    records = [
        {"id": e.id, "hardness": "easy", "correct": True, "gold_failed": False, "note": "",
         "status": "ok", "attempts": 1, "first_attempt_failed": False, "tokens": 10,
         "latency_ms": 5.0}
        for e in runnable
    ]  # fmt: skip
    summary = summarize(records, failures)
    assert summary["questions"] == 2 and summary["execution_accuracy"] == 100.0
    assert summary["excluded_gold_failures"] == 1 and "dev-x" in summary["gold_failures"]
    # Levels with no questions (likely with --limit) are null, never NaN: the JSON stays valid.
    assert summary["by_hardness"]["extra"]["execution_accuracy"] is None
    json.dumps(summary, allow_nan=False)
    meta = {"finished_at": "now", "hardware": {"cpu": "cpu"}}
    text = benchmark.render_markdown("m", summary, meta)
    assert "| extra | 0 | - |" in text and "(dev-x)" in text


def test_benchmark_stops_with_a_resumable_error_when_the_api_is_down(
    shop_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _fixture_spider(tmp_path, shop_db)

    def down(messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]) -> AIMessage:
        raise openai.OpenAIError("connection refused")

    monkeypatch.setattr(benchmark, "MAX_CONSECUTIVE_API_FAILURES", 2)
    preds = tmp_path / "run" / "predictions.jsonl"
    with pytest.raises(BenchmarkInterrupted, match="rerun the same command"):
        evaluate(
            layout.examples("dev"), layout, ScriptedChatModel(responder=down), QuarryConfig(),
            preds, trace_dir=None,
        )  # fmt: skip
    assert preds.read_text(encoding="utf-8") == "", "unreachable questions are not recorded"


def test_catalog_from_tables_json_and_report_rendering(shop_db: Path, tmp_path: Path) -> None:
    layout = _fixture_spider(tmp_path, shop_db)
    catalog = catalog_from_schema(layout.schemas["shop"])
    products = catalog.require_table("products")
    assert products.column("product_id") is not None
    assert (products.column("product_id") or products.columns[0]).description == "product id"
    report = {
        "measured_at": "now",
        "hardware": {"cpu": "cpu", "logical_cpus": 1, "os": "os", "python": "3.12"},
        "redteam": {
            "total_cases": 1, "total_passed": 1, "database_changed": 0,
            "by_category": {"benign": {"sqlite": {"cases": 1, "passed": 1}}},
            "engine_only_outcomes": {}, "engine_only_ran": [], "scenarios": [],
        },
        "injection": {
            "development_set": {
                "injections_flagged": 1, "injections": 1, "benign_flagged": 0, "benign": 1,
            },
            "database_text": {"flagged": 0, "distinct_values": 3, "false_positive_rate": 0.0},
        },
        "latency": {"database": "db", "tools": {},
                    "guard_check": {"n": 1, "p50_ms": 1.0, "p95_ms": 1.0},
                    "static_check": {"n": 1, "p50_ms": 1.0, "p95_ms": 1.0},
                    "schema_link": {"n": 1, "p50_ms": 1.0, "p95_ms": 1.0},
                    "catalog_introspection_cold": {"n": 1, "p50_ms": 1.0, "p95_ms": 1.0}},
    }  # fmt: skip
    text = render_markdown(report)
    assert "## Red team" in text and "| benign | 1/1 | - |" in text
