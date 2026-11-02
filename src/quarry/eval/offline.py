"""Everything quarry can measure without an LLM, in one command: ``quarry eval offline``.

* schema-linking recall@k of the gold tables and columns on Spider dev (three variants);
* the SQL guard on Spider's gold queries (dev, train_spider, and train_others once the
  full archive is present): how many it wrongly blocks;
* the static check on dev gold (false alarms), and dev gold run through the full sandbox;
* the Spider difficulty labels this project assigns, against the official totals;
* the red-team suite (both engines) and the planted-injection agent scenario;
* the injection detector on its development strings, on real database text, and on an
  external labelled set it was never tuned on (deepset/prompt-injections);
* latency of the guard, checks, tools, executor, linker and a full offline agent run.

Results go to ``docs/results/offline.json`` and ``docs/results/offline.md``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import duckdb
from rich.console import Console

from quarry.agent import AgentDeps, QuarryAgent
from quarry.agent.checks import static_errors
from quarry.config import QuarryConfig
from quarry.datasets import RemoteFile, fetch
from quarry.db import Catalog, Column, Database, SafeExecutor, Table
from quarry.db.engines import SQLiteEngine, decode_text
from quarry.demo.script import DEMO_QUESTIONS, DemoResponder
from quarry.errors import DatasetError, QueryFailed
from quarry.eval.hardness import hardness
from quarry.eval.matching import gold_schema_items
from quarry.eval.redteam import (
    BENIGN_TEXTS,
    INJECTIONS,
    AttackResult,
    run_attacks,
    run_injection_scenario,
)
from quarry.eval.spider import OFFICIAL_DEV_HARDNESS, Example, SpiderLayout, SpiderSchema
from quarry.eval.stats import hardware, latency_summary, pct
from quarry.guard import InjectionDetector, SQLGuard
from quarry.linking import HashingEmbedder, SchemaLinker
from quarry.llm import ScriptedChatModel

_DEEPSET_REV = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
_DEEPSET_BASE = (
    f"https://huggingface.co/datasets/deepset/prompt-injections/resolve/{_DEEPSET_REV}/data/"
)
DEEPSET_FILES = (
    RemoteFile(
        name="deepset-prompt-injections-train.parquet",
        url=_DEEPSET_BASE + "train-00000-of-00001-9564e8b05b4757ab.parquet",
        sha256="2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
        size_bytes=40_323,
        source_page="https://huggingface.co/datasets/deepset/prompt-injections",
    ),
    RemoteFile(
        name="deepset-prompt-injections-test.parquet",
        url=_DEEPSET_BASE + "test-00000-of-00001-701d16158af87368.parquet",
        sha256="39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
        size_bytes=10_892,
        source_page="https://huggingface.co/datasets/deepset/prompt-injections",
    ),
)

TABLE_KS = (1, 3, 5)
COLUMN_KS = (10, 20, 30)


# ------------------------------------------------------------------ Spider catalogs


def catalog_from_schema(schema: SpiderSchema) -> Catalog:
    """A catalog built from ``tables.json`` alone (names and natural names, no values)."""
    tables = []
    for name in schema.tables:
        columns = tuple(
            Column(name=col, type="", description=natural)
            for table, col, natural in schema.columns
            if table == name
        )
        tables.append(Table(name=name, columns=columns))
    return Catalog("sqlite", tuple(tables))


def spider_catalog(layout: SpiderLayout, db_id: str, *, sample_values: int) -> Catalog:
    schema = layout.schemas[db_id]
    path = layout.db_path(db_id)
    if path.exists():
        return Database(path, sample_values=sample_values, docs=schema.docs()).catalog()
    return catalog_from_schema(schema)


def _schema_dict(catalog: Catalog) -> dict[str, object]:
    return {t.name: {c.name: (c.type or "TEXT") for c in t.columns} for t in catalog.tables}


def _strip_samples(catalog: Catalog) -> Catalog:
    tables = tuple(
        replace(t, columns=tuple(replace(c, samples=()) for c in t.columns)) for t in catalog.tables
    )
    return Catalog(catalog.dialect, tables)


# ------------------------------------------------------------------ schema linking


def linking_recall(layout: SpiderLayout, examples: Sequence[Example]) -> dict[str, Any]:
    has_dbs = all(layout.db_path(db).exists() for db in {e.db_id for e in examples})
    catalogs = {
        db: spider_catalog(layout, db, sample_values=8) for db in {e.db_id for e in examples}
    }
    variants: dict[str, Callable[[Catalog], SchemaLinker]] = {
        "bm25 (names + descriptions)": lambda c: SchemaLinker(_strip_samples(c)),
    }
    if has_dbs:
        variants["bm25 + sampled values"] = SchemaLinker
        variants["bm25 + values + hashing embeddings (shipped)"] = lambda c: SchemaLinker(
            c, HashingEmbedder()
        )
    else:
        variants["bm25 + hashing embeddings (no values: databases absent)"] = lambda c: (
            SchemaLinker(_strip_samples(c), HashingEmbedder())
        )

    gold = {}
    for ex in examples:
        tables, columns = gold_schema_items(ex.gold_sql, _schema_dict(catalogs[ex.db_id]))
        if tables:
            gold[ex.id] = (tables, columns)

    out: dict[str, Any] = {"examples": len(gold), "with_sampled_values": has_dbs, "variants": {}}
    for label, make in variants.items():
        linkers = {db: make(c) for db, c in catalogs.items()}
        table_hits = dict.fromkeys(TABLE_KS, 0)
        table_mean = dict.fromkeys(TABLE_KS, 0.0)
        col_hits = dict.fromkeys(COLUMN_KS, 0)
        col_mean = dict.fromkeys(COLUMN_KS, 0.0)
        selected_hits = 0
        selected_size = 0
        latencies: list[float] = []
        col_n = 0
        for ex in examples:
            if ex.id not in gold:
                continue
            g_tables, g_cols = gold[ex.id]
            start = time.perf_counter()
            result = linkers[ex.db_id].link(ex.question)
            latencies.append((time.perf_counter() - start) * 1000)
            ranked_tables = [t.name.lower() for t in result.tables]
            ranked_cols = [(c.table.lower(), c.column.lower()) for c in result.columns]
            for k in TABLE_KS:
                top = set(ranked_tables[:k])
                table_hits[k] += g_tables <= top
                table_mean[k] += len(g_tables & top) / len(g_tables)
            if g_cols:
                col_n += 1
                for k in COLUMN_KS:
                    top_c = set(ranked_cols[:k])
                    col_hits[k] += g_cols <= top_c
                    col_mean[k] += len(g_cols & top_c) / len(g_cols)
            selected = {s.lower() for s in result.selected}
            selected_hits += g_tables <= selected
            selected_size += len(selected)
        n = len(gold)
        out["variants"][label] = {
            "table_recall_all_at_k": {k: pct(table_hits[k], n) for k in TABLE_KS},
            "table_recall_mean_at_k": {k: round(100 * table_mean[k] / n, 2) for k in TABLE_KS},
            "column_recall_all_at_k": {k: pct(col_hits[k], col_n) for k in COLUMN_KS},
            "column_recall_mean_at_k": {
                k: round(100 * col_mean[k] / col_n, 2) if col_n else None for k in COLUMN_KS
            },
            "selected_set_recall": pct(selected_hits, n),
            "selected_set_mean_size": round(selected_size / n, 2),
            "latency": latency_summary(latencies),
        }
    out["random_order_baseline"] = _random_baseline(gold, catalogs, examples)
    out["mean_tables_per_db"] = round(
        sum(len(c.tables) for c in catalogs.values()) / len(catalogs), 2
    )
    out["mean_columns_per_db"] = round(
        sum(sum(len(t.columns) for t in c.tables) for c in catalogs.values()) / len(catalogs), 2
    )
    return out


def _random_baseline(
    gold: dict[str, tuple[set[str], set[tuple[str, str]]]],
    catalogs: dict[str, Catalog],
    examples: Sequence[Example],
) -> dict[str, Any]:
    """Expected all@k if tables and columns were ranked in a uniformly random order.

    P(all g gold items land in the top k of n) = C(n - g, k - g) / C(n, k). Spider
    databases are small (about four tables), so this puts the table numbers in context.
    """
    tables = dict.fromkeys(TABLE_KS, 0.0)
    columns = dict.fromkeys(COLUMN_KS, 0.0)
    n_col = 0
    for ex in examples:
        if ex.id not in gold:
            continue
        g_tables, g_cols = gold[ex.id]
        catalog = catalogs[ex.db_id]
        n_t = len(catalog.tables)
        n_c = sum(len(t.columns) for t in catalog.tables)
        for k in TABLE_KS:
            tables[k] += _p_all_in_top(n_t, len(g_tables), k)
        if g_cols:
            n_col += 1
            for k in COLUMN_KS:
                columns[k] += _p_all_in_top(n_c, len(g_cols), k)
    n = len(gold)
    return {
        "table_recall_all_at_k": {k: round(100 * v / n, 2) for k, v in tables.items()},
        "column_recall_all_at_k": {k: round(100 * v / n_col, 2) for k, v in columns.items()},
    }


def _p_all_in_top(n: int, g: int, k: int) -> float:
    if g > n:
        return 0.0
    if k >= n:
        return 1.0
    if g > k:
        return 0.0
    return math.comb(n - g, k - g) / math.comb(n, k)


# ------------------------------------------------------------------ guard and checks on gold


def guard_on_gold(layout: SpiderLayout, splits: Sequence[str] = ("dev", "train")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    latencies: list[float] = []
    for split in splits:
        examples = layout.examples(split)
        guards = {db: SQLGuard("sqlite", schema.tables) for db, schema in layout.schemas.items()}
        blocked: list[dict[str, str]] = []
        rules: Counter[str] = Counter()
        for ex in examples:
            start = time.perf_counter()
            verdict = guards[ex.db_id].check(ex.gold_sql)
            latencies.append((time.perf_counter() - start) * 1000)
            if not verdict.allowed:
                rules.update(v.rule for v in verdict.violations)
                blocked.append(
                    {
                        "id": ex.id,
                        "db_id": ex.db_id,
                        "sql": ex.gold_sql,
                        "violations": "; ".join(str(v) for v in verdict.violations),
                    }
                )
        out[split] = {
            "queries": len(examples),
            # which files the split came from: the authors' mirror has no train_others
            "by_file": dict(Counter(ex.id.rsplit("-", 1)[0] for ex in examples)),
            "allowed": len(examples) - len(blocked),
            "pass_rate": pct(len(examples) - len(blocked), len(examples)),
            "blocked_by_rule": dict(rules),
            "blocked": blocked,
        }
    out["guard_latency"] = latency_summary(latencies)
    return out


def static_check_on_gold(layout: SpiderLayout, examples: Sequence[Example]) -> dict[str, Any]:
    catalogs = {
        db: spider_catalog(layout, db, sample_values=0) for db in {e.db_id for e in examples}
    }
    flagged: list[dict[str, str]] = []
    latencies: list[float] = []
    for ex in examples:
        start = time.perf_counter()
        errors = static_errors(ex.gold_sql, catalogs[ex.db_id])
        latencies.append((time.perf_counter() - start) * 1000)
        if errors:
            flagged.append({"id": ex.id, "sql": ex.gold_sql, "error": errors[0]})
    return {
        "queries": len(examples),
        "false_alarms": len(flagged),
        "false_alarm_rate": pct(len(flagged), len(examples)),
        "flagged": flagged,
        "latency": latency_summary(latencies),
    }


def gold_through_sandbox(layout: SpiderLayout, examples: Sequence[Example]) -> dict[str, Any]:
    """Run every dev gold query through guard + read-only engine, and once with neither."""
    sandbox_ok = raw_ok = 0
    differ: list[dict[str, str]] = []
    latencies: list[float] = []
    executors: dict[str, SafeExecutor] = {}
    for ex in examples:
        path = layout.db_path(ex.db_id)
        if ex.db_id not in executors:
            executors[ex.db_id] = SafeExecutor(
                Database(path, sample_values=0), row_limit=100_000, timeout_ms=30_000
            )
        run = executors[ex.db_id].run(ex.gold_sql)
        latencies.append(run.elapsed_ms)
        sandbox_ok += run.ok
        raw_error = _raw_sqlite_error(path, ex.gold_sql)
        raw_ok += raw_error is None
        if run.ok != (raw_error is None):
            differ.append({"id": ex.id, "sql": ex.gold_sql, "sandbox_error": run.error or ""})
    return {
        "queries": len(examples),
        "ran_in_sandbox": sandbox_ok,
        "ran_unsandboxed": raw_ok,
        "sandbox_only_failures": differ,
        "executor_latency": latency_summary(latencies),
    }


def _raw_sqlite_error(path: Path, sql: str) -> str | None:
    import sqlite3

    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.text_factory = decode_text
    try:
        conn.execute(sql).fetchmany(100_000)
    except sqlite3.Error as exc:
        return str(exc)
    finally:
        conn.close()
    return None


def hardness_distribution(examples: Sequence[Example]) -> dict[str, Any]:
    counts = Counter(hardness(e.gold_sql) for e in examples)
    ours = {k: counts.get(k, 0) for k in OFFICIAL_DEV_HARDNESS}
    return {
        "this_implementation": ours,
        "official": OFFICIAL_DEV_HARDNESS,
        "total_abs_difference": sum(abs(ours[k] - v) for k, v in OFFICIAL_DEV_HARDNESS.items()),
    }


# ------------------------------------------------------------------ injection detector


def _distinct_text(paths: Sequence[Path], limit_per_column: int = 2_000) -> list[str]:
    values: set[str] = set()
    for path in paths:
        database = Database(path, sample_values=0)
        engine = database.engine
        for table in database.catalog().tables:
            for column in table.columns:
                quoted_t = '"' + table.name.replace('"', '""') + '"'
                quoted_c = '"' + column.name.replace('"', '""') + '"'
                sql = (
                    f"SELECT DISTINCT {quoted_c} FROM {quoted_t} "
                    f"WHERE typeof({quoted_c}) = 'text' LIMIT {limit_per_column}"
                    if isinstance(engine, SQLiteEngine)
                    else f"SELECT DISTINCT CAST({quoted_c} AS VARCHAR) FROM {quoted_t} "
                    f"LIMIT {limit_per_column}"
                )
                try:
                    rows = engine.run(sql, row_limit=limit_per_column, timeout_ms=10_000).rows
                except QueryFailed:
                    continue
                values.update(str(r[0]) for r in rows if isinstance(r[0], str))
    return sorted(values)


def _load_deepset(data_root: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for remote in DEEPSET_FILES:
        path = fetch(remote, data_root / "raw")
        con = duckdb.connect()
        try:
            rows += [
                (str(t), int(label))
                for t, label in con.execute(
                    "SELECT text, label FROM read_parquet(?)", [str(path)]
                ).fetchall()
            ]
        finally:
            con.close()
    return rows


def injection_eval(
    data_root: Path, text_sources: Sequence[Path], *, external: bool = True
) -> dict[str, Any]:
    detector = InjectionDetector()
    out: dict[str, Any] = {
        "development_set": {
            "injections_flagged": sum(detector.scan(t) is not None for t in INJECTIONS),
            "injections": len(INJECTIONS),
            "benign_flagged": sum(detector.scan(t) is not None for t in BENIGN_TEXTS),
            "benign": len(BENIGN_TEXTS),
        }
    }
    texts = _distinct_text(text_sources)
    flagged = [t for t in texts if detector.scan(t) is not None]
    out["database_text"] = {
        "distinct_values": len(texts),
        "flagged": len(flagged),
        "false_positive_rate": pct(len(flagged), len(texts)),
        "flagged_examples": flagged[:20],
        "sources": [p.name for p in text_sources],
    }
    if external:
        try:
            labelled = _load_deepset(data_root)
        except DatasetError as exc:
            out["external"] = {"skipped": str(exc)}
            return out
        positives = [t for t, label in labelled if label == 1]
        negatives = [t for t, label in labelled if label == 0]
        tp = sum(detector.scan(t) is not None for t in positives)
        fp = sum(detector.scan(t) is not None for t in negatives)
        out["external"] = {
            "dataset": "deepset/prompt-injections (train + test, revision "
            + _DEEPSET_REV[:7]
            + ")",
            "injections": len(positives),
            "benign": len(negatives),
            "recall": pct(tp, len(positives)),
            "false_positive_rate": pct(fp, len(negatives)),
            "precision": pct(tp, tp + fp),
            "note": "Rules were written before this set was loaded and were not tuned on it. "
            "Many of its injections are German or are jailbreak role-play prompts, which "
            "this English, database-oriented rule set does not target.",
        }
    return out


# ------------------------------------------------------------------ red team


def redteam_eval() -> dict[str, Any]:
    results = run_attacks()
    table: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dialect, category), group in _group(results).items():
        table[category][dialect] = {
            "cases": len(group),
            "passed": sum(r.passed for r in group),
            "outcomes": dict(Counter(r.outcome for r in group)),
        }
    hostile = [r for r in results if r.category not in {"benign", "resource"}]
    engine_only = {
        dialect: dict(Counter(r.engine_only for r in hostile if r.dialect == dialect))
        for dialect in ("sqlite", "duckdb")
    }
    engine_only_ran = sorted({f"{r.dialect}: {r.sql}" for r in hostile if r.engine_only == "ran"})
    scenarios = [asdict(run_injection_scenario(detector_on=on)) for on in (True, False)]
    return {
        "by_category": dict(table),
        "total_cases": len(results),
        "total_passed": sum(r.passed for r in results),
        "database_changed": sum(not r.database_unchanged for r in results),
        "engine_only_outcomes": engine_only,
        "engine_only_ran": engine_only_ran,
        "failures": [asdict(r) for r in results if not r.passed],
        "scenarios": scenarios,
    }


def _group(results: Sequence[AttackResult]) -> dict[tuple[str, str], list[AttackResult]]:
    grouped: dict[tuple[str, str], list[AttackResult]] = defaultdict(list)
    for r in results:
        grouped[(r.dialect, r.category)].append(r)
    return grouped


# ------------------------------------------------------------------ latency


def _time(fn: Callable[[], object], repeats: int) -> list[float]:
    fn()  # warm-up (catalog caches, first connection)
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        out.append((time.perf_counter() - start) * 1000)
    return out


def latency_eval(warehouse: Path, repeats: int = 30) -> dict[str, Any]:
    cfg = QuarryConfig()
    db = Database(warehouse, sample_values=cfg.linking.sample_values)
    deps = AgentDeps(llm=ScriptedChatModel(responder=DemoResponder.load()), database=db, config=cfg)
    toolbox = deps.toolbox
    q1 = DemoResponder.load().questions[0].sql[-1]
    out: dict[str, Any] = {
        "database": f"{warehouse.name} ({db.dialect}, 64,000 customers)",
        "tools": {
            "list_tables": latency_summary(_time(toolbox.list_tables, repeats)),
            "describe_table": latency_summary(
                _time(lambda: toolbox.describe_table("outcomes"), repeats)
            ),
            "sample_rows": latency_summary(
                _time(lambda: toolbox.sample_rows("customers", 5), repeats)
            ),
            "run_sql (3-table join, 64k rows)": latency_summary(
                _time(lambda: toolbox.run_sql(q1), repeats)
            ),
        },
        "guard_check": latency_summary(_time(lambda: deps.executor.guard.check(q1), repeats)),
        "static_check": latency_summary(_time(lambda: static_errors(q1, db.catalog()), repeats)),
        "schema_link": latency_summary(_time(lambda: deps.linker.link(DEMO_QUESTIONS[0]), repeats)),
        "catalog_introspection_cold": latency_summary(
            _time(lambda: Database(warehouse, sample_values=8).catalog(), 5)
        ),
    }
    with tempfile.TemporaryDirectory() as traces:
        # Tracing on, as in normal use: every run appends its spans to a JSONL file.
        agent = QuarryAgent(deps, trace_dir=Path(traces))
        out["offline_agent_run (scripted model, 5 LLM calls, 1 repair, JSONL trace)"] = (
            latency_summary(_time(lambda: agent.ask(DEMO_QUESTIONS[0]), AGENT_RUNS))
        )
    return out


AGENT_RUNS = 15


# ------------------------------------------------------------------ orchestration


def run_offline_suite(data_root: Path, out_dir: Path, *, console: Console) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    warehouse = data_root / "warehouse.sqlite"
    if not warehouse.exists():
        raise DatasetError(f"{warehouse} is missing; run `quarry demo build` first")
    report: dict[str, Any] = {
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
    }
    layout = SpiderLayout(data_root / "spider")
    spider_ready = (layout.root / "dev.json").exists() and (layout.root / "tables.json").exists()
    dbs_ready = spider_ready and all(layout.db_path(d).exists() for d in layout.dev_db_ids())
    if spider_ready:
        dev = layout.examples("dev")
        console.print("[bold]schema linking[/] on Spider dev")
        report["linking"] = linking_recall(layout, dev)
        console.print("[bold]guard[/] on Spider gold SQL (dev + train)")
        report["guard_on_gold"] = guard_on_gold(layout)
        console.print("[bold]static check[/] on Spider dev gold")
        report["static_check_on_gold"] = static_check_on_gold(layout, dev)
        report["hardness"] = hardness_distribution(dev)
        if dbs_ready:
            console.print("[bold]sandbox[/] executing Spider dev gold")
            report["gold_in_sandbox"] = gold_through_sandbox(layout, dev)
    else:
        report["spider"] = "skipped: run `quarry data spider` first"
    console.print("[bold]red team[/]")
    report["redteam"] = redteam_eval()
    console.print("[bold]injection detector[/]")
    sources = [warehouse] + ([layout.db_path(d) for d in layout.dev_db_ids()] if dbs_ready else [])
    report["injection"] = injection_eval(data_root, sources)
    console.print("[bold]latency[/]")
    report["latency"] = latency_eval(warehouse)
    json_path = out_dir / "offline.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "offline.md").write_text(render_markdown(report), encoding="utf-8")
    return json_path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    hw = report["hardware"]
    lines = [
        "# quarry: offline measurements (no LLM)",
        "",
        f"Measured {report['measured_at']} on {hw['cpu']} ({hw['logical_cpus']} logical CPUs), "
        f"{hw['os']}, Python {hw['python']}. Regenerate with `uv run quarry eval offline`.",
        "",
    ]
    if "linking" in report:
        link = report["linking"]
        lines += [
            "## Schema linking on Spider dev",
            "",
            f"{link['examples']} questions; {link['mean_tables_per_db']} tables and "
            f"{link['mean_columns_per_db']} columns per database on average. 'All' = every gold "
            "table (column) is in the top k; 'mean' = average share of gold items in the top k.",
            "",
            "| Variant | Tables all@1 | all@3 | all@5 | Columns all@10 | all@20 | all@30 "
            "| Selected set (top-5 + bridges) | p50 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, v in link["variants"].items():
            t, c = v["table_recall_all_at_k"], v["column_recall_all_at_k"]
            lines.append(
                f"| {label} | {_fmt(t[1])} | {_fmt(t[3])} | {_fmt(t[5])} | {_fmt(c[10])} | "
                f"{_fmt(c[20])} | {_fmt(c[30])} | {_fmt(v['selected_set_recall'])} "
                f"(avg {v['selected_set_mean_size']} tables) | {_fmt(v['latency']['p50_ms'])} |"
            )
        base = link["random_order_baseline"]
        t, c = base["table_recall_all_at_k"], base["column_recall_all_at_k"]
        lines.append(
            f"| *random order (expected)* | {_fmt(t[1])} | {_fmt(t[3])} | {_fmt(t[5])} | "
            f"{_fmt(c[10])} | {_fmt(c[20])} | {_fmt(c[30])} | - | - |"
        )
        lines.append("")
    if "guard_on_gold" in report:
        g = report["guard_on_gold"]
        lines += [
            "## SQL guard on Spider gold queries",
            "",
            "| Split | Files | Queries | Allowed | Pass rate |",
            "|---|---|---:|---:|---:|",
        ]
        for split in ("dev", "train"):
            if split in g:
                s = g[split]
                files = ", ".join(f"{k} ({v:,})" for k, v in s.get("by_file", {}).items())
                lines.append(
                    f"| {split} | {files} | {s['queries']} | {s['allowed']} | "
                    f"{_fmt(s['pass_rate'])}% |"
                )
        if "train_others" not in g.get("train", {}).get("by_file", {"train_others": 0}):
            lines += [
                "",
                "train_others (1,659 queries) is not checked: it ships only in the full Spider "
                "archive, which the authors' mirror used here does not include.",
            ]
        lines.append("")
        for split in ("dev", "train"):
            for b in g.get(split, {}).get("blocked", [])[:10]:
                lines.append(f"- blocked `{b['id']}`: {b['violations']} - `{b['sql'][:120]}`")
        lines.append("")
    if "static_check_on_gold" in report:
        s = report["static_check_on_gold"]
        lines += [
            "## Static check on Spider dev gold",
            "",
            f"{s['false_alarms']} of {s['queries']} gold queries flagged "
            f"({_fmt(s['false_alarm_rate'])}% false alarms).",
            "",
        ]
    if "gold_in_sandbox" in report:
        s = report["gold_in_sandbox"]
        lines += [
            "## Spider dev gold through the full sandbox",
            "",
            f"{s['ran_in_sandbox']} of {s['queries']} ran through guard + read-only engine; "
            f"{s['ran_unsandboxed']} run on a plain read-only connection; "
            f"{len(s['sandbox_only_failures'])} fail only in the sandbox.",
            "",
        ]
    if "hardness" in report:
        h = report["hardness"]
        lines += [
            "## Spider difficulty labels (dev)",
            "",
            "| | easy | medium | hard | extra |",
            "|---|---:|---:|---:|---:|",
            "| this implementation | "
            + " | ".join(
                str(h["this_implementation"][k]) for k in ("easy", "medium", "hard", "extra")
            )
            + " |",
            "| official | "
            + " | ".join(str(h["official"][k]) for k in ("easy", "medium", "hard", "extra"))
            + " |",
            "",
        ]
    r = report["redteam"]
    lines += [
        "## Red team",
        "",
        f"{r['total_passed']} of {r['total_cases']} cases passed; the database file changed "
        f"{r['database_changed']} times.",
        "",
        "| Category | SQLite | DuckDB |",
        "|---|---:|---:|",
    ]
    for category, by_dialect in r["by_category"].items():
        cells = [
            f"{by_dialect[d]['passed']}/{by_dialect[d]['cases']}" if d in by_dialect else "-"
            for d in ("sqlite", "duckdb")
        ]
        lines.append(f"| {category} | {cells[0]} | {cells[1]} |")
    lines += ["", "Engine alone (guard bypassed), hostile cases:", ""]
    for dialect, counts in r["engine_only_outcomes"].items():
        lines.append(f"- {dialect}: {counts}")
    for item in r["engine_only_ran"]:
        lines.append(f"  - ran without the guard: `{item}`")
    lines += [
        "",
        "Planted-injection scenario (a scripted model that runs any of three planted write "
        "commands it reads in rows):",
        "",
    ]
    for s in r["scenarios"]:
        lines.append(
            f"- detector {'on' if s['detector_on'] else 'off'}: "
            f"{s['withheld_values']} values withheld, "
            f"{s['writes_attempted']} writes attempted, {s['writes_executed']} executed, "
            f"database unchanged: {s['database_unchanged']}, status {s['status']}"
        )
    inj = report["injection"]
    d = inj["development_set"]
    db_text = inj["database_text"]
    lines += [
        "",
        "## Injection detector",
        "",
        "- Development strings (used while writing the rules): "
        f"{d['injections_flagged']}/{d['injections']} "
        f"injections flagged, {d['benign_flagged']}/{d['benign']} benign notes flagged.",
        f"- Real database text: {db_text['flagged']} of {db_text['distinct_values']} "
        "distinct values "
        f"flagged ({_fmt(db_text['false_positive_rate'])}%).",
    ]
    ext = inj.get("external", {})
    if "recall" in ext:
        lines.append(
            f"- External, unseen ({ext['dataset']}): recall {_fmt(ext['recall'])}% on "
            f"{ext['injections']} injections, false-positive rate "
            f"{_fmt(ext['false_positive_rate'])}% on "
            f"{ext['benign']} benign prompts, precision {_fmt(ext['precision'])}%. {ext['note']}"
        )
    lat = report["latency"]
    lines += [
        "",
        f"## Latency ({lat['database']})",
        "",
        "| Operation | n | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    rows = {**{f"tool: {k}": v for k, v in lat["tools"].items()}}
    for key in ("guard_check", "static_check", "schema_link", "catalog_introspection_cold"):
        rows[key] = lat[key]
    for key, value in lat.items():
        if key.startswith("offline_agent_run"):
            rows[key] = value
    for key, v in rows.items():
        lines.append(f"| {key} | {v['n']} | {_fmt(v['p50_ms'])} | {_fmt(v['p95_ms'])} |")
    lines.append("")
    return "\n".join(lines)
