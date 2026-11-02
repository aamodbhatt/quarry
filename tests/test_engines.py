from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import sqlite3
from pathlib import Path

import pytest

from quarry.db import Database, SafeExecutor, render_schema
from quarry.db.engines import DuckDBEngine, SQLiteEngine, detect_dialect, open_engine, to_scalar
from quarry.errors import DatabaseError, QueryFailed, QueryTimeout
from quarry.guard import WITHHELD, InjectionDetector


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dialect_detection(warehouse: Path, warehouse_duckdb: Path, tmp_path: Path) -> None:
    assert detect_dialect(warehouse) == "sqlite"
    assert detect_dialect(warehouse_duckdb) == "duckdb"
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database at all")
    with pytest.raises(DatabaseError):
        detect_dialect(junk)
    with pytest.raises(DatabaseError, match="quarry demo build"):
        open_engine(tmp_path / "missing.sqlite")


SQLITE_REFUSALS = [
    "DELETE FROM customers",
    "PRAGMA writable_schema = 1",
    "ATTACH 'x.db' AS x",
    "SELECT randomblob(10)",
    "BEGIN",
    "CREATE TEMP TABLE t (x)",
    "SELECT * FROM sqlite_master",
]
DUCKDB_REFUSALS = [
    "DELETE FROM customers",
    "DROP TABLE outcomes",
    "SELECT * FROM read_csv('pyproject.toml')",
    "SET enable_external_access = true",
    "INSTALL httpfs",
]


@pytest.mark.parametrize("sql", SQLITE_REFUSALS)
def test_sqlite_engine_refuses_without_the_guard(warehouse: Path, sql: str) -> None:
    before = _digest(warehouse)
    with pytest.raises(QueryFailed):
        SQLiteEngine(warehouse).run(sql, row_limit=5, timeout_ms=1000)
    assert _digest(warehouse) == before


@pytest.mark.parametrize("sql", DUCKDB_REFUSALS)
def test_duckdb_engine_refuses_without_the_guard(warehouse_duckdb: Path, sql: str) -> None:
    before = _digest(warehouse_duckdb)
    with pytest.raises(QueryFailed):
        DuckDBEngine(warehouse_duckdb).run(sql, row_limit=5, timeout_ms=2000)
    assert _digest(warehouse_duckdb) == before


@pytest.mark.parametrize("engine_cls", [SQLiteEngine, DuckDBEngine])
def test_timeouts_interrupt_runaway_queries(
    engine_cls: type[SQLiteEngine] | type[DuckDBEngine], warehouse: Path, warehouse_duckdb: Path
) -> None:
    path = warehouse if engine_cls is SQLiteEngine else warehouse_duckdb
    sql = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r"
    with pytest.raises(QueryTimeout):
        engine_cls(path).run(sql, row_limit=5, timeout_ms=300)


def test_row_limit_truncates_and_flags(warehouse: Path) -> None:
    result = SQLiteEngine(warehouse).run("SELECT * FROM customers", row_limit=7, timeout_ms=2000)
    assert result.row_count == 7
    assert result.truncated
    small = SQLiteEngine(warehouse).run("SELECT * FROM campaigns", row_limit=7, timeout_ms=2000)
    assert not small.truncated


def test_both_engines_agree(warehouse: Path, warehouse_duckdb: Path) -> None:
    sql = (
        "SELECT c.name, COUNT(*) AS n, SUM(o.converted) AS conv FROM campaign_sends s "
        "JOIN campaigns c ON c.campaign_id = s.campaign_id "
        "JOIN outcomes o ON o.send_id = s.send_id GROUP BY c.name ORDER BY c.name"
    )
    a = SQLiteEngine(warehouse).run(sql, row_limit=10, timeout_ms=5000)
    b = DuckDBEngine(warehouse_duckdb).run(sql, row_limit=10, timeout_ms=5000)
    assert [tuple(r) for r in a.rows] == [tuple(r) for r in b.rows]


def test_catalog_introspection(warehouse: Path, warehouse_duckdb: Path) -> None:
    for path in (warehouse, warehouse_duckdb):
        catalog = Database(path).catalog()
        assert set(catalog.table_names) == {"campaigns", "customers", "campaign_sends", "outcomes"}
        sends = catalog.require_table("campaign_sends")
        assert {fk.ref_table for fk in sends.foreign_keys} == {"customers", "campaigns"}
        assert catalog.neighbours("campaigns") == {"campaign_sends"}
        assert catalog.require_table("outcomes").column("converted") is not None
        channel = catalog.require_table("customers").column("channel")
        assert channel is not None and "Web" in channel.samples
        converted = catalog.require_table("outcomes").column("converted")
        assert converted is not None and "conversion" in converted.description
    with pytest.raises(KeyError, match="tables:"):
        Database(warehouse).catalog().require_table("nope")


def test_render_schema_withholds_injected_samples(shop_db: Path) -> None:
    catalog = Database(shop_db).catalog()
    text = render_schema(catalog, ["reviews", "products"], detector=InjectionDetector())
    assert "DROP TABLE" not in text
    assert WITHHELD in text
    assert text.index("TABLE reviews") < text.index("TABLE products")
    assert "FOREIGN KEY product_id -> products.product_id" in text


def test_safe_executor_reports_each_failure_mode(shop_db: Path) -> None:
    executor = SafeExecutor(Database(shop_db), row_limit=5, timeout_ms=300)
    assert executor.run("SELECT name FROM products").ok
    blocked = executor.run("DROP TABLE orders")
    assert blocked.blocked and blocked.error and "guard" in blocked.error
    failed = executor.run("SELECT nope FROM products")
    assert not failed.ok and not failed.blocked and "nope" in (failed.error or "")
    slow = executor.run(
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r"
    )
    assert slow.timed_out


def test_to_scalar_conversions() -> None:
    assert to_scalar(decimal.Decimal("1.5")) == 1.5
    assert to_scalar(dt.date(2020, 1, 2)) == "2020-01-02"
    assert to_scalar(b"\x00\x01") == "<2 bytes>"
    assert to_scalar(float("nan")) == "nan"
    assert to_scalar(True) is True
    assert to_scalar([1]) == "[1]"


def test_sqlite_text_that_is_not_utf8_is_replaced_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "latin1.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE players (name TEXT)")
    conn.execute("INSERT INTO players VALUES (CAST(X'4D7565C76C6C6572' AS TEXT))")  # 'Mue\xc7ller'
    conn.commit()
    conn.close()
    result = SQLiteEngine(path).run("SELECT name FROM players", row_limit=5, timeout_ms=1000)
    assert result.rows == (("Mue\ufffdller",),)
