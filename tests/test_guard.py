from __future__ import annotations

import pytest

from quarry.errors import QueryRejected
from quarry.guard import SQLGuard
from quarry.guard.policy import MAX_SQL_CHARS, Dialect

TABLES = ["customers", "orders", "campaigns"]


@pytest.fixture
def guard() -> SQLGuard:
    return SQLGuard("sqlite", TABLES)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM customers",
        "select count(*) from orders where amount > 10",
        "WITH c AS (SELECT * FROM campaigns) SELECT name FROM c",
        "SELECT name FROM campaigns UNION SELECT name FROM customers",
        "SELECT a.x FROM customers AS a JOIN orders AS b ON a.id = b.customer_id",
        "SELECT ';' AS semicolon FROM customers",
        "SELECT 'DROP TABLE customers' AS text",
        "SELECT COUNT(*) FROM customers;",
        "-- comment\nSELECT 1",
        "SELECT strftime('%Y', created) AS y, total(amount) FROM orders GROUP BY 1",
        "SELECT row_number() OVER (ORDER BY id) FROM customers",
        "SELECT * FROM (SELECT id FROM customers) AS sub",
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 5) "
        "SELECT x FROM r",
    ],
)
def test_read_only_queries_pass(guard: SQLGuard, sql: str) -> None:
    verdict = guard.check(sql)
    assert verdict.allowed, verdict.violations


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("DROP TABLE customers", "statement_type"),
        ("DELETE FROM customers", "statement_type"),
        ("UPDATE orders SET amount = 0", "statement_type"),
        ("INSERT INTO campaigns VALUES (1)", "statement_type"),
        ("CREATE TABLE x AS SELECT 1", "statement_type"),
        ("PRAGMA writable_schema = 1", "statement_type"),
        ("ATTACH DATABASE 'x.db' AS x", "statement_type"),
        ("VACUUM", "statement_type"),
        ("REINDEX", "statement_type"),
        ("SELECT 1; DROP TABLE customers", "multiple_statements"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("WITH d AS (DELETE FROM customers RETURNING *) SELECT * FROM d", "forbidden_clause"),
        ("SELECT * INTO backup FROM customers", "forbidden_clause"),
        ("SELECT * FROM customers FOR UPDATE", "forbidden_clause"),
        ("SELECT load_extension('evil.so')", "function"),
        ("SELECT readfile('/etc/passwd')", "function"),
        ("SELECT randomblob(100)", "function"),
        ("SELECT * FROM pragma_table_info('customers')", "table_function"),
        ("SELECT * FROM read_csv('x.csv')", "table_function"),
        ("SELECT * FROM sqlite_master", "unknown_table"),
        ("SELECT * FROM other.customers", "qualified_table"),
        ("SELECT * FROM 'secrets.csv'", "file_access"),
        ("SELECT * FROM (", "parse_error"),
        ("", "empty"),
    ],
)
def test_dangerous_statements_are_refused(guard: SQLGuard, sql: str, rule: str) -> None:
    verdict = guard.check(sql)
    assert not verdict.allowed
    assert rule in {v.rule for v in verdict.violations}


def test_unknown_table_suggests_a_close_name(guard: SQLGuard) -> None:
    verdict = guard.check("SELECT * FROM customer")
    assert any("did you mean customers" in v.detail for v in verdict.violations)


def test_cte_names_shadow_the_table_allowlist(guard: SQLGuard) -> None:
    assert guard.check("WITH secret AS (SELECT 1 AS x) SELECT x FROM secret").allowed


@pytest.mark.parametrize(
    ("sql", "allowed"),
    [
        # in scope: the body, its subqueries, later CTEs, nested WITHs, the recursive branch
        ("WITH c AS (SELECT 1 AS v) SELECT * FROM (SELECT * FROM c) AS s", True),
        ("WITH a AS (SELECT 1 AS v), b AS (SELECT * FROM a) SELECT * FROM b", True),
        ("WITH c AS (SELECT 1 AS v) SELECT * FROM (WITH d AS (SELECT * FROM c) SELECT * FROM d)",
         True),
        ("WITH RECURSIVE r(v) AS (SELECT 1 UNION ALL SELECT v + 1 FROM r WHERE v < 3) "
         "SELECT v FROM r", True),
        # out of scope: a CTE inside one subquery must not vouch for the same name elsewhere
        ("SELECT b.* FROM (WITH sqlite_master AS (SELECT 1 AS x) SELECT x FROM sqlite_master) "
         "AS a, sqlite_master AS b", False),
        ("SELECT * FROM (WITH c AS (SELECT 1 AS v) SELECT * FROM c) AS s, (SELECT * FROM c) AS t",
         False),
        ('SELECT b.* FROM (WITH "secrets.csv" AS (SELECT 1 AS x) SELECT x FROM "secrets.csv") '
         'AS a, "secrets.csv" AS b', False),
        # a qualified name is the stored object, never a CTE
        ("WITH sqlite_master AS (SELECT 1 AS x) SELECT * FROM main.sqlite_master", False),
        # a CTE does not see itself unless recursive, nor later siblings, nor the anchor
        ("WITH sqlite_master AS (SELECT * FROM sqlite_master) SELECT * FROM sqlite_master",
         False),
        ("WITH b AS (SELECT * FROM a), a AS (SELECT 1 AS v) SELECT * FROM b", False),
        ("WITH RECURSIVE r(v) AS (SELECT * FROM r UNION ALL SELECT 1) SELECT v FROM r", False),
    ],
)  # fmt: skip
@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_cte_names_count_only_where_the_cte_is_in_scope(
    dialect: Dialect, sql: str, allowed: bool
) -> None:
    verdict = SQLGuard(dialect, TABLES).check(sql)
    assert verdict.allowed is allowed, verdict.violations


def test_duckdb_specific_refusals() -> None:
    guard = SQLGuard("duckdb", TABLES)
    for sql in ("INSTALL httpfs", "LOAD httpfs", "COPY customers TO 'x.csv'", "SET threads = 1"):
        assert not guard.check(sql).allowed, sql
    assert not guard.check("SELECT * FROM glob('*')").allowed
    assert guard.check("SELECT epoch(now()) AS t FROM customers").allowed


def test_too_long_statements_are_refused(guard: SQLGuard) -> None:
    sql = "SELECT 1 AS x" + " " * MAX_SQL_CHARS
    assert guard.check(sql + "-- end").violations[0].rule == "too_long"


def test_raise_if_blocked(guard: SQLGuard) -> None:
    guard.check("SELECT 1").raise_if_blocked()
    with pytest.raises(QueryRejected) as info:
        guard.check("DROP TABLE customers").raise_if_blocked()
    assert info.value.reasons


def test_verdict_reports_tables_and_functions(guard: SQLGuard) -> None:
    verdict = guard.check("SELECT COUNT(*), upper(name) FROM customers JOIN orders ON 1 = 1")
    assert verdict.tables == {"customers", "orders"}
    assert {"count", "upper"} <= verdict.functions


def test_without_a_catalog_file_paths_are_still_refused() -> None:
    guard = SQLGuard("duckdb")
    assert not guard.check("SELECT * FROM 'data/x.parquet'").allowed
    assert guard.check("SELECT * FROM anything").allowed
