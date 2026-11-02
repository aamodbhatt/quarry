"""Property tests: nothing outside the allowlist ever reaches a database engine.

A spy engine records every statement the guard lets through. Each recorded statement is
then judged by an oracle that shares no code with the guard and never uses sqlglot: the
database engine's own parser and planner.

* **SQLite**: the statement is compiled (``EXPLAIN``, so it never runs) on a separate
  read-only connection whose authorizer records every action SQLite asks permission for.
  Allowed: ``SELECT``, reads of the user tables, recursive CTEs, and calls to a short,
  fixed list of pure built-ins. A second statement in the text is a failure too.
* **DuckDB**: ``duckdb.extract_statements`` must find exactly one ``SELECT``; the planner's
  ``EXPLAIN (FORMAT JSON)`` may scan only user tables (no catalog views, no table
  functions, no files); DuckDB's own parse tree may call only pure functions.

Statements come from a grammar built to probe the guard's scoping logic: CTEs defined in
inner subqueries, names that shadow catalog views or look like files, table functions,
qualified names, recursion, unions and joins of subqueries; plus writes, stacked
statements, comment tricks and arbitrary text. Both engines run the same property, and the
database file must stay byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlglot import exp

from quarry.db import Catalog, Database, QueryResult, SafeExecutor
from quarry.db.engines import Dialect, Engine
from quarry.eval.redteam import export_redteam_duckdb
from quarry.guard import SQLGuard
from quarry.guard.policy import _FORBIDDEN_NODE_NAMES, FORBIDDEN_NODES, RULES

TABLES = ["campaigns", "customers", "campaign_sends", "outcomes", "customer_notes"]
COLUMNS = ["name", "channel", "customer_id", "campaign_id", "spend_usd", "visited"]


class SpyEngine:
    """Records every statement that gets past the guard, then runs it for real."""

    def __init__(self, inner: Engine) -> None:
        self.inner = inner
        self.dialect: Dialect = inner.dialect
        self.seen: list[str] = []

    def run(self, sql: str, *, row_limit: int, timeout_ms: int) -> QueryResult:
        self.seen.append(sql)
        return self.inner.run(sql, row_limit=row_limit, timeout_ms=timeout_ms)

    def introspect(self, *, sample_values: int) -> Catalog:
        return self.inner.introspect(sample_values=sample_values)


# ------------------------------------------------------------------------- oracles

# Pure built-ins the generators below may call, as each engine's own parser names them.
SQLITE_PURE = frozenset("count sum avg min max total upper lower round abs length random".split())
DUCKDB_PURE = frozenset(
    "count count_star sum avg min max upper lower round abs length random + - * /".split()
)
_ALLOWED_SQLITE_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}


def sqlite_oracle(path: Path, sql: str) -> list[str]:
    """What SQLite itself says the statement would do that the allowlist forbids."""
    problems: list[str] = []

    def authorize(
        action: int, arg1: str | None, arg2: str | None, db: str | None, _trigger: str | None
    ) -> int:
        if action in _ALLOWED_SQLITE_ACTIONS:
            return sqlite3.SQLITE_OK
        # A read with no database name is a CTE built inside the statement itself.
        if action == sqlite3.SQLITE_READ and (
            db is None or (db == "main" and arg1 is not None and arg1.lower() in TABLES)
        ):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and arg2 is not None and arg2.lower() in SQLITE_PURE:
            return sqlite3.SQLITE_OK
        problems.append(f"authorizer action {action} on {arg1!r}/{arg2!r}")
        return sqlite3.SQLITE_DENY

    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.set_authorizer(authorize)
        try:
            conn.execute("EXPLAIN " + sql)  # compiles the statement without running it
        except (sqlite3.ProgrammingError, sqlite3.Warning) as exc:
            problems.append(f"more than one statement: {exc}")
        except sqlite3.Error:
            pass  # SQLite cannot compile it, so it cannot run either
    finally:
        conn.close()
    return problems


_DUCKDB_CONFIG: dict[str, str | bool | int | float | list[str]] = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "lock_configuration": True,
}


def _plan_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("children", []):
        yield from _plan_nodes(child)


def _parsed_functions(tree: Any) -> Iterator[str]:
    if isinstance(tree, dict):
        if tree.get("class") == "FUNCTION":
            yield str(tree.get("function_name", "")).lower()
        for value in tree.values():
            yield from _parsed_functions(value)
    elif isinstance(tree, list):
        for value in tree:
            yield from _parsed_functions(value)


def duckdb_oracle(path: Path, sql: str) -> list[str]:
    """What DuckDB's parser and planner say the statement would do that is forbidden."""
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error:
        return []  # DuckDB cannot parse it, so it cannot run
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        return [f"statements: {[str(s.type) for s in statements]}"]
    problems: list[str] = []
    conn = duckdb.connect(str(path), read_only=True, config=_DUCKDB_CONFIG)
    try:
        serialized = conn.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
        parsed = json.loads(serialized[0]) if serialized else {}
        problems += [f"calls {f}()" for f in _parsed_functions(parsed) if f not in DUCKDB_PURE]
        try:
            rows = conn.execute("EXPLAIN (FORMAT JSON) " + sql).fetchall()
        except duckdb.PermissionException as exc:
            return [*problems, f"reaches outside the database: {exc}"]
        except duckdb.Error:
            return problems  # does not bind, so it cannot run
        for root in json.loads(rows[0][1]):
            for node in _plan_nodes(root):
                info = node.get("extra_info") or {}
                if "Table" in info:
                    table = str(info["Table"]).rsplit(".", 1)[-1].lower()
                    if table not in TABLES:
                        problems.append(f"scans {info['Table']}")
                elif "Function" in info:
                    problems.append(f"table function {info['Function']}")
    finally:
        conn.close()
    return problems


# ---------------------------------------------------------------------- generators

CATALOG = [
    "sqlite_master",
    "sqlite_schema",
    "sqlite_temp_master",
    "duckdb_tables",
    "duckdb_settings",
    "duckdb_views",
    "information_schema.tables",
    "main.sqlite_master",
    "main.duckdb_tables",
    "temp.sqlite_master",
]
FILES = ['"secrets.csv"', "'secrets.csv'", '"/etc/passwd"', '"data.parquet"', "'s3://b/x.csv'"]
TABLE_FUNCTIONS = [
    "read_csv('secrets.csv')",
    "pragma_table_info('customers')",
    "glob('*')",
    "range(3)",
    "duckdb_tables()",
    "query('SELECT 1')",
    "json_each('[1]')",
]
# Names a CTE may take: plain ones, user tables, catalog views and file-like names.
CTE_NAMES = ["c", "d", "customers", "campaigns", "sqlite_master", "duckdb_tables", '"secrets.csv"']
PROJECTIONS = [
    "*",
    "1 AS v",
    "COUNT(*) AS n",
    "UPPER('a') AS u",
    "ROUND(1.5) AS r",
    "load_extension('x') AS f",
    "readfile('x') AS f",
    "getenv('HOME') AS f",
    "current_setting('threads') AS f",
    "random() AS f",
]

names = st.sampled_from(CTE_NAMES)
projections = st.sampled_from(PROJECTIONS)
sources = st.one_of(
    st.sampled_from(TABLES),
    st.sampled_from(CTE_NAMES),
    st.sampled_from(CATALOG),
    st.sampled_from(FILES),
    st.sampled_from(TABLE_FUNCTIONS),
)
leaf_queries = st.builds(lambda p, s: f"SELECT {p} FROM {s}", projections, sources)


def _extend(inner: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.one_of(
        # a CTE, used (or not) by the body
        st.builds(lambda n, q, p, s: f"WITH {n} AS ({q}) SELECT {p} FROM {s}",
                  names, inner, projections, st.one_of(names, sources)),
        # a CTE defined inside a subquery, with the same name used outside it
        st.builds(lambda n, q, p: f"SELECT b.* FROM (WITH {n} AS ({q}) SELECT {p} FROM {n}) AS a, "
                                  f"{n} AS b", names, inner, projections),
        st.builds(lambda q, s: f"SELECT a.* FROM ({q}) AS a, {s} AS b", inner, sources),
        st.builds(lambda q1, q2: f"SELECT 1 AS v FROM ({q1}) AS a UNION ALL "
                                 f"SELECT 1 AS v FROM ({q2}) AS b", inner, inner),
        st.builds(lambda s, q: f"SELECT * FROM {s} WHERE 1 IN (SELECT 1 FROM ({q}) AS q)",
                  sources, inner),
        st.builds(lambda n, s, t: f"WITH RECURSIVE {n}(v) AS (SELECT 1 UNION ALL SELECT v + 1 "
                                  f"FROM {s} WHERE v < 3) SELECT COUNT(*) AS n FROM {t}",
                  names, st.one_of(names, sources), st.one_of(names, sources)),
    )  # fmt: skip


scoped_queries = st.recursive(leaf_queries, _extend, max_leaves=4)

dangerous = st.sampled_from(
    [
        "DROP TABLE customers",
        "DELETE FROM outcomes",
        "UPDATE outcomes SET spend_usd = 0",
        "INSERT INTO campaigns VALUES (9, 'x', NULL, 0, 'y')",
        "CREATE TABLE t AS SELECT 1",
        "ALTER TABLE customers ADD COLUMN x TEXT",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE 'x.db' AS x",
        "DETACH x",
        "VACUUM",
        "SELECT load_extension('x')",
        "SELECT readfile('/etc/passwd')",
        "SELECT * FROM sqlite_master",
        "WITH d AS (DELETE FROM customers RETURNING *) SELECT * FROM d",
        "SELECT * INTO b FROM customers",
        "BEGIN",
        "COMMIT",
        "REPLACE INTO campaigns VALUES (1, 'x', NULL, 0, 'y')",
        "SELECT * FROM pragma_table_info('customers')",
    ]
)
separators = st.sampled_from([";", "; ", ";\n", " ; -- x\n", "/* c */;", ";;"])


@st.composite
def stacked(draw: st.DrawFn) -> str:
    parts = draw(st.lists(st.one_of(scoped_queries, dangerous), min_size=1, max_size=3))
    glue = draw(separators)
    prefix = draw(st.sampled_from(["", " ", "-- note\n", "/* hi */ "]))
    return prefix + glue.join(parts)


fragments = st.lists(
    st.sampled_from(
        ["SELECT", "DROP", "TABLE", "customers", ";", "--", "/*", "*/", "'", "FROM", "WHERE",
         "PRAGMA", "ATTACH", "(", ")", "*", ",", "1", "load_extension", "WITH", "AS", "DELETE",
         "sqlite_master", "duckdb_tables"]
    ),
    max_size=12,
).map(" ".join)  # fmt: skip

any_statement = st.one_of(
    scoped_queries, scoped_queries, stacked(), dangerous, fragments, st.text(max_size=80)
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def engine_db(
    request: pytest.FixtureRequest, redteam_db: Path, tmp_path: Path
) -> tuple[Path, Callable[[Path, str], list[str]]]:
    if request.param == "sqlite":
        return redteam_db, sqlite_oracle
    return export_redteam_duckdb(redteam_db, tmp_path / "redteam.duckdb"), duckdb_oracle


@pytest.mark.parametrize("engine_db", ["sqlite", "duckdb"], indirect=True)
@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(sql=any_statement)
def test_nothing_outside_the_allowlist_reaches_the_engine(
    engine_db: tuple[Path, Callable[[Path, str], list[str]]], sql: str
) -> None:
    path, oracle = engine_db
    database = Database(path, sample_values=0)
    spy = SpyEngine(database.engine)
    database.engine = spy
    executor = SafeExecutor(database, row_limit=10, timeout_ms=500)
    before = _digest(path)

    executor.run(sql)

    for seen in spy.seen:
        assert oracle(path, seen) == [], seen
    assert _digest(path) == before


@pytest.mark.parametrize(
    ("sql", "flag_sqlite", "flag_duckdb"),
    [
        # The CTE-scope bypass the guard once let through: both oracles must catch it.
        (
            "SELECT b.* FROM (WITH sqlite_master AS (SELECT 1 AS x) SELECT x FROM sqlite_master) "
            "AS a, sqlite_master AS b",
            True,
            True,
        ),
        # duckdb_tables does not exist in SQLite (cannot compile there, so nothing to flag).
        (
            "SELECT b.* FROM (WITH duckdb_tables AS (SELECT 1 AS x) SELECT x FROM duckdb_tables) "
            "AS a, duckdb_tables AS b",
            False,
            True,
        ),
        ("SELECT 1; SELECT 2", True, True),
        ("SELECT COUNT(*) AS n FROM customers", False, False),
        (
            "WITH c AS (SELECT 1 AS v) SELECT * FROM (WITH d AS (SELECT * FROM c) SELECT * FROM d)",
            False,
            False,
        ),
    ],
)
def test_oracles_flag_what_the_allowlist_forbids(
    redteam_db: Path, tmp_path: Path, sql: str, flag_sqlite: bool, flag_duckdb: bool
) -> None:
    assert bool(sqlite_oracle(redteam_db, sql)) == flag_sqlite
    duck = export_redteam_duckdb(redteam_db, tmp_path / "r.duckdb")
    assert bool(duckdb_oracle(duck, sql)) == flag_duckdb


@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(sql=dangerous)
def test_dangerous_statements_never_execute(redteam_db: Path, sql: str) -> None:
    database = Database(redteam_db, sample_values=0)
    spy = SpyEngine(database.engine)
    database.engine = spy
    run = SafeExecutor(database, row_limit=10, timeout_ms=500).run(sql)
    assert run.blocked
    assert spy.seen == []


safe_select = st.builds(
    lambda col, table, limit: f"SELECT {col} FROM {table} LIMIT {limit}",
    st.sampled_from(["*", "COUNT(*)", "name", "1"]),
    st.sampled_from(TABLES),
    st.integers(1, 5),
)


@settings(max_examples=150, deadline=None)
@given(sql=safe_select)
def test_safe_selects_are_allowed_and_stable(sql: str) -> None:
    guard = SQLGuard("sqlite", TABLES)
    first = guard.check(sql)
    assert first.allowed
    assert guard.check(sql) == first


@settings(max_examples=200, deadline=None)
@given(text=st.one_of(st.text(max_size=200), fragments, scoped_queries))
def test_guard_verdicts_are_well_formed_on_any_text(text: str) -> None:
    for dialect in ("sqlite", "duckdb"):
        verdict = SQLGuard(dialect, TABLES).check(text)
        assert {v.rule for v in verdict.violations} <= RULES
        assert verdict.tables <= set(TABLES)
        assert SQLGuard(dialect, TABLES).check(text) == verdict


@settings(max_examples=60, deadline=None)
@given(column=st.sampled_from(COLUMNS), table=st.sampled_from(TABLES), stmt=dangerous)
def test_appending_anything_to_a_safe_select_is_refused(column: str, table: str, stmt: str) -> None:
    guard = SQLGuard("sqlite", TABLES)
    assert not guard.check(f"SELECT {column} FROM {table}; {stmt}").allowed


def test_every_forbidden_node_type_exists_in_this_sqlglot() -> None:
    # The guard resolves these at import and refuses to load if one is missing (fail closed).
    assert len(FORBIDDEN_NODES) == len(_FORBIDDEN_NODE_NAMES)
    assert all(issubclass(cls, exp.Expr) for cls in FORBIDDEN_NODES)
