"""Read-only database engines: the second guardrail layer.

Everything here assumes the SQL may be hostile even after the AST guard accepted it:

* **SQLite** files are opened with ``mode=ro`` (URI), ``PRAGMA query_only=ON``, extension
  loading off, tight ``sqlite3`` limits (value size, SQL length, no attached databases),
  a progress handler that enforces the time budget, and an *authorizer* that only lets
  ``SELECT``, column reads, recursive CTEs and allowlisted functions through. Writes,
  ``PRAGMA``, ``ATTACH``, transactions and unknown functions fail at prepare time.
* **DuckDB** files are opened ``read_only`` with ``enable_external_access=false`` (no file
  or network access), extension autoloading off, a memory cap and a locked configuration
  so a statement cannot switch any of that back on. A timer interrupts slow statements.

Each query gets its own short-lived connection, so no state leaks between queries.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import duckdb

from quarry.db.catalog import Catalog, Column, ForeignKey, Table
from quarry.errors import DatabaseError, QueryFailed, QueryTimeout

Dialect = Literal["sqlite", "duckdb"]
Scalar = None | bool | int | float | str


@dataclass(frozen=True)
class QueryResult:
    """Rows from one statement, already converted to JSON-safe scalars."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Scalar, ...], ...]
    truncated: bool
    elapsed_ms: float

    @property
    def row_count(self) -> int:
        return len(self.rows)


class Engine(Protocol):
    dialect: Dialect

    def run(self, sql: str, *, row_limit: int, timeout_ms: int) -> QueryResult:
        """Execute one statement on a fresh read-only connection."""
        ...

    def introspect(self, *, sample_values: int) -> Catalog:
        """Read the schema with trusted, internal queries."""
        ...


def to_scalar(value: Any) -> Scalar:
    """Convert a driver value to something JSON can carry without losing meaning."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.date | dt.datetime | dt.time):
        return value.isoformat()
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(bytes(value))} bytes>"
    return str(value)


def detect_dialect(path: Path) -> Dialect:
    """Pick the engine from the file header (SQLite magic, or DuckDB's ``DUCK`` at byte 8)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError as exc:
        raise DatabaseError(f"cannot open database {path}: {exc}") from exc
    if head.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if head[8:12] == b"DUCK":
        return "duckdb"
    raise DatabaseError(f"{path} is neither a SQLite nor a DuckDB database file")


# ---------------------------------------------------------------------------- SQLite

# Functions the SQLite authorizer lets through (lower-case). Built-in, pure, bounded.
SQLITE_FUNCTIONS: frozenset[str] = frozenset(
    """
    abs avg char coalesce concat concat_ws count cume_dist date datetime dense_rank
    first_value format glob group_concat hex ifnull iif if instr julianday lag last_value
    lead length like likelihood likely lower ltrim max min nth_value ntile nullif
    octet_length percent_rank printf quote random rank replace round row_number rtrim sign
    soundex strftime string_agg substr substring sum time timediff total trim typeof unhex
    unicode unixepoch unlikely upper acos acosh asin asinh atan atan2 atanh ceil ceiling cos
    cosh degrees exp floor ln log log10 log2 mod pi pow power radians sin sinh sqrt tan tanh
    trunc json json_array json_array_length json_extract json_object json_quote json_type
    json_valid json_group_array json_group_object ->> ->
    """.split()
)

_SQLITE_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}


def decode_text(raw: bytes) -> str:
    """SQLite TEXT to ``str``. Real databases (Spider's ``wta_1`` among them) hold bytes that
    are not valid UTF-8; replacing them beats failing every query that touches the row."""
    return raw.decode("utf-8", errors="replace")


def _sqlite_authorizer(
    action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None
) -> int:
    if action in _SQLITE_ALLOWED_ACTIONS:
        if (
            action == sqlite3.SQLITE_READ
            and arg1 is not None
            and arg1.lower().startswith("sqlite_")
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and arg2 is not None and arg2.lower() in SQLITE_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLiteEngine:
    dialect: Dialect = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._uri = path.resolve().as_uri() + "?mode=ro"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        except sqlite3.Error as exc:
            raise DatabaseError(f"cannot open {self.path} read-only: {exc}") from exc
        with closing(conn):
            conn.text_factory = decode_text
            conn.execute("PRAGMA query_only = ON")
            conn.enable_load_extension(False)
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 10_000_000)
            conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
            conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            yield conn

    def run(self, sql: str, *, row_limit: int, timeout_ms: int) -> QueryResult:
        start = time.perf_counter()
        deadline = start + timeout_ms / 1000
        timed_out = False

        def progress() -> int:
            nonlocal timed_out
            if time.perf_counter() > deadline:
                timed_out = True
                return 1
            return 0

        with self._connect() as conn:
            conn.set_authorizer(_sqlite_authorizer)
            conn.set_progress_handler(progress, 2_000)
            try:
                cursor = conn.execute(sql)
                columns = tuple(d[0] for d in cursor.description or ())
                raw = cursor.fetchmany(row_limit + 1)
            except sqlite3.Error as exc:
                if timed_out:
                    raise QueryTimeout(f"query exceeded {timeout_ms} ms and was stopped") from exc
                raise QueryFailed(_sqlite_message(exc)) from exc
        rows = tuple(tuple(to_scalar(v) for v in row) for row in raw[:row_limit])
        elapsed = (time.perf_counter() - start) * 1000
        return QueryResult(columns, rows, len(raw) > row_limit, round(elapsed, 3))

    def introspect(self, *, sample_values: int) -> Catalog:
        with self._connect() as conn:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            tables = [self._table(conn, name, sample_values) for name in names]
        return Catalog("sqlite", tuple(tables))

    def _table(self, conn: sqlite3.Connection, name: str, sample_values: int) -> Table:
        quoted = quote_ident(name)
        info = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
        fks = conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
        pk_cols = {r[1] for r in info if r[5]}
        foreign_keys = tuple(
            ForeignKey(column=str(r[3]), ref_table=str(r[2]), ref_column=str(r[4] or r[3]))
            for r in fks
        )
        try:
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        except sqlite3.Error:
            row_count = None
        columns = []
        for _cid, col_name, col_type, _notnull, _default, _pk in info:
            samples: tuple[str, ...] = ()
            if sample_values and _is_texty(str(col_type)):
                samples = _sample_sqlite(conn, quoted, quote_ident(col_name), sample_values)
            columns.append(
                Column(
                    name=str(col_name),
                    type=str(col_type).upper(),
                    primary_key=col_name in pk_cols,
                    samples=samples,
                )
            )
        return Table(name, tuple(columns), foreign_keys, row_count)


def _sample_sqlite(conn: sqlite3.Connection, table: str, column: str, k: int) -> tuple[str, ...]:
    query = (
        f"SELECT v, COUNT(*) AS n FROM (SELECT {column} AS v FROM {table} "
        f"WHERE {column} IS NOT NULL LIMIT 20000) GROUP BY v ORDER BY n DESC, v LIMIT {int(k)}"
    )
    try:
        return tuple(str(r[0]) for r in conn.execute(query) if str(r[0]).strip())
    except sqlite3.Error:
        return ()


def _sqlite_message(exc: sqlite3.Error) -> str:
    text = str(exc)
    if "not authorized" in text or "prohibited" in text:
        return f"blocked by the read-only sandbox ({text})"
    return text


# ---------------------------------------------------------------------------- DuckDB


class DuckDBEngine:
    dialect: Dialect = "duckdb"

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        config: dict[str, str | bool | int | float | list[str]] = {
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "threads": 2,
            "memory_limit": "512MB",
            "lock_configuration": True,
        }
        try:
            conn = duckdb.connect(str(self.path), read_only=True, config=config)
        except duckdb.Error as exc:
            raise DatabaseError(f"cannot open {self.path} read-only: {exc}") from exc
        try:
            yield conn
        finally:
            conn.close()

    def run(self, sql: str, *, row_limit: int, timeout_ms: int) -> QueryResult:
        start = time.perf_counter()
        with self._connect() as conn:
            timer = threading.Timer(timeout_ms / 1000, conn.interrupt)
            timer.start()
            try:
                cursor = conn.execute(sql)
                columns = tuple(d[0] for d in cursor.description or ())
                raw = cursor.fetchmany(row_limit + 1)
            except duckdb.InterruptException as exc:
                raise QueryTimeout(f"query exceeded {timeout_ms} ms and was stopped") from exc
            except duckdb.Error as exc:
                raise QueryFailed(str(exc).splitlines()[0]) from exc
            finally:
                timer.cancel()
        rows = tuple(tuple(to_scalar(v) for v in row) for row in raw[:row_limit])
        elapsed = (time.perf_counter() - start) * 1000
        return QueryResult(columns, rows, len(raw) > row_limit, round(elapsed, 3))

    def introspect(self, *, sample_values: int) -> Catalog:
        with self._connect() as conn:
            names = [
                str(r[0])
                for r in conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main' ORDER BY table_name"
                ).fetchall()
            ]
            constraints = conn.execute(
                "SELECT table_name, constraint_type, constraint_column_names, "
                "referenced_table, referenced_column_names FROM duckdb_constraints() "
                "WHERE schema_name = 'main'"
            ).fetchall()
            tables = [self._table(conn, name, constraints, sample_values) for name in names]
        return Catalog("duckdb", tuple(tables))

    def _table(
        self,
        conn: duckdb.DuckDBPyConnection,
        name: str,
        constraints: list[tuple[Any, ...]],
        sample_values: int,
    ) -> Table:
        quoted = quote_ident(name)
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
            [name],
        ).fetchall()
        pk_cols: set[str] = set()
        fks: list[ForeignKey] = []
        for t_name, c_type, c_cols, ref_table, ref_cols in constraints:
            if t_name != name:
                continue
            if c_type == "PRIMARY KEY":
                pk_cols.update(str(c) for c in c_cols)
            elif c_type == "FOREIGN KEY" and ref_table:
                for src, dst in zip(c_cols, ref_cols or c_cols, strict=False):
                    fks.append(ForeignKey(str(src), str(ref_table), str(dst)))
        count_row = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()
        columns = []
        for col_name, col_type in cols:
            samples: tuple[str, ...] = ()
            if sample_values and _is_texty(str(col_type)):
                query = (
                    f"SELECT v, COUNT(*) AS n FROM (SELECT {quote_ident(col_name)} AS v FROM "
                    f"{quoted} WHERE {quote_ident(col_name)} IS NOT NULL LIMIT 20000) "
                    f"GROUP BY v ORDER BY n DESC, v LIMIT {int(sample_values)}"
                )
                samples = tuple(str(r[0]) for r in conn.execute(query).fetchall())
            columns.append(
                Column(str(col_name), str(col_type).upper(), "", col_name in pk_cols, samples)
            )
        row_count = int(count_row[0]) if count_row else None
        return Table(name, tuple(columns), tuple(fks), row_count)


# ---------------------------------------------------------------------------- helpers


def quote_ident(name: str) -> str:
    """Quote an identifier for SQLite and DuckDB (both use double quotes)."""
    return '"' + name.replace('"', '""') + '"'


def _is_texty(declared: str) -> bool:
    upper = declared.upper()
    return not upper or any(k in upper for k in ("CHAR", "TEXT", "CLOB", "STRING", "VARCHAR"))


def open_engine(path: Path) -> Engine:
    """Return the engine for a SQLite or DuckDB file."""
    if not path.exists():
        raise DatabaseError(
            f"database {path} does not exist. Build the demo with `quarry demo build` "
            "or pass --db <file.sqlite|file.duckdb>."
        )
    dialect = detect_dialect(path)
    return SQLiteEngine(path) if dialect == "sqlite" else DuckDBEngine(path)
