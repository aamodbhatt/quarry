"""The SQL allowlist, enforced on the sqlglot AST before anything reaches a database.

A statement is allowed only if **all** of these hold:

1. it parses, and it is exactly one statement;
2. its root is a ``SELECT`` (optionally with ``WITH``) or a set operation of ``SELECT``\\ s;
3. no node anywhere in the tree is a write, DDL, transaction, ``PRAGMA``, ``ATTACH``,
   ``COPY``, ``SET``, ``INSTALL``/``LOAD``, ``SELECT ... INTO``, row lock or raw command
   (this also catches DML hidden in a CTE such as ``WITH d AS (DELETE ... RETURNING *)``);
4. every function is on an explicit allowlist of pure functions (so ``load_extension``,
   ``readfile``, ``read_csv``, ``query()`` and anything unknown are refused);
5. every ``FROM``/``JOIN`` source is a plain table name (no table-valued functions, no
   ``FROM 'file.csv'``), and, when the catalog is known, it is a table of that catalog or a
   CTE that is visible *at that point* of the statement (see :func:`visible_ctes`).

The guard is the first of two layers. The engine layer (read-only connections, the SQLite
authorizer, DuckDB with external access disabled) refuses the same things again, so a
bug in one layer is not enough to write to the database. See ``quarry.db.engines``.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Literal, get_args

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from quarry.errors import QueryRejected

Dialect = Literal["sqlite", "duckdb"]

MAX_SQL_CHARS = 20_000

# Statement and clause types that must never appear anywhere in an allowed tree, by sqlglot
# class name. All must exist (checked at import below): a rename must not shrink the list.
_FORBIDDEN_NODE_NAMES = (
    "DML",  # Insert, Update, Delete, Merge, Copy
    "DDL",  # Create and friends
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Copy",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Command",  # anything sqlglot could only keep as raw text (VACUUM, LOAD, EXPLAIN, ...)
    "Pragma",
    "Attach",
    "Detach",
    "Set",
    "Use",
    "Transaction",
    "Commit",
    "Rollback",
    "Install",
    "LoadData",
    "Describe",
    "Summarize",
    "Show",
    "Lock",
    "Into",
    "Kill",
    "Analyze",
    "Grant",
    "Revoke",
    "Refresh",
    "Cache",
    "Uncache",
    "Export",
)


def _resolve_node_types(names: Iterable[str]) -> tuple[type[exp.Expr], ...]:
    """Look the node classes up, failing closed: a missing class means sqlglot changed shape
    under the guard, so the guard refuses to load instead of silently checking less."""
    resolved: list[type[exp.Expr]] = []
    missing: list[str] = []
    for name in names:
        cls = getattr(exp, name, None)
        if isinstance(cls, type) and issubclass(cls, exp.Expr):
            resolved.append(cls)
        else:
            missing.append(name)
    if missing:
        raise ImportError(
            f"sqlglot {version('sqlglot')} has no node type(s) {', '.join(missing)}; the SQL "
            "guard will not run on an AST it was not written for (install sqlglot>=30.27,<31)"
        )
    return tuple(resolved)


FORBIDDEN_NODES: tuple[type[exp.Expr], ...] = _resolve_node_types(_FORBIDDEN_NODE_NAMES)

# sqlglot models some operators (AND, OR, XOR, ``->>``, ...) as Func subclasses. They are
# operators, not function calls, and none of them can reach a file or the network.
OPERATORS: tuple[type[exp.Expr], ...] = (exp.Connector, exp.Binary, exp.Unary, exp.Predicate)

# Canonical sqlglot names (``Func.sql_name()``) of functions sqlglot models explicitly.
# Every entry is a pure function: no I/O, no side effects, bounded output.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    """
    COUNT COUNT_IF SUM AVG MIN MAX GROUP_CONCAT STRING_AGG MEDIAN STDDEV STDDEV_POP
    STDDEV_SAMP VARIANCE VARIANCE_POP PERCENTILE_CONT PERCENTILE_DISC APPROX_DISTINCT
    ANY_VALUE CORR COVAR_POP COVAR_SAMP BOOL_AND BOOL_OR QUANTILE APPROX_QUANTILE MODE
    ARG_MAX ARG_MIN LOGICAL_AND LOGICAL_OR ROW_NUMBER RANK DENSE_RANK PERCENT_RANK CUME_DIST
    NTILE LAG LEAD FIRST_VALUE LAST_VALUE NTH_VALUE CASE IF COALESCE NULLIF CAST TRY_CAST
    GREATEST LEAST TYPEOF ABS ROUND FLOOR CEIL SIGN SQRT CBRT POWER EXP LN LOG LOG2 LOG10
    TRUNC RAND PI DEGREES RADIANS SIN COS TAN ASIN ACOS ATAN ATAN2 MOD SAFE_DIVIDE LENGTH
    LOWER UPPER SUBSTRING TRIM REPLACE STR_POSITION CONCAT CONCAT_WS HEX UNHEX UNICODE CHR
    LEFT RIGHT PAD REVERSE INITCAP SPLIT_PART REPEAT REGEXP_LIKE REGEXP_EXTRACT
    REGEXP_REPLACE STARTS_WITH ENDS_WITH CONTAINS ASCII FORMAT LIKE ILIKE GLOB DATE DATETIME
    TIME TIMESTAMP TIME_TO_STR STR_TO_TIME STR_TO_DATE TS_OR_DS_TO_DATE
    TS_OR_DS_TO_TIMESTAMP TIME_STR_TO_TIME TIME_TO_UNIX UNIX_TO_TIME DATE_TRUNC
    TIMESTAMP_TRUNC EXTRACT DATE_ADD DATE_SUB DATE_DIFF DATEDIFF TIMESTAMP_DIFF YEAR MONTH
    DAY DAY_OF_WEEK DAY_OF_MONTH DAY_OF_YEAR WEEK_OF_YEAR QUARTER CURRENT_DATE
    CURRENT_TIMESTAMP CURRENT_TIME LAST_DAY MAKE_DATE JSON_EXTRACT JSON_EXTRACT_SCALAR
    JSON_ARRAY_LENGTH JSON_TYPE JSON_VALID
    """.split()
)

# Functions sqlglot does not model (``exp.Anonymous``), allowed per dialect by lower-case
# name. Kept to built-ins documented as pure by SQLite and DuckDB.
ALLOWED_ANONYMOUS: dict[Dialect, frozenset[str]] = {
    "sqlite": frozenset(
        """
        total julianday datetime strftime time unixepoch timediff printf format quote glob
        like likelihood likely unlikely soundex octet_length iif instr ltrim rtrim ifnull
        char unhex mod pi degrees radians sin cos tan asin acos atan atan2 log2 log10 trunc
        ceiling pow substr concat_ws json json_array json_object json_quote json_group_array
        json_group_object
        """.split()
    ),
    "duckdb": frozenset(
        """
        total epoch epoch_ms now today date_part datepart dayname monthname strptime
        strftime age make_timestamp isodow dayofweek dayofyear week weekofyear yearweek
        printf format instr ltrim rtrim ifnull regexp_matches string_split list_value len
        strlen substr ceiling pow log2 log10 trunc nullif fsum favg product histogram
        entropy kurtosis skewness regr_slope regr_intercept regr_r2 bit_count even gamma
        lgamma factorial isnan isinf isfinite greatest least
        """.split()
    ),
}

# Named only to produce a sharper message; anything not allowlisted is refused anyway.
FILE_OR_SYSTEM_FUNCTIONS: frozenset[str] = frozenset(
    """
    load_extension readfile writefile edit fts3_tokenizer zipfile sqlar_compress
    sqlar_uncompress read_csv read_csv_auto read_parquet parquet_scan read_json
    read_json_auto read_ndjson read_text read_blob glob query query_table getenv
    current_setting pg_read_file lo_import lo_export sniff_csv randomblob zeroblob read_xlsx
    st_read iceberg_scan delta_scan
    """.split()
)


Rule = Literal[
    "empty",
    "too_long",
    "parse_error",
    "multiple_statements",
    "statement_type",
    "forbidden_clause",
    "function",
    "table_function",
    "qualified_table",
    "file_access",
    "unknown_table",
]
RULES: frozenset[str] = frozenset(get_args(Rule))


@dataclass(frozen=True)
class Violation:
    """One reason a statement was refused."""

    rule: Rule
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


@dataclass(frozen=True)
class GuardVerdict:
    """Outcome of :meth:`SQLGuard.check`."""

    sql: str
    violations: tuple[Violation, ...] = ()
    tables: frozenset[str] = field(default_factory=frozenset)
    functions: frozenset[str] = field(default_factory=frozenset)

    @property
    def allowed(self) -> bool:
        return not self.violations

    def raise_if_blocked(self) -> None:
        if self.violations:
            raise QueryRejected([str(v) for v in self.violations])


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name)
    return node.sql_name()


class SQLGuard:
    """Checks statements against the read-only allowlist for one dialect and catalog."""

    def __init__(self, dialect: Dialect, known_tables: Iterable[str] | None = None) -> None:
        self.dialect: Dialect = dialect
        self._known: dict[str, str] | None = (
            None if known_tables is None else {t.lower(): t for t in known_tables}
        )

    def check(self, sql: str) -> GuardVerdict:
        text = sql.strip()
        if not text:
            return GuardVerdict(sql, (Violation("empty", "no SQL was given"),))
        if len(text) > MAX_SQL_CHARS:
            return GuardVerdict(
                sql, (Violation("too_long", f"statement exceeds {MAX_SQL_CHARS} characters"),)
            )
        try:
            statements = _parse_quietly(text, self.dialect)
        except (ParseError, TokenError) as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return GuardVerdict(sql, (Violation("parse_error", first_line),))
        if len(statements) != 1:
            detail = (
                "no statement found"
                if not statements
                else f"{len(statements)} statements found; send exactly one SELECT"
            )
            return GuardVerdict(sql, (Violation("multiple_statements", detail),))
        return self._check_tree(sql, statements[0])

    def _check_tree(self, sql: str, root: exp.Expr) -> GuardVerdict:
        violations: list[Violation] = []
        if not isinstance(root, exp.Select | exp.SetOperation):
            violations.append(
                Violation(
                    "statement_type",
                    f"{type(root).__name__.upper()} is not allowed; only SELECT/WITH queries run",
                )
            )
        functions: set[str] = set()
        for node in root.walk():
            if isinstance(node, FORBIDDEN_NODES):
                violations.append(
                    Violation("forbidden_clause", f"{type(node).__name__.upper()} is not allowed")
                )
            elif isinstance(node, exp.Func) and not isinstance(node, OPERATORS):
                name = _function_name(node)
                functions.add(name.lower())
                if not self._function_allowed(node, name):
                    kind = (
                        "touches files or the system"
                        if name.lower() in FILE_OR_SYSTEM_FUNCTIONS
                        else "is not on the allowlist"
                    )
                    violations.append(Violation("function", f"{name}() {kind}"))
        tables = self._check_sources(root, violations)
        return GuardVerdict(
            sql, tuple(_dedupe(violations)), frozenset(tables), frozenset(functions)
        )

    def _function_allowed(self, node: exp.Func, name: str) -> bool:
        if isinstance(node, exp.Anonymous):
            return name.lower() in ALLOWED_ANONYMOUS[self.dialect]
        return name.upper() in ALLOWED_FUNCTIONS

    def _check_sources(self, root: exp.Expr, violations: list[Violation]) -> set[str]:
        tables: set[str] = set()
        for table in root.find_all(exp.Table):
            source = table.this
            if not isinstance(source, exp.Identifier):
                label = source.sql(dialect=self.dialect)[:60] if source is not None else "?"
                violations.append(
                    Violation("table_function", f"{label} is not a plain table reference")
                )
                continue
            name = source.name
            schema = table.db
            if table.args.get("catalog") is not None or (schema and schema.lower() != "main"):
                violations.append(
                    Violation("qualified_table", f"{table.sql()} reaches outside the database")
                )
                continue
            lowered = name.lower()
            # Only a bare name can mean a CTE (``main.x`` is always the stored object), and
            # only where that CTE is in scope: a CTE defined inside one subquery must not
            # vouch for the same name elsewhere in the statement.
            if not schema and lowered in visible_ctes(table):
                continue
            if _FILE_LIKE.search(name):
                violations.append(Violation("file_access", f"{name!r} looks like a file path"))
                continue
            if self._known is not None and lowered not in self._known:
                violations.append(
                    Violation("unknown_table", _unknown_table_detail(name, self._known))
                )
                continue
            tables.add(lowered)
        return tables


# A path separator, a URL scheme or a file extension: DuckDB would read such a "table" from
# disk or the network (a replacement scan), so it is refused whatever the catalog says.
_FILE_LIKE = re.compile(r"[/\\:]|\.[A-Za-z0-9]{1,8}$")


def visible_ctes(node: exp.Expr) -> set[str]:
    """Lower-case names of the CTEs a table reference at ``node`` can resolve to.

    Walks outwards scope by scope. A CTE is visible in the body of the query whose ``WITH``
    defines it (including that body's subqueries) and in the CTEs that follow it in the same
    ``WITH``. It refers to itself only from a non-anchor branch of a ``WITH RECURSIVE``
    union. Anything wider would let a CTE in one subquery vouch for a catalog table or a
    file with the same name somewhere else in the statement.
    """
    visible: set[str] = set()
    child: exp.Expr = node
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.With):
            if isinstance(child, exp.CTE) and child.arg_key == "expressions":
                visible.update(
                    cte.alias_or_name.lower() for cte in parent.expressions[: child.index]
                )
                if parent.args.get("recursive") and _in_recursive_branch(node, child):
                    visible.add(child.alias_or_name.lower())
            # Skip the query that owns this WITH: its CTEs were handled just above.
            owner = parent.parent
            if owner is None:
                break
            child, parent = owner, owner.parent
            continue
        with_ = parent.args.get("with_")
        if isinstance(with_, exp.With) and child is not with_:
            visible.update(cte.alias_or_name.lower() for cte in with_.expressions)
        child, parent = parent, parent.parent
    return visible


def _in_recursive_branch(node: exp.Expr, cte: exp.CTE) -> bool:
    """True if ``node`` sits in a recursive (non-anchor) branch of the CTE's UNION body."""
    body = cte.this
    if not isinstance(body, exp.Union):
        return False
    anchor: exp.Expr = body
    while isinstance(anchor, exp.Union):
        anchor = anchor.this
    ancestor: exp.Expr | None = node
    while ancestor is not None and ancestor is not cte:
        if ancestor is anchor:
            return False
        ancestor = ancestor.parent
    return ancestor is cte


def _unknown_table_detail(name: str, known: dict[str, str]) -> str:
    close = difflib.get_close_matches(name.lower(), list(known), n=3, cutoff=0.6)
    hint = f"; did you mean {', '.join(known[c] for c in close)}?" if close else ""
    return f"{name} is not a table in this database{hint}"


def _parse_quietly(sql: str, dialect: Dialect) -> list[exp.Expr]:
    """Parse without sqlglot's 'falling back to Command' warnings leaking to the console."""
    logger = logging.getLogger("sqlglot")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        return [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    finally:
        logger.setLevel(previous)


def _dedupe(violations: list[Violation]) -> list[Violation]:
    seen: set[Violation] = set()
    ordered: list[Violation] = []
    for violation in violations:
        if violation not in seen:
            seen.add(violation)
            ordered.append(violation)
    return ordered
