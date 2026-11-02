"""Checks around execution: static (before) and result sanity (after).

* :func:`static_errors` asks sqlglot's ``qualify`` to resolve every column against the
  catalog, which catches misspelt columns and wrong table aliases without touching the
  database. It is advisory: if sqlglot itself cannot handle a construct, nothing is
  reported and the database gets the final word.
* :func:`suspicions` looks at a result for the classic silent failures (no rows, all
  NULLs, raw rows where the question wanted a number). Each kind fires at most once per
  question, so a legitimately empty answer is accepted on the second try.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, SqlglotError
from sqlglot.optimizer.qualify import qualify

from quarry.db.catalog import Catalog
from quarry.db.engines import Scalar

_AGGREGATE_CUES = re.compile(
    r"\b(how many|number of|count|total|sum|average|avg|mean|median|rate|ratio|percent|"
    r"percentage|proportion|share)\b",
    re.IGNORECASE,
)
_LIST_CUES = re.compile(r"\b(list|show all|which|each|every|per|by)\b", re.IGNORECASE)


def static_errors(sql: str, catalog: Catalog) -> list[str]:
    """Unknown columns or tables that sqlglot can prove, as model-readable messages.

    SQLite reads a double-quoted name that matches no column as a string literal (a
    legacy behaviour plenty of real SQL relies on, e.g. ``WHERE city = "Paris"``). For
    SQLite, such a name is turned into a literal and the check repeated, but only where
    it is the value side of a comparison with a column (``=``, ``<>``, ``<``, ``LIKE``,
    ``IN (...)``, ``BETWEEN``). Anywhere else, for example ``SELECT "nmae" FROM t``, it is
    far more likely a misspelt identifier, and it is reported.
    """
    schema: dict[str, object] = {
        t.name: {c.name: (c.type or "TEXT") for c in t.columns} for t in catalog.tables
    }
    logger = logging.getLogger("sqlglot")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        tree = sqlglot.parse_one(sql, read=catalog.dialect)
        for _ in range(_MAX_LITERAL_FIXES):
            try:
                qualify(
                    tree.copy(),
                    schema=schema,
                    dialect=catalog.dialect,
                    validate_qualify_columns=True,
                    quote_identifiers=False,
                )
            except OptimizeError as exc:
                message = str(exc).splitlines()[0]
                unresolved = _UNRESOLVED.search(message)
                if (
                    catalog.dialect == "sqlite"
                    and unresolved
                    and _quoted_to_literal(tree, unresolved.group(1))
                ):
                    continue
                if re.search(
                    r"could not be resolved|Unknown table|Unknown column|Ambiguous", message
                ):
                    return [message]
                return []
            return []
    except (ParseError, SqlglotError, KeyError, ValueError, TypeError, AttributeError):
        return []
    finally:
        logger.setLevel(previous)
    return []


_MAX_LITERAL_FIXES = 8
_UNRESOLVED = re.compile(r"Column '([^']+)' could not be resolved")


def _quoted_to_literal(tree: exp.Expr, name: str) -> bool:
    """Turn ``name`` into a string where it is compared with a column; report if changed."""
    changed = False
    for column in list(tree.find_all(exp.Column)):
        ident = column.this
        if (
            isinstance(ident, exp.Identifier)
            and ident.quoted
            and not column.table
            and ident.name.lower() == name.lower()
            and _compared_with_a_column(column)
        ):
            column.replace(exp.Literal.string(ident.name))
            changed = True
    return changed


_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike, exp.Glob)


def _compared_with_a_column(node: exp.Expr) -> bool:
    """True if ``node`` is a value operand whose other side is (or contains) a column."""
    while isinstance(node.parent, exp.Paren):
        node = node.parent
    parent = node.parent
    if isinstance(parent, _COMPARISONS):
        other = parent.expression if node is parent.this else parent.this
    elif (isinstance(parent, exp.In) and node.arg_key == "expressions") or (
        isinstance(parent, exp.Between) and node.arg_key in {"low", "high"}
    ):
        other = parent.this
    else:
        return False
    if other is None:
        return False
    return any(not _is_bare_quoted(c) for c in other.find_all(exp.Column))


def _is_bare_quoted(column: exp.Column) -> bool:
    return isinstance(column.this, exp.Identifier) and column.this.quoted and not column.table


@dataclass(frozen=True)
class Suspicion:
    kind: str
    message: str


def suspicions(
    question: str,
    sql: str,
    dialect: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Scalar]],
    already: Sequence[str],
) -> list[Suspicion]:
    """Result patterns that usually mean the query is wrong, minus kinds already raised."""
    found: list[Suspicion] = []
    if not rows:
        found.append(
            Suspicion(
                "empty",
                "The query returned no rows. Check filter values against the sampled values "
                "(exact spelling and case), join keys and date formats. If an empty answer is "
                "correct, submit the same query again.",
            )
        )
    elif all(v is None for row in rows for v in row):
        found.append(
            Suspicion(
                "all_null",
                "Every value in the result is NULL, which usually means an aggregate ran over "
                "no rows or a join failed. Fix it, or resubmit the same query if NULL is right.",
            )
        )
    elif len(rows) > 1 and _wants_single_number(question) and not _aggregates(sql, dialect):
        found.append(
            Suspicion(
                "raw_rows",
                f"The question asks for a number but the query returned {len(rows)} raw rows "
                "with no aggregate. Aggregate in SQL, or resubmit if rows are really wanted.",
            )
        )
    return [s for s in found if s.kind not in already]


def _wants_single_number(question: str) -> bool:
    return bool(_AGGREGATE_CUES.search(question)) and not _LIST_CUES.search(question)


def _aggregates(sql: str, dialect: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError:
        return True
    return tree.find(exp.AggFunc) is not None or tree.find(exp.Group) is not None
