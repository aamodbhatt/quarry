"""Execution-accuracy comparison and gold schema extraction.

``results_match`` follows the Spider test-suite convention: rows are compared as
multisets unless the gold query orders its output, and a prediction may list the same
columns in a different order.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from collections import Counter
from collections.abc import Sequence

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from quarry.db.engines import Scalar

Row = tuple[Scalar, ...]
MAX_PERMUTED_COLUMNS = 6


def has_top_level_order(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return False
    return tree.args.get("order") is not None


def _norm(value: Scalar) -> Scalar:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _rows(rows: Sequence[Sequence[Scalar]]) -> list[Row]:
    return [tuple(_norm(v) for v in row) for row in rows]


def results_match(
    gold: Sequence[Sequence[Scalar]],
    pred: Sequence[Sequence[Scalar]],
    *,
    order_matters: bool,
) -> bool:
    """Same rows (as multisets, or as lists when ordered), allowing a column permutation."""
    g, p = _rows(gold), _rows(pred)
    if len(g) != len(p):
        return False
    if not g:
        return True
    width = len(g[0])
    if any(len(r) != width for r in p):
        return False
    if _equal(g, p, order_matters):
        return True
    if width > MAX_PERMUTED_COLUMNS:
        return False
    gold_columns = [Counter(r[i] for r in g) for i in range(width)]
    pred_columns = [Counter(r[i] for r in p) for i in range(width)]
    for perm in itertools.permutations(range(width)):
        if any(gold_columns[i] != pred_columns[j] for i, j in enumerate(perm)):
            continue
        permuted = [tuple(r[j] for j in perm) for r in p]
        if _equal(g, permuted, order_matters):
            return True
    return False


def _equal(gold: list[Row], pred: list[Row], order_matters: bool) -> bool:
    if order_matters:
        return gold == pred
    return Counter(gold) == Counter(pred)


def gold_schema_items(sql: str, schema: dict[str, object]) -> tuple[set[str], set[tuple[str, str]]]:
    """Tables and (table, column) pairs a gold query reads, lower-cased.

    Columns are resolved per scope with sqlglot, so aliases such as ``T1`` map back to
    their tables even when subqueries reuse them.
    """
    logger = logging.getLogger("sqlglot")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        with contextlib.suppress(SqlglotError):
            tree = qualify(
                tree,
                schema=schema,
                dialect="sqlite",
                validate_qualify_columns=False,
                quote_identifiers=False,
            )
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        tables = {
            t.name.lower() for t in tree.find_all(exp.Table) if t.name.lower() not in cte_names
        }
        columns: set[tuple[str, str]] = set()
        for scope in traverse_scope(tree):
            for column in scope.columns:
                source = scope.sources.get(column.table)
                if isinstance(source, exp.Table) and source.name.lower() not in cte_names:
                    columns.add((source.name.lower(), column.name.lower()))
    except SqlglotError:
        return set(), set()
    finally:
        logger.setLevel(previous)
    return tables, columns
