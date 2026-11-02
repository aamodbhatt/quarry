"""Spider's difficulty levels (easy / medium / hard / extra), recomputed on the sqlglot AST.

Spider labels difficulty from counts over its own SQL parse (``evaluation.py``,
``eval_hardness``). This is a re-implementation of those rules on sqlglot, including two
quirks of the official code so the counts line up:

* in ``count_others`` the WHERE and HAVING "aggregate" counts look at the first field of
  each condition, which is the NOT flag, so negated conditions count as aggregates;
* the HAVING list also contains the ``and``/``or`` connectors, and each one counts.

The official dev split is 248 easy / 446 medium / 174 hard / 166 extra. The offline
report compares this version's per-level totals with those; it does not check
question-by-question agreement, so matching totals are evidence, not proof, that every
label agrees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

Hardness = str  # "easy" | "medium" | "hard" | "extra"


@dataclass(frozen=True)
class Components:
    component1: int
    component2: int
    others: int


def _conditions(node: exp.Expr | None) -> tuple[list[exp.Expr], int]:
    """Flatten AND/OR chains (not entering subqueries); return conditions and connector count."""
    if node is None:
        return [], 0
    if isinstance(node, exp.Paren) and not isinstance(node.this, exp.Query):
        return _conditions(node.this)
    if isinstance(node, exp.And | exp.Or):
        left, lc = _conditions(node.left)
        right, rc = _conditions(node.right)
        return left + right, lc + rc + 1
    return [node], 0


def _ors(node: exp.Expr | None) -> int:
    if node is None:
        return 0
    if isinstance(node, exp.Paren) and not isinstance(node.this, exp.Query):
        return _ors(node.this)
    if isinstance(node, exp.And | exp.Or):
        return _ors(node.left) + _ors(node.right) + (1 if isinstance(node, exp.Or) else 0)
    return 0


def _is_like(cond: exp.Expr) -> bool:
    inner = cond.this if isinstance(cond, exp.Not) else cond
    return isinstance(inner, exp.Like)


def _is_negated(cond: exp.Expr) -> bool:
    return isinstance(cond, exp.Not)


def _subqueries(cond: exp.Expr) -> int:
    """Top-level subqueries used as values in one condition (Spider's val1/val2 SQL)."""
    return sum(
        1
        for node in cond.walk(prune=lambda n: isinstance(n, exp.Subquery | exp.Query))
        if isinstance(node, exp.Subquery | exp.Query)
    )


def _has_agg(node: exp.Expr) -> bool:
    target = node.unalias() if isinstance(node, exp.Alias) else node
    if isinstance(target, exp.Distinct):
        return any(isinstance(e, exp.AggFunc) for e in target.expressions)
    return isinstance(target, exp.AggFunc)


def _leftmost_select(node: exp.Expr) -> exp.Select | None:
    while isinstance(node, exp.SetOperation):
        node = node.this
    if isinstance(node, exp.Subquery):
        return _leftmost_select(node.this)
    return node if isinstance(node, exp.Select) else None


def components(sql: str) -> Components | None:
    """The three counts Spider's hardness rule uses, or None if the SQL does not parse."""
    logger = logging.getLogger("sqlglot")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return None
    finally:
        logger.setLevel(previous)
    select = _leftmost_select(tree)
    if select is None:
        return None
    set_ops = 1 if isinstance(tree, exp.SetOperation) else 0

    where_node = select.args.get("where")
    having_node = select.args.get("having")
    where, _ = _conditions(where_node.this if where_node else None)
    having, having_connectors = _conditions(having_node.this if having_node else None)
    joins = select.args.get("joins") or []
    join_conds: list[exp.Expr] = []
    for join in joins:
        on = join.args.get("on")
        conds, _ = _conditions(on)
        join_conds += conds

    group = select.args.get("group")
    group_items = list(group.expressions) if group else []
    order = select.args.get("order")
    order_items = [o.this for o in order.expressions] if order else []

    c1 = 0
    c1 += 1 if where else 0
    c1 += 1 if group_items else 0
    c1 += 1 if order_items else 0
    c1 += 1 if select.args.get("limit") else 0
    c1 += len(joins)  # table units - 1
    c1 += _ors(where_node.this if where_node else None)
    c1 += _ors(having_node.this if having_node else None)
    c1 += sum(_ors(j.args.get("on")) for j in joins)
    c1 += sum(1 for c in join_conds + where + having if _is_like(c))

    c2 = set_ops + sum(_subqueries(c) for c in join_conds + where + having)

    agg = sum(1 for e in select.expressions if _has_agg(e))
    agg += sum(1 for c in where if _is_negated(c))
    agg += sum(1 for g in group_items if _has_agg(g))
    agg += sum(1 for o in order_items if _has_agg(o))
    agg += sum(1 for c in having if _is_negated(c)) + having_connectors
    others = 0
    others += 1 if agg > 1 else 0
    others += 1 if len(select.expressions) > 1 else 0
    others += 1 if len(where) > 1 else 0
    others += 1 if len(group_items) > 1 else 0
    return Components(c1, c2, others)


def hardness(sql: str) -> Hardness:
    """Spider's ``eval_hardness`` rule; unparseable SQL is labelled ``extra``."""
    parts = components(sql)
    if parts is None:
        return "extra"
    c1, c2, others = parts.component1, parts.component2, parts.others
    if c1 <= 1 and others == 0 and c2 == 0:
        return "easy"
    if (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0):
        return "medium"
    if (
        (others > 2 and c1 <= 2 and c2 == 0)
        or (2 < c1 <= 3 and others <= 2 and c2 == 0)
        or (c1 <= 1 and others == 0 and c2 <= 1)
    ):
        return "hard"
    return "extra"
