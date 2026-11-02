"""Schema linking: which tables and columns does this question need?

Large schemas do not fit in a prompt, and even small ones distract the model. The linker
ranks every table and column for the question and the agent shows the model only the top
of that ranking (plus anything it explicitly asks for with ``describe_table``).

Signals, fused with reciprocal-rank fusion so no score needs calibrating:

* BM25 over one document per column (table name, column name twice, description and
  sampled values) and one per table (name twice, description, every column name);
* exact value hits: a sampled value that appears verbatim in the question
  ("Mens E-Mail", "Web") is strong evidence for its column;
* optionally, cosine similarity from an :class:`~quarry.linking.embeddings.Embedder`.

Tables are then expanded with *bridge* tables: if two selected tables are not joined by a
foreign key but share a neighbour, the neighbour is added so the join path is visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quarry.db.catalog import Catalog
from quarry.linking.bm25 import BM25
from quarry.linking.embeddings import Embedder, cosine
from quarry.linking.text import tokenize

RRF_K = 60
CONTEXT_WEIGHT = 0.5
VALUE_HIT_BONUS = 3.0


@dataclass(frozen=True)
class ScoredTable:
    name: str
    score: float


@dataclass(frozen=True)
class ScoredColumn:
    table: str
    column: str
    score: float


@dataclass(frozen=True)
class LinkResult:
    """Full rankings plus the table set the agent will show the model."""

    tables: tuple[ScoredTable, ...]
    columns: tuple[ScoredColumn, ...]
    selected: tuple[str, ...]
    value_hits: tuple[tuple[str, str, str], ...] = ()

    def top_tables(self, k: int) -> list[str]:
        return [t.name for t in self.tables[:k]]

    def top_columns(self, k: int) -> list[tuple[str, str]]:
        return [(c.table, c.column) for c in self.columns[:k]]

    def columns_by_table(self, k: int) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for col in self.columns[:k]:
            out.setdefault(col.table, []).append(col.column)
        return out


class SchemaLinker:
    """Ranks a catalog's tables and columns for natural-language questions."""

    def __init__(self, catalog: Catalog, embedder: Embedder | None = None) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self._columns: list[tuple[str, str]] = []
        col_docs: list[list[str]] = []
        col_texts: list[str] = []
        tab_docs: list[list[str]] = []
        tab_texts: list[str] = []
        self._values: list[tuple[int, str]] = []
        for table in catalog.tables:
            col_names = " ".join(c.name for c in table.columns)
            tab_text = f"{table.name} {table.name} {table.description} {col_names}"
            tab_docs.append(tokenize(tab_text))
            tab_texts.append(f"{table.name} {table.description}")
            for col in table.columns:
                text = f"{table.name} {col.name} {col.name} {col.description} " + " ".join(
                    col.samples
                )
                index = len(self._columns)
                self._columns.append((table.name, col.name))
                col_docs.append(tokenize(text))
                col_texts.append(f"{col.name} {col.description} {table.name}")
                for value in col.samples:
                    if len(value) >= 3 and not _is_number(value):
                        self._values.append((index, value))
        self._col_bm25 = BM25(col_docs)
        self._tab_bm25 = BM25(tab_docs)
        self._col_vecs = embedder.embed(col_texts) if embedder else []
        self._tab_vecs = embedder.embed(tab_texts) if embedder else []

    def link(
        self, question: str, *, context: str = "", top_tables: int = 5, max_bridges: int = 2
    ) -> LinkResult:
        query = tokenize(question)
        ctx = tokenize(context) if context else []
        col_lex = _blend(self._col_bm25.scores(query), self._col_bm25.scores(ctx) if ctx else None)
        tab_lex = _blend(self._tab_bm25.scores(query), self._tab_bm25.scores(ctx) if ctx else None)

        hits = self._value_hits(question)
        for index, _value in hits:
            col_lex[index] += VALUE_HIT_BONUS

        col_rankings = [col_lex]
        tab_rankings = [tab_lex, self._best_column_per_table(col_lex)]
        if self.embedder is not None:
            q_vec = self.embedder.embed([f"{question} {context}".strip()])[0]
            col_rankings.append([cosine(q_vec, v) for v in self._col_vecs])
            tab_rankings.append([cosine(q_vec, v) for v in self._tab_vecs])

        col_fused = _rrf(col_rankings)
        tab_fused = _rrf(tab_rankings)
        columns = sorted(
            (ScoredColumn(t, c, s) for (t, c), s in zip(self._columns, col_fused, strict=True)),
            key=lambda x: -x.score,
        )
        tables = sorted(
            (ScoredTable(t.name, s) for t, s in zip(self.catalog.tables, tab_fused, strict=True)),
            key=lambda x: -x.score,
        )
        selected = self._with_bridges([t.name for t in tables[:top_tables]], max_bridges)
        value_hits = tuple((self._columns[i][0], self._columns[i][1], value) for i, value in hits)
        return LinkResult(tuple(tables), tuple(columns), tuple(selected), value_hits)

    def _value_hits(self, question: str) -> list[tuple[int, str]]:
        lowered = question.lower()
        hits: list[tuple[int, str]] = []
        for index, value in self._values:
            pattern = r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])"
            if value.lower() in lowered and re.search(pattern, lowered):
                hits.append((index, value))
        return hits

    def _best_column_per_table(self, col_scores: list[float]) -> list[float]:
        best: dict[str, float] = {}
        for (table, _), score in zip(self._columns, col_scores, strict=True):
            best[table] = max(best.get(table, 0.0), score)
        return [best.get(t.name, 0.0) for t in self.catalog.tables]

    def _with_bridges(self, chosen: list[str], max_bridges: int) -> list[str]:
        """Add tables that connect two chosen tables through one foreign-key hop."""
        out = list(chosen)
        lowered = {c.lower() for c in chosen}
        core = chosen[:3]
        added = 0
        for i, a in enumerate(core):
            for b in core[i + 1 :]:
                if added >= max_bridges:
                    return out
                na, nb = self.catalog.neighbours(a), self.catalog.neighbours(b)
                if b.lower() in na:
                    continue
                for bridge in sorted(na & nb):
                    if bridge not in lowered:
                        table = self.catalog.table(bridge)
                        if table is not None:
                            out.append(table.name)
                            lowered.add(bridge)
                            added += 1
                        break
        return out


def _blend(primary: list[float], secondary: list[float] | None) -> list[float]:
    if secondary is None:
        return list(primary)
    return [p + CONTEXT_WEIGHT * s for p, s in zip(primary, secondary, strict=True)]


def _rrf(rankings: list[list[float]]) -> list[float]:
    """Reciprocal-rank fusion; items with a zero score in a ranking get nothing from it."""
    fused = [0.0] * len(rankings[0])
    for scores in rankings:
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        for rank, index in enumerate(order):
            if scores[index] > 0:
                fused[index] += 1.0 / (RRF_K + rank + 1)
    return fused


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
