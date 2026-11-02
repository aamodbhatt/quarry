"""In-memory description of a database schema: tables, columns, keys, docs and sample values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from quarry.guard.injection import WITHHELD, InjectionDetector


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str = ""
    primary_key: bool = False
    samples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    row_count: int | None = None
    description: str = ""

    def column(self, name: str) -> Column | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


@dataclass(frozen=True)
class Catalog:
    """Everything the agent may know about one database."""

    dialect: str
    tables: tuple[Table, ...]
    _index: dict[str, Table] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {t.name.lower(): t for t in self.tables})

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def table(self, name: str) -> Table | None:
        return self._index.get(name.lower())

    def require_table(self, name: str) -> Table:
        table = self.table(name)
        if table is None:
            known = ", ".join(self.table_names) or "none"
            raise KeyError(f"no table named {name!r}; tables: {known}")
        return table

    def with_docs(self, docs: Mapping[str, TableDocs]) -> Catalog:
        """Return a copy with table/column descriptions merged in (case-insensitive)."""
        lowered = {k.lower(): v for k, v in docs.items()}
        tables: list[Table] = []
        for table in self.tables:
            doc = lowered.get(table.name.lower())
            if doc is None:
                tables.append(table)
                continue
            col_docs = {k.lower(): v for k, v in doc.columns.items()}
            columns = tuple(
                replace(c, description=col_docs.get(c.name.lower(), c.description))
                for c in table.columns
            )
            tables.append(
                replace(table, columns=columns, description=doc.description or table.description)
            )
        return Catalog(self.dialect, tuple(tables))

    def neighbours(self, name: str) -> set[str]:
        """Tables joined to ``name`` by a foreign key in either direction (lower-case)."""
        lowered = name.lower()
        out: set[str] = set()
        for table in self.tables:
            for fk in table.foreign_keys:
                if table.name.lower() == lowered:
                    out.add(fk.ref_table.lower())
                elif fk.ref_table.lower() == lowered:
                    out.add(table.name.lower())
        out.discard(lowered)
        return out & set(self._index)


@dataclass(frozen=True)
class TableDocs:
    description: str = ""
    columns: dict[str, str] = field(default_factory=dict)


def render_schema(
    catalog: Catalog,
    tables: Iterable[str] | None = None,
    *,
    columns: Mapping[str, Iterable[str]] | None = None,
    detector: InjectionDetector | None = None,
    max_columns: int = 30,
    max_samples: int = 5,
) -> str:
    """Render a compact, prompt-friendly schema.

    ``tables`` limits the output to those tables (in that order). For wide tables only the
    keys plus the ``columns`` picked by schema linking are shown. Sample values pass
    through ``detector`` so an injected row never reaches a prompt verbatim.
    """
    names = list(tables) if tables is not None else catalog.table_names
    picked = {k.lower(): {c.lower() for c in v} for k, v in (columns or {}).items()}
    blocks: list[str] = []
    for name in names:
        table = catalog.table(name)
        if table is None:
            continue
        rows = f", {table.row_count:,} rows" if table.row_count is not None else ""
        header = f"TABLE {_clean(table.name, detector)} ({len(table.columns)} columns{rows})"
        if table.description:
            header += f" -- {_clean(table.description, detector)}"
        lines = [header]
        fk_cols = {fk.column.lower() for fk in table.foreign_keys}
        shown = list(table.columns)
        if len(shown) > max_columns:
            keep = picked.get(table.name.lower(), set())
            shown = [
                c
                for c in table.columns
                if c.primary_key or c.name.lower() in fk_cols or c.name.lower() in keep
            ][:max_columns]
            lines.append(f"  (showing {len(shown)} of {len(table.columns)} columns)")
        for col in shown:
            lines.append("  " + _render_column(col, detector, max_samples))
        for fk in table.foreign_keys:
            lines.append(f"  FOREIGN KEY {fk.column} -> {fk.ref_table}.{fk.ref_column}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _clean(text: str, detector: InjectionDetector | None) -> str:
    """Identifiers and descriptions are database content too: withhold suspected injections."""
    return WITHHELD if detector is not None and detector.is_suspicious(text) else text


def _render_column(col: Column, detector: InjectionDetector | None, max_samples: int) -> str:
    text = f"{_clean(col.name, detector)} {col.type or 'ANY'}"
    if col.primary_key:
        text += " PRIMARY KEY"
    notes: list[str] = []
    if col.description and col.description.lower() != col.name.lower():
        notes.append(_clean(col.description, detector))
    if col.samples and max_samples:
        values = [_clean(v, detector) for v in col.samples[:max_samples]]
        notes.append("e.g. " + ", ".join(_quote(v) for v in values))
    if notes:
        text += " -- " + "; ".join(notes)
    return text


def _quote(value: str, limit: int = 40) -> str:
    clipped = value if len(value) <= limit else value[: limit - 1] + "…"
    return "'" + clipped.replace("'", "''") + "'"
