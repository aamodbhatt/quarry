"""A read-only database handle and the guarded executor every caller goes through."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from quarry.db.catalog import Catalog, TableDocs
from quarry.db.engines import Dialect, Engine, QueryResult, open_engine
from quarry.errors import DatabaseError, QueryFailed, QueryTimeout
from quarry.guard.policy import GuardVerdict, SQLGuard


def docs_path_for(db_path: Path) -> Path:
    """Sidecar file with table/column descriptions: ``<db file>.docs.json``."""
    return db_path.with_name(db_path.name + ".docs.json")


def load_docs(path: Path) -> dict[str, TableDocs]:
    """Read a docs sidecar: ``{"tables": {"t": {"description": "...", "columns": {...}}}}``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatabaseError(f"cannot read schema docs {path}: {exc}") from exc
    tables = raw.get("tables", {}) if isinstance(raw, dict) else {}
    out: dict[str, TableDocs] = {}
    for name, entry in tables.items():
        if not isinstance(entry, dict):
            continue
        columns = entry.get("columns", {})
        out[str(name)] = TableDocs(
            description=str(entry.get("description", "")),
            columns={str(k): str(v) for k, v in columns.items()}
            if isinstance(columns, dict)
            else {},
        )
    return out


class Database:
    """One SQLite or DuckDB file, opened read-only, with a lazily built catalog."""

    def __init__(
        self,
        path: Path,
        *,
        sample_values: int = 8,
        docs: dict[str, TableDocs] | None = None,
    ) -> None:
        self.path = path
        self.engine: Engine = open_engine(path)
        self._sample_values = sample_values
        self._docs = docs
        self._catalog: Catalog | None = None

    @property
    def dialect(self) -> Dialect:
        return self.engine.dialect

    @property
    def name(self) -> str:
        return self.path.stem

    def catalog(self) -> Catalog:
        if self._catalog is None:
            catalog = self.engine.introspect(sample_values=self._sample_values)
            docs = self._docs
            sidecar = docs_path_for(self.path)
            if docs is None and sidecar.exists():
                docs = load_docs(sidecar)
            self._catalog = catalog.with_docs(docs) if docs else catalog
        return self._catalog


@dataclass(frozen=True)
class Execution:
    """What happened to one statement: refused, failed, timed out or returned rows."""

    sql: str
    verdict: GuardVerdict
    result: QueryResult | None = None
    error: str | None = None
    timed_out: bool = False
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def blocked(self) -> bool:
        return not self.verdict.allowed


class SafeExecutor:
    """Guard first, then the read-only engine. The only path from SQL text to a database."""

    def __init__(self, database: Database, *, row_limit: int, timeout_ms: int) -> None:
        self.database = database
        self.row_limit = row_limit
        self.timeout_ms = timeout_ms
        self._guard: SQLGuard | None = None

    @property
    def guard(self) -> SQLGuard:
        if self._guard is None:
            self._guard = SQLGuard(
                self.database.dialect, known_tables=self.database.catalog().table_names
            )
        return self._guard

    def run(self, sql: str, *, row_limit: int | None = None) -> Execution:
        start = time.perf_counter()
        verdict = self.guard.check(sql)
        if not verdict.allowed:
            reasons = "; ".join(str(v) for v in verdict.violations)
            return Execution(sql, verdict, error=f"refused by the SQL guard: {reasons}")
        try:
            result = self.database.engine.run(
                sql, row_limit=row_limit or self.row_limit, timeout_ms=self.timeout_ms
            )
        except QueryTimeout as exc:
            return Execution(sql, verdict, error=str(exc), timed_out=True, elapsed_ms=_ms(start))
        except QueryFailed as exc:
            return Execution(sql, verdict, error=str(exc), elapsed_ms=_ms(start))
        return Execution(sql, verdict, result=result, elapsed_ms=_ms(start))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
