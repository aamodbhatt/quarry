"""Spider 1.0: download, unpack the parts quarry uses, and load examples and schemas.

Why Spider and not BIRD mini-dev: BIRD's official mini-dev archive (``minidev.zip``) is
801 MB, and its eleven SQLite databases alone unpack to about 1.5 GB, both over this
project's 500 MB data budget. Spider's official ``spider_data.zip`` is 206 MB and its 20
dev databases are a few megabytes, so the whole benchmark runs on a laptop.

Only what is needed is unpacked into ``data/spider``: ``dev.json``, ``train_spider.json``,
``train_others.json``, ``tables.json`` and the 20 dev databases. Train-split gold SQL is
still checked against the guard using the schemas in ``tables.json``.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from quarry.datasets import RemoteFile, fetch
from quarry.db.catalog import TableDocs
from quarry.errors import DatasetError

SPIDER = RemoteFile(
    name="spider_data.zip",
    url=(
        "https://drive.usercontent.google.com/download"
        "?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
    ),
    # Google Drive's quota blocked every download while this was written, so the hash
    # is pinned on first use (size from the server's Content-Range: 205,800,266 bytes).
    sha256=None,
    size_bytes=205_800_266,
    source_page="https://yale-lily.github.io/spider",
)

_KEEP_FILES = ("dev.json", "train_spider.json", "train_others.json", "tables.json")

# Published sizes of Spider dev by difficulty (Yu et al., 2018; the official leaderboard).
OFFICIAL_DEV_HARDNESS = {"easy": 248, "medium": 446, "hard": 174, "extra": 166}


@dataclass(frozen=True)
class Example:
    """One benchmark question with its gold SQL."""

    id: str
    split: str
    db_id: str
    question: str
    gold_sql: str


@dataclass(frozen=True)
class SpiderSchema:
    """A database schema as ``tables.json`` describes it."""

    db_id: str
    tables: tuple[str, ...]
    columns: tuple[tuple[str, str, str], ...]  # (table, original column name, natural name)

    def docs(self) -> dict[str, TableDocs]:
        """Natural-language column names as descriptions (Spider has no other docs)."""
        out: dict[str, TableDocs] = {t: TableDocs() for t in self.tables}
        for table, column, natural in self.columns:
            out[table].columns[column] = natural
        return out


class SpiderLayout:
    """Paths and loaders for an unpacked ``data/spider`` directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _load(self, name: str) -> list[dict[str, Any]]:
        path = self.root / name
        if not path.exists():
            raise DatasetError(f"{path} is missing; run `quarry data spider` first")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise DatasetError(f"{path} is not a JSON list")
        return data

    def examples(self, split: str) -> list[Example]:
        files = {"dev": ["dev.json"], "train": ["train_spider.json", "train_others.json"]}
        if split not in files:
            raise ValueError(f"unknown split {split!r}; use 'dev' or 'train'")
        out: list[Example] = []
        for name in files[split]:
            for i, row in enumerate(self._load(name)):
                out.append(
                    Example(
                        id=f"{split}-{len(out):05d}" if split == "dev" else f"{name[:-5]}-{i:05d}",
                        split=split,
                        db_id=str(row["db_id"]),
                        question=str(row["question"]),
                        gold_sql=str(row["query"]),
                    )
                )
        return out

    @cached_property
    def schemas(self) -> dict[str, SpiderSchema]:
        out: dict[str, SpiderSchema] = {}
        for entry in self._load("tables.json"):
            tables = tuple(str(t) for t in entry["table_names_original"])
            columns = tuple(
                (tables[t_idx], str(name), str(natural))
                for (t_idx, name), (_, natural) in zip(
                    entry["column_names_original"], entry["column_names"], strict=True
                )
                if t_idx >= 0
            )
            out[str(entry["db_id"])] = SpiderSchema(str(entry["db_id"]), tables, columns)
        return out

    def db_path(self, db_id: str) -> Path:
        return self.root / "database" / db_id / f"{db_id}.sqlite"

    def dev_db_ids(self) -> list[str]:
        return sorted({e.db_id for e in self.examples("dev")})

    def ready(self) -> bool:
        if not all((self.root / name).exists() for name in _KEEP_FILES):
            return False
        return all(self.db_path(db).exists() for db in self.dev_db_ids())


def prepare_spider(data_root: Path) -> SpiderLayout:
    """Download (once, checksummed) and unpack Spider into ``data_root/spider``."""
    layout = SpiderLayout(data_root / "spider")
    if layout.ready():
        return layout
    archive = fetch(SPIDER, data_root / "raw")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        dev_member = next((n for n in names if n.endswith("/dev.json") or n == "dev.json"), None)
        if dev_member is None:
            raise DatasetError(f"{archive} does not contain dev.json")
        prefix = dev_member[: -len("dev.json")]
        layout.root.mkdir(parents=True, exist_ok=True)
        for name in _KEEP_FILES:
            (layout.root / name).write_bytes(bundle.read(prefix + name))
        for db_id in layout.dev_db_ids():
            member = f"{prefix}database/{db_id}/{db_id}.sqlite"
            if member not in names:
                raise DatasetError(f"{archive} has no database {member}")
            target = layout.db_path(db_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
    return layout


# The official archive sits on Google Drive, whose per-file download quota is often
# exhausted. Questions, gold SQL and schemas are also published by Spider's authors on
# the Hugging Face hub (xlangai/spider) and in the test-suite evaluation repository
# (taoyds/test-suite-sql-eval). Those are enough for every metric that needs no database
# file (guard pass rate, static-check false alarms, difficulty labels, name-only schema
# linking); value sampling, gold execution and the LLM benchmark still need the archive.
_HF_REV = "0c350918f3f29ec754f1181c65cdce76cd6c133c"
_TS_REV = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
SPIDER_METADATA = (
    RemoteFile(
        name="spider-dev.parquet",
        url=f"https://huggingface.co/datasets/xlangai/spider/resolve/{_HF_REV}/spider/validation-00000-of-00001.parquet",
        sha256="c3e2a46303899a2d4afe3f6a3a62e59f8d589f241b3cbfb52356479b1f054888",
        size_bytes=125_887,
        source_page="https://huggingface.co/datasets/xlangai/spider",
    ),
    RemoteFile(
        name="spider-train.parquet",
        url=f"https://huggingface.co/datasets/xlangai/spider/resolve/{_HF_REV}/spider/train-00000-of-00001.parquet",
        sha256="cb4b681558f6f8f428e516fb94c5a1cb19c5a0a0c153c0618c8cc4a28115d4cb",
        size_bytes=831_359,
        source_page="https://huggingface.co/datasets/xlangai/spider",
    ),
    RemoteFile(
        name="spider-tables.json",
        url=f"https://raw.githubusercontent.com/taoyds/test-suite-sql-eval/{_TS_REV}/tables.json",
        sha256="b68de9166952871d64554486fca4b25ff88509e983857a6af428250ffd58b67f",
        size_bytes=979_746,
        source_page="https://github.com/taoyds/test-suite-sql-eval",
    ),
)


def prepare_spider_metadata(data_root: Path) -> SpiderLayout:
    """Questions, gold SQL and schemas only (no database files), from the authors' mirrors."""
    import duckdb

    layout = SpiderLayout(data_root / "spider")
    layout.root.mkdir(parents=True, exist_ok=True)
    dev, train, tables = (fetch(remote, data_root / "raw") for remote in SPIDER_METADATA)
    con = duckdb.connect()
    try:
        for source, target in ((dev, "dev.json"), (train, "train_spider.json")):
            rows = con.execute(
                "SELECT db_id, question, query FROM read_parquet(?)", [str(source)]
            ).fetchall()
            payload = [{"db_id": d, "question": q, "query": s} for d, q, s in rows]
            (layout.root / target).write_text(json.dumps(payload), encoding="utf-8")
    finally:
        con.close()
    (layout.root / "train_others.json").write_text("[]", encoding="utf-8")
    (layout.root / "tables.json").write_bytes(tables.read_bytes())
    return layout
