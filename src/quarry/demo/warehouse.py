"""Build the demo "marketing warehouse" from Kevin Hillstrom's e-mail experiment.

The source is one CSV row per customer (64,000 customers who bought in the last twelve
months). A third were e-mailed men's merchandise, a third women's merchandise and a third
nothing, at random, and visits, purchases and spend were tracked for two weeks.

quarry normalises it into a small star schema so questions need real joins::

    campaigns (3) <- campaign_sends (64,000) -> customers (64,000)
                            ^
                            outcomes (64,000, one per send)

Only two transformations touch values: the source spells one zip area "Surburban"
(stored as "Suburban"), and 0/1 flags keep their meaning under clearer names. Nothing is
invented: there are no dates, costs or IDs beyond row order in the source.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from quarry.db.database import docs_path_for
from quarry.errors import DatasetError

EXPECTED_HEADER = [
    "recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie",
    "channel", "segment", "visit", "conversion", "spend",
]  # fmt: skip

CAMPAIGNS = (
    (1, "Mens E-Mail", "Men's merchandise", 0,
     "E-mail featuring men's merchandise, sent to a random third of the customers."),
    (2, "Womens E-Mail", "Women's merchandise", 0,
     "E-mail featuring women's merchandise, sent to a random third of the customers."),
    (3, "No E-Mail", None, 1,
     "Hold-out control group: a random third of the customers received no e-mail."),
)  # fmt: skip

SCHEMA = """
CREATE TABLE campaigns (
    campaign_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    merchandise TEXT,
    is_control INTEGER NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    recency_months INTEGER NOT NULL,
    history_segment TEXT NOT NULL,
    history_usd REAL NOT NULL,
    bought_mens INTEGER NOT NULL,
    bought_womens INTEGER NOT NULL,
    zip_area TEXT NOT NULL,
    is_new_customer INTEGER NOT NULL,
    channel TEXT NOT NULL
);
CREATE TABLE campaign_sends (
    send_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    campaign_id INTEGER NOT NULL REFERENCES campaigns (campaign_id)
);
CREATE TABLE outcomes (
    send_id INTEGER PRIMARY KEY REFERENCES campaign_sends (send_id),
    visited INTEGER NOT NULL,
    converted INTEGER NOT NULL,
    spend_usd REAL NOT NULL
);
"""

INDEXES = """
CREATE INDEX idx_sends_customer ON campaign_sends (customer_id);
CREATE INDEX idx_sends_campaign ON campaign_sends (campaign_id);
"""

DOCS: dict[str, dict[str, object]] = {
    "campaigns": {
        "description": "The three arms of the e-mail experiment (two e-mails and a control).",
        "columns": {
            "name": "campaign name: 'Mens E-Mail', 'Womens E-Mail' or 'No E-Mail'",
            "merchandise": "merchandise featured in the e-mail; NULL for the control group",
            "is_control": "1 for the hold-out group that received no e-mail, else 0",
        },
    },
    "customers": {
        "description": "Customers who purchased within the twelve months before the campaign.",
        "columns": {
            "recency_months": "months since the customer's last purchase (1-12)",
            "history_segment": "bucket of dollars spent in the past year, e.g. '1) $0 - $100'",
            "history_usd": "dollars spent in the past year",
            "bought_mens": "1 if the customer bought men's merchandise in the past year",
            "bought_womens": "1 if the customer bought women's merchandise in the past year",
            "zip_area": "'Urban', 'Suburban' or 'Rural'",
            "is_new_customer": "1 if the customer is new in the past twelve months",
            "channel": "channels purchased from in the past year: 'Phone', 'Web', 'Multichannel'",
        },
    },
    "campaign_sends": {
        "description": "Which campaign arm each customer was randomly assigned to (one row each).",
        "columns": {},
    },
    "outcomes": {
        "description": "Results tracked for two weeks after the send (one row per send).",
        "columns": {
            "visited": "1 if the customer visited the website within two weeks",
            "converted": "1 if the customer purchased within two weeks (conversion)",
            "spend_usd": "dollars spent within two weeks (revenue)",
        },
    },
}


@dataclass(frozen=True)
class WarehouseSummary:
    path: Path
    customers: int
    sends: int
    campaigns: int


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise DatasetError(f"{csv_path} does not have the Hillstrom header {EXPECTED_HEADER}")
        return list(reader)


def _tables(rows: list[dict[str, str]]) -> dict[str, list[tuple[object, ...]]]:
    campaign_ids = {name: cid for cid, name, *_ in CAMPAIGNS}
    customers: list[tuple[object, ...]] = []
    sends: list[tuple[object, ...]] = []
    outcomes: list[tuple[object, ...]] = []
    for i, row in enumerate(rows, start=1):
        segment = row["segment"]
        if segment not in campaign_ids:
            raise DatasetError(f"row {i}: unknown segment {segment!r}")
        zip_area = "Suburban" if row["zip_code"] == "Surburban" else row["zip_code"]
        customers.append(
            (
                i,
                int(row["recency"]),
                row["history_segment"],
                float(row["history"]),
                int(row["mens"]),
                int(row["womens"]),
                zip_area,
                int(row["newbie"]),
                row["channel"],
            )
        )
        sends.append((i, i, campaign_ids[segment]))
        outcomes.append((i, int(row["visit"]), int(row["conversion"]), float(row["spend"])))
    return {
        "campaigns": [tuple(c) for c in CAMPAIGNS],
        "customers": customers,
        "campaign_sends": sends,
        "outcomes": outcomes,
    }


def build_warehouse(csv_path: Path, out: Path) -> WarehouseSummary:
    """Write the SQLite warehouse (and its ``.docs.json`` sidecar) from the Hillstrom CSV."""
    tables = _tables(_read_rows(csv_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".part")
    partial.unlink(missing_ok=True)
    conn = sqlite3.connect(partial)
    try:
        conn.executescript("PRAGMA foreign_keys = ON;" + SCHEMA)
        for name, rows in tables.items():
            marks = ", ".join("?" * len(rows[0]))
            conn.executemany(f"INSERT INTO {name} VALUES ({marks})", rows)
        conn.executescript(INDEXES)
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    partial.replace(out)
    _write_docs(out)
    return WarehouseSummary(
        out, len(tables["customers"]), len(tables["campaign_sends"]), len(CAMPAIGNS)
    )


def export_duckdb(sqlite_path: Path, out: Path) -> Path:
    """Copy the warehouse into a DuckDB file with the same schema (for the DuckDB engine).

    Rows travel through temporary CSV files because DuckDB bulk-loads CSV far faster
    than row-by-row inserts. This is trusted build code, not the query sandbox.
    """
    out.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{sqlite_path.resolve().as_posix()}?mode=ro", uri=True)
    target = duckdb.connect(str(out))
    try:
        target.execute(SCHEMA)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("campaigns", "customers", "campaign_sends", "outcomes"):
                staged = Path(tmp) / f"{name}.csv"
                with staged.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerows(source.execute(f"SELECT * FROM {name}"))
                target.execute(
                    f"INSERT INTO {name} SELECT * FROM read_csv(?, header = false)",
                    [staged.as_posix()],
                )
    finally:
        target.close()
        source.close()
    _write_docs(out)
    return out


def _write_docs(db_path: Path) -> None:
    docs_path_for(db_path).write_text(json.dumps({"tables": DOCS}, indent=2), encoding="utf-8")
