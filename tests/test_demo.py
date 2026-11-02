from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quarry.agent import open_agent
from quarry.config import QuarryConfig
from quarry.datasets import HILLSTROM, sha256_of, verify
from quarry.demo import DEMO_QUESTIONS, DemoResponder, build_warehouse
from quarry.errors import DatasetError


def test_warehouse_shape(warehouse: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(warehouse)
    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("campaigns", "customers", "campaign_sends", "outcomes")
    }
    zips = {r[0] for r in conn.execute("SELECT DISTINCT zip_area FROM customers")}
    conn.close()
    assert counts == {"campaigns": 3, "customers": 600, "campaign_sends": 600, "outcomes": 600}
    assert "Surburban" not in zips and "Suburban" in zips
    assert (warehouse.parent / "warehouse.sqlite.docs.json").exists()


def test_bad_csv_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Hillstrom header"):
        build_warehouse(bad, tmp_path / "w.sqlite")


def test_checksum_mismatch_is_explained(tmp_path: Path) -> None:
    fake = tmp_path / HILLSTROM.name
    fake.write_text("not the file", encoding="utf-8")
    assert sha256_of(fake) != HILLSTROM.sha256
    with pytest.raises(DatasetError, match="SHA-256"):
        verify(fake, HILLSTROM)


def test_offline_demo_runs_every_scripted_question(warehouse: Path, config: QuarryConfig) -> None:
    before = hashlib.sha256(warehouse.read_bytes()).hexdigest()
    with open_agent(config, db_path=warehouse, offline=True) as agent:
        outcomes = [agent.ask(q, thread_id="demo") for q in DEMO_QUESTIONS]
        unscripted = agent.ask("What is the airspeed of a swallow?", thread_id="other")
    assert [o.status for o in outcomes] == ["ok", "ok", "ok", "ok", "refused"]
    first = outcomes[0]
    assert [a["stage"] for a in first.attempts] == ["check", "accepted"]
    assert first.rows and first.chart is not None
    assert "converted best" in first.answer
    assert outcomes[3].chart is not None and outcomes[3].chart["mark"]["type"] == "line"
    assert unscripted.status == "unanswerable"
    assert hashlib.sha256(warehouse.read_bytes()).hexdigest() == before


def test_demo_script_loads_and_matches() -> None:
    responder = DemoResponder.load()
    ids = [q.id for q in responder.questions]
    assert ids[0] == "best-campaign" and "delete-attempt" in ids
    for question in DEMO_QUESTIONS:
        assert any(q.matches(question) for q in responder.questions), question


def test_trust_on_first_use_pins_the_first_download(tmp_path: Path) -> None:
    from quarry.datasets import RemoteFile

    data = tmp_path / "blob.bin"
    data.write_bytes(b"x" * 10)
    remote = RemoteFile(
        "blob.bin", "https://example.invalid/blob", None, 10, "https://example.invalid"
    )
    verify(data, remote)
    assert (tmp_path / "blob.bin.sha256").read_text(encoding="utf-8").strip() == sha256_of(data)
    data.write_bytes(b"y" * 10)
    with pytest.raises(DatasetError, match="SHA-256"):
        verify(data, remote)
    data.write_bytes(b"short")
    with pytest.raises(DatasetError, match="bytes"):
        verify(data, remote)
