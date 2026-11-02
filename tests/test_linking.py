from __future__ import annotations

import math
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from quarry.db import Catalog, Column, Database, ForeignKey, Table
from quarry.linking import HashingEmbedder, SchemaLinker
from quarry.linking.bm25 import BM25
from quarry.linking.embeddings import cosine
from quarry.linking.text import split_identifier, stem, tokenize


def test_tokenizer_splits_identifiers_and_stems() -> None:
    assert split_identifier("customerID_v2") == "customer ID v2"
    assert tokenize("How many customers converted?") == ["customer", "convert"]
    assert stem("categories") == "category"
    assert stem("class") == "class"


def test_bm25_prefers_the_matching_document() -> None:
    index = BM25([["customer", "channel"], ["campaign", "name"], ["order", "price"]])
    scores = index.scores(["campaign"])
    assert scores.index(max(scores)) == 1
    assert BM25([]).scores(["x"]) == []


@settings(max_examples=50, deadline=None)
@given(st.lists(st.text(alphabet="abcxyz_ ", min_size=1, max_size=30), min_size=1, max_size=5))
def test_hashing_embeddings_are_unit_length_and_deterministic(texts: list[str]) -> None:
    embedder = HashingEmbedder(dim=64)
    first, second = embedder.embed(texts), embedder.embed(texts)
    assert first == second
    for vector in first:
        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == 0.0 or abs(norm - 1.0) < 1e-9


def test_hashing_embeddings_capture_subword_overlap() -> None:
    embedder = HashingEmbedder()
    spend, spending, channel = embedder.embed(["spend", "spending", "channel"])
    assert cosine(spend, spending) > cosine(spend, channel)


def test_linker_ranks_the_right_table_and_finds_values(warehouse: Path) -> None:
    linker = SchemaLinker(Database(warehouse).catalog(), HashingEmbedder())
    result = linker.link("Conversion rate for the Mens E-Mail campaign by channel")
    assert {"campaigns", "outcomes", "customers"} <= set(result.top_tables(4))
    assert ("campaigns", "name", "Mens E-Mail") in result.value_hits
    assert ("outcomes", "converted") in result.top_columns(10)


def test_linker_adds_bridge_tables() -> None:
    a = Table("artists", (Column("artist_id", "INTEGER", primary_key=True), Column("name", "TEXT")))
    s = Table(
        "songs",
        (
            Column("song_id", "INTEGER"),
            Column("artist_id", "INTEGER"),
            Column("album_id", "INTEGER"),
        ),
        (
            ForeignKey("artist_id", "artists", "artist_id"),
            ForeignKey("album_id", "albums", "album_id"),
        ),
    )
    b = Table("albums", (Column("album_id", "INTEGER", primary_key=True), Column("title", "TEXT")))
    linker = SchemaLinker(Catalog("sqlite", (a, s, b)))
    result = linker.link("album title and artist name", top_tables=2)
    assert set(result.top_tables(2)) == {"artists", "albums"}
    assert "songs" in result.selected


def test_context_helps_follow_up_questions(warehouse: Path) -> None:
    linker = SchemaLinker(Database(warehouse).catalog())
    bare = linker.link("break that down")
    followed = linker.link("break that down", context="conversion rate by campaign")
    assert followed.top_tables(2) != bare.top_tables(2) or followed.tables[0].score > 0
