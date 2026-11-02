"""Embeddings behind a small interface.

:class:`Embedder` is what the schema linker needs: texts in, unit vectors out. The shipped
implementation, :class:`HashingEmbedder`, hashes character n-grams into a fixed number of
buckets. It downloads nothing and runs in microseconds, and it catches sub-word overlap
that BM25's whole-token matching misses ("nationality" vs "nation", "spend" vs
"spending"). A neural model can be dropped in by implementing the same protocol.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from quarry.linking.text import split_identifier

Vector = list[float]


class Embedder(Protocol):
    """Maps texts to L2-normalised vectors of a fixed dimension."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def cosine(a: Vector, b: Vector) -> float:
    """Dot product of two unit vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class HashingEmbedder:
    """Character n-gram feature hashing with sublinear term frequency."""

    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (3, 4, 5)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngrams = ngrams

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> Vector:
        counts: dict[int, int] = {}
        for word in split_identifier(text).lower().split():
            padded = f"<{word}>"
            for n in self.ngrams:
                for i in range(max(1, len(padded) - n + 1)):
                    bucket = _bucket(padded[i : i + n], self.dim)
                    counts[bucket] = counts.get(bucket, 0) + 1
        vector = [0.0] * self.dim
        for bucket, count in counts.items():
            vector[bucket] = 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


@lru_cache(maxsize=200_000)
def _bucket(gram: str, dim: int) -> int:
    """Stable across processes (unlike ``hash``), so vectors are reproducible."""
    digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim
