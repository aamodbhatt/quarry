"""Okapi BM25 over small in-memory corpora (one document per table or column)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


class BM25:
    """Classic BM25 with the Lucene-style non-negative IDF.

    Schema corpora are tiny (tens to thousands of documents), so a plain inverted index
    in dictionaries is fast enough and keeps the dependency list short.
    """

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tf = [Counter(doc) for doc in documents]
        self._len = [len(doc) for doc in documents]
        n = len(documents)
        self._avg_len = (sum(self._len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self._tf):
            for term in tf:
                self._postings.setdefault(term, []).append(i)

    def __len__(self) -> int:
        return len(self._tf)

    def scores(self, query: Sequence[str]) -> list[float]:
        """Score every document against ``query`` (repeated terms count once)."""
        out = [0.0] * len(self._tf)
        if not self._avg_len:
            return out
        for term in set(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                freq = self._tf[i][term]
                norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg_len)
                out[i] += idf * freq * (self.k1 + 1) / (freq + norm)
        return out
