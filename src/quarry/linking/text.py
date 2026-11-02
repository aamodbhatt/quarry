"""Tokenisation shared by BM25 and the hashing embedder.

Identifiers are split the way people read them (``customerID`` -> ``customer id``,
``history_segment`` -> ``history segment``), lower-cased and lightly stemmed so that
"customers" in a question meets ``customer_id`` in a schema.
"""

from __future__ import annotations

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the of in on at to for from by with and or is are was were be been being do does
    did what which who whom whose when where why how many much all any each every some there
    their them they it its this that these those me my our we you your i show list give tell
    find return get please number count total than then also only just as into per vs versus
    """.split()
)


def split_identifier(text: str) -> str:
    """``customerID_v2`` -> ``customer ID v2`` (camel case and separators become spaces)."""
    return _CAMEL.sub(" ", text).replace("_", " ").replace("-", " ").replace(".", " ")


def stem(token: str) -> str:
    """A deliberately small suffix stripper; good enough to align plurals and -ing forms."""
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, replacement in (
        ("ies", "y"),
        ("sses", "ss"),
        ("ing", ""),
        ("ers", "er"),
        ("ed", ""),
        ("es", "e"),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith("ss"):
                return token
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Split, lower-case, drop stopwords and stem."""
    words = _WORD.findall(split_identifier(text).lower())
    return [stem(w) for w in words if keep_stopwords or w not in STOPWORDS]
