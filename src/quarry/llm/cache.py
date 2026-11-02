"""A persistent LLM response cache in SQLite.

Plugs into LangChain's standard ``cache=`` hook, so a cached call skips the network and
the rate limiter entirely. The benchmark uses it to make reruns free and resumable: the
key is the exact serialized prompt plus the model's parameters.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import warnings
from pathlib import Path
from typing import Any

from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_core.load.serializable import Serializable
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation

_CACHEABLE: list[type[Serializable]] = [
    Generation,
    ChatGeneration,
    ChatGenerationChunk,
    AIMessage,
    AIMessageChunk,
]


class SQLiteLLMCache(BaseCache):
    """Thread-safe ``(prompt, llm_string) -> generations`` store."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            " key TEXT PRIMARY KEY, llm_string TEXT NOT NULL, value TEXT NOT NULL)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\x1f{prompt}".encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM llm_cache WHERE key = ?", (self._key(prompt, llm_string),)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            # Only generation and message classes may be revived from the cache file.
            generations: list[Generation] = loads(row[0], allowed_objects=_CACHEABLE)
        return list(generations)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        payload = dumps(list(return_val))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, llm_string, value) VALUES (?, ?, ?)",
                (self._key(prompt, llm_string), llm_string, payload),
            )
            self._conn.commit()

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
