"""Prompt-injection detector for text that comes back *from* the database.

Database rows are untrusted input: anyone who can write a row (a support ticket, a product
review, a CRM note) can try to talk to the agent through it. Every value the model is
shown from a tool (sampled values, ``sample_rows``, ``run_sql`` previews) passes through
:meth:`InjectionDetector.scan` first, and flagged cells are replaced by a placeholder.

This is a transparent, rule-based scorer, not a classifier. Each signal carries a weight;
a cell is flagged when the weights of the distinct signals it trips sum to at least the
threshold. It is a tripwire, not the safety boundary: whatever the model does after
reading a row, the SQL guard and the read-only engine still refuse every write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WITHHELD = "[withheld: possible prompt injection]"

_I = re.IGNORECASE
_IM = re.IGNORECASE | re.MULTILINE

# (name, weight, pattern). Weights >= 1.0 flag on their own; weaker ones need company.
_SIGNALS: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        1.0,
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|skip)\b[^.\n]{0,40}?"
            r"\b(previous|prior|above|earlier|all|any|your|the|these|those|system)\b[^.\n]{0,30}?"
            r"\b(instructions?|prompts?|rules?|directions?|guidelines?|context|messages?|"
            r"constraints?|polic(y|ies)|question|request|task|user)\b",
            _I,
        ),
    ),
    (
        "role_hijack",
        1.0,
        re.compile(
            r"\byou are (now|no longer)\b|\bfrom now on,? you\b|\bnew instructions?\s*:|"
            r"\bdeveloper mode\b|\bjailbreak|\bDAN mode\b|\bmaintenance mode\b|\bgod mode\b",
            _I,
        ),
    ),
    (
        "persona_request",
        0.5,
        re.compile(r"\b(act as|pretend (to be|you are)|roleplay as|you must now)\b", _I),
    ),
    (
        "prompt_markup",
        1.0,
        re.compile(
            r"<\|?(im_start|im_end|system|endoftext|eot_id|start_header_id)\|?>|\[/?INST\]|"
            r"^\s*(system|assistant)\s*:|#{2,}\s*(system|instructions?)\b|</?(system|tool_call)>",
            _IM,
        ),
    ),
    (
        "tool_invocation",
        1.0,
        re.compile(
            r"\b(run_sql|describe_table|sample_rows|list_tables|tool_calls?|function_call)\b|"
            r"\bcall (the |a )?\w+ (tool|function)\b|\binvoke (the )?\w+ tool\b",
            _I,
        ),
    ),
    (
        "sql_write",
        1.0,
        re.compile(
            r"\b(drop\s+(table|database|schema|view|index)|delete\s+from|truncate\s+table|"
            r"insert\s+into|update\s+\w+\s+set|alter\s+table|attach\s+database|"
            r"create\s+(table|trigger)|grant\s+all|load_extension|writefile)\b",
            _I,
        ),
    ),
    (
        "address_model",
        0.6,
        re.compile(
            r"\b(dear|attention|note to( the)?|hey|hi)\s+(ai|assistant|agent|model|llm|chatbot|"
            r"language model|bot)\b|"
            r"\b(ai|assistant|agent|language model|llm|chatbot)s?\b[^.\n]{0,30}?"
            r"\b(must|should|shall|will|are required to|need to)\b",
            _I,
        ),
    ),
    (
        "system_prompt",
        0.8,
        re.compile(
            r"\b(system prompt|your (initial |original |hidden )?(instructions|prompt)|"
            r"reveal (your|the) (prompt|instructions|rules))\b",
            _I,
        ),
    ),
    (
        "send_elsewhere",
        0.8,
        re.compile(
            r"\b(send|post|upload|forward|exfiltrate|leak|email)\b[^.\n]{0,40}?\b(to|at)\b"
            r"[^.\n]{0,25}?(https?://|[\w.+-]+@[\w-]+\.\w+)",
            _I,
        ),
    ),
    # Alone this is just a tracking link; it only matters next to another signal.
    ("url_with_params", 0.4, re.compile(r"https?://\S+[?&][\w-]+=", _I)),
    (
        "secrecy",
        0.6,
        re.compile(
            r"\b(do not|don't|never)\s+(tell|mention|reveal|show|inform|alert)\b[^.\n]{0,30}?"
            r"\b(user|anyone|human|operator)\b",
            _I,
        ),
    ),
    (
        "invisible_text",
        0.7,
        re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]|[\U000e0000-\U000e007f]"),
    ),
    ("encoded_payload", 0.4, re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")),
)


@dataclass(frozen=True)
class InjectionFinding:
    """A flagged value, with the signals that fired."""

    excerpt: str
    score: float
    signals: tuple[str, ...]


class InjectionDetector:
    """Scores free text for instructions aimed at the model."""

    def __init__(self, threshold: float = 1.0, max_chars: int = 4_000) -> None:
        self.threshold = threshold
        self.max_chars = max_chars

    def scan(self, text: str) -> InjectionFinding | None:
        """Return a finding when ``text`` looks like an injection, else ``None``."""
        if len(text) < 8:
            return None
        window = text[: self.max_chars]
        fired = [(name, weight) for name, weight, rx in _SIGNALS if rx.search(window)]
        score = round(sum(weight for _, weight in fired), 3)
        if score < self.threshold:
            return None
        return InjectionFinding(
            excerpt=_excerpt(window), score=score, signals=tuple(name for name, _ in fired)
        )

    def is_suspicious(self, value: object) -> bool:
        return isinstance(value, str) and self.scan(value) is not None


def _excerpt(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
