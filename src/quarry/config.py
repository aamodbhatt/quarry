"""Typed configuration loaded from TOML (``configs/quarry.toml`` by default).

Every knob that changes behaviour lives here so a run is described by one file plus the
command line. Secrets never live in the TOML: the OpenRouter key comes from the
``OPENROUTER_API_KEY`` environment variable (or a git-ignored ``.env``).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quarry.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("configs/quarry.toml")
API_KEY_ENV = "OPENROUTER_API_KEY"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMConfig(_Section):
    """Chat model served through OpenRouter's OpenAI-compatible API."""

    model: str = "google/gemma-4-31b-it:free"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    request_timeout_s: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    # Client-side token bucket. OpenRouter's free tier allows roughly 20 requests/minute.
    requests_per_second: float = Field(default=0.3, gt=0)


class AgentConfig(_Section):
    """Budgets and loop bounds for one question (one graph run)."""

    max_repairs: int = Field(default=3, ge=0, description="SQL attempts after the first one")
    max_explore_calls: int = Field(default=4, ge=0, description="schema tool calls while planning")
    max_steps: int = Field(default=16, gt=0, description="LLM calls + tool calls per question")
    max_tokens: int = Field(
        default=40_000, gt=0, description="prompt + completion tokens per question"
    )
    history_turns: int = Field(default=4, ge=0, description="previous turns shown for follow-ups")
    preview_rows: int = Field(default=20, gt=0, description="result rows shown to the model")


class SQLConfig(_Section):
    """Sandbox limits applied to every statement, whoever wrote it."""

    row_limit: int = Field(default=500, gt=0)
    timeout_ms: int = Field(default=5_000, gt=0)
    max_value_chars: int = Field(default=500, gt=0, description="longer cell values are clipped")


class LinkingConfig(_Section):
    """Schema linking: how much of the schema the model sees up front."""

    top_tables: int = Field(default=5, gt=0)
    top_columns: int = Field(default=25, gt=0)
    sample_values: int = Field(default=8, ge=0, description="distinct values sampled per column")
    use_embeddings: bool = Field(default=True, description="fuse a hashing embedder with BM25")


class TracingConfig(_Section):
    """Where JSONL traces go and which optional exporters are on."""

    dir: Path = Path("runs/traces")
    langfuse: bool = False
    langsmith: bool = False


class ServerConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8097, gt=0, lt=65536)


class QuarryConfig(_Section):
    """Root of the TOML document."""

    database: Path = Path("data/warehouse.sqlite")
    memory_path: Path = Path("runs/memory.sqlite")
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    sql: SQLConfig = SQLConfig()
    linking: LinkingConfig = LinkingConfig()
    tracing: TracingConfig = TracingConfig()
    server: ServerConfig = ServerConfig()


def load_config(path: Path | None = None) -> QuarryConfig:
    """Load and validate a config file; ``None`` means the default path, or defaults if absent."""
    target = path or DEFAULT_CONFIG_PATH
    if not target.exists():
        if path is not None:
            raise ConfigError(f"config file {target} does not exist")
        return QuarryConfig()
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{target} is not valid TOML: {exc}") from exc
    try:
        return QuarryConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{target} is invalid:\n{exc}") from exc


def api_key(env_mode: Literal["required", "optional"] = "required") -> str | None:
    """Return the OpenRouter key from the environment (after ``.env`` is loaded)."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        return key
    if env_mode == "optional":
        return None
    raise ConfigError(
        f"{API_KEY_ENV} is not set. Put it in .env (see .env.example) or export it, "
        "or pass --offline to use the scripted demo model."
    )
