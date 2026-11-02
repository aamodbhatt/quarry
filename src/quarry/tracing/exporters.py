"""Optional export of runs to Langfuse or LangSmith (both off by default).

The local JSONL trace is always written. These exporters are LangChain callback handlers
attached to the graph run when both the config flag *and* the provider's credentials are
present, so nothing leaves the machine unless the operator opts in twice.

* Langfuse: ``[tracing] langfuse = true`` plus ``LANGFUSE_PUBLIC_KEY``,
  ``LANGFUSE_SECRET_KEY`` (and ``LANGFUSE_HOST`` for self-hosted). Needs the extra:
  ``uv sync --extra langfuse``.
* LangSmith: ``[tracing] langsmith = true`` plus ``LANGSMITH_API_KEY``
  (``LANGSMITH_PROJECT`` defaults to ``quarry``).
"""

from __future__ import annotations

import importlib
import os

from langchain_core.callbacks import BaseCallbackHandler

from quarry.config import TracingConfig
from quarry.errors import ConfigError


def export_callbacks(cfg: TracingConfig) -> list[BaseCallbackHandler]:
    handlers: list[BaseCallbackHandler] = []
    if cfg.langfuse:
        handlers.append(_langfuse_handler())
    if cfg.langsmith:
        handlers.append(_langsmith_handler())
    return handlers


def _langfuse_handler() -> BaseCallbackHandler:
    missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"[tracing] langfuse = true but {', '.join(missing)} is not set")
    try:
        module = importlib.import_module("langfuse.langchain")
    except ImportError as exc:
        raise ConfigError(
            "Langfuse export needs the optional dependency: uv sync --extra langfuse"
        ) from exc
    handler = module.CallbackHandler()
    if not isinstance(handler, BaseCallbackHandler):
        raise ConfigError("langfuse.langchain.CallbackHandler is not a LangChain callback")
    return handler


def _langsmith_handler() -> BaseCallbackHandler:
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise ConfigError("[tracing] langsmith = true but LANGSMITH_API_KEY is not set")
    from langchain_core.tracers import LangChainTracer

    return LangChainTracer(project_name=os.environ.get("LANGSMITH_PROJECT", "quarry"))
