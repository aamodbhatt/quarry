"""Build the chat model: OpenRouter through ChatOpenAI, or the scripted offline model."""

from __future__ import annotations

from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from quarry.config import LLMConfig, api_key
from quarry.llm.scripted import Responder, ScriptedChatModel

# OpenRouter uses these optional headers to attribute traffic to an app.
APP_HEADERS = {"HTTP-Referer": "https://github.com/aamodbhatt/quarry", "X-Title": "quarry"}


def build_openrouter_model(cfg: LLMConfig, *, cache: BaseCache | None = None) -> BaseChatModel:
    """ChatOpenAI pointed at OpenRouter. Raises ``ConfigError`` if the key is missing."""
    key = api_key("required")
    assert key is not None  # noqa: S101 - api_key("required") raises instead of returning None
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=SecretStr(key),
        temperature=cfg.temperature,
        timeout=cfg.request_timeout_s,
        max_retries=cfg.max_retries,
        # OpenRouter documents ``max_tokens``; ChatOpenAI would rename its own field to
        # ``max_completion_tokens``, which not every upstream provider honours.
        extra_body={"max_tokens": cfg.max_output_tokens},
        default_headers=APP_HEADERS,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=cfg.requests_per_second,
            check_every_n_seconds=0.05,
            max_bucket_size=1,
        ),
        cache=cache,
        stream_usage=True,
    )


def build_chat_model(
    cfg: LLMConfig,
    *,
    offline: bool,
    responder: Responder | None = None,
    cache: BaseCache | None = None,
) -> BaseChatModel:
    """The online model, or the scripted one when ``offline`` (``responder`` required)."""
    if offline:
        if responder is None:
            raise ValueError("offline mode needs a responder (e.g. the demo script)")
        return ScriptedChatModel(responder=responder)
    return build_openrouter_model(cfg, cache=cache)
