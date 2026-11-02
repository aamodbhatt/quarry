"""Chat models: OpenRouter (online), the scripted model (tests and offline demo), caching."""

from quarry.llm.cache import SQLiteLLMCache
from quarry.llm.factory import build_chat_model, build_openrouter_model
from quarry.llm.scripted import (
    SCRIPTED_LLM_TYPE,
    QueueResponder,
    Responder,
    ScriptedChatModel,
    tool_call_message,
)

__all__ = [
    "SCRIPTED_LLM_TYPE",
    "QueueResponder",
    "Responder",
    "SQLiteLLMCache",
    "ScriptedChatModel",
    "build_chat_model",
    "build_openrouter_model",
    "tool_call_message",
]
