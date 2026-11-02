"""A scripted chat model with the same interface as the real one.

``ScriptedChatModel`` is a LangChain ``BaseChatModel``: it supports ``bind_tools``,
returns ``AIMessage``\\ s with ``tool_calls`` and reports ``usage_metadata``, so the agent
graph cannot tell it from ChatOpenAI. What it says comes from a *responder*:

* :class:`QueueResponder` replays a fixed list of messages (unit tests);
* :class:`quarry.demo.script.DemoResponder` follows a script of demo questions (offline
  demo mode and the README recording).

Token counts it reports are estimates (characters / 4) and traces mark them as such.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

Responder = Callable[[Sequence[BaseMessage], Sequence[dict[str, Any]]], AIMessage]
"""Gets the prompt messages and the bound tool schemas, returns the next reply."""

SCRIPTED_LLM_TYPE = "quarry-scripted"


class ScriptedChatModel(BaseChatModel):
    """Chat model whose replies come from a :data:`Responder`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Responder
    model_name: str = "scripted-demo"

    @property
    def _llm_type(self) -> str:
        return SCRIPTED_LLM_TYPE

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[dict[str, Any]] = list(kwargs.get("tools") or [])
        reply = self.responder(messages, tools)
        prompt_chars = sum(len(_text(m)) for m in messages) + len(json.dumps(tools))
        out_chars = len(_text(reply)) + len(json.dumps(reply.tool_calls, default=str))
        tokens_in, tokens_out = max(1, prompt_chars // 4), max(1, out_chars // 4)
        usage = UsageMetadata(
            input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out
        )
        message = reply.model_copy(
            update={"usage_metadata": usage, "response_metadata": {"model_name": self.model_name}}
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        schemas = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=schemas, tool_choice=tool_choice, **kwargs)


class QueueResponder:
    """Replays prepared replies in order; raises when the script runs out."""

    def __init__(self, replies: Sequence[AIMessage | str]) -> None:
        self._replies = [AIMessage(content=r) if isinstance(r, str) else r for r in replies]
        self.calls: list[Sequence[BaseMessage]] = []

    def __call__(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
    ) -> AIMessage:
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("QueueResponder ran out of scripted replies")
        return self._replies.pop(0)

    @property
    def remaining(self) -> int:
        return len(self._replies)


def tool_call_message(
    name: str, args: dict[str, Any], call_id: str, content: str = ""
) -> AIMessage:
    """Build an assistant message that calls one tool (handy in scripts and tests)."""
    return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": call_id}])


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)
