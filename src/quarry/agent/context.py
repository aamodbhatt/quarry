"""Dependencies the graph nodes need, passed as LangGraph run context (not state)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from quarry.config import QuarryConfig
from quarry.db import Database, SafeExecutor
from quarry.guard import InjectionDetector
from quarry.linking import HashingEmbedder, SchemaLinker
from quarry.llm.scripted import SCRIPTED_LLM_TYPE
from quarry.tools import ToolBox
from quarry.tracing import Trace


@dataclass
class AgentDeps:
    """Everything that stays fixed across questions for one database and model."""

    llm: BaseChatModel
    database: Database
    config: QuarryConfig
    detector: InjectionDetector = field(default_factory=InjectionDetector)

    @cached_property
    def executor(self) -> SafeExecutor:
        return SafeExecutor(
            self.database,
            row_limit=self.config.sql.row_limit,
            timeout_ms=self.config.sql.timeout_ms,
        )

    @cached_property
    def toolbox(self) -> ToolBox:
        return ToolBox(
            self.database,
            self.executor,
            self.detector,
            preview_rows=self.config.agent.preview_rows,
            max_value_chars=self.config.sql.max_value_chars,
        )

    @cached_property
    def tools(self) -> dict[str, BaseTool]:
        return self.toolbox.by_name()

    @cached_property
    def linker(self) -> SchemaLinker:
        embedder = HashingEmbedder() if self.config.linking.use_embeddings else None
        return SchemaLinker(self.database.catalog(), embedder)

    @property
    def model_name(self) -> str:
        name = getattr(self.llm, "model_name", None)
        return str(name) if name else type(self.llm).__name__

    @property
    def tokens_estimated(self) -> bool:
        return self.llm._llm_type == SCRIPTED_LLM_TYPE


@dataclass
class RunContext:
    """Run-scoped context: the shared dependencies plus this run's trace."""

    deps: AgentDeps
    trace: Trace
