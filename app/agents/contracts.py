from __future__ import annotations

from typing import Protocol

from app.agents.document_models import (
    Document,
    KnowledgeReadyDocument,
    NLPBackendKind,
    NLPExtraction,
    ParsedDocument,
)
from app.core.entities import ExecutionPlan, TaskIntent
from app.knowledge.models import KnowledgeUpdateRequest, KnowledgeUpdateResult


class PlannerAgentContract(Protocol):
    async def plan(self, intent: TaskIntent) -> ExecutionPlan:
        """Planner Agent: intent understanding, task decomposition, and tool planning."""


class ReviewerAgentContract(Protocol):
    async def review(self, plan: ExecutionPlan, outputs: dict[str, object]) -> dict[str, object]:
        """Reviewer Agent: validate sources, evidence, extraction quality, and confidence."""


class ParserAgentContract(Protocol):
    async def parse(self, document: Document) -> ParsedDocument:
        """Parser Agent: parse, clean, normalize, and extract document metadata."""


class NLPAgentContract(Protocol):
    async def analyze(
        self,
        document: ParsedDocument,
        *,
        backend: NLPBackendKind | None = None,
    ) -> NLPExtraction:
        """NLP Agent: call NLP MCP tools and return structured extraction."""

    async def prepare_for_knowledge_agent(
        self,
        document: ParsedDocument,
        *,
        backend: NLPBackendKind | None = None,
    ) -> KnowledgeReadyDocument:
        """Return a Knowledge Agent-ready document payload."""


class KnowledgeAgentContract(Protocol):
    async def update(
        self,
        payload: KnowledgeReadyDocument,
        request: KnowledgeUpdateRequest | None = None,
    ) -> KnowledgeUpdateResult:
        """Update graph, vector index, metadata, relationships, timeline, and evidence."""
