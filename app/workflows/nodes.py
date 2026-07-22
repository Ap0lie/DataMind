from __future__ import annotations

from typing import Protocol

from app.agents.document_models import KnowledgeReadyDocument, NLPExtraction, ParsedDocument
from app.core.entities import ExecutionPlan, TaskIntent
from app.knowledge.models import KnowledgeUpdateResult
from app.workflows.models import CrawlOutput, ReportResult, ReviewResult, SearchOutput


class PlannerNode(Protocol):
    async def plan(self, task: TaskIntent) -> ExecutionPlan:
        """Create an ExecutionPlan from task intent."""


class SearchNode(Protocol):
    async def search(self, plan: ExecutionPlan) -> tuple[SearchOutput, ...]:
        """Search candidate sources."""


class CrawlNode(Protocol):
    async def crawl(self, search_results: tuple[SearchOutput, ...]) -> tuple[CrawlOutput, ...]:
        """Crawl candidate sources."""


class ParserNode(Protocol):
    async def parse(self, crawl_results: tuple[CrawlOutput, ...]) -> tuple[ParsedDocument, ...]:
        """Parse crawled content into normalized documents."""


class NLPNode(Protocol):
    async def analyze(self, documents: tuple[ParsedDocument, ...]) -> tuple[NLPExtraction, ...]:
        """Extract NLP structures from parsed documents."""


class KnowledgeNode(Protocol):
    async def update(
        self,
        documents: tuple[ParsedDocument, ...],
        extractions: tuple[NLPExtraction, ...],
    ) -> tuple[tuple[KnowledgeReadyDocument, ...], KnowledgeUpdateResult]:
        """Update knowledge graph, vector index, metadata, relationships, and timeline."""


class ReviewerNode(Protocol):
    async def review(self, state_summary: dict[str, object]) -> ReviewResult:
        """Validate sources, extraction, summary, evidence, and confidence."""


class ReportNode(Protocol):
    async def report(self, state_summary: dict[str, object]) -> ReportResult:
        """Generate a final report."""


class WorkflowAgents(Protocol):
    planner: PlannerNode
    search: SearchNode
    crawl: CrawlNode
    parser: ParserNode
    nlp: NLPNode
    knowledge: KnowledgeNode
    reviewer: ReviewerNode
    report: ReportNode
