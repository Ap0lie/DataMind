from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import NotRequired, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agents.document_models import KnowledgeReadyDocument, NLPExtraction, ParsedDocument
from app.core.entities import ExecutionPlan, TaskIntent
from app.harness.models import TokenUsage
from app.knowledge.models import KnowledgeUpdateResult


class WorkflowModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    REVIEW_RETRY = "review_retry"
    COMPLETED = "completed"


class AgentNodeName(StrEnum):
    PLANNER = "planner"
    SEARCH = "search"
    CRAWL = "crawl"
    PARSER = "parser"
    NLP = "nlp"
    KNOWLEDGE = "knowledge"
    REVIEWER = "reviewer"
    REPORT = "report"


class WorkflowError(WorkflowModel):
    node: AgentNodeName
    message: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowCheckpoint(WorkflowModel):
    checkpoint_id: UUID = Field(default_factory=uuid4)
    node: AgentNodeName
    status: WorkflowStatus
    state_keys: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowTraceEntry(WorkflowModel):
    trace_id: UUID
    node: AgentNodeName
    message: str = Field(min_length=1)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewResult(WorkflowModel):
    passed: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    findings: tuple[str, ...] = Field(default_factory=tuple)


class ReportResult(WorkflowModel):
    title: str = Field(min_length=1)
    markdown: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SearchOutput(WorkflowModel):
    query: str = Field(min_length=1)
    urls: tuple[str, ...] = Field(default_factory=tuple)


class CrawlOutput(WorkflowModel):
    url: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_type: str = Field(min_length=1)


class WorkflowState(TypedDict):
    task: TaskIntent
    trace_id: UUID
    status: WorkflowStatus
    retry_counts: dict[str, int]
    checkpoints: tuple[WorkflowCheckpoint, ...]
    trace: tuple[WorkflowTraceEntry, ...]
    errors: tuple[WorkflowError, ...]
    plan: NotRequired[ExecutionPlan | None]
    search_results: NotRequired[tuple[SearchOutput, ...]]
    crawl_results: NotRequired[tuple[CrawlOutput, ...]]
    parsed_documents: NotRequired[tuple[ParsedDocument, ...]]
    nlp_extractions: NotRequired[tuple[NLPExtraction, ...]]
    knowledge_documents: NotRequired[tuple[KnowledgeReadyDocument, ...]]
    knowledge_result: NotRequired[KnowledgeUpdateResult | None]
    review: NotRequired[ReviewResult | None]
    report: NotRequired[ReportResult | None]
    max_node_retries: NotRequired[int]
    max_review_retries: NotRequired[int]
    review_retries: NotRequired[int]
