from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class AgentModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentFormat(StrEnum):
    HTML = "html"
    MARKDOWN = "markdown"
    PDF_TEXT = "pdf_text"
    RSS_ITEM = "rss_item"
    PLAIN_TEXT = "plain_text"


class NLPBackendKind(StrEnum):
    RULE_BASED = "rule_based"
    LOCAL_MODEL = "local_model"
    LLM = "llm"


class Document(AgentModel):
    document_id: UUID = Field(default_factory=uuid4)
    source_url: HttpUrl
    format: DocumentFormat
    content: str = Field(min_length=1)
    content_type: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ParsedDocument(AgentModel):
    document_id: UUID
    source_url: HttpUrl
    title: str | None = None
    normalized_text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    links: tuple[HttpUrl, ...] = Field(default_factory=tuple)
    content_hash: str | None = None
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Entity(AgentModel):
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    canonical_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(AgentModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Keyword(AgentModel):
    text: str = Field(min_length=1)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class Topic(AgentModel):
    label: str = Field(min_length=1)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class Summary(AgentModel):
    text: str = Field(min_length=1)
    compression_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    model: str | None = None


class LanguageDetection(AgentModel):
    language: str = Field(min_length=2)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SentimentAnalysis(AgentModel):
    sentiment: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NLPExtraction(AgentModel):
    document_id: UUID
    source_url: HttpUrl
    backend: NLPBackendKind
    entities: tuple[Entity, ...] = Field(default_factory=tuple)
    relations: tuple[Relation, ...] = Field(default_factory=tuple)
    keywords: tuple[Keyword, ...] = Field(default_factory=tuple)
    topics: tuple[Topic, ...] = Field(default_factory=tuple)
    summary: Summary | None = None
    language: LanguageDetection | None = None
    sentiment: SentimentAnalysis | None = None
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeReadyDocument(AgentModel):
    document: ParsedDocument
    extraction: NLPExtraction

    @property
    def source_url(self) -> HttpUrl:
        return self.document.source_url
