from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.agents.document_models import NLPBackendKind


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeUpdateStatus(StrEnum):
    UPDATED = "updated"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    FAILED = "failed"


class EntityMergeStrategy(StrEnum):
    EXACT_CANONICAL_ID = "exact_canonical_id"
    NORMALIZED_NAME_AND_LABEL = "normalized_name_and_label"


class EvidenceRef(KnowledgeModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    source_url: HttpUrl
    quote: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeEntity(KnowledgeModel):
    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeRelation(KnowledgeModel):
    relation_id: str = Field(min_length=1)
    subject_entity_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_entity_id: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentMetadata(KnowledgeModel):
    document_id: UUID
    source_url: HttpUrl
    title: str | None = None
    content_hash: str | None = None
    language: str | None = None
    sentiment: str | None = None
    topics: tuple[str, ...] = Field(default_factory=tuple)
    keywords: tuple[str, ...] = Field(default_factory=tuple)
    nlp_backend: NLPBackendKind
    parsed_at: datetime
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRecord(KnowledgeModel):
    vector_id: str = Field(min_length=1)
    document_id: UUID
    text: str = Field(min_length=1)
    embedding: tuple[float, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TimelineEvent(KnowledgeModel):
    event_id: str = Field(min_length=1)
    document_id: UUID
    source_url: HttpUrl
    title: str = Field(min_length=1)
    occurred_at: datetime
    entity_ids: tuple[str, ...] = Field(default_factory=tuple)
    relation_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityLink(KnowledgeModel):
    mention_text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    merge_strategy: EntityMergeStrategy = EntityMergeStrategy.NORMALIZED_NAME_AND_LABEL


class KnowledgeUpdateRequest(KnowledgeModel):
    payload_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    force: bool = False


class KnowledgeUpdateResult(KnowledgeModel):
    document_id: UUID
    status: KnowledgeUpdateStatus
    entities_upserted: int = 0
    relations_upserted: int = 0
    vectors_upserted: int = 0
    timeline_events_upserted: int = 0
    skipped_reason: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
