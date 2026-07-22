from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.agents.document_models import KnowledgeReadyDocument
from app.knowledge.models import (
    DocumentMetadata,
    EntityLink,
    KnowledgeEntity,
    KnowledgeRelation,
    TimelineEvent,
    VectorRecord,
)


class EntityLinker(Protocol):
    async def link(self, payload: KnowledgeReadyDocument) -> tuple[EntityLink, ...]:
        """Resolve extracted entity mentions to canonical entity ids."""


class EntityMerger(Protocol):
    async def merge(
        self,
        payload: KnowledgeReadyDocument,
        links: tuple[EntityLink, ...],
    ) -> tuple[KnowledgeEntity, ...]:
        """Merge linked mentions into canonical KnowledgeEntity objects."""


class KnowledgeGraphRepository(Protocol):
    """Neo4j-backed repository contract."""

    async def upsert_entities(self, entities: tuple[KnowledgeEntity, ...]) -> None:
        """Upsert canonical entities into Neo4j."""

    async def upsert_relations(self, relations: tuple[KnowledgeRelation, ...]) -> None:
        """Upsert relationships into Neo4j."""

    async def upsert_timeline_events(self, events: tuple[TimelineEvent, ...]) -> None:
        """Upsert timeline events into Neo4j."""


class VectorIndexRepository(Protocol):
    """FAISS-backed vector index contract."""

    async def upsert_vectors(self, records: tuple[VectorRecord, ...]) -> None:
        """Upsert document vectors into FAISS."""

    async def delete_document_vectors(self, document_id: UUID) -> None:
        """Remove stale vectors for an incrementally updated document."""


class DocumentMetadataRepository(Protocol):
    """PostgreSQL-backed document metadata contract."""

    async def get(self, document_id: UUID) -> DocumentMetadata | None:
        """Load indexed document metadata."""

    async def upsert(self, metadata: DocumentMetadata) -> None:
        """Upsert document metadata."""


class RelationshipDiscoveryService(Protocol):
    async def discover(
        self,
        payload: KnowledgeReadyDocument,
        links: tuple[EntityLink, ...],
    ) -> tuple[KnowledgeRelation, ...]:
        """Discover explicit and inferred relationships for graph updates."""


class TimelineBuilder(Protocol):
    async def build(
        self,
        payload: KnowledgeReadyDocument,
        entities: tuple[KnowledgeEntity, ...],
        relations: tuple[KnowledgeRelation, ...],
    ) -> tuple[TimelineEvent, ...]:
        """Build timeline events from document metadata, entities, relations, and evidence."""


class EmbeddingService(Protocol):
    async def embed_document(self, payload: KnowledgeReadyDocument) -> VectorRecord:
        """Create a vector record for FAISS indexing."""
