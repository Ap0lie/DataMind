from __future__ import annotations

from uuid import UUID

from app.knowledge.models import (
    DocumentMetadata,
    KnowledgeEntity,
    KnowledgeRelation,
    TimelineEvent,
    VectorRecord,
)


class InMemoryDocumentMetadataRepository:
    def __init__(self) -> None:
        self.items: dict[UUID, DocumentMetadata] = {}

    async def get(self, document_id: UUID) -> DocumentMetadata | None:
        return self.items.get(document_id)

    async def upsert(self, metadata: DocumentMetadata) -> None:
        self.items[metadata.document_id] = metadata


class InMemoryKnowledgeGraphRepository:
    def __init__(self) -> None:
        self.entities: dict[str, KnowledgeEntity] = {}
        self.relations: dict[str, KnowledgeRelation] = {}
        self.timeline_events: dict[str, TimelineEvent] = {}

    async def upsert_entities(self, entities: tuple[KnowledgeEntity, ...]) -> None:
        for entity in entities:
            self.entities[entity.entity_id] = entity

    async def upsert_relations(self, relations: tuple[KnowledgeRelation, ...]) -> None:
        for relation in relations:
            self.relations[relation.relation_id] = relation

    async def upsert_timeline_events(self, events: tuple[TimelineEvent, ...]) -> None:
        for event in events:
            self.timeline_events[event.event_id] = event


class InMemoryVectorIndexRepository:
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.deleted_document_ids: list[UUID] = []

    async def upsert_vectors(self, records: tuple[VectorRecord, ...]) -> None:
        for record in records:
            self.records[record.vector_id] = record

    async def delete_document_vectors(self, document_id: UUID) -> None:
        self.deleted_document_ids.append(document_id)
        self.records = {
            vector_id: record
            for vector_id, record in self.records.items()
            if record.document_id != document_id
        }
