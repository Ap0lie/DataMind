from __future__ import annotations

from app.agents.document_models import KnowledgeReadyDocument
from app.knowledge.contracts import (
    DocumentMetadataRepository,
    EmbeddingService,
    EntityLinker,
    EntityMerger,
    KnowledgeGraphRepository,
    RelationshipDiscoveryService,
    TimelineBuilder,
    VectorIndexRepository,
)
from app.knowledge.models import (
    KnowledgeUpdateRequest,
    KnowledgeUpdateResult,
    KnowledgeUpdateStatus,
)
from app.knowledge.services import build_document_metadata


class KnowledgeAgent:
    def __init__(
        self,
        *,
        entity_linker: EntityLinker,
        entity_merger: EntityMerger,
        graph_repository: KnowledgeGraphRepository,
        vector_repository: VectorIndexRepository,
        metadata_repository: DocumentMetadataRepository,
        relationship_discovery: RelationshipDiscoveryService,
        timeline_builder: TimelineBuilder,
        embedding_service: EmbeddingService,
    ) -> None:
        self._entity_linker = entity_linker
        self._entity_merger = entity_merger
        self._graph_repository = graph_repository
        self._vector_repository = vector_repository
        self._metadata_repository = metadata_repository
        self._relationship_discovery = relationship_discovery
        self._timeline_builder = timeline_builder
        self._embedding_service = embedding_service

    async def update(
        self,
        payload: KnowledgeReadyDocument,
        request: KnowledgeUpdateRequest | None = None,
    ) -> KnowledgeUpdateResult:
        update_request = request or KnowledgeUpdateRequest(document_id=payload.document.document_id)
        existing_metadata = await self._metadata_repository.get(payload.document.document_id)
        if (
            existing_metadata is not None
            and existing_metadata.content_hash == payload.document.content_hash
            and not update_request.force
        ):
            return KnowledgeUpdateResult(
                document_id=payload.document.document_id,
                status=KnowledgeUpdateStatus.SKIPPED_UNCHANGED,
                skipped_reason="Document content hash is unchanged.",
            )

        links = await self._entity_linker.link(payload)
        entities = await self._entity_merger.merge(payload, links)
        relations = await self._relationship_discovery.discover(payload, links)
        timeline_events = await self._timeline_builder.build(payload, entities, relations)
        vector = await self._embedding_service.embed_document(payload)
        metadata = build_document_metadata(payload)

        if existing_metadata is not None:
            await self._vector_repository.delete_document_vectors(payload.document.document_id)

        await self._metadata_repository.upsert(metadata)
        await self._graph_repository.upsert_entities(entities)
        await self._graph_repository.upsert_relations(relations)
        await self._graph_repository.upsert_timeline_events(timeline_events)
        await self._vector_repository.upsert_vectors((vector,))

        return KnowledgeUpdateResult(
            document_id=payload.document.document_id,
            status=KnowledgeUpdateStatus.UPDATED,
            entities_upserted=len(entities),
            relations_upserted=len(relations),
            vectors_upserted=1,
            timeline_events_upserted=len(timeline_events),
        )
