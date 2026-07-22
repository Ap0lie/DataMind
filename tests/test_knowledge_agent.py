from __future__ import annotations

from uuid import UUID

import pytest

from app.agents.document_models import (
    Entity,
    Keyword,
    KnowledgeReadyDocument,
    LanguageDetection,
    NLPBackendKind,
    NLPExtraction,
    ParsedDocument,
    Relation,
    SentimentAnalysis,
    Summary,
    Topic,
)
from app.knowledge.agent import KnowledgeAgent
from app.knowledge.memory import (
    InMemoryDocumentMetadataRepository,
    InMemoryKnowledgeGraphRepository,
    InMemoryVectorIndexRepository,
)
from app.knowledge.models import KnowledgeUpdateRequest, KnowledgeUpdateStatus
from app.knowledge.services import (
    DefaultEntityMerger,
    DefaultRelationshipDiscoveryService,
    DefaultTimelineBuilder,
    DeterministicEmbeddingService,
    DeterministicEntityLinker,
)


def make_payload(content_hash: str = "hash-v1") -> KnowledgeReadyDocument:
    document_id = UUID("00000000-0000-0000-0000-000000000100")
    document = ParsedDocument(
        document_id=document_id,
        source_url="https://example.com/datamind",
        title="DataMind MCP Runtime",
        normalized_text="DataMind uses MCP. DeepSeek provides model access.",
        content_hash=content_hash,
        metadata={"source": "unit-test"},
    )
    extraction = NLPExtraction(
        document_id=document_id,
        source_url=document.source_url,
        backend=NLPBackendKind.LOCAL_MODEL,
        entities=(
            Entity(text="DataMind", label="PRODUCT", confidence=0.9),
            Entity(text="DeepSeek", label="ORG", confidence=0.85),
        ),
        relations=(
            Relation(
                subject="DataMind",
                predicate="uses",
                object="DeepSeek",
                confidence=0.7,
                evidence="DataMind uses MCP. DeepSeek provides model access.",
            ),
        ),
        keywords=(Keyword(text="datamind", score=0.9), Keyword(text="mcp", score=0.8)),
        topics=(Topic(label="web-intelligence", score=0.9),),
        summary=Summary(text="DataMind uses MCP.", compression_ratio=0.4),
        language=LanguageDetection(language="en", confidence=0.95),
        sentiment=SentimentAnalysis(sentiment="positive", confidence=0.8),
    )
    return KnowledgeReadyDocument(document=document, extraction=extraction)


def make_agent(
    *,
    graph: InMemoryKnowledgeGraphRepository | None = None,
    vectors: InMemoryVectorIndexRepository | None = None,
    metadata: InMemoryDocumentMetadataRepository | None = None,
) -> tuple[
    KnowledgeAgent,
    InMemoryKnowledgeGraphRepository,
    InMemoryVectorIndexRepository,
    InMemoryDocumentMetadataRepository,
]:
    graph_repo = graph or InMemoryKnowledgeGraphRepository()
    vector_repo = vectors or InMemoryVectorIndexRepository()
    metadata_repo = metadata or InMemoryDocumentMetadataRepository()
    return (
        KnowledgeAgent(
            entity_linker=DeterministicEntityLinker(),
            entity_merger=DefaultEntityMerger(),
            graph_repository=graph_repo,
            vector_repository=vector_repo,
            metadata_repository=metadata_repo,
            relationship_discovery=DefaultRelationshipDiscoveryService(),
            timeline_builder=DefaultTimelineBuilder(),
            embedding_service=DeterministicEmbeddingService(),
        ),
        graph_repo,
        vector_repo,
        metadata_repo,
    )


@pytest.mark.asyncio
async def test_knowledge_agent_updates_graph_vector_metadata_and_timeline() -> None:
    agent, graph, vectors, metadata = make_agent()

    result = await agent.update(make_payload())

    assert result.status == KnowledgeUpdateStatus.UPDATED
    assert result.entities_upserted == 2
    assert result.relations_upserted == 1
    assert result.vectors_upserted == 1
    assert result.timeline_events_upserted == 1
    assert len(graph.entities) == 2
    assert len(graph.relations) == 1
    assert len(graph.timeline_events) == 1
    assert len(vectors.records) == 1
    assert len(metadata.items) == 1


@pytest.mark.asyncio
async def test_knowledge_agent_skips_unchanged_document() -> None:
    agent, graph, vectors, metadata = make_agent()
    payload = make_payload()

    first = await agent.update(payload)
    second = await agent.update(payload)

    assert first.status == KnowledgeUpdateStatus.UPDATED
    assert second.status == KnowledgeUpdateStatus.SKIPPED_UNCHANGED
    assert second.skipped_reason == "Document content hash is unchanged."
    assert len(graph.entities) == 2
    assert len(vectors.records) == 1
    assert len(metadata.items) == 1


@pytest.mark.asyncio
async def test_knowledge_agent_force_updates_incremental_document_and_rebuilds_vectors() -> None:
    agent, graph, vectors, _ = make_agent()
    payload = make_payload()
    changed_payload = make_payload(content_hash="hash-v2")

    await agent.update(payload)
    result = await agent.update(
        changed_payload,
        request=KnowledgeUpdateRequest(
            document_id=changed_payload.document.document_id,
            force=True,
        ),
    )

    assert result.status == KnowledgeUpdateStatus.UPDATED
    assert vectors.deleted_document_ids == [payload.document.document_id]
    assert len(vectors.records) == 1
    assert len(graph.entities) == 2


@pytest.mark.asyncio
async def test_knowledge_agent_tracks_source_evidence() -> None:
    agent, graph, _, _ = make_agent()

    await agent.update(make_payload())

    relation = next(iter(graph.relations.values()))
    entity = next(iter(graph.entities.values()))
    assert relation.evidence
    assert str(relation.evidence[0].source_url) == "https://example.com/datamind"
    assert entity.evidence
    assert entity.evidence[0].document_id == UUID("00000000-0000-0000-0000-000000000100")
