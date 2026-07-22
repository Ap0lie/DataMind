from __future__ import annotations

import hashlib
import math
import re

from app.agents.document_models import KnowledgeReadyDocument
from app.knowledge.models import (
    DocumentMetadata,
    EntityLink,
    EntityMergeStrategy,
    EvidenceRef,
    KnowledgeEntity,
    KnowledgeRelation,
    TimelineEvent,
    VectorRecord,
)


class DeterministicEntityLinker:
    async def link(self, payload: KnowledgeReadyDocument) -> tuple[EntityLink, ...]:
        return tuple(
            EntityLink(
                mention_text=entity.text,
                label=entity.label,
                entity_id=_entity_id(entity.label, entity.canonical_id or entity.text),
                confidence=entity.confidence,
                merge_strategy=(
                    EntityMergeStrategy.EXACT_CANONICAL_ID
                    if entity.canonical_id
                    else EntityMergeStrategy.NORMALIZED_NAME_AND_LABEL
                ),
            )
            for entity in payload.extraction.entities
        )


class DefaultEntityMerger:
    async def merge(
        self,
        payload: KnowledgeReadyDocument,
        links: tuple[EntityLink, ...],
    ) -> tuple[KnowledgeEntity, ...]:
        merged: dict[str, KnowledgeEntity] = {}
        evidence = _evidence(payload, quote=payload.document.normalized_text[:240])
        for link in links:
            existing = merged.get(link.entity_id)
            aliases = (
                (link.mention_text,)
                if existing is None
                else (*existing.aliases, link.mention_text)
            )
            merged[link.entity_id] = KnowledgeEntity(
                entity_id=link.entity_id,
                name=existing.name if existing else link.mention_text,
                label=link.label,
                aliases=tuple(dict.fromkeys(aliases)),
                confidence=max(existing.confidence if existing else 0.0, link.confidence),
                evidence=(evidence,),
                metadata={"merge_strategy": link.merge_strategy.value},
            )
        return tuple(merged.values())


class DefaultRelationshipDiscoveryService:
    async def discover(
        self,
        payload: KnowledgeReadyDocument,
        links: tuple[EntityLink, ...],
    ) -> tuple[KnowledgeRelation, ...]:
        link_by_name = {link.mention_text.lower(): link for link in links}
        relations: list[KnowledgeRelation] = []
        for relation in payload.extraction.relations:
            subject = link_by_name.get(relation.subject.lower())
            object_ = link_by_name.get(relation.object.lower())
            if subject is None or object_ is None:
                continue
            relation_id = _relation_id(subject.entity_id, relation.predicate, object_.entity_id)
            relations.append(
                KnowledgeRelation(
                    relation_id=relation_id,
                    subject_entity_id=subject.entity_id,
                    predicate=relation.predicate,
                    object_entity_id=object_.entity_id,
                    confidence=relation.confidence,
                    evidence=(_evidence(payload, quote=relation.evidence),),
                    metadata=relation.metadata,
                )
            )
        return tuple(relations)


class DefaultTimelineBuilder:
    async def build(
        self,
        payload: KnowledgeReadyDocument,
        entities: tuple[KnowledgeEntity, ...],
        relations: tuple[KnowledgeRelation, ...],
    ) -> tuple[TimelineEvent, ...]:
        title = payload.document.title or f"Document {payload.document.document_id}"
        occurred_at = payload.document.parsed_at
        event_id = _stable_id(
            "event",
            str(payload.document.document_id),
            title,
            occurred_at.isoformat(),
        )
        summary_quote = payload.extraction.summary.text if payload.extraction.summary else None
        return (
            TimelineEvent(
                event_id=event_id,
                document_id=payload.document.document_id,
                source_url=payload.document.source_url,
                title=title,
                occurred_at=occurred_at,
                entity_ids=tuple(entity.entity_id for entity in entities),
                relation_ids=tuple(relation.relation_id for relation in relations),
                evidence=(_evidence(payload, quote=summary_quote),),
                metadata={"content_hash": payload.document.content_hash},
            ),
        )


class DeterministicEmbeddingService:
    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions

    async def embed_document(self, payload: KnowledgeReadyDocument) -> VectorRecord:
        values = [0.0 for _ in range(self._dimensions)]
        for token in re.findall(r"[A-Za-z0-9]+", payload.document.normalized_text.lower()):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            values[index] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        embedding = tuple(value / norm for value in values)
        return VectorRecord(
            vector_id=f"doc:{payload.document.document_id}",
            document_id=payload.document.document_id,
            text=payload.document.normalized_text,
            embedding=embedding,
            metadata={"source_url": str(payload.document.source_url)},
        )


def build_document_metadata(payload: KnowledgeReadyDocument) -> DocumentMetadata:
    extraction = payload.extraction
    return DocumentMetadata(
        document_id=payload.document.document_id,
        source_url=payload.document.source_url,
        title=payload.document.title,
        content_hash=payload.document.content_hash,
        language=extraction.language.language if extraction.language else None,
        sentiment=extraction.sentiment.sentiment if extraction.sentiment else None,
        topics=tuple(topic.label for topic in extraction.topics),
        keywords=tuple(keyword.text for keyword in extraction.keywords),
        nlp_backend=extraction.backend,
        parsed_at=payload.document.parsed_at,
        metadata=payload.document.metadata,
    )


def _evidence(payload: KnowledgeReadyDocument, quote: str | None = None) -> EvidenceRef:
    return EvidenceRef(
        document_id=payload.document.document_id,
        source_url=payload.document.source_url,
        quote=quote,
        confidence=0.8,
        extracted_at=payload.extraction.extracted_at,
    )


def _entity_id(label: str, value: str) -> str:
    return _stable_id("entity", label, _normalize(value))


def _relation_id(subject: str, predicate: str, object_: str) -> str:
    return _stable_id("relation", subject, predicate, object_)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
