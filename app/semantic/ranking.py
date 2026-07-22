from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.semantic.embedding import SemanticEmbeddingProvider, cosine_similarity


@dataclass(frozen=True)
class RankedCandidate:
    item: dict[str, Any]
    final_score: float
    lexical_score: float
    embedding_score: float
    type_score: float
    context_score: float
    model_revision: str

    def evidence(self) -> dict[str, Any]:
        return {
            "candidate_id": self.item.get("id"),
            "lexical_score": round(self.lexical_score, 4),
            "embedding_score": round(self.embedding_score, 4),
            "type_score": round(self.type_score, 4),
            "context_score": round(self.context_score, 4),
            "final_score": round(self.final_score, 4),
            "embedding_model_revision": self.model_revision,
        }


class SemanticCandidateRanker:
    def __init__(self, provider: SemanticEmbeddingProvider) -> None:
        self.provider = provider

    def rank(self, question: str, items: list[dict[str, Any]], *, expected_type: str) -> tuple[RankedCandidate, ...]:
        if not items:
            return ()
        candidate_texts = [_candidate_text(item) for item in items]
        vectors = self.provider.encode([question, *candidate_texts])
        query_vector = vectors[0] if vectors else ()
        normalized_question = _normalize(question)
        ranked = []
        for index, item in enumerate(items):
            terms = [_normalize(str(item.get("name") or "")), *(_normalize(str(alias)) for alias in item.get("aliases") or [])]
            lexical = max((1.0 if term == normalized_question else min(1.0, len(term) / max(len(normalized_question), 1)) if term and term in normalized_question else 0.0 for term in terms), default=0.0)
            embedding = cosine_similarity(query_vector, vectors[index + 1]) if len(vectors) > index + 1 else 0.0
            item_type = str(item.get("type") or ("metric" if item.get("formula") else "dimension"))
            type_score = 1.0 if expected_type in item_type or (expected_type == "metric" and item.get("formula")) else 0.5
            context = _context_score(question, expected_type)
            final = lexical * 0.45 + embedding * 0.35 + type_score * 0.10 + context * 0.10
            ranked.append(RankedCandidate(item, final, lexical, embedding, type_score, context, self.provider.model_revision))
        return tuple(sorted(ranked, key=lambda candidate: candidate.final_score, reverse=True))


def _candidate_text(item: dict[str, Any]) -> str:
    return "；".join(str(value) for value in (item.get("name"), " ".join(item.get("aliases") or []), item.get("description"), item.get("type")) if value)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _context_score(question: str, expected_type: str) -> float:
    lowered = question.lower()
    if expected_type == "metric" and any(token in lowered for token in ("总", "平均", "金额", "收入", "利润", "数量", "率")):
        return 1.0
    if expected_type == "dimension" and any(token in lowered for token in ("按", "各", "地区", "客户", "产品", "日期", "月份")):
        return 1.0
    return 0.5
