from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import ijson  # type: ignore[import-untyped]
from ijson.common import IncompleteJSONError  # type: ignore[import-untyped]

from app.tool_results.contracts import (
    CanonicalFact,
    ToolResultChunkSummary,
    ToolResultExcerpt,
    ToolResultProjection,
    ToolResultSummary,
)

_ASCII_TERM = re.compile(r"[a-zA-Z0-9_]{2,}")
_CJK_SEQUENCE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}")
_STOP_TERMS = {
    "about",
    "data",
    "result",
    "show",
    "the",
    "this",
    "tool",
    "分析",
    "数据",
    "查看",
    "结果",
}


@dataclass(frozen=True)
class ProjectionPolicy:
    max_chars: int = 12_000
    max_items: int = 32
    scan_max_bytes: int = 33_554_432


@dataclass(frozen=True)
class _Candidate:
    path: str
    text: str
    source: str
    score: float


class _BoundedReader:
    """Limit decompressed reads before the streaming parser can buffer them."""

    def __init__(self, source: gzip.GzipFile, limit: int) -> None:
        self._source = source
        self._limit = limit
        self.bytes_read = 0
        self.limit_reached = False

    def read(self, size: int = -1) -> bytes:
        remaining = self._limit - self.bytes_read
        if remaining <= 0:
            self.limit_reached = True
            return b""
        requested = remaining if size < 0 else min(size, remaining)
        content = self._source.read(requested)
        self.bytes_read += len(content)
        return content


def build_tool_result_projection(
    *,
    artifact_id: UUID,
    storage_root: Path,
    storage_path: str,
    artifact_size_bytes: int,
    summary: ToolResultSummary,
    chunks: tuple[ToolResultChunkSummary, ...],
    query: str,
    policy: ProjectionPolicy,
) -> ToolResultProjection:
    normalized_query = " ".join(query.split())[:1_000]
    query_hash = hashlib.sha256(normalized_query.casefold().encode()).hexdigest()
    terms = _query_terms(normalized_query)
    candidates: list[_Candidate] = []

    for fact in summary.canonical_facts:
        text = _bounded_text(fact.value)
        score = _score(fact.path, text, terms, base=8.0)
        if score > 0:
            candidates.append(_Candidate(fact.path, text, "canonical_fact", score))
    for index, row in enumerate(summary.preview):
        text = _bounded_text(row)
        score = _score(f"preview[{index}]", text, terms, base=5.0)
        if score > 0:
            candidates.append(_Candidate(f"preview[{index}]", text, "summary_preview", score))
    for chunk in chunks:
        if not chunk.verified:
            continue
        for quote_index, quote in enumerate(chunk.source_quotes):
            path = f"chunk[{chunk.chunk_index}].quote[{quote_index}]"
            score = _score(path, quote, terms, base=6.0)
            if score > 0:
                candidates.append(_Candidate(path, _bounded_text(quote), "verified_chunk", score))

    scanned_bytes = 0
    scan_truncated = False
    if not _enough(candidates, terms, policy):
        streamed, scanned_bytes, scan_truncated = _stream_candidates(
            root=storage_root,
            storage_path=storage_path,
            terms=terms,
            max_bytes=policy.scan_max_bytes,
            max_items=policy.max_items * 4,
        )
        candidates.extend(streamed)

    excerpts = _fit_candidates(candidates, policy)
    selected_facts = _fit_facts(summary.canonical_facts, terms, excerpts, policy)
    excerpts, selected_facts, evidence_ids, context_size = _fit_projection(
        excerpts,
        selected_facts,
        summary.evidence_ids,
        max_bytes=policy.max_chars,
    )
    selected_paths = tuple(item.path for item in excerpts)
    more_available = scan_truncated or artifact_size_bytes > max(scanned_bytes, context_size)
    return ToolResultProjection(
        artifact_id=artifact_id,
        query_hash=query_hash,
        headline=f"Selected {len(excerpts)} exact excerpt(s) for the requested focus.",
        excerpts=tuple(excerpts),
        canonical_facts=selected_facts,
        evidence_ids=evidence_ids,
        selected_paths=selected_paths,
        scanned_bytes=scanned_bytes,
        context_size_bytes=context_size,
        truncated=scan_truncated or len(candidates) > len(excerpts),
        more_available=more_available,
    )


def _query_terms(query: str) -> tuple[str, ...]:
    lowered = query.casefold()
    output = {item for item in _ASCII_TERM.findall(lowered) if item not in _STOP_TERMS}
    for sequence in _CJK_SEQUENCE.findall(lowered):
        output.add(sequence)
        output.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(sorted(output, key=lambda item: (-len(item), item)))


def _score(path: str, text: str, terms: tuple[str, ...], *, base: float) -> float:
    if not terms:
        return base
    path_value = path.casefold()
    text_value = text.casefold()
    path_hits = sum(1 for term in terms if term in path_value)
    text_hits = sum(1 for term in terms if term in text_value)
    return base + path_hits * 4.0 + text_hits * 2.0 if path_hits or text_hits else 0.0


def _enough(
    candidates: list[_Candidate],
    terms: tuple[str, ...],
    policy: ProjectionPolicy,
) -> bool:
    if len(candidates) < min(8, policy.max_items):
        return False
    if not terms:
        return True
    searchable = tuple(
        f"{candidate.path} {candidate.text}".casefold() for candidate in candidates
    )
    return all(any(term in value for value in searchable) for term in terms)


def _stream_candidates(
    *,
    root: Path,
    storage_path: str,
    terms: tuple[str, ...],
    max_bytes: int,
    max_items: int,
) -> tuple[list[_Candidate], int, bool]:
    root_resolved = root.resolve()
    target = (root / storage_path).resolve()
    if not target.is_relative_to(root_resolved):
        raise ValueError("Tool artifact path is outside the configured store.")

    heap: list[tuple[float, int, _Candidate]] = []
    sequence = 0
    reader: _BoundedReader
    with gzip.open(target, "rb") as source:
        reader = _BoundedReader(source, max_bytes)
        try:
            for prefix, event, value in ijson.parse(reader, use_float=True):
                if event not in {"string", "number", "boolean", "null"}:
                    continue
                text = _bounded_text(value)
                score = _score(prefix or "$", text, terms, base=1.0)
                if score <= 0:
                    continue
                candidate = _Candidate(prefix or "$", text, "artifact", score)
                entry = (score, sequence, candidate)
                sequence += 1
                if len(heap) < max_items:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
        except IncompleteJSONError:
            if not reader.limit_reached:
                raise
    return (
        [item[2] for item in sorted(heap, reverse=True)],
        reader.bytes_read,
        reader.limit_reached,
    )


def _fit_candidates(
    candidates: Iterable[_Candidate], policy: ProjectionPolicy
) -> list[ToolResultExcerpt]:
    selected: list[ToolResultExcerpt] = []
    seen: set[tuple[str, str]] = set()
    used = 0
    ordered = sorted(candidates, key=lambda item: (-item.score, item.path, item.text))
    for candidate in ordered:
        signature = (candidate.path, candidate.text)
        if signature in seen:
            continue
        encoded = len(candidate.path.encode()) + len(candidate.text.encode()) + 64
        if selected and used + encoded > policy.max_chars:
            break
        seen.add(signature)
        selected.append(
            ToolResultExcerpt(
                path=candidate.path,
                text=candidate.text,
                source=candidate.source,
                score=candidate.score,
            )
        )
        used += encoded
        if len(selected) >= policy.max_items:
            break
    return selected


def _fit_facts(
    facts: tuple[CanonicalFact, ...],
    terms: tuple[str, ...],
    excerpts: list[ToolResultExcerpt],
    policy: ProjectionPolicy,
) -> tuple[CanonicalFact, ...]:
    used = sum(len(item.path.encode()) + len(item.text.encode()) + 64 for item in excerpts)
    selected: list[CanonicalFact] = []
    ranked = sorted(
        facts,
        key=lambda item: -_score(item.path, _bounded_text(item.value), terms, base=8.0),
    )
    for fact in ranked:
        if terms and _score(fact.path, _bounded_text(fact.value), terms, base=8.0) <= 0:
            continue
        encoded = len(json.dumps(fact.model_dump(mode="json"), ensure_ascii=False).encode())
        if used + encoded > policy.max_chars:
            break
        selected.append(fact)
        used += encoded
        if len(selected) >= 16:
            break
    return tuple(selected)


def _bounded_text(value: Any, limit: int = 1_200) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 24]}... [excerpt trimmed]"


def _fit_projection(
    excerpts: list[ToolResultExcerpt],
    facts: tuple[CanonicalFact, ...],
    evidence_ids: tuple[str, ...],
    *,
    max_bytes: int,
) -> tuple[
    list[ToolResultExcerpt],
    tuple[CanonicalFact, ...],
    tuple[str, ...],
    int,
]:
    selected_excerpts = list(excerpts)
    selected_facts = list(facts)
    selected_evidence = list(evidence_ids)

    while _projection_size(selected_excerpts, selected_facts, selected_evidence) > max_bytes:
        if selected_facts:
            selected_facts.pop()
            continue
        if len(selected_excerpts) > 1:
            selected_excerpts.pop()
            continue
        if len(selected_evidence) > 1:
            selected_evidence.pop()
            continue
        if selected_excerpts and len(selected_excerpts[0].text) > 160:
            excerpt = selected_excerpts[0]
            selected_excerpts[0] = excerpt.model_copy(
                update={"text": _bounded_text(excerpt.text, max(160, len(excerpt.text) // 2))}
            )
            continue
        if selected_excerpts:
            selected_excerpts.clear()
            continue
        if selected_evidence:
            selected_evidence.clear()
            continue
        raise ValueError("Tool result continuation budget is too small for its envelope.")

    size = _projection_size(selected_excerpts, selected_facts, selected_evidence)
    return (
        selected_excerpts,
        tuple(selected_facts),
        tuple(selected_evidence),
        size,
    )


def _projection_size(
    excerpts: list[ToolResultExcerpt],
    facts: list[CanonicalFact],
    evidence_ids: list[str],
) -> int:
    payload = {
        "headline": f"Selected {len(excerpts)} exact excerpt(s) for the requested focus.",
        "excerpts": [item.model_dump(mode="json") for item in excerpts],
        "canonical_facts": [item.model_dump(mode="json") for item in facts],
        "evidence_ids": evidence_ids,
        "truncated": True,
        "more_available": True,
        "verified": True,
    }
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
