from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.memory.models import GuardedMemoryRecord

_STATUSES = {"active", "pending", "superseded", "stale", "dormant", "recycled"}
_MEMORY_TYPES = {
    "preference",
    "terminology",
    "metric_definition",
    "business_context",
    "workflow_preference",
}
_EXPLICIT_MARKERS = (
    "请记住",
    "记住",
    "以后",
    "从今以后",
    "默认",
    "我偏好",
    "我喜欢",
    "定义为",
    "口径是",
    "remember",
    "from now on",
    "by default",
    "i prefer",
    "defined as",
)
_SENSITIVE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|密码|口令)\s*[:=：]\s*\S+",
        re.I,
    ),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b\d{17}[\dXx]\b"),
    re.compile(r"\b\d{16,19}\b"),
)


@dataclass(frozen=True)
class MemoryGuardResult:
    accepted: tuple[GuardedMemoryRecord, ...]
    rejected_codes: tuple[str, ...]


class DataMindMemoryGuard:
    """Validate model-formed memory against the user message and fixed scope."""

    def validate(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        source_message: Mapping[str, Any],
        source_message_id: UUID,
    ) -> MemoryGuardResult:
        source_text = str(source_message.get("content") or "")
        if (
            str(source_message.get("message_id") or "") != str(source_message_id)
            or str(source_message.get("role") or "") != "user"
        ):
            return MemoryGuardResult((), ("invalid_source_message",))
        if sensitive_memory_reason(source_text):
            return MemoryGuardResult((), ("sensitive_source",))

        accepted: list[GuardedMemoryRecord] = []
        rejected: list[str] = []
        expected_sources = (str(source_message_id),)
        explicit = any(marker in source_text.casefold() for marker in _EXPLICIT_MARKERS)
        for raw in tuple(records)[:3]:
            memory_type = str(raw.get("memory_type") or "")
            if memory_type not in _MEMORY_TYPES:
                rejected.append("unsupported_memory_type")
                continue
            source_ids = tuple(str(value) for value in raw.get("source_message_ids") or ())
            if source_ids != expected_sources:
                rejected.append("source_mismatch")
                continue
            evidence = _clean(str(raw.get("evidence") or ""))
            content = _clean(str(raw.get("content") or evidence))
            if (
                not evidence
                or evidence not in source_text
                or sensitive_memory_reason(evidence)
                or sensitive_memory_reason(content)
            ):
                rejected.append("invalid_evidence")
                continue
            entity_key = _identifier(str(raw.get("entity_key") or ""))
            predicate = _identifier(str(raw.get("predicate") or "value")) or "value"
            typed_value = raw.get("typed_value")
            if not entity_key or not isinstance(typed_value, Mapping) or "value" not in typed_value:
                rejected.append("invalid_structure")
                continue
            if str(raw.get("scope") or "conversation") != "conversation":
                rejected.append("scope_expansion")
                continue
            accepted.append(
                GuardedMemoryRecord(
                    memory_type=memory_type,
                    entity_key=entity_key,
                    predicate=predicate,
                    typed_value=dict(typed_value),
                    unit=str(raw.get("unit") or "").strip() or None,
                    content=content,
                    evidence=evidence,
                    source_message_ids=source_ids,
                    confidence=min(0.95, max(0.5, float(raw.get("confidence") or 0.65))),
                    explicit=explicit,
                    correction=bool(raw.get("correction")) and explicit,
                    valid_from=_optional_text(raw.get("valid_from")),
                    valid_to=_optional_text(raw.get("valid_to")),
                )
            )
        return MemoryGuardResult(tuple(accepted), tuple(rejected))


def validate_store_value(value: dict[str, Any]) -> None:
    memory_type = str(value.get("memory_type") or "").strip()
    content = str(value.get("content") or "").strip()
    status = str(value.get("status") or "active")
    if not memory_type:
        raise ValueError("Memory value requires memory_type.")
    if not content:
        raise ValueError("Memory value requires content.")
    if status not in _STATUSES:
        raise ValueError("Unsupported memory status.")
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError("Memory value must be JSON serializable.") from exc


def sensitive_memory_reason(text: str) -> str | None:
    if any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS):
        return "检测到凭证、Token 或高风险个人信息，未写入长期记忆。"
    return None


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_:\-.\u4e00-\u9fff]+", "_", value.casefold()).strip("_")
    return normalized[:160]


def _clean(value: str) -> str:
    return " ".join(value.split())[:2_000]


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text[:64] or None
