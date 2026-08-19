from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any
from uuid import UUID

from app.memory.models import MemoryAgent, ParsedMemoryNamespace

_AGENT_MEMORY_TYPES: dict[MemoryAgent, frozenset[str]] = {
    "kimi": frozenset(
        {
            "preference",
            "terminology",
            "metric_definition",
            "business_context",
            "workflow_preference",
        }
    ),
    "planner": frozenset({"metric_definition", "business_context", "analysis_experience"}),
    "sql": frozenset(),
    "python": frozenset(),
    "reviewer": frozenset({"metric_definition", "business_context", "analysis_experience"}),
    "report": frozenset({"preference", "workflow_preference", "analysis_experience"}),
}


def memory_store_key(memory: Mapping[str, Any]) -> str:
    memory_type = str(memory.get("memory_type") or "business_context").strip()
    subject_key = str(
        memory.get("subject_key") or memory.get("normalized_key") or memory.get("memory_id")
    ).strip()
    if not subject_key:
        raise ValueError("Memory record requires a stable subject key.")
    return f"{memory_type}:{subject_key}"


def parse_memory_store_key(key: str) -> tuple[str, str]:
    memory_type, separator, subject_key = str(key).partition(":")
    if not separator or not memory_type or not subject_key:
        raise ValueError("Memory key must use '<memory_type>:<subject_key>'.")
    return memory_type, subject_key


def memory_to_store_value(memory: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in memory.items()}


def store_value_to_memory(value: Mapping[str, Any]) -> dict[str, Any]:
    memory = dict(value)
    for key in ("memory_id", "scope_id", "source_conversation_id", "source_message_id", "source_job_id"):
        if memory.get(key):
            memory[key] = UUID(str(memory[key]))
    return memory


def project_agent_memories(
    agent: MemoryAgent,
    memories: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Expose only the memory fields and kinds approved for one Agent role."""
    allowed = _AGENT_MEMORY_TYPES[agent]
    if not allowed:
        return ()
    output: list[dict[str, Any]] = []
    for memory in memories:
        memory_type = str(memory.get("memory_type") or "")
        if memory_type not in allowed:
            continue
        structured = memory.get("structured_value")
        structured = dict(structured) if isinstance(structured, Mapping) else {}
        if memory_type == "analysis_experience":
            structured = {
                key: structured.get(key)
                for key in (
                    "analysis_contract",
                    "semantic_model_id",
                    "semantic_model_version",
                    "join_plan",
                    "relationship_plan",
                    "tool_sequence",
                    "result_summary",
                    "report_id",
                )
            }
        output.append(
            {
                key: memory.get(key)
                for key in (
                    "memory_id",
                    "memory_kind",
                    "memory_type",
                    "scope_type",
                    "scope_id",
                    "subject_key",
                    "entity_key",
                    "predicate",
                    "content",
                    "unit",
                    "confidence",
                    "explicit",
                    "source_message_ids",
                    "source_kind",
                    "source_job_id",
                    "relevance_score",
                    "utility_score",
                    "final_score",
                    "recall_reason",
                    "usage_id",
                )
                if memory.get(key) is not None
            }
            | {"structured_value": structured}
        )
    return tuple(output)


def repository_fields_from_store_value(
    *,
    namespace: ParsedMemoryNamespace,
    key: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    memory_type, key_subject = parse_memory_store_key(key)
    value_type = str(value.get("memory_type") or memory_type)
    if value_type != memory_type:
        raise ValueError("Memory key and value use different memory types.")
    subject_key = str(value.get("subject_key") or key_subject).strip()
    if subject_key != key_subject:
        raise ValueError("Memory key and value use different subject keys.")
    scope_id = None if namespace.scope_type == "user" else UUID(namespace.scope_key)
    return {
        "memory_type": memory_type,
        "scope_type": namespace.scope_type,
        "scope_id": scope_id,
        "normalized_key": str(value.get("normalized_key") or subject_key),
        "subject_key": subject_key,
        "entity_key": str(value.get("entity_key") or subject_key),
        "predicate": str(value.get("predicate") or "value"),
        "content": str(value.get("content") or "").strip(),
        "structured_value": dict(value.get("structured_value") or {}),
        "typed_value": dict(value.get("typed_value") or {}),
        "unit": str(value["unit"]) if value.get("unit") is not None else None,
        "memory_kind": namespace.memory_kind,
        "explicit": bool(value.get("explicit", True)),
        "confidence": float(value.get("confidence", 1.0)),
        "status": str(value.get("status") or "active"),
        "pinned": bool(value.get("pinned", False)),
        "source_conversation_id": _optional_uuid(value.get("source_conversation_id")),
        "source_message_id": _optional_uuid(value.get("source_message_id")),
        "application_policy": str(value.get("application_policy") or "relevant"),
        "source_kind": str(value.get("source_kind") or "langmem_store"),
        "source_job_id": _optional_uuid(value.get("source_job_id")),
        "correction": bool(value.get("correction", False)),
    }


def matches_filter(value: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = value.get(key)
        if isinstance(expected, Mapping):
            if not all(_compare(actual, operator, operand) for operator, operand in expected.items()):
                return False
        elif actual != expected:
            return False
    return True


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "$eq":
        return bool(actual == expected)
    if operator == "$ne":
        return bool(actual != expected)
    if operator == "$gt":
        return actual is not None and bool(actual > expected)
    if operator == "$gte":
        return actual is not None and bool(actual >= expected)
    if operator == "$lt":
        return actual is not None and bool(actual < expected)
    if operator == "$lte":
        return actual is not None and bool(actual <= expected)
    raise ValueError(f"Unsupported memory filter operator: {operator}")


def _optional_uuid(value: Any) -> UUID | None:
    return UUID(str(value)) if value not in (None, "") else None


def _json_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    return value
