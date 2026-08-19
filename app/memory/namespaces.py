from __future__ import annotations

from typing import cast
from uuid import UUID

from app.memory.models import (
    MemoryKind,
    MemoryNamespace,
    MemoryScopeType,
    ParsedMemoryNamespace,
)

_SCOPE_TYPES = {"user", "dataset", "dataset_group", "report"}
_MEMORY_KINDS = {"semantic", "episodic"}


def build_memory_namespace(
    *,
    user_id: str,
    scope_type: str,
    scope_id: UUID | str | None,
    memory_kind: str,
) -> MemoryNamespace:
    resolved_user = str(user_id).strip()
    if not resolved_user:
        raise ValueError("Memory namespace requires a user id.")
    if "." in resolved_user or resolved_user == "langgraph":
        raise ValueError("Memory user id is not a valid LangGraph namespace label.")
    if scope_type not in _SCOPE_TYPES:
        raise ValueError("Unsupported memory scope type.")
    if memory_kind not in _MEMORY_KINDS:
        raise ValueError("Unsupported memory kind.")
    if scope_type == "user":
        if scope_id is not None:
            raise ValueError("User-scoped memory cannot have a scope id.")
        scope_key = "user"
    else:
        if scope_id is None:
            raise ValueError("Asset-scoped memory requires a scope id.")
        scope_key = str(UUID(str(scope_id)))
    return (
        resolved_user,
        scope_type,
        scope_key,
        memory_kind,
    )


def parse_memory_namespace(namespace: tuple[str, ...]) -> ParsedMemoryNamespace:
    if len(namespace) != 4:
        raise ValueError("Memory namespace must contain exactly four components.")
    user_id, scope_type, scope_key, memory_kind = (str(value).strip() for value in namespace)
    if not user_id:
        raise ValueError("Memory namespace requires a user id.")
    if "." in user_id or user_id == "langgraph":
        raise ValueError("Memory user id is not a valid LangGraph namespace label.")
    if scope_type not in _SCOPE_TYPES:
        raise ValueError("Unsupported memory scope type.")
    if memory_kind not in _MEMORY_KINDS:
        raise ValueError("Unsupported memory kind.")
    if scope_type == "user":
        if scope_key != "user":
            raise ValueError("User-scoped memory must use the 'user' scope key.")
    else:
        scope_key = str(UUID(scope_key))
    return ParsedMemoryNamespace(
        user_id=user_id,
        scope_type=cast(MemoryScopeType, scope_type),
        scope_key=scope_key,
        memory_kind=cast(MemoryKind, memory_kind),
    )


def namespace_for_memory(user_id: str, memory: dict[str, object]) -> MemoryNamespace:
    return build_memory_namespace(
        user_id=user_id,
        scope_type=str(memory["scope_type"]),
        scope_id=(
            str(memory["scope_id"])
            if memory.get("scope_id") is not None
            else None
        ),
        memory_kind=str(memory.get("memory_kind") or "semantic"),
    )


def namespace_has_prefix(namespace: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(prefix) <= len(namespace) and namespace[: len(prefix)] == prefix
