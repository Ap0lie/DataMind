from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MemoryKind = Literal["semantic", "episodic"]
MemoryScopeType = Literal["user", "dataset", "dataset_group", "report"]
MemoryAgent = Literal["kimi", "planner", "sql", "python", "reviewer", "report"]
type MemoryNamespace = tuple[str, str, str, str]


@dataclass(frozen=True)
class ParsedMemoryNamespace:
    user_id: str
    scope_type: MemoryScopeType
    scope_key: str
    memory_kind: MemoryKind


@dataclass(frozen=True)
class GuardedMemoryRecord:
    memory_type: str
    entity_key: str
    predicate: str
    typed_value: dict[str, Any]
    unit: str | None
    content: str
    evidence: str
    source_message_ids: tuple[str, ...]
    confidence: float
    explicit: bool
    correction: bool
    valid_from: str | None = None
    valid_to: str | None = None
