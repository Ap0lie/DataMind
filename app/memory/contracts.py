from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID


class VersionedMemoryRepository(Protocol):
    """Persistence surface required by the LangGraph Store adapter."""

    user_id: str

    def get(self, memory_id: UUID, *, include_recycled: bool = True) -> dict[str, Any]: ...

    def list(
        self,
        *,
        scope_type: str | None = None,
        scope_id: UUID | None = None,
        memory_type: str | None = None,
        memory_kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> tuple[dict[str, Any], ...]: ...

    def save(
        self,
        *,
        memory_type: str,
        scope_type: str,
        scope_id: UUID | None,
        normalized_key: str,
        content: str,
        explicit: bool,
        confidence: float,
        status: str,
        pinned: bool = False,
        source_conversation_id: UUID | None = None,
        source_message_id: UUID | None = None,
        memory_kind: str = "semantic",
        subject_key: str | None = None,
        structured_value: dict[str, Any] | None = None,
        entity_key: str | None = None,
        predicate: str = "value",
        typed_value: dict[str, Any] | None = None,
        unit: str | None = None,
        application_policy: str = "relevant",
        source_kind: str = "user_message",
        source_job_id: UUID | None = None,
        correction: bool = False,
    ) -> dict[str, Any]: ...

    def recycle(self, memory_id: UUID, *, retention_days: int) -> dict[str, Any]: ...
