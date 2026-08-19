from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

from app.memory.contracts import VersionedMemoryRepository
from app.memory.guards import validate_store_value
from app.memory.namespaces import (
    namespace_for_memory,
    namespace_has_prefix,
    parse_memory_namespace,
)
from app.memory.projections import (
    matches_filter,
    memory_store_key,
    memory_to_store_value,
    parse_memory_store_key,
    repository_fields_from_store_value,
)


class DataMindMemoryStore(BaseStore):
    """LangGraph Store adapter over DataMind's versioned memory repository."""

    def __init__(
        self,
        repository: VersionedMemoryRepository,
        *,
        recycle_retention_days: int = 30,
    ) -> None:
        self.repository = repository
        self.recycle_retention_days = recycle_retention_days

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        return [self._execute(operation) for operation in ops]

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return await asyncio.to_thread(self.batch, tuple(ops))

    def _execute(self, operation: Any) -> Any:
        if isinstance(operation, GetOp):
            return self._get(operation)
        if isinstance(operation, PutOp):
            return self._put(operation)
        if isinstance(operation, SearchOp):
            return self._search(operation)
        if isinstance(operation, ListNamespacesOp):
            return self._list_namespaces(operation)
        raise TypeError(f"Unsupported memory store operation: {type(operation).__name__}")

    def _get(self, operation: GetOp) -> Item | None:
        namespace = parse_memory_namespace(operation.namespace)
        self._authorize_namespace(namespace.user_id)
        memory = self._find_current(operation.namespace, operation.key)
        return self._item(memory) if memory is not None else None

    def _put(self, operation: PutOp) -> None:
        namespace = parse_memory_namespace(operation.namespace)
        self._authorize_namespace(namespace.user_id)
        if operation.ttl is not None:
            raise NotImplementedError("DataMind Memory Store does not support per-item TTL.")
        current = self._find_current(operation.namespace, operation.key)
        if operation.value is None:
            if current is not None and current["status"] != "recycled":
                self.repository.recycle(
                    current["memory_id"],
                    retention_days=self.recycle_retention_days,
                )
            return None
        self.put_versioned(operation.namespace, operation.key, operation.value)
        return None

    def put_versioned(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist through the Store boundary and return the resulting version."""
        parsed = parse_memory_namespace(namespace)
        self._authorize_namespace(parsed.user_id)
        validate_store_value(value)
        fields = repository_fields_from_store_value(
            namespace=parsed,
            key=key,
            value=value,
        )
        return self.repository.save(**fields)

    def _search(self, operation: SearchOp) -> list[SearchItem]:
        self._validate_prefix(operation.namespace_prefix)
        records = self._search_records(operation)
        results: list[SearchItem] = []
        for memory in records:
            namespace = namespace_for_memory(self.repository.user_id, memory)
            if not namespace_has_prefix(namespace, operation.namespace_prefix):
                continue
            value = memory_to_store_value(memory)
            if not matches_filter(value, operation.filter):
                continue
            score = _search_score(operation.query, memory)
            if operation.query and score <= 0:
                continue
            results.append(self._search_item(memory, score))
        results.sort(
            key=lambda item: (
                item.score or 0.0,
                bool(item.value.get("pinned")),
                item.updated_at,
            ),
            reverse=True,
        )
        start = max(0, operation.offset)
        return results[start : start + max(0, operation.limit)]

    def _search_records(self, operation: SearchOp) -> tuple[dict[str, Any], ...]:
        if len(operation.namespace_prefix) != 4:
            return tuple(self.repository.list(limit=500))
        namespace = parse_memory_namespace(operation.namespace_prefix)
        scope_id = None if namespace.scope_type == "user" else UUID(namespace.scope_key)
        filters: dict[str, Any] = {
            "scope_type": namespace.scope_type,
            "scope_id": scope_id,
            "memory_kind": namespace.memory_kind,
            "limit": 500,
        }
        if operation.filter and isinstance(operation.filter.get("status"), str):
            filters["status"] = operation.filter["status"]
        return tuple(self.repository.list(**filters))

    def _list_namespaces(self, operation: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = {
            tuple(namespace_for_memory(self.repository.user_id, memory))
            for memory in self.repository.list(limit=500)
        }
        filtered: list[tuple[str, ...]] = [
            namespace
            for namespace in namespaces
            if _matches_namespace_conditions(namespace, operation.match_conditions)
        ]
        if operation.max_depth is not None:
            filtered = [namespace[: operation.max_depth] for namespace in filtered]
        ordered = sorted(set(filtered))
        start = max(0, operation.offset)
        return ordered[start : start + max(0, operation.limit)]

    def _find_current(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> dict[str, Any] | None:
        parsed = parse_memory_namespace(namespace)
        memory_type, subject_key = parse_memory_store_key(key)
        scope_id = None if parsed.scope_type == "user" else UUID(parsed.scope_key)
        candidates = self.repository.list(
            scope_type=parsed.scope_type,
            scope_id=scope_id,
            memory_type=memory_type,
            memory_kind=parsed.memory_kind,
            limit=500,
        )
        status_priority = {"active": 0, "pending": 1, "dormant": 2, "stale": 3}
        matching = [
            memory
            for memory in candidates
            if memory["subject_key"] == subject_key
            and memory["status"] not in {"superseded", "recycled"}
        ]
        return min(
            matching,
            key=lambda memory: (
                status_priority.get(str(memory["status"]), 10),
                -int(memory.get("version") or 1),
            ),
            default=None,
        )

    def _item(self, memory: Mapping[str, Any]) -> Item:
        return Item(
            namespace=namespace_for_memory(self.repository.user_id, dict(memory)),
            key=memory_store_key(memory),
            value=memory_to_store_value(memory),
            created_at=_datetime(memory["created_at"]),
            updated_at=_datetime(memory["updated_at"]),
        )

    def _search_item(self, memory: Mapping[str, Any], score: float) -> SearchItem:
        item = self._item(memory)
        return SearchItem(
            namespace=item.namespace,
            key=item.key,
            value=item.value,
            created_at=item.created_at,
            updated_at=item.updated_at,
            score=score,
        )

    def _validate_prefix(self, prefix: tuple[str, ...]) -> None:
        if not prefix:
            raise ValueError("Memory Store requires a user-scoped namespace prefix.")
        self._authorize_namespace(str(prefix[0]))
        if len(prefix) > 4:
            raise ValueError("Memory namespace prefix is too deep.")
        if len(prefix) == 4:
            parse_memory_namespace(prefix)

    def _authorize_namespace(self, user_id: str) -> None:
        if user_id != self.repository.user_id:
            raise PermissionError("Memory namespace does not belong to this repository user.")


def _search_score(query: str | None, memory: Mapping[str, Any]) -> float:
    if not query:
        return 1.0
    query_tokens = _tokens(query)
    memory_tokens = _tokens(
        " ".join(
            str(memory.get(key) or "")
            for key in ("content", "normalized_key", "subject_key", "entity_key", "predicate")
        )
    )
    lexical = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
    durable = 0.15 if memory.get("application_policy") == "always" else 0.0
    pinned = 0.05 if memory.get("pinned") else 0.0
    utility = float(memory.get("utility_score") or 0.5) * 0.05
    return round(min(1.0, lexical + durable + pinned + utility), 6)


def _tokens(value: str) -> set[str]:
    normalized = str(value).casefold()
    words = set(re.findall(r"[a-z0-9_]+", normalized))
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", normalized)
    words.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    return {token for token in words if token}


def _matches_namespace_conditions(
    namespace: tuple[str, ...],
    conditions: tuple[Any, ...] | None,
) -> bool:
    for condition in conditions or ():
        path = tuple(str(value) for value in condition.path)
        if condition.match_type == "prefix":
            candidate = namespace[: len(path)]
        elif condition.match_type == "suffix":
            candidate = namespace[-len(path) :] if path else ()
        else:
            return False
        if len(candidate) != len(path) or any(
            expected != "*" and expected != actual
            for actual, expected in zip(candidate, path, strict=True)
        ):
            return False
    return True


def _datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
