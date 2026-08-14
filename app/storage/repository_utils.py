from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def recycle_times() -> tuple[str, str]:
    from app.core.settings import get_settings

    now = datetime.now(UTC)
    return now.isoformat(), (
        now + timedelta(days=get_settings().assistant_recycle_retention_days)
    ).isoformat()


def dedupe_uuids(values: tuple[UUID, ...]) -> tuple[UUID, ...]:
    return tuple(dict.fromkeys(values))


def validate_relationship_graph_structure(
    relationships: tuple[dict[str, Any], ...],
    *,
    expected_root: UUID | None,
) -> UUID | None:
    enabled = tuple(item for item in relationships if item.get("enabled") is not False)
    if not enabled:
        return expected_root

    edges: list[tuple[UUID, UUID]] = []
    seen_edges: set[tuple[UUID, UUID]] = set()
    right_ids: set[UUID] = set()
    for relationship in enabled:
        left_id = UUID(str(relationship.get("left_dataset_id") or ""))
        right_id = UUID(str(relationship.get("right_dataset_id") or ""))
        edge = (left_id, right_id)
        if left_id == right_id:
            raise ValueError("Relationship cannot join a dataset to itself.")
        if edge in seen_edges:
            raise ValueError("Relationship plan contains a duplicate edge.")
        if right_id in right_ids:
            raise ValueError("Each joined dataset can have only one parent relationship.")
        seen_edges.add(edge)
        right_ids.add(right_id)
        edges.append(edge)

    roots = {left_id for left_id, _ in edges if left_id not in right_ids}
    if len(roots) != 1:
        raise ValueError("Relationship plan must contain exactly one acyclic root dataset.")
    root = next(iter(roots))
    if expected_root is not None and root != expected_root:
        raise ValueError("Relationship plan root must match the primary dataset.")

    connected = {root}
    remaining = list(edges)
    while remaining:
        progressed = False
        for edge in tuple(remaining):
            left_id, right_id = edge
            if left_id not in connected:
                continue
            if right_id in connected:
                raise ValueError("Relationship plan must not contain cycles or redundant paths.")
            connected.add(right_id)
            remaining.remove(edge)
            progressed = True
        if not progressed:
            raise ValueError("Every relationship must be reachable from the root dataset.")
    return root
