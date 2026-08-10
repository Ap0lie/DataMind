from __future__ import annotations

from collections import deque
from typing import Any


def build_relationship_graph(definition: dict[str, Any]) -> dict[str, Any]:
    nodes = tuple(
        {
            "entity_id": str(entity.get("id") or entity.get("entity_id") or ""),
            "dataset_id": str(entity.get("dataset_id") or ""),
            "name": str(entity.get("name") or ""),
            "entity_type": str(entity.get("entity_type") or "unknown"),
            "grain": str(entity.get("grain") or "one row per source record"),
            "primary_key": entity.get("primary_key"),
        }
        for entity in definition.get("entities") or ()
        if isinstance(entity, dict)
    )
    edges = tuple(
        _graph_edge(relationship)
        for relationship in definition.get("relationships") or ()
        if isinstance(relationship, dict) and relationship.get("enabled", True)
    )
    return {"nodes": nodes, "edges": edges}


def plan_relationship_path(
    definition: dict[str, Any],
    *,
    metric_ids: tuple[str, ...],
    dimension_ids: tuple[str, ...],
) -> dict[str, Any]:
    graph = build_relationship_graph(definition)
    metrics = {
        str(item.get("id")): item
        for item in definition.get("metrics") or ()
        if isinstance(item, dict)
    }
    dimensions = {
        str(item.get("id")): item
        for item in definition.get("dimensions") or ()
        if isinstance(item, dict)
    }
    entities = {node["entity_id"]: node for node in graph["nodes"]}
    metric_source_entities = _metric_entities(metrics, metric_ids)
    dimension_entities = {
        str(dimensions[item].get("entity_id") or dimensions[item].get("entity") or "")
        for item in dimension_ids
        if item in dimensions
    }
    fact_entities = tuple(
        entity_id
        for entity_id, entity in entities.items()
        if entity["entity_type"] == "fact"
    )
    metric_fact_entities = tuple(
        item for item in metric_source_entities if item in fact_entities
    )
    roots = (
        metric_fact_entities
        or (fact_entities if len(fact_entities) == 1 else ())
        or tuple(item for item in metric_source_entities if item in entities)
    )
    targets = tuple(
        dict.fromkeys(
            item
            for item in (*metric_source_entities, *dimension_entities)
            if item in entities and item not in roots
        )
    )
    selected_edges: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    for target in targets:
        path = _shortest_path(graph["edges"], roots, target)
        if not path:
            warnings.append(f"No declared relationship path reaches entity {target}.")
            continue
        current = path[0]["from_entity_id"]
        for item in path:
            edge = item["edge"]
            strategy, safe, reason = _join_strategy(
                edge,
                from_entity_id=current,
                target_is_dimension=item["to_entity_id"] in dimension_entities,
            )
            selected_edges.append(edge)
            steps.append(
                {
                    "relationship_id": edge["relationship_id"],
                    "from_entity_id": current,
                    "to_entity_id": item["to_entity_id"],
                    "strategy": strategy,
                    "safe": safe,
                    "reason": reason,
                }
            )
            if not safe:
                warnings.append(reason)
            current = item["to_entity_id"]
    selected_edges = list(
        {edge["relationship_id"]: edge for edge in selected_edges}.values()
    )
    safe = bool(roots) and len(targets) == len(
        {
            step["to_entity_id"]
            for step in steps
            if step["to_entity_id"] in targets
        }
    ) and all(step["safe"] for step in steps)
    return {
        "metric_entity_ids": roots,
        "metric_source_entity_ids": metric_source_entities,
        "dimension_entity_ids": tuple(sorted(dimension_entities)),
        "metric_grain": tuple(
            entities[item]["grain"] for item in roots if item in entities
        ),
        "output_grain": dimension_ids
        or tuple(entities[item]["grain"] for item in roots if item in entities)
        or ("dataset",),
        "join_path": tuple(selected_edges),
        "steps": tuple(steps),
        "safe": safe,
        "warnings": tuple(dict.fromkeys(warnings)),
    }


def _graph_edge(relationship: dict[str, Any]) -> dict[str, Any]:
    left = str(
        relationship.get("left_entity_id")
        or relationship.get("left_entity")
        or ""
    )
    right = str(
        relationship.get("right_entity_id")
        or relationship.get("right_entity")
        or ""
    )
    return {
        "relationship_id": str(
            relationship.get("id")
            or f"{left}:{relationship.get('left_field_id') or relationship.get('left_field')}->{right}:{relationship.get('right_field_id') or relationship.get('right_field')}"
        ),
        "left_entity_id": left,
        "right_entity_id": right,
        "left_field_id": str(
            relationship.get("left_field_id")
            or relationship.get("left_field")
            or ""
        ),
        "right_field_id": str(
            relationship.get("right_field_id")
            or relationship.get("right_field")
            or ""
        ),
        "cardinality": str(relationship.get("cardinality") or "unknown"),
        "join_type": str(relationship.get("join_type") or "left"),
        "deduplication_strategy": relationship.get("deduplication_strategy"),
        "risk_note": str(relationship.get("risk_note") or ""),
    }


def _metric_entities(
    metrics: dict[str, dict[str, Any]], metric_ids: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []

    def visit(expression: Any, stack: set[str]) -> None:
        if not isinstance(expression, dict):
            return
        if expression.get("op") == "field":
            entity_id = str(expression.get("entity_id") or expression.get("entity") or "")
            if entity_id:
                result.append(entity_id)
        if expression.get("op") == "metric_ref":
            target = str(expression.get("metric_id") or expression.get("metric") or "")
            if target in metrics and target not in stack:
                visit(metrics[target].get("formula"), stack | {target})
        for value in expression.values():
            if isinstance(value, (dict, list, tuple)):
                if isinstance(value, dict):
                    visit(value, stack)
                else:
                    for item in value:
                        visit(item, stack)

    for metric_id in metric_ids:
        if metric_id in metrics:
            visit(metrics[metric_id].get("formula"), {metric_id})
    return tuple(dict.fromkeys(result))


def _shortest_path(
    edges: tuple[dict[str, Any], ...],
    roots: tuple[str, ...],
    target: str,
) -> list[dict[str, Any]]:
    queue = deque((root, []) for root in roots)
    visited = set(roots)
    while queue:
        entity_id, path = queue.popleft()
        if entity_id == target:
            return path
        for edge in edges:
            if edge["left_entity_id"] == entity_id:
                next_id = edge["right_entity_id"]
            elif edge["right_entity_id"] == entity_id:
                next_id = edge["left_entity_id"]
            else:
                continue
            if next_id in visited:
                continue
            visited.add(next_id)
            queue.append(
                (
                    next_id,
                    [
                        *path,
                        {
                            "edge": edge,
                            "from_entity_id": entity_id,
                            "to_entity_id": next_id,
                        },
                    ],
                )
            )
    return []


def _join_strategy(
    edge: dict[str, Any],
    *,
    from_entity_id: str,
    target_is_dimension: bool,
) -> tuple[str, bool, str]:
    cardinality = edge["cardinality"]
    forward = from_entity_id == edge["left_entity_id"]
    from_side, to_side = _cardinality_sides(cardinality, forward=forward)
    deduplication = str(edge.get("deduplication_strategy") or "")
    if from_side == "many" and to_side == "one":
        return "direct_join", True, "Many-to-one traversal preserves metric grain."
    if from_side == "one" and to_side == "one":
        return "direct_join", True, "One-to-one traversal preserves metric grain."
    if from_side == "one" and to_side == "many":
        if deduplication:
            return (
                "deduplicate_before_join",
                True,
                f"Child rows must be reduced with {deduplication} before the join.",
            )
        if not target_is_dimension:
            return (
                "semi_join",
                False,
                "Use an existence filter; this path requires semi-join compilation before execution.",
            )
        return (
            "pre_aggregate_before_join",
            False,
            "One-to-many traversal would duplicate the metric; define an allocation or deduplication rule.",
        )
    if from_side == "many" and to_side == "many":
        if deduplication:
            return (
                "pre_aggregate_before_join",
                False,
                f"Both sides require key-grain aggregation using {deduplication} before execution.",
            )
        return (
            "blocked",
            False,
            "Many-to-many traversal is blocked until an explicit pre-aggregation rule is published.",
        )
    return (
        "blocked",
        False,
        "Unknown relationship cardinality cannot be used for deterministic aggregation.",
    )


def _cardinality_sides(cardinality: str, *, forward: bool) -> tuple[str, str]:
    sides = {
        "one_to_one": ("one", "one"),
        "one_to_many": ("one", "many"),
        "many_to_one": ("many", "one"),
        "many_to_many": ("many", "many"),
    }.get(cardinality, ("unknown", "unknown"))
    return sides if forward else (sides[1], sides[0])
