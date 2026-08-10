from __future__ import annotations

from typing import Any

from app.schemas.analysis import (
    AnalysisContractResponse,
    AnalysisLineageResponse,
    ChartResponse,
    InsightFindingResponse,
    LineageEdgeResponse,
    LineageNodeResponse,
    MultiDatasetProfileResponse,
    PlannerMetadataResponse,
)


def build_analysis_lineage(
    *,
    contract: AnalysisContractResponse,
    planner_metadata: PlannerMetadataResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None,
    findings: tuple[InsightFindingResponse, ...],
    charts: tuple[ChartResponse, ...],
    report_id: str | None = None,
) -> AnalysisLineageResponse:
    nodes: dict[str, LineageNodeResponse] = {}
    edges: dict[tuple[str, str, str], LineageEdgeResponse] = {}

    def add_node(node: LineageNodeResponse) -> None:
        nodes[node.node_id] = node

    def add_edge(source: str, target: str, relation: str) -> None:
        if source in nodes and target in nodes:
            edges[(source, target, relation)] = LineageEdgeResponse(
                source_node_id=source,
                target_node_id=target,
                relation=relation,
            )

    source_map = (
        multi_dataset_context.column_source_map if multi_dataset_context else {}
    )
    relevant_fields = tuple(
        dict.fromkeys(
            (
                *((contract.metric,) if contract.metric else ()),
                *contract.dimensions,
                *((contract.time_field,) if contract.time_field else ()),
                *(
                    str(field)
                    for chart in charts
                    for field in (chart.spec.get("x"), chart.spec.get("y"))
                    if field
                ),
            )
        )
    )
    for field in relevant_fields:
        node_id = f"field:{field}"
        add_node(
            LineageNodeResponse(
                node_id=node_id,
                node_type="field",
                label=field,
                source_ref=source_map.get(field, "primary_dataset"),
            )
        )

    metric_node_id = f"metric:{contract.metric}" if contract.metric else "metric:count"
    add_node(
        LineageNodeResponse(
            node_id=metric_node_id,
            node_type="metric",
            label=contract.metric or "row_count",
            metadata={
                "grain": list(contract.grain),
                "semantic_model_id": str(planner_metadata.semantic_model_id)
                if planner_metadata and planner_metadata.semantic_model_id
                else None,
                "semantic_model_version": planner_metadata.semantic_model_version
                if planner_metadata
                else None,
            },
        )
    )
    if contract.metric:
        add_edge(f"field:{contract.metric}", metric_node_id, "defines")

    for index, finding in enumerate(findings):
        node_id = f"finding:{index}"
        add_node(
            LineageNodeResponse(
                node_id=node_id,
                node_type="finding",
                label=finding.title,
                source_ref=finding.data_source,
                metadata={"evidence": finding.evidence},
            )
        )
        add_edge(metric_node_id, node_id, "supports")

    for index, chart in enumerate(charts):
        node_id = f"chart:{index}"
        add_node(
            LineageNodeResponse(
                node_id=node_id,
                node_type="chart",
                label=chart.title,
                metadata={"chart_type": chart.chart_type},
            )
        )
        chart_fields = tuple(
            str(field)
            for field in (chart.spec.get("x"), chart.spec.get("y"))
            if field
        )
        for field in chart_fields:
            add_edge(f"field:{field}", node_id, "visualizes")
        if not chart_fields:
            add_edge(metric_node_id, node_id, "visualizes")

    if report_id:
        report_node_id = f"report:{report_id}"
        add_node(
            LineageNodeResponse(
                node_id=report_node_id,
                node_type="report",
                label="DataMind 分析报告",
                source_ref=report_id,
            )
        )
        for node in tuple(nodes.values()):
            if node.node_type in {"finding", "chart"}:
                add_edge(node.node_id, report_node_id, "included_in")

    semantic_plan: dict[str, Any] = (
        planner_metadata.semantic_plan if planner_metadata else {}
    )
    context_graph, context_grain = _multi_dataset_lineage(multi_dataset_context)
    semantic_graph = semantic_plan.get("relationship_graph") or {}
    semantic_grain = semantic_plan.get("grain_plan") or {}
    return AnalysisLineageResponse(
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
        relationship_graph=(
            context_graph
            if context_graph.get("edges") and not semantic_graph.get("edges")
            else semantic_graph
        ),
        grain_plan=(
            context_grain
            if context_grain.get("steps") and not semantic_grain.get("steps")
            else semantic_grain
        ),
    )


def _multi_dataset_lineage(
    context: MultiDatasetProfileResponse | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if context is None or not context.join_plan:
        return {}, {}
    datasets = (context.primary_dataset, *context.additional_datasets)
    names = {str(item.dataset_id): item.name for item in datasets}
    nodes = tuple(
        {
            "entity_id": str(item.dataset_id),
            "dataset_id": str(item.dataset_id),
            "name": item.name,
            "entity_type": (
                "fact"
                if item.dataset_id == context.primary_dataset.dataset_id
                else "dimension"
            ),
            "grain": "one row per source record",
            "primary_key": None,
        }
        for item in datasets
    )
    edges = []
    steps = []
    join_summaries = {
        (
            str(item.get("left_dataset_id") or ""),
            str(item.get("right_dataset_id") or ""),
        ): item
        for item in context.join_summary.get("joins", ())
        if isinstance(item, dict)
    }
    for index, join in enumerate(context.join_plan, start=1):
        left_id = str(join.left_dataset_id)
        right_id = str(join.right_dataset_id)
        summary = join_summaries.get((left_id, right_id), {})
        relationship_id = f"runtime_join_{index}"
        expansion = float(summary.get("row_expansion_ratio") or 1)
        right_unique = bool(summary.get("right_key_unique"))
        safe = summary.get("status", "joined") == "joined" and expansion <= 1.05
        reason = (
            f"{names.get(left_id, left_id)}.{join.left_column} "
            f"{join.join_type} join {names.get(right_id, right_id)}.{join.right_column}"
        )
        edges.append(
            {
                "relationship_id": relationship_id,
                "left_entity_id": left_id,
                "right_entity_id": right_id,
                "left_field_id": join.left_column,
                "right_field_id": join.right_column,
                "cardinality": "many_to_one" if right_unique else "unknown",
                "join_type": join.join_type,
                "deduplication_strategy": None,
                "risk_note": "" if safe else "Join may expand the analysis grain.",
            }
        )
        steps.append(
            {
                "relationship_id": relationship_id,
                "from_entity_id": left_id,
                "to_entity_id": right_id,
                "strategy": "direct_join",
                "safe": safe,
                "reason": reason,
            }
        )
    warnings = tuple(
        step["reason"] for step in steps if not bool(step["safe"])
    )
    return (
        {"nodes": nodes, "edges": tuple(edges)},
        {
            "metric_entity_ids": (str(context.primary_dataset.dataset_id),),
            "metric_source_entity_ids": (str(context.primary_dataset.dataset_id),),
            "dimension_entity_ids": tuple(
                str(item.dataset_id) for item in context.additional_datasets
            ),
            "metric_grain": ("one row per primary source record",),
            "output_grain": ("joined dataset",),
            "join_path": tuple(edges),
            "steps": tuple(steps),
            "safe": all(bool(step["safe"]) for step in steps),
            "warnings": warnings,
        },
    )
