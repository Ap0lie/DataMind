from __future__ import annotations

from uuid import uuid4

from app.analysis.lineage import build_analysis_lineage
from app.schemas.analysis import (
    AnalysisContractResponse,
    ChartResponse,
    DatasetJoinConfig,
    DatasetReferenceResponse,
    InsightFindingResponse,
    MultiDatasetProfileResponse,
    PlannerMetadataResponse,
)
from app.semantic.relationship_graph import plan_relationship_path


def test_relationship_planner_selects_only_safe_shortest_path() -> None:
    definition = _definition(cardinality="many_to_one")
    result = plan_relationship_path(
        definition,
        metric_ids=("revenue",),
        dimension_ids=("customer_region",),
    )

    assert result["safe"] is True
    assert [item["relationship_id"] for item in result["join_path"]] == [
        "orders_customers"
    ]
    assert result["steps"][0]["strategy"] == "direct_join"


def test_relationship_planner_blocks_one_to_many_grain_expansion() -> None:
    definition = _definition(cardinality="one_to_many")
    result = plan_relationship_path(
        definition,
        metric_ids=("revenue",),
        dimension_ids=("customer_region",),
    )

    assert result["safe"] is False
    assert result["steps"][0]["strategy"] == "pre_aggregate_before_join"
    assert "duplicate" in result["warnings"][0]


def test_relationship_planner_allows_explicit_deduplication() -> None:
    definition = _definition(
        cardinality="one_to_many",
        deduplication_strategy="latest_by_updated_at",
    )
    result = plan_relationship_path(
        definition,
        metric_ids=("revenue",),
        dimension_ids=("customer_region",),
    )

    assert result["safe"] is True
    assert result["steps"][0]["strategy"] == "deduplicate_before_join"


def test_analysis_lineage_reaches_report_artifact() -> None:
    dataset_id = uuid4()
    contract = AnalysisContractResponse(
        objective="Compare revenue by region",
        population="All imported orders",
        dataset_ids=(dataset_id,),
        analysis_type="comparison",
        metric="amount",
        dimensions=("region",),
        grain=("region",),
        method="grouped comparison",
    )
    planner = PlannerMetadataResponse(
        confidence=0.9,
        semantic_plan={
            "relationship_graph": {"nodes": [{"entity_id": "orders"}], "edges": []},
            "grain_plan": {"safe": True, "steps": []},
        },
    )
    lineage = build_analysis_lineage(
        contract=contract,
        planner_metadata=planner,
        multi_dataset_context=None,
        findings=(
            InsightFindingResponse(
                title="North leads",
                content="North has the highest revenue.",
                data_source="SQL",
                evidence="evidence_id:sql-1",
            ),
        ),
        charts=(
            ChartResponse(
                title="Revenue by region",
                chart_type="bar",
                spec={"x": "region", "y": "amount"},
                data=(),
            ),
        ),
        report_id="report-1",
    )

    report_node = next(item for item in lineage.nodes if item.node_type == "report")
    incoming = {
        edge.source_node_id
        for edge in lineage.edges
        if edge.target_node_id == report_node.node_id
    }
    assert {"finding:0", "chart:0"} <= incoming
    assert lineage.grain_plan["safe"] is True


def test_analysis_lineage_uses_executed_runtime_join_without_semantic_graph() -> None:
    orders_id = uuid4()
    customers_id = uuid4()
    contract = AnalysisContractResponse(
        objective="Compare sales by segment",
        population="All imported orders",
        dataset_ids=(orders_id, customers_id),
        analysis_type="comparison",
        metric="amount",
        dimensions=("customers__segment",),
        grain=("customers__segment",),
        method="grouped comparison",
    )
    context = MultiDatasetProfileResponse(
        primary_dataset=DatasetReferenceResponse(
            dataset_id=orders_id,
            name="orders.csv",
            status="cleaned",
            row_count=5,
            column_count=3,
            columns=("order_id", "customer_id", "amount"),
        ),
        additional_datasets=(
            DatasetReferenceResponse(
                dataset_id=customers_id,
                name="customers.csv",
                status="cleaned",
                row_count=5,
                column_count=2,
                columns=("customer_id", "segment"),
            ),
        ),
        join_plan=(
            DatasetJoinConfig(
                left_dataset_id=orders_id,
                right_dataset_id=customers_id,
                left_column="customer_id",
                right_column="customer_id",
                join_type="left",
            ),
        ),
        join_summary={
            "joins": [
                {
                    "status": "joined",
                    "left_dataset_id": str(orders_id),
                    "right_dataset_id": str(customers_id),
                    "right_key_unique": True,
                    "row_expansion_ratio": 1.0,
                }
            ]
        },
        column_source_map={"customers__segment": "customers.csv"},
    )

    lineage = build_analysis_lineage(
        contract=contract,
        planner_metadata=PlannerMetadataResponse(confidence=0.8),
        multi_dataset_context=context,
        findings=(),
        charts=(),
        report_id="runtime-report",
    )

    assert len(lineage.relationship_graph["edges"]) == 1
    assert lineage.relationship_graph["edges"][0]["left_field_id"] == "customer_id"
    assert lineage.grain_plan["safe"] is True
    assert len(lineage.grain_plan["steps"]) == 1


def _definition(
    *,
    cardinality: str,
    deduplication_strategy: str | None = None,
) -> dict[str, object]:
    relationship = {
        "id": "orders_customers",
        "left_entity_id": "orders",
        "right_entity_id": "customers",
        "left_field_id": "orders_customer_id",
        "right_field_id": "customers_customer_id",
        "cardinality": cardinality,
        "enabled": True,
    }
    if deduplication_strategy:
        relationship["deduplication_strategy"] = deduplication_strategy
    return {
        "entities": [
            {
                "id": "orders",
                "dataset_id": "orders-dataset",
                "name": "Orders",
                "entity_type": "fact",
                "grain": "one row per order",
            },
            {
                "id": "customers",
                "dataset_id": "customers-dataset",
                "name": "Customers",
                "entity_type": "dimension",
                "grain": "one row per customer",
            },
            {
                "id": "products",
                "dataset_id": "products-dataset",
                "name": "Products",
                "entity_type": "dimension",
                "grain": "one row per product",
            },
        ],
        "relationships": [
            relationship,
            {
                "id": "orders_products",
                "left_entity_id": "orders",
                "right_entity_id": "products",
                "left_field_id": "orders_product_id",
                "right_field_id": "products_product_id",
                "cardinality": "many_to_one",
                "enabled": True,
            },
        ],
        "metrics": [
            {
                "id": "revenue",
                "formula": {
                    "op": "sum",
                    "args": [
                        {
                            "op": "field",
                            "entity_id": "orders",
                            "field_id": "orders_amount",
                        }
                    ],
                },
            }
        ],
        "dimensions": [
            {
                "id": "customer_region",
                "entity_id": "customers",
                "field_id": "customers_region",
            }
        ],
    }
