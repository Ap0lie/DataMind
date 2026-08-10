from __future__ import annotations

from uuid import uuid4

from app.analysis.services import DatasetProfiler, PlannedAnalysis
from app.analysis.workflow import (
    _analysis_fast_path_eligible,
    _planner_metadata,
    _sanitize_model_plan,
)
from app.schemas.analysis import AnalysisAggregationResponse, AnalysisFilterResponse


def test_selected_business_dimension_precedes_identifier_candidates() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {
                "order_id": "O-1",
                "customer_id": "C-1",
                "segment": "enterprise",
                "status": "paid",
                "amount": 120,
            },
            {
                "order_id": "O-2",
                "customer_id": "C-2",
                "segment": "consumer",
                "status": "paid",
                "amount": 80,
            },
        ],
    )
    plan = PlannedAnalysis(
        route="hybrid",
        category_column="segment",
        metric_column="amount",
        time_column=None,
        steps=("group",),
        requested_dimensions=("segment",),
    )

    metadata = _planner_metadata(
        question="比较不同客户分群的销售额",
        profile=profile,
        planned_analysis=plan,
        source="rules",
        error=None,
    )

    assert metadata.candidate_dimensions[0] == "segment"
    assert metadata.candidate_metrics[0] == "amount"
    assert "order_id" not in metadata.candidate_metrics


def test_model_plan_cannot_add_time_grain_without_time_intent() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {"region": "East", "order_date": "2026-07-01", "sales": 100},
            {"region": "West", "order_date": "2026-07-02", "sales": 80},
        ],
    )
    model_plan = PlannedAnalysis(
        route="sql",
        category_column="region",
        metric_column="sales",
        time_column="order_date",
        steps=("group",),
        requested_dimensions=("region",),
    )

    sanitized = _sanitize_model_plan(model_plan, profile, allow_time=False)

    assert sanitized.category_column == "region"
    assert sanitized.time_column is None
    assert _analysis_fast_path_eligible(
        profile=profile,
        planned_analysis=sanitized,
        multi_dataset_context=None,
        multimodal_inputs=(),
    )


def test_filter_and_metric_fields_do_not_reenter_candidate_dimensions() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {
                "customer_state": "SP",
                "seller_state": "RJ",
                "order_status": "delivered",
                "payment_type": "credit_card",
                "payment_value": 120,
            },
            {
                "customer_state": "RJ",
                "seller_state": "SP",
                "order_status": "delivered",
                "payment_type": "voucher",
                "payment_value": 80,
            },
        ],
    )
    plan = PlannedAnalysis(
        route="sql",
        category_column="customer_state",
        metric_column="payment_value",
        time_column=None,
        steps=("group",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="payment_value", alias="total_payment"
            ),
        ),
        filters=(
            AnalysisFilterResponse(
                column="order_status", operator="=", value="delivered"
            ),
        ),
        requested_dimensions=("customer_state",),
    )

    metadata = _planner_metadata(
        question="按客户州统计已交付订单总支付金额",
        profile=profile,
        planned_analysis=plan,
        source="rules",
        error=None,
    )

    assert metadata.candidate_dimensions == ("customer_state",)
