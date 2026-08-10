from __future__ import annotations

from uuid import uuid4

import pandas as pd

from app.analysis.query_intent import infer_query_intent, infer_source_aggregations
from app.analysis.services import DatasetProfiler, _plan, _run_sql
from app.schemas.analysis import DatasetColumnProfile, DatasetProfileResponse


def _profile() -> DatasetProfileResponse:
    columns = (
        "customers_dataset_csv__customer_state",
        "sellers_dataset_csv__seller_state",
        "products_dataset_csv__product_category_name",
        "product_translation_csv__product_category_name_english",
        "orders_dataset_csv__order_status",
        "order_payments_dataset_csv__payment_type",
        "order_items_dataset_csv__price",
        "order_items_dataset_csv__freight_value",
        "order_payments_dataset_csv__payment_value",
        "products_dataset_csv__product_name_lenght",
    )
    return DatasetProfileResponse(
        dataset_id=uuid4(),
        row_count=3,
        column_count=len(columns),
        missing_value_count=0,
        missing_value_ratio=0,
        duplicate_row_count=0,
        numeric_columns=columns[6:],
        categorical_columns=columns[:6],
        columns=tuple(
            DatasetColumnProfile(
                name=name,
                dtype="float64" if name in columns[6:] else "object",
                missing_count=0,
                distinct_count=3,
                is_numeric=name in columns[6:],
            )
            for name in columns
        ),
        sample_records=(
            {
                "customers_dataset_csv__customer_state": "SP",
                "sellers_dataset_csv__seller_state": "SP",
                "orders_dataset_csv__order_status": "delivered",
                "order_payments_dataset_csv__payment_type": "credit_card",
            },
        ),
    )


def test_customer_state_question_prefers_customer_entity_dimension() -> None:
    intent = infer_query_intent(
        "按 customers.customer_state 分析销售额，同时参考商品数据",
        _profile(),
    )

    assert intent.dimensions[0] == "customers_dataset_csv__customer_state"
    assert intent.required_dimensions == ("customers_dataset_csv__customer_state",)


def test_seller_state_question_prefers_seller_entity_dimension() -> None:
    intent = infer_query_intent("Compare sales by seller state", _profile())

    assert intent.dimensions[0] == "sellers_dataset_csv__seller_state"
    assert intent.required_dimensions == ("sellers_dataset_csv__seller_state",)


def test_ambiguous_dimension_remains_a_candidate_until_planner_resolves_it() -> None:
    intent = infer_query_intent("按州分析销售额", _profile())

    assert intent.required_dimensions == ()
    assert set(intent.candidate_dimensions[:2]) == {
        "customers_dataset_csv__customer_state",
        "sellers_dataset_csv__seller_state",
    }


def test_payment_amount_question_requires_payment_metric() -> None:
    intent = infer_query_intent("按客户州统计总支付金额", _profile())

    assert intent.required_metric == "order_payments_dataset_csv__payment_value"
    assert intent.aggregations[0].operation == "sum"
    assert intent.aggregations[0].column == intent.required_metric
    assert "products_dataset_csv__product_name_lenght" in intent.candidate_metrics


def test_generic_total_does_not_require_an_arbitrary_numeric_metric() -> None:
    intent = infer_query_intent("按客户州统计总体规模", _profile())

    assert intent.required_metric is None


def test_payment_metric_and_order_filter_are_not_promoted_to_dimensions() -> None:
    intent = infer_query_intent(
        "仅使用 customers、orders、order_payments 三张表，过滤 "
        "order_status=delivered，按 customer_state 统计 payment_value 总额，"
        "并给出总体支付总额和 SP 州支付总额。不要使用 order_items、reviews、"
        "products、sellers 或 geolocation，也不要按 order_status 或 "
        "payment_type 分组。",
        _profile(),
    )

    assert intent.required_dimensions == ("customers_dataset_csv__customer_state",)
    assert intent.candidate_dimensions == ()
    assert intent.dimensions == ("customers_dataset_csv__customer_state",)
    assert intent.required_metric == "order_payments_dataset_csv__payment_value"
    assert "sellers_dataset_csv__seller_state" not in intent.dimensions
    assert tuple(item.model_dump() for item in intent.filters) == (
        {
            "column": "orders_dataset_csv__order_status",
            "operator": "=",
            "value": "delivered",
        },
    )


def test_explicit_equality_filter_does_not_depend_on_sampled_value() -> None:
    profile = _profile().model_copy(
        update={
            "sample_records": (
                {
                    "orders_dataset_csv__order_status": "processing",
                    "order_payments_dataset_csv__payment_type": "credit_card",
                },
            )
        }
    )

    intent = infer_query_intent(
        "按 customer_state 汇总 payment_value，order_status=delivered",
        profile,
    )

    assert tuple(item.model_dump() for item in intent.filters) == (
        {
            "column": "orders_dataset_csv__order_status",
            "operator": "=",
            "value": "delivered",
        },
    )


def test_short_sample_value_is_not_matched_inside_an_english_word() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {"segment": "A", "amount": 10},
            {"segment": "B", "amount": 20},
        ],
    )

    intent = infer_query_intent(
        "Analyze customer order amount by segment.",
        profile,
    )

    assert intent.filters == ()
    assert intent.required_dimensions == ("segment",)


def test_multi_source_totals_are_kept_as_separate_contract_metrics() -> None:
    aggregations = infer_source_aggregations(
        "分别计算 order_items 的商品收入和运费、order_payments 的支付总额",
        (
            ("order_items", ("order_id", "price", "freight_value")),
            ("order_payments", ("order_id", "payment_value")),
        ),
    )

    assert {(item.operation, item.column) for item in aggregations} == {
        ("sum", "order_items__price"),
        ("sum", "order_items__freight_value"),
        ("sum", "order_payments__payment_value"),
    }


def test_monthly_aov_uses_distinct_orders_and_a_derived_ratio() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {"order_id": "O1", "price": 10, "on_time": True, "order_purchase_timestamp": "2026-01-01"},
            {"order_id": "O1", "price": 20, "on_time": False, "order_purchase_timestamp": "2026-01-02"},
            {"order_id": "O2", "price": 30, "on_time": True, "order_purchase_timestamp": "2026-02-01"},
            {"order_id": "O3", "price": 40, "on_time": None, "order_purchase_timestamp": "2026-02-02"},
        ],
    )

    plan = _plan(
        "以 price 之和定义 GMV，以唯一 order_id 定义订单数。计算 GMV、订单数、客单价、准时率并分析月度趋势。",
        profile,
    )
    result = _run_sql(pd.DataFrame(profile.sample_records), plan)

    assert plan.time_column == "order_purchase_timestamp"
    assert plan.time_grain == "month"
    assert plan.derived_metrics == ("average_order_value",)
    assert {(item.operation, item.column) for item in plan.aggregations} >= {
        ("sum", "price"),
        ("count_distinct", "order_id"),
        ("avg", "on_time"),
    }
    overall = next(row for row in result.rows if row["period"] == "ALL")
    assert overall["total_price"] == 100
    assert overall["order_count"] == 3
    assert overall["average_order_value"] == 33.333333
    assert overall["on_time_rate"] == 0.666667
    assert "STRFTIME" in result.sql
    assert "GROUPING SETS" in result.sql
    assert 'AS "average_order_value"' in result.sql
    assert "LIMIT" not in result.sql


def test_gmv_prefers_price_over_a_precomputed_total_price_column() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {"order_id": "O1", "price": 10, "total_price": 12},
            {"order_id": "O2", "price": 20, "total_price": 24},
        ],
    )

    intent = infer_query_intent("分析订单数和 GMV", profile)

    assert {(item.operation, item.column) for item in intent.aggregations} == {
        ("sum", "price"),
        ("count_distinct", "order_id"),
    }
