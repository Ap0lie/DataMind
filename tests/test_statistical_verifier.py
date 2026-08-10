from __future__ import annotations

from uuid import uuid4

import pandas as pd

from app.analysis.analysis_contract import build_analysis_contract
from app.analysis.services import DatasetProfiler, PlannedAnalysis
from app.analysis.statistical_verifier import (
    qualify_observational_findings,
    reportable_findings,
    statistical_validation_issues,
    verify_statistical_analysis,
)
from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisContractResponse,
    AnalysisFilterResponse,
    DatasetReferenceResponse,
    InsightFindingResponse,
    MultiDatasetProfileResponse,
    PlannerMetadataResponse,
    SQLAnalysisResponse,
)


def test_analysis_contract_uses_plan_scope_and_server_budget() -> None:
    dataset_id = uuid4()
    profile = _profile(
        dataset_id,
        [
            {"region": "A", "amount": 10, "created_at": "2026-01-01"},
            {"region": "B", "amount": 20, "created_at": "2026-01-02"},
        ],
    )

    contract = build_analysis_contract(
        question="比较不同地区的销售额趋势",
        dataset_id=dataset_id,
        additional_dataset_ids=(),
        profile=profile,
        plan=PlannedAnalysis(
            route="hybrid",
            category_column="region",
            metric_column="amount",
            time_column="created_at",
            steps=("aggregate", "compare"),
        ),
        planner_metadata=PlannerMetadataResponse(confidence=0.9),
        multi_dataset_context=None,
    )

    assert contract.analysis_type == "trend"
    assert contract.metric == "amount"
    assert contract.grain == ("region", "created_at")
    assert contract.causal_claim_allowed is False
    assert contract.analysis_budget["max_tool_calls"] >= 1


def test_explicit_dimension_excludes_unrelated_planner_candidates() -> None:
    dataset_id = uuid4()
    profile = _profile(
        dataset_id,
        [
            {
                "customers_dataset_csv__customer_state": "SP",
                "products_dataset_csv__product_category_name": "books",
                "product_translation_csv__product_category_name_english": "books",
                "amount": 10,
            }
        ],
    )

    contract = build_analysis_contract(
        question="按 customers.customer_state 分析销售额，同时参考商品数据",
        dataset_id=dataset_id,
        additional_dataset_ids=(),
        profile=profile,
        plan=PlannedAnalysis(
            route="sql",
            category_column="products_dataset_csv__product_category_name",
            metric_column="amount",
            time_column=None,
            steps=("aggregate",),
            requested_dimensions=(
                "products_dataset_csv__product_category_name",
                "product_translation_csv__product_category_name_english",
            ),
        ),
        planner_metadata=PlannerMetadataResponse(confidence=0.9),
        multi_dataset_context=None,
    )

    assert contract.dimensions == ("customers_dataset_csv__customer_state",)


def test_statistical_verifier_supports_comparison_with_effect_and_interval() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "segment": ["A", "A", "A", "B", "B", "B"],
            "amount": [12, 14, 16, 8, 9, 10],
        }
    )
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    finding = InsightFindingResponse(
        title="A 组更高",
        content="A 组平均销售额 14，高于 B 组的 9。",
        data_source="tool_evidence",
        evidence="evidence_id:ev_1",
    )

    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount", dimensions=("segment",)),
        profile=profile,
        dataframe=frame,
        findings=(finding,),
        evidence=(
            {
                "evidence_id": "ev_1",
                "status": "succeeded",
                "result": {
                    "sql": (
                        "SELECT segment, AVG(amount) AS avg FROM dataset "
                        "GROUP BY segment"
                    ),
                    "rows": [
                        {"segment": "A", "avg": 14},
                        {"segment": "B", "avg": 9},
                    ],
                },
            },
        ),
    )

    verdict = result.finding_verdicts[0]
    assert result.status == "passed"
    assert result.numeric_evidence_coverage == 1
    assert verdict.sample_size == 6
    assert verdict.effect_size is not None
    assert verdict.confidence_interval is not None


def test_statistical_verifier_accepts_descriptive_ranking_from_sql_evidence() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"segment": ["A", "B"], "amount": [10, 20]})
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    sql_result = SQLAnalysisResponse(
        sql="SELECT segment, SUM(amount) AS amount FROM dataset GROUP BY segment",
        rows=(
            {"segment": "B", "amount": 20},
            {"segment": "A", "amount": 10},
        ),
        explanation="按分组汇总。",
    )

    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount", dimensions=("segment",)),
        profile=profile,
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="分组表现排名",
                content="按销售额排名前两位的分组为 B（20）、A（10）。",
                data_source="sql_result.rows",
                evidence="segment x amount",
            ),
        ),
        sql_result=sql_result,
    )

    assert result.status == "passed"
    assert result.numeric_evidence_coverage == 1
    assert next(check for check in result.checks if check.code == "comparison_support").status == "not_applicable"


def test_grouping_sets_cover_a_monthly_time_dimension() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2026-01-01", "2026-02-01"],
            "order_id": ["O1", "O2"],
            "price": [10.0, 20.0],
            "on_time": [True, False],
        }
    )
    contract = AnalysisContractResponse(
        objective="计算 GMV、订单数、客单价、准时率并分析月度趋势。",
        population="测试订单",
        dataset_ids=(dataset_id,),
        analysis_type="trend",
        metric="price",
        time_field="order_purchase_timestamp",
        aggregations=(
            AnalysisAggregationResponse(operation="sum", column="price", alias="total_price"),
            AnalysisAggregationResponse(
                operation="count_distinct", column="order_id", alias="order_count"
            ),
            AnalysisAggregationResponse(operation="avg", column="on_time", alias="on_time_rate"),
        ),
        grain=("order_purchase_timestamp",),
        method="按月聚合并计算整体指标。",
    )
    sql_result = SQLAnalysisResponse(
        sql=(
            "SELECT COALESCE(STRFTIME(TRY_CAST(order_purchase_timestamp AS TIMESTAMP), "
            "'%Y-%m'), 'ALL') AS period, SUM(price) AS total_price, "
            "COUNT(DISTINCT order_id) AS order_count, AVG(on_time) AS on_time_rate "
            "FROM dataset GROUP BY GROUPING SETS "
            "((STRFTIME(TRY_CAST(order_purchase_timestamp AS TIMESTAMP), '%Y-%m')), ())"
        ),
        rows=(
            {"period": "2026-01", "total_price": 10.0, "order_count": 1, "on_time_rate": 1.0},
            {"period": "2026-02", "total_price": 20.0, "order_count": 1, "on_time_rate": 0.0},
            {"period": "ALL", "total_price": 30.0, "order_count": 2, "on_time_rate": 0.5},
        ),
        explanation="月度与整体指标。",
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=sql_result,
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "passed"
    assert coverage.details["covered_by"] == "sql_statement_1"


def test_weighted_aov_is_supported_by_monthly_sql_rows() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2026-01-01", "2026-02-01"],
            "order_id": ["O1", "O2"],
            "price": [100.0, 500.0],
        }
    )
    contract = AnalysisContractResponse(
        objective="计算 GMV、订单数、客单价并分析月度趋势。",
        population="测试订单",
        dataset_ids=(dataset_id,),
        analysis_type="trend",
        metric="price",
        time_field="order_purchase_timestamp",
        aggregations=(
            AnalysisAggregationResponse(operation="sum", column="price", alias="gmv"),
            AnalysisAggregationResponse(
                operation="count_distinct", column="order_id", alias="order_count"
            ),
        ),
        grain=("order_purchase_timestamp",),
        method="按月聚合。",
    )
    sql_result = SQLAnalysisResponse(
        sql=(
            "SELECT STRFTIME(order_purchase_timestamp, '%Y-%m') AS month, "
            "SUM(price) AS gmv, COUNT(DISTINCT order_id) AS order_count, "
            "SUM(price) / COUNT(DISTINCT order_id) AS aov FROM dataset "
            "GROUP BY STRFTIME(order_purchase_timestamp, '%Y-%m')"
        ),
        rows=(
            {"month": "2026-01", "gmv": 100.0, "order_count": 10, "aov": 10.0},
            {"month": "2026-02", "gmv": 500.0, "order_count": 20, "aov": 25.0},
        ),
        explanation="月度指标。",
    )
    result = verify_statistical_analysis(
        contract=contract,
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="核心指标概览",
                content="GMV=600.00；订单数=30；客单价=20.00。",
                data_source="sql_result.rows",
                evidence="月度结果加权汇总。",
            ),
        ),
        sql_result=sql_result,
    )

    assert result.finding_verdicts[0].status == "passed"


def test_statistical_verifier_requires_evidence_for_numeric_claims() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [10, 20, 30]})
    profile = _profile(dataset_id, frame.to_dict(orient="records"))

    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=profile,
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="无证据数字",
                content="总销售额为 60。",
                data_source="tool_evidence",
            ),
        ),
    )

    assert result.status == "failed"
    assert result.requires_replan is True
    assert result.numeric_evidence_coverage == 0
    assert any(issue.finding_ref == "analysis" for issue in statistical_validation_issues(result))


def test_statistical_verifier_rejects_results_that_do_not_answer_contract() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "amount": [10, 20],
            "status": ["completed", "cancelled"],
            "segment": ["A", "B"],
        }
    )
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    contract = AnalysisContractResponse(
        objective="按客户细分统计已完成订单的总销售额、订单数和平均订单金额",
        population="测试记录",
        dataset_ids=(dataset_id,),
        analysis_type="descriptive",
        metric="amount",
        dimensions=("segment",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="amount", alias="total_amount"
            ),
            AnalysisAggregationResponse(
                operation="count_distinct", column="order_id", alias="order_count"
            ),
            AnalysisAggregationResponse(
                operation="avg", column="amount", alias="average_amount"
            ),
        ),
        filters=(
            AnalysisFilterResponse(column="status", value="completed"),
        ),
        grain=("segment",),
        method="确定性测试",
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                'SELECT "order_id" AS category, '
                'SUM(CAST("amount" AS DOUBLE)) AS total_amount '
                'FROM dataset GROUP BY "order_id"'
            ),
            rows=({"category": "O1", "total_amount": 10},),
            explanation="错误的回归样例。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details == {
        "dimensions": ["segment"],
        "unexpected_dimensions": ["order_id"],
        "filters": ["status=completed"],
        "aggregations": ["count_distinct(order_id)", "avg(amount)"],
    }
    assert result.requires_replan is True


def test_statistical_verifier_reports_filtered_population() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "amount": [10, 20],
            "status": ["completed", "cancelled"],
        }
    )
    filtered = frame.loc[frame["status"] == "completed"]
    profile = _profile(dataset_id, frame.to_dict(orient="records"))

    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=profile,
        dataframe=filtered,
        findings=(),
    )

    population = next(
        check for check in result.checks if check.code == "population_non_empty"
    )
    assert population.details == {"row_count": 1}
    assert population.message == "分析总体包含 1 行记录。"


def test_observational_causal_language_is_qualified_before_verification() -> None:
    dataset_id = uuid4()
    contract = _contract(dataset_id, metric="amount")
    findings = (
        InsightFindingResponse(
            title="相关性",
            content="折扣导致销售额增长。",
            data_source="python_result.statistics",
            evidence="python_result",
            confidence="high",
        ),
    )

    qualified = qualify_observational_findings(findings, contract)

    assert "不能证明因果" in qualified[0].content
    assert "导致" not in qualified[0].content
    assert qualified[0].confidence == "medium"


def test_join_expansion_without_native_grain_evidence_requires_replan() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {"order_id": ["O1", "O1", "O2"], "amount": [10, 10, 20]}
    )
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    reference = DatasetReferenceResponse(
        dataset_id=dataset_id,
        name="orders.csv",
        status="cleaned",
        row_count=3,
        column_count=2,
        columns=("order_id", "amount"),
    )
    context = MultiDatasetProfileResponse(
        primary_dataset=reference,
        join_summary={
            "mode": "joined",
            "row_expansion_ratio": 1.5,
            "skipped_join_count": 0,
        },
        joined_profile=profile,
    )

    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=profile,
        dataframe=frame,
        findings=(),
        multi_dataset_context=context,
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "failed"
    assert result.requires_replan is True


def test_contract_coverage_cannot_be_spliced_across_sql_statements() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP", "RJ"],
            "order_status": ["delivered", "delivered"],
            "payment_type": ["credit_card", "voucher"],
            "payment_value": [100.0, 50.0],
        }
    )
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    contract = AnalysisContractResponse(
        objective="按客户州统计已交付订单支付总额",
        population="已交付订单",
        dataset_ids=(dataset_id,),
        analysis_type="descriptive",
        metric="payment_value",
        dimensions=("customer_state",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="payment_value", alias="total_payment"
            ),
        ),
        filters=(AnalysisFilterResponse(column="order_status", value="delivered"),),
        grain=("customer_state",),
        method="确定性测试",
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment "
                "FROM dataset GROUP BY customer_state; "
                "SELECT payment_type, SUM(payment_value) AS total_payment "
                "FROM dataset WHERE order_status = 'delivered' GROUP BY payment_type"
            ),
            rows=(),
            explanation="两个不完整结果。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["filters"] == ["order_status=delivered"]


def test_explicit_request_cannot_pass_with_an_empty_analysis_contract() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    contract = AnalysisContractResponse(
        objective=(
            "过滤 order_status=delivered，按 customer_state 统计 "
            "payment_value 总额。"
        ),
        population="测试记录",
        dataset_ids=(dataset_id,),
        analysis_type="descriptive",
        metric="payment_value",
        dimensions=(),
        aggregations=(),
        filters=(),
        grain=("dataset",),
        method="错误的空契约",
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment "
                "FROM dataset WHERE order_status = 'delivered' "
                "GROUP BY customer_state"
            ),
            rows=({"customer_state": "SP", "total_payment": 100.0},),
            explanation="SQL 本身正确，但不能替空分析契约洗白。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["contract_incomplete"] == [
        "维度",
        "过滤条件",
        "聚合指标",
    ]


def test_successful_contract_coverage_records_the_single_covering_query() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment "
                "FROM dataset WHERE order_status = 'delivered' "
                "GROUP BY customer_state"
            ),
            rows=({"customer_state": "SP", "total_payment": 100.0},),
            explanation="完整口径查询。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "passed"
    assert coverage.details == {
        "required_dimensions": ["customer_state"],
        "required_filters": ["order_status=delivered"],
        "required_aggregations": ["sum(payment_value)"],
        "covered_by": "sql_statement_1",
    }


def test_contract_coverage_rejects_extra_group_dimensions() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_type": ["credit_card"],
            "payment_value": [100.0],
        }
    )
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    contract = AnalysisContractResponse(
        objective="按客户州统计已交付订单支付总额",
        population="已交付订单",
        dataset_ids=(dataset_id,),
        analysis_type="descriptive",
        metric="payment_value",
        dimensions=("customer_state",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="payment_value", alias="total_payment"
            ),
        ),
        filters=(AnalysisFilterResponse(column="order_status", value="delivered"),),
        grain=("customer_state",),
        method="确定性测试",
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, payment_type, "
                "SUM(payment_value) AS total_payment FROM dataset "
                "WHERE order_status = 'delivered' "
                "GROUP BY customer_state, payment_type"
            ),
            rows=(),
            explanation="错误的额外分组。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["unexpected_dimensions"] == ["payment_type"]


def test_join_grain_requires_native_evidence_matching_full_olist_contract() -> None:
    payments_id = uuid4()
    orders_id = uuid4()
    customers_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    profile = _profile(payments_id, frame.to_dict(orient="records"))
    context = MultiDatasetProfileResponse(
        primary_dataset=DatasetReferenceResponse(
            dataset_id=payments_id,
            name="order_payments.csv",
            status="cleaned",
            row_count=1,
            column_count=1,
            columns=("payment_value",),
        ),
        join_summary={"mode": "joined", "row_expansion_ratio": 1.05},
        joined_profile=profile,
    )
    contract = AnalysisContractResponse(
        objective="按客户州统计已交付订单支付总额",
        population="已交付订单",
        dataset_ids=(payments_id, orders_id, customers_id),
        analysis_type="descriptive",
        metric="payment_value",
        dimensions=("customer_state",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="payment_value", alias="total_payment"
            ),
        ),
        filters=(AnalysisFilterResponse(column="order_status", value="delivered"),),
        grain=("customer_state",),
        method="确定性测试",
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        evidence=(
            {
                "evidence_id": "ev_unfiltered",
                "status": "succeeded",
                "result": {
                    "native_grain": True,
                    "source_dataset_id": str(payments_id),
                    "metric": "payment_value",
                    "aggregation": "sum",
                    "grain": ["dataset"],
                    "filters": [],
                },
            },
        ),
        multi_dataset_context=context,
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "failed"
    assert join_check.details["row_expansion_ratio"] == 1.05
    assert join_check.details["native_grain_candidate_count"] == 1
    assert join_check.details["native_grain_match_count"] == 0


def test_numeric_finding_rejects_claim_not_present_in_cited_evidence() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    finding = InsightFindingResponse(
        title="错误总额",
        content="总额为 999。",
        data_source="tool_evidence",
        evidence="evidence_id:ev_100",
    )

    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(finding,),
        evidence=(
            {
                "evidence_id": "ev_100",
                "status": "succeeded",
                "result": {"rows": [{"total_amount": 100.0}]},
            },
        ),
    )

    verdict = result.finding_verdicts[0]
    assert verdict.status == "failed"
    assert result.numeric_evidence_coverage == 0
    assert any("999" in note and "不一致" in note for note in verdict.notes)
    assert reportable_findings((finding,), result) == ()


def test_correct_evidence_cannot_launder_wrong_cited_query() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [248_007.22],
        }
    )
    contract = _payment_contract(dataset_id)
    correct_sql = (
        "SELECT customer_state, SUM(payment_value) AS total_payment "
        "FROM dataset WHERE order_status = 'delivered' GROUP BY customer_state"
    )
    result = verify_statistical_analysis(
        contract=contract,
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="SP 已交付支付总额",
                content="SP 已交付支付总额为 248,007.22。",
                data_source="tool_evidence",
                evidence="evidence_id:ev_correct; evidence_id:ev_unfiltered",
            ),
        ),
        evidence=(
            {
                "evidence_id": "ev_correct",
                "status": "succeeded",
                "result": {
                    "sql": correct_sql,
                    "rows": [
                        {"customer_state": "SP", "total_payment": 248_007.22}
                    ],
                },
            },
            {
                "evidence_id": "ev_unfiltered",
                "status": "succeeded",
                "result": {
                    "sql": "SELECT SUM(payment_value) AS total_payment FROM dataset",
                    "rows": [{"total_payment": 180_096.80}],
                },
            },
        ),
        sql_result=SQLAnalysisResponse(
            sql=correct_sql,
            rows=({"customer_state": "SP", "total_payment": 248_007.22},),
            explanation="完整口径查询。",
        ),
    )

    verdict = result.finding_verdicts[0]
    assert verdict.status == "failed"
    assert any("ev_unfiltered" in note and "未独立满足" in note for note in verdict.notes)


def test_join_native_grain_evidence_cannot_validate_another_cited_result() -> None:
    payments_id = uuid4()
    orders_id = uuid4()
    customers_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [248_007.22],
        }
    )
    profile = _profile(payments_id, frame.to_dict(orient="records"))
    contract = _payment_contract(
        payments_id,
        dataset_ids=(payments_id, orders_id, customers_id),
    )
    sql = (
        "SELECT customer_state, SUM(payment_value) AS total_payment "
        "FROM dataset WHERE order_status = 'delivered' GROUP BY customer_state"
    )
    context = MultiDatasetProfileResponse(
        primary_dataset=DatasetReferenceResponse(
            dataset_id=payments_id,
            name="order_payments.csv",
            status="cleaned",
            row_count=1,
            column_count=1,
            columns=("payment_value",),
        ),
        join_summary={"mode": "joined", "row_expansion_ratio": 1.05},
        joined_profile=profile,
    )

    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="Join 结果",
                content="SP 已交付支付总额为 248,007.22。",
                data_source="tool_evidence",
                evidence="evidence_id:ev_joined",
            ),
        ),
        evidence=(
            {
                "evidence_id": "ev_joined",
                "status": "succeeded",
                "result": {
                    "sql": sql,
                    "rows": [
                        {"customer_state": "SP", "total_payment": 248_007.22}
                    ],
                },
            },
            {
                "evidence_id": "ev_native",
                "status": "succeeded",
                "result": {
                    "native_grain": True,
                    "source_dataset_id": str(payments_id),
                    "metric": "payment_value",
                    "aggregation": "sum",
                    "grain": ["customer_state"],
                    "filters": [
                        {
                            "column": "order_status",
                            "operator": "=",
                            "value": "delivered",
                        }
                    ],
                    "rows": [
                        {"customer_state": "SP", "sum_payment_value": 200_000.0}
                    ],
                },
            },
        ),
        sql_result=SQLAnalysisResponse(
            sql=sql,
            rows=({"customer_state": "SP", "total_payment": 248_007.22},),
            explanation="Join 后查询。",
        ),
        multi_dataset_context=context,
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "passed"
    verdict = result.finding_verdicts[0]
    assert verdict.status == "failed"
    assert any("未引用同口径原生粒度证据" in note for note in verdict.notes)


def test_contract_coverage_rejects_undeclared_filters() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "seller_state": ["RJ"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment "
                "FROM dataset WHERE order_status = 'delivered' "
                "AND customer_state = 'MS' AND seller_state = 'SP' "
                "GROUP BY customer_state"
            ),
            rows=(),
            explanation="包含未声明过滤。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["unexpected_filters"] == [
        "customer_state=MS",
        "seller_state=SP",
    ]


def test_contract_coverage_rejects_required_filter_or_true() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment "
                "FROM dataset WHERE order_status = 'delivered' OR 1 = 1 "
                "GROUP BY customer_state"
            ),
            rows=(),
            explanation="OR true 绕过过滤。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["filters"] == ["order_status=delivered"]
    assert coverage.details["unexpected_filters"]


def test_relationship_metadata_cannot_support_an_analysis_metric_claim() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="伪造匹配率总额",
                content="销售总额为 95%。",
                data_source="tool_evidence",
                evidence="evidence_id:relationship_ev_1",
            ),
        ),
        evidence=(
            {
                "evidence_id": "relationship_ev_1",
                "status": "succeeded",
                "result": {"relationships": [{"match_rate": 0.95}]},
            },
        ),
    )

    assert result.finding_verdicts[0].status == "failed"
    assert result.numeric_evidence_coverage == 0


def test_source_row_metadata_cannot_support_a_metric_claim() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    contract = AnalysisContractResponse(
        objective="计算总额",
        population="测试记录",
        dataset_ids=(dataset_id,),
        analysis_type="descriptive",
        metric="amount",
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="amount", alias="total_amount"
            ),
        ),
        grain=("dataset",),
        method="确定性测试",
    )
    result = verify_statistical_analysis(
        contract=contract,
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="误用源表行数",
                content="总额为 3。",
                data_source="tool_evidence",
                evidence="evidence_id:source_ev_1",
            ),
        ),
        evidence=(
            {
                "evidence_id": "source_ev_1",
                "status": "succeeded",
                "result": {
                    "native_grain": True,
                    "source_dataset_id": str(dataset_id),
                    "source_row_count": 3,
                    "filtered_row_count": 3,
                    "metric": "amount",
                    "aggregation": "sum",
                    "grain": ["dataset"],
                    "filters": [],
                    "rows": [{"sum_amount": 100.0}],
                },
            },
        ),
    )

    assert result.finding_verdicts[0].status == "failed"
    assert result.numeric_evidence_coverage == 0


def test_chinese_text_without_space_still_extracts_the_full_number() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
                InsightFindingResponse(
                    title="无空格伪造数字",
                    content="总额为5,790。",
                    data_source="tool_evidence",
                    evidence="evidence_id:ev_790",
                ),
            ),
        evidence=(
            {
                "evidence_id": "ev_790",
                "status": "succeeded",
                "result": {
                    "sql": "SELECT SUM(amount) AS total_amount FROM dataset",
                    "rows": [{"total_amount": 790.0}],
                },
            },
        ),
    )

    assert result.finding_verdicts[0].status == "failed"
    assert any("5,790" in note for note in result.finding_verdicts[0].notes)


def test_unused_cte_aggregation_cannot_cover_outer_result() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "WITH bogus AS (SELECT SUM(payment_value) AS x FROM dataset) "
                "SELECT customer_state, COUNT(*) AS c FROM dataset "
                "WHERE order_status = 'delivered' GROUP BY customer_state"
            ),
            rows=(),
            explanation="未使用 CTE 不得提供聚合。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["aggregations"] == ["sum(payment_value)"]


def test_hidden_cte_filter_cannot_narrow_the_contract_population() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP", "MS"],
            "order_status": ["delivered", "delivered"],
            "payment_value": [100.0, 50.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "WITH d AS (SELECT * FROM dataset WHERE customer_state = 'MS') "
                "SELECT customer_state, SUM(payment_value) AS total_payment FROM d "
                "WHERE order_status = 'delivered' GROUP BY customer_state"
            ),
            rows=(),
            explanation="内层隐藏过滤。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert any("INNER WHERE" in item for item in coverage.details["unexpected_filters"])


def test_having_filter_is_not_an_authorized_population_predicate() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP", "MS"],
            "order_status": ["delivered", "delivered"],
            "payment_value": [100.0, 50.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment "
                "FROM dataset WHERE order_status = 'delivered' "
                "GROUP BY customer_state HAVING customer_state = 'MS'"
            ),
            rows=(),
            explanation="HAVING 隐藏过滤。",
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert any("HAVING" in item for item in coverage.details["unexpected_filters"])


def test_group_label_must_match_the_same_evidence_row_as_the_claim_value() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP", "MS"],
            "order_status": ["delivered", "delivered"],
            "payment_value": [100.0, 50.0],
        }
    )
    sql = (
        "SELECT customer_state, SUM(payment_value) AS total_payment FROM dataset "
        "WHERE order_status = 'delivered' GROUP BY customer_state"
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="错配州标签",
                content="SP 支付总额为50。",
                data_source="tool_evidence",
                evidence="evidence_id:ev_states",
            ),
        ),
        evidence=(
            {
                "evidence_id": "ev_states",
                "status": "succeeded",
                "result": {
                    "sql": sql,
                    "rows": [
                        {"customer_state": "SP", "total_payment": 100.0},
                        {"customer_state": "MS", "total_payment": 50.0},
                    ],
                },
            },
        ),
        sql_result=SQLAnalysisResponse(sql=sql, rows=(), explanation="按州汇总。"),
    )

    assert result.finding_verdicts[0].status == "failed"
    assert result.numeric_evidence_coverage == 0


def test_combined_sql_cannot_borrow_a_value_from_another_query_index() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(dataset_id),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="跨查询借值",
                content="SP 已交付支付总额为999。",
                data_source="sql_result.rows",
                evidence="combined sql",
            ),
        ),
        sql_result=SQLAnalysisResponse(
            sql=(
                "SELECT customer_state, SUM(payment_value) AS total_payment FROM dataset "
                "WHERE order_status = 'delivered' GROUP BY customer_state; "
                "SELECT SUM(payment_value) AS unrelated_total FROM dataset"
            ),
            rows=(
                {
                    "evidence_id": "ev_1",
                    "query_index": 1,
                    "customer_state": "SP",
                    "total_payment": 100.0,
                },
                {
                    "evidence_id": "ev_2",
                    "query_index": 2,
                    "unrelated_total": 999.0,
                },
            ),
            explanation="合并 SQL。",
        ),
    )

    assert result.finding_verdicts[0].status == "failed"


def test_artifact_sibling_claim_values_remain_bound_to_the_sql_contract() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"segment": ["A"], "amount": [100.0]})
    sql = "SELECT segment, SUM(amount) AS total_amount FROM dataset GROUP BY segment"
    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount", dimensions=("segment",)),
        profile=_profile(dataset_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="工件化总额",
                content="总额为100。",
                data_source="tool_evidence",
                evidence="evidence_id:ev_artifact",
            ),
        ),
        evidence=(
            {
                "evidence_id": "ev_artifact",
                "status": "succeeded",
                "result": None,
                "claim_result": {
                    "sql_result": {"sql": sql, "rows": [], "explanation": ""},
                    "claim_values": [100.0],
                },
            },
        ),
        sql_result=SQLAnalysisResponse(sql=sql, rows=(), explanation="按分组汇总。"),
    )

    assert result.finding_verdicts[0].status == "passed"


def test_payments_root_without_row_expansion_does_not_require_native_guard() -> None:
    payments_id, orders_id, customers_id = uuid4(), uuid4(), uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    profile = _profile(payments_id, frame.to_dict(orient="records"))
    sql = (
        "SELECT customer_state, SUM(payment_value) AS total_payment FROM dataset "
        "WHERE order_status = 'delivered' GROUP BY customer_state"
    )
    result = verify_statistical_analysis(
        contract=_payment_contract(
            payments_id,
            dataset_ids=(payments_id, orders_id, customers_id),
        ),
        profile=profile,
        dataframe=frame,
        findings=(),
        sql_result=SQLAnalysisResponse(
            sql=sql,
            rows=({"customer_state": "SP", "total_payment": 100.0},),
            explanation="payments root + many-to-one dimensions",
        ),
        multi_dataset_context=MultiDatasetProfileResponse(
            primary_dataset=DatasetReferenceResponse(
                dataset_id=payments_id,
                name="order_payments.csv",
                status="cleaned",
                row_count=1,
                column_count=1,
                columns=("payment_value",),
            ),
            join_summary={"mode": "joined", "row_expansion_ratio": 1.0},
            joined_profile=profile,
        ),
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "passed"
    assert join_check.details["native_grain_evidence"] is False


def test_lineage_finding_cannot_publish_an_unfiltered_native_amount() -> None:
    payments_id, orders_id, customers_id = uuid4(), uuid4(), uuid4()
    frame = pd.DataFrame(
        {
            "customer_state": ["SP"],
            "order_status": ["delivered"],
            "payment_value": [100.0],
        }
    )
    profile = _profile(payments_id, frame.to_dict(orient="records"))
    contract = _payment_contract(
        payments_id,
        dataset_ids=(payments_id, orders_id, customers_id),
    )
    correct_native = {
        "native_grain": True,
        "source_dataset_id": str(payments_id),
        "source_dataset": "order_payments",
        "metric": "payment_value",
        "aggregation": "sum",
        "grain": ["customer_state"],
        "filters": [
            {"column": "order_status", "operator": "=", "value": "delivered"}
        ],
        "rows": [{"customer_state": "SP", "sum_payment_value": 100.0}],
    }
    wrong_native = {
        **correct_native,
        "grain": ["dataset"],
        "filters": [],
        "rows": [{"sum_payment_value": 180_096.80}],
    }
    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(
            InsightFindingResponse(
                title="错误原生总额",
                content=(
                    "原生粒度结果：order_payments.payment_value 的 "
                    "SUM=180,096.80。"
                ),
                data_source="tool_evidence.source_relationships_and_native_aggregates",
                evidence="evidence_id:ev_wrong_native",
            ),
        ),
        evidence=(
            {
                "evidence_id": "ev_correct_native",
                "status": "succeeded",
                "result": correct_native,
            },
            {
                "evidence_id": "ev_wrong_native",
                "status": "succeeded",
                "result": wrong_native,
            },
        ),
        multi_dataset_context=MultiDatasetProfileResponse(
            primary_dataset=DatasetReferenceResponse(
                dataset_id=payments_id,
                name="order_payments.csv",
                status="cleaned",
                row_count=1,
                column_count=1,
                columns=("payment_value",),
            ),
            join_summary={"mode": "joined", "row_expansion_ratio": 1.05},
            joined_profile=profile,
        ),
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "passed"
    assert result.finding_verdicts[0].status == "failed"


def test_artifact_native_contract_result_still_satisfies_join_guard() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    result = verify_statistical_analysis(
        contract=_contract(dataset_id, metric="amount"),
        profile=profile,
        dataframe=frame,
        findings=(),
        evidence=(
            {
                "evidence_id": "source_ev_artifact",
                "status": "succeeded",
                "result": None,
                "claim_result": {
                    "native_grain": True,
                    "source_dataset_id": str(dataset_id),
                    "source_dataset": "orders",
                    "metric": "amount",
                    "aggregation": "sum",
                    "grain": ["dataset"],
                    "filters": [],
                    "rows": [],
                    "claim_values": [100.0],
                },
            },
        ),
        multi_dataset_context=MultiDatasetProfileResponse(
            primary_dataset=DatasetReferenceResponse(
                dataset_id=dataset_id,
                name="orders.csv",
                status="cleaned",
                row_count=1,
                column_count=1,
                columns=("amount",),
            ),
            join_summary={"mode": "joined", "row_expansion_ratio": 1.05},
            joined_profile=profile,
        ),
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "passed"
    assert join_check.details["native_grain_match_count"] == 1


def test_one_native_operation_cannot_cover_a_multi_operation_contract() -> None:
    dataset_id = uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    profile = _profile(dataset_id, frame.to_dict(orient="records"))
    contract = AnalysisContractResponse(
        objective="总额与均值",
        population="测试记录",
        dataset_ids=(dataset_id,),
        analysis_type="descriptive",
        metric="amount",
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="amount", alias="total_amount"
            ),
            AnalysisAggregationResponse(
                operation="avg", column="amount", alias="average_amount"
            ),
        ),
        grain=("dataset",),
        method="确定性测试",
    )
    result = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        evidence=(
            {
                "evidence_id": "source_ev_sum",
                "status": "succeeded",
                "result": {
                    "native_grain": True,
                    "source_dataset_id": str(dataset_id),
                    "metric": "amount",
                    "aggregation": "sum",
                    "grain": ["dataset"],
                    "filters": [],
                    "rows": [{"sum_amount": 100.0}],
                },
            },
        ),
        multi_dataset_context=MultiDatasetProfileResponse(
            primary_dataset=DatasetReferenceResponse(
                dataset_id=dataset_id,
                name="orders.csv",
                status="cleaned",
                row_count=1,
                column_count=1,
                columns=("amount",),
            ),
            join_summary={"mode": "joined", "row_expansion_ratio": 1.05},
            joined_profile=profile,
        ),
    )

    join_check = next(check for check in result.checks if check.code == "join_grain")
    assert join_check.status == "failed"


def test_named_native_source_totals_collectively_cover_a_multi_metric_contract() -> None:
    items_id, payments_id, orders_id = uuid4(), uuid4(), uuid4()
    frame = pd.DataFrame({"order_id": ["O1"]})
    profile = _profile(items_id, frame.to_dict(orient="records"))
    contract = AnalysisContractResponse(
        objective="分别计算商品收入、运费和支付总额",
        population="测试记录",
        dataset_ids=(items_id, payments_id, orders_id),
        analysis_type="descriptive",
        aggregations=tuple(
            AnalysisAggregationResponse(
                operation="sum", column=column, alias=f"total_{column}"
            )
            for column in (
                "order_items__price",
                "order_items__freight_value",
                "order_payments__payment_value",
            )
        ),
        grain=("dataset",),
        method="来源表原生粒度汇总",
    )
    evidence = tuple(
        {
            "evidence_id": f"source_{metric}",
            "tool_name": "aggregate_source_dataset",
            "status": "succeeded",
            "result": {
                "native_grain": True,
                "source_dataset_id": str(source_id),
                "source_dataset": source,
                "metric": metric,
                "aggregation": "sum",
                "grain": ["dataset"],
                "filters": [],
                "rows": [{f"sum_{metric}": value}],
            },
        }
        for source_id, source, metric, value in (
            (items_id, "order_items", "price", 100.0),
            (items_id, "order_items", "freight_value", 10.0),
            (payments_id, "order_payments", "payment_value", 110.0),
        )
    )
    context = MultiDatasetProfileResponse(
        primary_dataset=DatasetReferenceResponse(
            dataset_id=items_id,
            name="order_items",
            status="cleaned",
            row_count=1,
            column_count=3,
            columns=("order_id", "price", "freight_value"),
        ),
        join_summary={"mode": "joined", "row_expansion_ratio": 1.0, "skipped_join_count": 1},
        joined_profile=profile,
    )

    complete = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        evidence=evidence,
        multi_dataset_context=context,
    )
    incomplete = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=(),
        evidence=evidence[:-1],
        multi_dataset_context=context,
    )

    complete_coverage = next(
        check for check in complete.checks if check.code == "request_coverage"
    )
    incomplete_coverage = next(
        check for check in incomplete.checks if check.code == "request_coverage"
    )
    assert complete_coverage.status == "passed"
    assert complete_coverage.details["covered_by"] == "native_grain_evidence"
    assert next(check for check in complete.checks if check.code == "join_grain").status == "warning"
    assert incomplete_coverage.status == "failed"
    assert incomplete_coverage.details["aggregations"] == [
        "sum(order_payments__payment_value)"
    ]


def test_native_metric_cannot_impersonate_the_same_column_from_another_source() -> None:
    left_id, right_id = uuid4(), uuid4()
    frame = pd.DataFrame({"amount": [100.0]})
    contract = AnalysisContractResponse(
        objective="分别计算两张来源表的总额",
        population="测试记录",
        dataset_ids=(left_id, right_id),
        analysis_type="descriptive",
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="left_source__amount", alias="left_total"
            ),
            AnalysisAggregationResponse(
                operation="sum", column="right_source__amount", alias="right_total"
            ),
        ),
        grain=("dataset",),
        method="来源表原生粒度汇总",
    )
    result = verify_statistical_analysis(
        contract=contract,
        profile=_profile(left_id, frame.to_dict(orient="records")),
        dataframe=frame,
        findings=(),
        evidence=(
            {
                "evidence_id": "left_amount",
                "tool_name": "aggregate_source_dataset",
                "status": "succeeded",
                "result": {
                    "native_grain": True,
                    "source_dataset_id": str(left_id),
                    "source_dataset": "left_source",
                    "metric": "amount",
                    "aggregation": "sum",
                    "grain": ["dataset"],
                    "filters": [],
                    "rows": [{"sum_amount": 100.0}],
                },
            },
        ),
    )

    coverage = next(check for check in result.checks if check.code == "request_coverage")
    assert coverage.status == "failed"
    assert coverage.details["aggregations"] == ["sum(right_source__amount)"]


def _profile(dataset_id, records):
    return DatasetProfiler().profile(dataset_id=dataset_id, records=records)


def _contract(
    dataset_id,
    *,
    metric: str | None,
    dimensions: tuple[str, ...] = (),
) -> AnalysisContractResponse:
    return AnalysisContractResponse(
        objective="验证分析",
        population="测试记录",
        dataset_ids=(dataset_id,),
        analysis_type="comparison" if dimensions else "descriptive",
        metric=metric,
        dimensions=dimensions,
        grain=dimensions or ("dataset",),
        method="确定性测试",
        causal_claim_allowed=False,
    )


def _payment_contract(
    dataset_id,
    *,
    dataset_ids=None,
) -> AnalysisContractResponse:
    return AnalysisContractResponse(
        objective="按客户州统计已交付订单支付总额",
        population="已交付订单",
        dataset_ids=tuple(dataset_ids or (dataset_id,)),
        analysis_type="descriptive",
        metric="payment_value",
        dimensions=("customer_state",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum", column="payment_value", alias="total_payment"
            ),
        ),
        filters=(AnalysisFilterResponse(column="order_status", value="delivered"),),
        grain=("customer_state",),
        method="确定性测试",
    )
