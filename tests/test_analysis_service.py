from __future__ import annotations

from uuid import uuid4

import pandas as pd
import pytest

from app.analysis.services import AnalysisService, DatasetProfiler, _plan
from app.storage.dataset_store import DatasetStoreRepository


def test_dataset_profiler_returns_v1_profile(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 150, "profit": 30},
            {"region": "South", "sales": 150, "profit": 30},
        ],
    )

    profile = DatasetProfiler().profile(
        dataset_id=dataset.id,
        records=repository.read_raw_records(dataset.id),
    )

    assert profile.row_count == 3
    assert profile.column_count == 3
    assert profile.duplicate_row_count == 1
    assert profile.numeric_columns == ("sales", "profit")
    assert profile.categorical_columns == ("region",)


def test_planner_fallback_does_not_restore_a_negated_grouping_dimension() -> None:
    profile = DatasetProfiler().profile(
        dataset_id=uuid4(),
        records=[
            {
                "order_id": "O1",
                "payment_type": "credit_card",
                "payment_value": 10.0,
            }
        ],
    )

    plan = _plan(
        "按 customer_state 统计 payment_value 总额，不要按 payment_type 分组。",
        profile,
    )

    assert plan.metric_column == "payment_value"
    assert plan.category_column is None
    assert plan.requested_dimensions == ()


def test_dataset_profiler_excludes_identifier_like_numeric_columns(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="customers.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"customer_id": "1001", "customer_state": "CA", "sales": "20.5"},
            {"customer_id": "1002", "customer_state": "NY", "sales": "30.0"},
        ],
    )

    profile = DatasetProfiler().profile(
        dataset_id=dataset.id,
        records=repository.read_raw_records(dataset.id),
    )

    assert "customer_id" not in profile.numeric_columns
    assert "sales" in profile.numeric_columns


def test_dataset_profiler_counts_blank_csv_cells_as_missing(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    records = [
        {"order_id": "O1", "review": "", "amount": "10"},
        {"order_id": "O2", "review": "   ", "amount": ""},
        {"order_id": "O3", "review": "ok", "amount": "30"},
    ]

    profile = DatasetProfiler().profile(dataset_id=dataset.id, records=records)

    assert profile.missing_value_count == 3
    assert next(column for column in profile.columns if column.name == "review").missing_count == 2
    assert next(column for column in profile.columns if column.name == "amount").missing_count == 1
    assert "amount" in profile.numeric_columns


def test_analysis_service_generates_duckdb_sql_python_charts_and_report(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 250, "profit": 30},
            {"region": "North", "sales": 80, "profit": 10},
        ],
    )

    result = AnalysisService(repository).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    assert result.plan.route == "sql"
    assert result.sql_result is not None
    assert 'SUM(CAST("sales" AS DOUBLE))' in result.sql_result.sql
    assert result.sql_result.rows[0]["category"] == "South"
    assert result.python_result is not None
    assert result.python_result.charts
    chart_types = {chart.chart_type for chart in result.python_result.charts}
    assert {"bar", "pie"}.issubset(chart_types)
    assert not {"histogram", "box_plot", "correlation_heatmap"} & chart_types
    assert "# DataMind 分析报告" in result.report_markdown
    assert result.html_report is not None
    assert "<!doctype html>" in result.html_report
    assert "Executive Summary" not in result.html_report
    assert "Key Findings" in result.html_report
    reports = repository.list_reports(dataset.id)
    assert "html_report" in reports[0]["metadata"]
    assert "<html lang=\"zh-CN\">" in reports[0]["metadata"]["html_report"]


def test_analysis_service_counts_categories_when_metric_is_identifier(tmp_path) -> None:
    from app.analysis.services import PlannedAnalysis, _run_sql

    records = [
        {"customer_id": "06b8999e2fba1a1fbc88172c00ba8bc7", "customer_state": "CA"},
        {"customer_id": "a11b8999e2fba1a1fbc88172c00ba8bc7", "customer_state": "CA"},
        {"customer_id": "b22b8999e2fba1a1fbc88172c00ba8bc7", "customer_state": "NY"},
    ]
    plan = PlannedAnalysis(
        route="sql",
        category_column="customer_state",
        metric_column="customer_id",
        time_column=None,
        steps=(),
    )

    result = _run_sql(pd.DataFrame.from_records(records), plan)

    assert 'CAST("customer_id" AS DOUBLE)' not in result.sql
    assert "COUNT(*) AS row_count" in result.sql
    assert result.rows[0]["category"] == "CA"
    assert result.rows[0]["row_count"] == 2


def test_rule_planner_honors_dimension_filter_and_compound_metrics() -> None:
    from app.analysis.services import (
        DatasetProfiler,
        _build_final_insights,
        _format_report_charts,
        _plan,
        _run_python,
        _run_sql,
    )

    records = [
        {
            "order_id": "O-1001",
            "customer_id": "C-001",
            "amount": 128.5,
            "status": "completed",
            "segment": "Enterprise",
        },
        {
            "order_id": "O-1002",
            "customer_id": "C-002",
            "amount": 86.0,
            "status": "completed",
            "segment": "SMB",
        },
        {
            "order_id": "O-1003",
            "customer_id": "C-001",
            "amount": 214.9,
            "status": "completed",
            "segment": "Enterprise",
        },
        {
            "order_id": "O-1004",
            "customer_id": "C-003",
            "amount": 45.0,
            "status": "cancelled",
            "segment": "Consumer",
        },
        {
            "order_id": "O-1005",
            "customer_id": "C-004",
            "amount": 172.3,
            "status": "completed",
            "segment": "SMB",
        },
        {
            "order_id": "O-1006",
            "customer_id": "C-002",
            "amount": 99.9,
            "status": "completed",
            "segment": "SMB",
        },
    ]
    profile = DatasetProfiler().profile(dataset_id=uuid4(), records=records)

    plan = _plan(
        "按客户细分统计已完成订单的总销售额、订单数和平均订单金额",
        profile,
    )
    result = _run_sql(pd.DataFrame.from_records(records), plan)

    assert plan.category_column == "segment"
    assert plan.metric_column == "amount"
    assert [(item.operation, item.column) for item in plan.aggregations] == [
        ("sum", "amount"),
        ("count_distinct", "order_id"),
        ("avg", "amount"),
    ]
    assert [(item.column, item.value) for item in plan.filters] == [
        ("status", "completed")
    ]
    assert 'WHERE "status" = \'completed\'' in result.sql
    assert 'GROUP BY "segment"' in result.sql
    assert "O-1004" not in str(result.rows)
    assert result.rows[0]["category"] == "SMB"
    assert result.rows[0]["total_amount"] == pytest.approx(358.2)
    assert result.rows[0]["order_count"] == 3
    assert result.rows[0]["average_amount"] == pytest.approx(119.4)

    python_result = _run_python(
        pd.DataFrame.from_records(records),
        plan,
        result,
        question="按客户细分统计已完成订单",
    )
    findings = _build_final_insights(
        question="按客户细分统计已完成订单的总销售额、订单数和平均订单金额",
        python_result=python_result,
        sql_result=result,
    )
    charts = _format_report_charts(
        question="按客户细分统计已完成订单的总销售额、订单数和平均订单金额",
        charts=python_result.charts,
        sql_result=result,
        findings=findings,
    )

    assert python_result.statistics["rows"] == 5
    assert python_result.statistics["numeric_summary"]["amount"]["mean"] == pytest.approx(
        140.32
    )
    sql_bar = next(chart for chart in python_result.charts if chart.title == "SQL 结果图")
    sql_pie = next(chart for chart in python_result.charts if chart.title == "占比图")
    assert sql_bar.spec == {"x": "category", "y": "total_amount"}
    assert sql_pie.spec == {
        "names": "category",
        "values": "total_amount",
        "denominator_scope": "complete_query_result",
        "displayed_category_count": 2,
        "excluded_category_count": 0,
    }
    assert "全部返回类别" in sql_pie.explanation
    assert "平均金额=140.32" in findings[0].content
    assert "平均金额=291.10" not in findings[0].content
    assert charts[0].spec == {"x": "category", "y": "total_amount"}
    assert not any(chart.chart_type == "box_plot" for chart in charts)


def test_pie_chart_discloses_top_n_denominator() -> None:
    from app.analysis.services import PlannedAnalysis, _run_python
    from app.schemas.analysis import (
        AnalysisAggregationResponse,
        SQLAnalysisResponse,
    )

    plan = PlannedAnalysis(
        route="sql",
        category_column="customer_state",
        metric_column="payment_value",
        time_column=None,
        steps=(),
        requested_dimensions=("customer_state",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum",
                column="payment_value",
                alias="total_payment_value",
            ),
        ),
    )
    sql_result = SQLAnalysisResponse(
        sql=(
            "SELECT customer_state, SUM(payment_value) AS total_payment_value "
            "FROM dataset GROUP BY customer_state"
        ),
        rows=tuple(
            {
                "customer_state": f"S{index}",
                "total_payment_value": float(100 - index),
            }
            for index in range(10)
        ),
        explanation="按州汇总。",
    )
    result = _run_python(
        pd.DataFrame.from_records(
            [
                {"customer_state": "SP", "payment_value": 100.0},
                {"customer_state": "RJ", "payment_value": 50.0},
            ]
        ),
        plan,
        sql_result,
        question="按 customer_state 统计 payment_value 总额",
    )

    pie = next(chart for chart in result.charts if chart.title == "占比图")
    assert len(pie.data) == 8
    assert pie.spec["denominator_scope"] == "displayed_top_n"
    assert pie.spec["excluded_category_count"] == 2
    assert "前 8 类" in pie.explanation
    assert "不能将该百分比解释为全量占比" in pie.explanation


def test_python_baseline_handles_metric_reused_as_category_and_time() -> None:
    from app.analysis.services import PlannedAnalysis, _run_python

    frame = pd.DataFrame.from_records(
        [
            {"seller_id": "S1", "seller_zip_code_prefix": 13023, "seller_state": "SP"},
            {"seller_id": "S2", "seller_zip_code_prefix": 13844, "seller_state": "MG"},
            {"seller_id": "S3", "seller_zip_code_prefix": 13023, "seller_state": "SP"},
        ]
    )
    conflicting_plan = PlannedAnalysis(
        route="hybrid",
        category_column="seller_zip_code_prefix",
        metric_column="seller_zip_code_prefix",
        time_column="seller_zip_code_prefix",
        steps=(),
    )

    result = _run_python(frame, conflicting_plan, None, question="哪些卖家的潜力最大")

    chart_types = {chart.chart_type for chart in result.charts}
    assert "histogram" not in chart_types
    assert "line" not in chart_types
    assert "box_plot" not in chart_types


def test_python_baseline_adds_distribution_charts_only_with_enough_data() -> None:
    from app.analysis.services import PlannedAnalysis, _run_python

    frame = pd.DataFrame.from_records(
        [
            {
                "segment": "Enterprise" if index < 15 else "SMB",
                "sales_amount": 100 + index,
                "profit": 20 + index / 2,
                "quantity": 1 + index % 4,
            }
            for index in range(30)
        ]
    )
    plan = PlannedAnalysis(
        route="hybrid",
        category_column="segment",
        metric_column="sales_amount",
        time_column=None,
        steps=(),
    )

    result = _run_python(frame, plan, None, question="比较各客群销售额并分析分布")

    chart_types = {chart.chart_type for chart in result.charts}
    assert {"histogram", "box_plot", "correlation_heatmap"}.issubset(chart_types)


def test_python_fallback_preserves_olist_negative_table_and_dimension_contract() -> None:
    from app.analysis.services import PlannedAnalysis, _run_python

    frame = pd.DataFrame.from_records(
        [
            {
                "customer_id": f"{index:032x}",
                "customer_state": "SP" if index < 12 else "RJ",
                "payment_type": "credit_card" if index % 2 else "boleto",
                "payment_value": float(100 + index),
            }
            for index in range(24)
        ]
    )
    plan = PlannedAnalysis(
        route="sql",
        category_column="customer_state",
        metric_column="payment_value",
        time_column=None,
        steps=(
            "Python Agent: compute EDA, text statistics, keyword comparisons, and visualization specs.",
        ),
    )
    question = (
        "仅使用 customers、orders、order_payments 三张表，过滤 "
        "order_status=delivered，按 customer_state 统计 payment_value 总额。"
        "不要使用 reviews，也不要按 payment_type 分组。"
    )

    result = _run_python(frame, plan, None, question=question)
    rendered_charts = tuple(str(chart.model_dump(mode="json")) for chart in result.charts)

    assert result.text_analysis == ()
    assert all("payment_type" not in chart for chart in rendered_charts)
    assert all("customer_id" not in chart for chart in rendered_charts)
    assert all("reviews" not in chart for chart in rendered_charts)


def test_analysis_service_generates_line_chart_for_time_series(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"date": "2026-01", "region": "North", "sales": 100, "profit": 20},
            {"date": "2026-02", "region": "North", "sales": 120, "profit": 25},
            {"date": "2026-03", "region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisService(repository).run(
        dataset_id=dataset.id,
        question="Show monthly sales trend",
    )

    assert result.python_result is not None
    line_chart = next(
        chart for chart in result.python_result.charts if chart.chart_type == "line"
    )
    assert line_chart.spec == {"x": "date", "y": "sales"}
    assert len(line_chart.data) == 3


def test_report_fallback_prioritizes_verified_kpis_and_monthly_trend() -> None:
    from app.analysis.services import _build_final_insights, _format_report_charts
    from app.schemas.analysis import ChartResponse, PythonAnalysisResponse, SQLAnalysisResponse

    sql_result = SQLAnalysisResponse(
        sql="SELECT month, gmv, order_count, aov, on_time_rate FROM dataset",
        rows=(
            {
                "evidence_id": "ev_1",
                "query_index": 1,
                "month": "2026-01",
                "gmv": 100.0,
                "order_count": 10,
                "aov": 10.0,
                "on_time_rate": 0.95,
            },
            {
                "evidence_id": "ev_1",
                "query_index": 1,
                "month": "2026-02",
                "gmv": 200.0,
                "order_count": 20,
                "aov": 10.0,
                "on_time_rate": 0.8,
            },
            {
                "evidence_id": "ev_1",
                "query_index": 1,
                "month": "2026-03",
                "gmv": 150.0,
                "order_count": 15,
                "aov": 10.0,
                "on_time_rate": 0.92,
            },
        ),
        explanation="Verified monthly KPI query.",
    )
    python_result = PythonAnalysisResponse(
        statistics={"numeric_summary": {"on_time": {"mean": 0.9}}},
        insights=("数据集包含三个月记录。", "评论文本长度为 20。"),
        charts=(
            ChartResponse(
                title="评论关键词",
                chart_type="bar",
                spec={"x": "keyword", "y": "count"},
                data=({"keyword": "ok", "count": 3},),
            ),
        ),
    )
    question = "计算 GMV、订单数、客单价、准时率并分析月度趋势。"

    findings = _build_final_insights(
        question=question,
        python_result=python_result,
        sql_result=sql_result,
    )
    report_text = " ".join(finding.content for finding in findings)
    charts = _format_report_charts(
        question=question,
        sql_result=sql_result,
        charts=python_result.charts,
        findings=findings,
    )

    assert "GMV=450.00" in report_text
    assert "订单数=45" in report_text
    assert "客单价=10.00" in report_text
    assert "准时率=90.00%" in report_text
    assert "2026-02（200.00）" in report_text
    assert "准时率最低出现在 2026-02（80.00%）" in report_text
    assert findings[0].evidence.startswith("evidence_id:ev_1")
    assert len(charts) == 1
    assert charts[0].title == "GMV月度趋势"
    assert charts[0].spec == {"x": "month", "y": "gmv"}


def test_report_charts_reject_fields_outside_planner_scope() -> None:
    from app.analysis.services import PlannedAnalysis, _format_report_charts
    from app.schemas.analysis import ChartResponse

    plan = PlannedAnalysis(
        route="sql",
        category_column="customers__customer_state",
        metric_column="payments__payment_value",
        time_column=None,
        steps=("按客户州汇总付款金额",),
    )
    charts = _format_report_charts(
        question="按客户州分析付款金额",
        charts=(
            ChartResponse(
                title="客户州付款金额",
                chart_type="bar",
                spec={"x": "customers__customer_state", "y": "payments__payment_value"},
                data=({"customers__customer_state": "SP", "payments__payment_value": 10},),
            ),
            ChartResponse(
                title="卖家州付款金额",
                chart_type="bar",
                spec={"x": "sellers__seller_state", "y": "payments__payment_value"},
                data=({"sellers__seller_state": "SP", "payments__payment_value": 10},),
            ),
            ChartResponse(
                title="评论长度",
                chart_type="bar",
                spec={"x": "review_length", "y": "count"},
                data=({"review_length": 20, "count": 3},),
            ),
        ),
        findings=(),
        plan=plan,
    )

    assert [chart.title for chart in charts] == ["客户州付款金额"]


def test_analysis_service_python_agent_handles_text_review_data(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="reviews.json", source_type="json", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {
                "review": "A warm movie with excellent acting and beautiful pacing.",
                "sentiment": "positive",
            },
            {
                "review": "Terrible pacing, weak story, and boring scenes.",
                "sentiment": "negative",
            },
            {
                "review": "Excellent story with excellent acting and perfect ending.",
                "sentiment": "positive",
            },
        ],
    )

    result = AnalysisService(repository).run(
        dataset_id=dataset.id,
        question="比较正面和负面评论的文本长度、关键词和情绪表达差异",
    )

    assert result.plan.route == "hybrid"
    assert result.python_result is not None
    assert result.python_result.text_analysis
    assert result.python_result.text_analysis[0].text_column == "review"
    assert result.python_result.text_analysis[0].group_column == "sentiment"
    assert "review" in result.python_result.statistics["text_columns"]
    chart_types = {chart.chart_type for chart in result.python_result.charts}
    assert {"histogram", "bar"}.issubset(chart_types)
    assert any("平均文本长度" in insight for insight in result.python_result.insights)
