from __future__ import annotations

import pandas as pd

from app.analysis.services import AnalysisService, DatasetProfiler
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
    assert {"bar", "pie", "histogram", "box_plot", "correlation_heatmap"}.issubset(chart_types)
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
    assert "histogram" in chart_types
    assert "line" not in chart_types
    assert any(chart.chart_type == "box_plot" and chart.spec["x"] == "seller_id" for chart in result.charts)


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
