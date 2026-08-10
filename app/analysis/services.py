from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from itertools import islice
from math import cos, isfinite, isnan, pi, sin
from typing import Any
from uuid import UUID

import duckdb
import pandas as pd

from app.analysis.query_intent import (
    infer_query_intent,
    negated_grouping_columns,
    strip_negated_clauses,
)
from app.analysis.text_analysis import run_text_analysis_toolbox
from app.analysis.validators import validate_chart_specs, validate_findings_traceability
from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisFilterResponse,
    AnalysisFrameworkResponse,
    AnalysisHypothesisResponse,
    AnalysisPlanResponse,
    AnalysisReflectionResponse,
    AnalysisRoundPlanResponse,
    AnalysisRoundResponse,
    AnalysisRunResponse,
    ChartResponse,
    DatasetColumnProfile,
    DatasetProfileResponse,
    InsightFindingResponse,
    PythonAnalysisResponse,
    SQLAnalysisResponse,
    StructuredReportResponse,
    ValidationIssueResponse,
)
from app.storage.dataset_store import DatasetStoreRepository


@dataclass(frozen=True)
class PlannedAnalysis:
    route: str
    category_column: str | None
    metric_column: str | None
    time_column: str | None
    steps: tuple[str, ...]
    aggregations: tuple[AnalysisAggregationResponse, ...] = ()
    filters: tuple[AnalysisFilterResponse, ...] = ()
    requested_dimensions: tuple[str, ...] = ()
    derived_metrics: tuple[str, ...] = ()
    time_grain: str | None = None


class DatasetProfiler:
    def profile(
        self,
        *,
        dataset_id: UUID,
        records: list[dict[str, Any]],
        sample_limit: int = 20,
    ) -> DatasetProfileResponse:
        df = _dataframe(records)
        columns: list[DatasetColumnProfile] = []
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []

        for column in df.columns:
            series = _column_series(df, str(column))
            numeric_series = pd.to_numeric(series, errors="coerce")
            is_numeric = bool(
                series.notna().any() and numeric_series.notna().sum() == series.notna().sum()
                and not _looks_like_identifier_column(str(column))
            )
            if is_numeric:
                numeric_columns.append(str(column))
            else:
                categorical_columns.append(str(column))

            valid_numeric = numeric_series.dropna()
            columns.append(
                DatasetColumnProfile(
                    name=str(column),
                    dtype=str(series.dtype),
                    missing_count=int(series.isna().sum()),
                    distinct_count=int(series.dropna().nunique()),
                    is_numeric=is_numeric,
                    min_value=(
                        _float_or_none(valid_numeric.min())
                        if is_numeric and not valid_numeric.empty
                        else None
                    ),
                    max_value=(
                        _float_or_none(valid_numeric.max())
                        if is_numeric and not valid_numeric.empty
                        else None
                    ),
                    mean=(
                        _float_or_none(valid_numeric.mean())
                        if is_numeric and not valid_numeric.empty
                        else None
                    ),
                )
            )

        missing_count = int(df.isna().sum().sum()) if not df.empty else 0
        total_cells = max(int(df.shape[0] * df.shape[1]), 1)
        return DatasetProfileResponse(
            dataset_id=dataset_id,
            row_count=int(df.shape[0]),
            column_count=int(df.shape[1]),
            missing_value_count=missing_count,
            missing_value_ratio=missing_count / total_cells,
            duplicate_row_count=int(df.duplicated().sum()) if not df.empty else 0,
            numeric_columns=tuple(numeric_columns),
            categorical_columns=tuple(categorical_columns),
            columns=tuple(columns),
            sample_records=tuple(_records(df.head(sample_limit))),
        )


def apply_column_metadata_to_profile(
    profile: DatasetProfileResponse,
    column_metadata: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> DatasetProfileResponse:
    metadata_by_name = {
        str(item.get("column_name")): item
        for item in column_metadata
        if isinstance(item, dict) and item.get("column_name")
    }
    if not metadata_by_name:
        return profile

    columns: list[DatasetColumnProfile] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in profile.columns:
        metadata = metadata_by_name.get(column.name, {})
        role = str(metadata.get("role") or "").strip()
        effective_type = str(
            metadata.get("override_type")
            or metadata.get("inferred_type")
            or column.dtype
        ).strip()
        is_ignored = role == "ignore"
        is_identifier = role == "id"
        is_metric = role == "metric" or effective_type.lower() in {
            "number",
            "numeric",
            "integer",
            "float",
            "decimal",
        }
        is_numeric = bool(is_metric and not is_ignored and not is_identifier)
        columns.append(
            DatasetColumnProfile(
                name=column.name,
                dtype=effective_type or column.dtype,
                missing_count=column.missing_count,
                distinct_count=column.distinct_count,
                is_numeric=is_numeric,
                min_value=column.min_value if is_numeric else None,
                max_value=column.max_value if is_numeric else None,
                mean=column.mean if is_numeric else None,
            )
        )
        if is_ignored:
            continue
        if is_numeric:
            numeric_columns.append(column.name)
        else:
            categorical_columns.append(column.name)

    return DatasetProfileResponse(
        dataset_id=profile.dataset_id,
        row_count=profile.row_count,
        column_count=profile.column_count,
        missing_value_count=profile.missing_value_count,
        missing_value_ratio=profile.missing_value_ratio,
        duplicate_row_count=profile.duplicate_row_count,
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        columns=tuple(columns),
        sample_records=profile.sample_records,
    )


class AnalysisService:
    def __init__(self, repository: DatasetStoreRepository) -> None:
        self._repository = repository
        self._profiler = DatasetProfiler()

    def run(self, *, dataset_id: UUID, question: str) -> AnalysisRunResponse:
        records = self._repository.read_analysis_records(dataset_id)
        if not records:
            raise RuntimeError("Dataset has no raw records to analyze.")

        df = _dataframe(records)
        profile = apply_column_metadata_to_profile(
            self._profiler.profile(dataset_id=dataset_id, records=records),
            self._repository.list_column_metadata(dataset_id),
        )
        plan = _plan(question, profile)

        sql_result: SQLAnalysisResponse | None = None
        if plan.route in {"sql", "hybrid"}:
            sql_result = _run_sql(df, plan)

        python_result = _run_python(df, plan, sql_result, question=question)

        report = _report_markdown(
            question=question,
            profile=profile,
            plan=plan,
            sql_result=sql_result,
            python_result=python_result,
        )
        rounds = _build_analysis_rounds(question=question, plan=plan, python_result=python_result)
        final_insights = _build_final_insights(
            question=question,
            python_result=python_result,
            sql_result=sql_result,
        )
        validation_issues = _build_validation_issues(
            findings=final_insights,
            charts=python_result.charts,
        )
        structured_report = _structured_report(
            question=question,
            profile=profile,
            sql_result=sql_result,
            python_result=python_result,
            rounds=rounds,
            final_insights=final_insights,
            validation_issues=validation_issues,
        )
        html_report = render_structured_report_html(
            structured_report,
            title="DataMind 分析报告",
        )
        self._repository.save_report(
            dataset_id=dataset_id,
            title="DataMind 分析报告",
            markdown=report,
            metadata={
                "question": question,
                "route": plan.route,
                "structured_report": structured_report.model_dump(mode="json"),
                "html_report": html_report,
            },
        )
        for chart in python_result.charts:
            self._repository.save_chart(
                dataset_id=dataset_id,
                title=chart.title,
                chart_type=chart.chart_type,
                chart_spec=chart.spec,
                chart_data=list(chart.data),
            )

        return AnalysisRunResponse(
            dataset_id=dataset_id,
            question=question,
            plan=AnalysisPlanResponse(route=plan.route, steps=plan.steps),
            profile=profile,
            sql_result=sql_result,
            python_result=python_result,
            rounds=rounds,
            final_insights=final_insights,
            validation_issues=validation_issues,
            structured_report=structured_report,
            html_report=html_report,
            report_markdown=report,
        )


def _plan(question: str, profile: DatasetProfileResponse) -> PlannedAnalysis:
    semantic_question = strip_negated_clauses(question)
    text = semantic_question.casefold()
    numeric = list(profile.numeric_columns)
    categorical = list(profile.categorical_columns)
    intent = infer_query_intent(question, profile)
    forbidden_dimensions = set(
        negated_grouping_columns(question, tuple(categorical))
    )
    eligible_categories = [
        column for column in categorical if column not in forbidden_dimensions
    ]
    time_column = _pick_time_column(profile)
    metric = intent.metric or _pick_metric_column(text, numeric)
    category = (
        intent.required_dimensions[0]
        if intent.required_dimensions
        else None
        if intent.aggregations
        else intent.dimensions[0]
        if intent.dimensions
        else _pick_category_column(text, eligible_categories)
    )

    wants_python = any(
        token in text for token in ("correlation", "distribution", "histogram", "box", "eda")
    )
    wants_python = wants_python or any(
        token in question for token in ("相关", "分布", "探索", "画像", "统计")
    )
    wants_text_analysis = any(
        token in text
        for token in (
            "comment",
            "feedback",
            "keyword",
            "negative",
            "positive",
            "review",
            "sentiment",
            "text",
        )
    ) or any(token in question for token in ("关键词", "正面", "负面", "评论", "评价", "文本", "情绪"))
    wants_python = wants_python or wants_text_analysis
    wants_trend = any(token in text for token in ("trend", "monthly", "date", "time")) or any(
        token in question for token in ("趋势", "月度", "按月", "月份", "日期", "时间")
    )
    route = "python" if wants_python and not category else "sql"
    if wants_python and (category or wants_trend):
        route = "hybrid"

    steps = (
        "Planner Agent: understand the question and inspect dataset schema.",
        f"User question: {question}",
        f"SQL Agent: generate DuckDB query for {metric or 'available numeric metric'}.",
        "Python Agent: compute EDA, text statistics, keyword comparisons, and visualization specs.",
        "Report Agent: summarize results as Markdown.",
    )
    return PlannedAnalysis(
        route=route,
        category_column=category,
        metric_column=metric,
        time_column=time_column if wants_trend else None,
        steps=steps,
        aggregations=intent.aggregations,
        filters=intent.filters,
        requested_dimensions=intent.required_dimensions,
        derived_metrics=intent.derived_metrics,
        time_grain="month"
        if wants_trend and any(token in text for token in ("月度", "按月", "月份", "monthly"))
        else None,
    )


def _run_sql(df: pd.DataFrame, plan: PlannedAnalysis) -> SQLAnalysisResponse:
    if plan.aggregations or plan.filters:
        return _run_intent_sql(df, plan)

    metric = _safe_metric_column(df, plan.metric_column)
    if metric is None:
        category = plan.category_column or _first_categorical(df)
        if plan.time_column is not None and plan.time_column in df.columns:
            time_col = plan.time_column
            sql = (
                f'SELECT "{time_col}" AS period, COUNT(*) AS row_count '
                "FROM dataset "
                f'GROUP BY "{time_col}" '
                f'ORDER BY "{time_col}" '
                "LIMIT 50"
            )
            explanation = f"未检测到可聚合的数值指标，因此按 {time_col} 统计记录数。"
        elif category is not None:
            sql = (
                f'SELECT "{category}" AS category, COUNT(*) AS row_count '
                "FROM dataset "
                f'GROUP BY "{category}" '
                "ORDER BY row_count DESC "
                "LIMIT 20"
            )
            explanation = f"未检测到可聚合的数值指标，因此按 {category} 统计记录数。"
        else:
            sql = "SELECT COUNT(*) AS row_count FROM dataset"
            explanation = "未检测到可聚合的数值指标，因此统计数据集记录数。"
    elif plan.time_column is not None:
        time_col = plan.time_column
        metric_alias = _alias(metric)
        sql = (
            f'SELECT "{time_col}" AS period, '
            f'SUM(CAST("{metric}" AS DOUBLE)) AS total_{metric_alias} '
            "FROM dataset "
            f'GROUP BY "{time_col}" '
            f'ORDER BY "{time_col}" '
            "LIMIT 50"
        )
        explanation = f"Grouped records by {time_col} and summed {metric} to show a trend."
    else:
        category = plan.category_column or _first_categorical(df)
        if category is None:
            sql = f'SELECT SUM(CAST("{metric}" AS DOUBLE)) AS total_{_alias(metric)} FROM dataset'
            explanation = f"Summed {metric} across the whole dataset."
        else:
            metric_alias = _alias(metric)
            sql = (
                f'SELECT "{category}" AS category, '
                f'SUM(CAST("{metric}" AS DOUBLE)) AS total_{metric_alias} '
                "FROM dataset "
                f'GROUP BY "{category}" '
                f"ORDER BY total_{metric_alias} DESC "
                "LIMIT 20"
            )
            explanation = f"Grouped records by {category} and ranked them by total {metric}."

    connection = duckdb.connect(":memory:")
    try:
        connection.register("dataset", df)
        rows = _records(connection.execute(sql).fetchdf())
    finally:
        connection.close()
    return SQLAnalysisResponse(sql=sql, rows=tuple(rows), explanation=explanation)


def _run_intent_sql(
    df: pd.DataFrame,
    plan: PlannedAnalysis,
) -> SQLAnalysisResponse:
    sql, explanation = _build_intent_query(df, plan)
    connection = duckdb.connect(":memory:")
    try:
        connection.register("dataset", df)
        rows = _records(connection.execute(sql).fetchdf())
    finally:
        connection.close()
    return SQLAnalysisResponse(sql=sql, rows=tuple(rows), explanation=explanation)


def _build_intent_query(
    df: pd.DataFrame,
    plan: PlannedAnalysis,
) -> tuple[str, str]:
    dimensions = [
        column
        for column in (plan.time_column, plan.category_column)
        if column and column in df.columns
    ]
    dimensions = list(dict.fromkeys(dimensions))
    select_parts: list[str] = []
    group_parts: list[str] = []
    for column in dimensions:
        output_alias = (
            "period"
            if column == plan.time_column
            else "category"
            if len(dimensions) == 1
            else _alias(column)
        )
        expression = _quote_identifier(column)
        if column == plan.time_column and plan.time_grain == "month":
            expression = f"STRFTIME(TRY_CAST({expression} AS TIMESTAMP), '%Y-%m')"
        selected_expression = (
            f"COALESCE({expression}, 'ALL')"
            if column == plan.time_column and plan.time_grain
            else expression
        )
        select_parts.append(f"{selected_expression} AS {_quote_identifier(output_alias)}")
        group_parts.append(expression)

    aggregation_aliases: list[str] = []
    aggregation_expressions: dict[tuple[str, str | None], str] = {}
    for aggregation in plan.aggregations:
        expression = _aggregation_sql(df, aggregation)
        if expression is None:
            continue
        aggregation_expressions[(aggregation.operation, aggregation.column)] = expression
        alias = _alias(aggregation.alias)
        aggregation_aliases.append(alias)
        select_parts.append(f"{expression} AS {_quote_identifier(alias)}")
    if "average_order_value" in plan.derived_metrics:
        total = next(
            (
                expression
                for (operation, _), expression in aggregation_expressions.items()
                if operation == "sum"
            ),
            None,
        )
        orders = next(
            (
                expression
                for (operation, _), expression in aggregation_expressions.items()
                if operation == "count_distinct"
            ),
            None,
        )
        if total and orders:
            aggregation_aliases.append("average_order_value")
            select_parts.append(
                f"ROUND(CAST(({total}) AS DOUBLE) / NULLIF(({orders}), 0), 6) "
                'AS "average_order_value"'
            )
    if not aggregation_aliases:
        select_parts.append('COUNT(*) AS "row_count"')
        aggregation_aliases.append("row_count")

    where_parts = [
        _filter_sql(condition)
        for condition in plan.filters
        if condition.column in df.columns
    ]
    sql = f"SELECT {', '.join(select_parts)} FROM dataset"
    if where_parts:
        sql += f" WHERE {' AND '.join(where_parts)}"
    if plan.time_column in dimensions and len(group_parts) == 1 and plan.time_grain:
        sql += f" GROUP BY GROUPING SETS (({group_parts[0]}), ())"
    elif group_parts:
        sql += f" GROUP BY {', '.join(group_parts)}"
    if plan.time_column in dimensions:
        sql += ' ORDER BY "period" ASC'
    elif group_parts and aggregation_aliases:
        sql += f" ORDER BY {_quote_identifier(aggregation_aliases[0])} DESC"
    explanation = (
        "按用户明确要求执行 "
        f"{', '.join(item.operation for item in plan.aggregations) or 'count'} 聚合"
        f"，维度：{', '.join(dimensions) or '全体'}"
        f"，过滤条件：{len(where_parts)} 个"
        f"，派生指标：{', '.join(plan.derived_metrics) or '无'}。"
    )
    return sql, explanation


def _aggregation_sql(
    df: pd.DataFrame,
    aggregation: AnalysisAggregationResponse,
) -> str | None:
    operation = aggregation.operation
    column = aggregation.column
    if operation == "count" and not column:
        return "COUNT(*)"
    if not column or column not in df.columns:
        return None
    quoted = _quote_identifier(column)
    if operation == "count":
        return f"COUNT({quoted})"
    if operation == "count_distinct":
        return f"COUNT(DISTINCT {quoted})"
    if operation == "avg" and column.rsplit("__", 1)[-1].casefold() == "on_time":
        return (
            "ROUND(AVG(CASE WHEN LOWER(TRIM(CAST("
            f"{quoted} AS VARCHAR))) IN ('true', '1', 'yes', 'y', 'on_time', 'ontime') "
            "THEN 1.0 WHEN LOWER(TRIM(CAST("
            f"{quoted} AS VARCHAR))) IN ('false', '0', 'no', 'n', 'late', 'delayed') "
            "THEN 0.0 ELSE NULL END), 6)"
        )
    if operation not in {"sum", "avg", "min", "max"}:
        return None
    if not _is_metric_column(df, column):
        return None
    expression = f"{operation.upper()}(CAST({quoted} AS DOUBLE))"
    return f"ROUND({expression}, 6)"


def _filter_sql(condition: AnalysisFilterResponse) -> str:
    return (
        f"{_quote_identifier(condition.column)} {condition.operator} "
        f"{_sql_literal(condition.value)}"
    )


def _quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _sql_literal(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _run_python(
    df: pd.DataFrame,
    plan: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    question: str = "",
) -> PythonAnalysisResponse:
    df = _apply_plan_filters(df, plan.filters)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    statistics: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_summary": (
            _jsonable(numeric_df.describe().to_dict()) if not numeric_df.empty else {}
        ),
    }

    insights: list[str] = [
        f"数据集包含 {df.shape[0]} 行、{df.shape[1]} 列。",
        f"当前分析路线为 {plan.route}。",
    ]
    if plan.metric_column:
        metric_series = pd.to_numeric(_column_series(df, plan.metric_column), errors="coerce").dropna()
        if not metric_series.empty:
            mean = metric_series.mean()
            maximum = metric_series.max()
            insights.append(
                f"{plan.metric_column} 平均值为 {mean:.2f}，最大值为 {maximum:.2f}。"
            )

    charts: list[ChartResponse] = []
    if sql_result and sql_result.rows:
        row_keys = tuple(sql_result.rows[0].keys())
        primary_aggregation = next(
            (
                item
                for item in plan.aggregations
                if item.alias in row_keys
                and item.operation in {"sum", "count", "count_distinct"}
            ),
            None,
        ) or next(
            (item for item in plan.aggregations if item.alias in row_keys),
            None,
        )
        chart_metric = (
            primary_aggregation.alias if primary_aggregation is not None else row_keys[-1]
        )
        charts.append(
            ChartResponse(
                title="SQL 结果图",
                chart_type="bar",
                spec={"x": row_keys[0], "y": chart_metric},
                data=sql_result.rows,
            )
        )
        if (
            len(row_keys) >= 2
            and primary_aggregation is not None
            and primary_aggregation.operation in {"sum", "count", "count_distinct"}
        ):
            displayed_rows = tuple(sql_result.rows[:8])
            excluded_count = max(0, len(sql_result.rows) - len(displayed_rows))
            denominator_scope = (
                "displayed_top_n" if excluded_count else "complete_query_result"
            )
            charts.append(
                ChartResponse(
                    title="占比图",
                    chart_type="pie",
                    spec={
                        "names": row_keys[0],
                        "values": chart_metric,
                        "denominator_scope": denominator_scope,
                        "displayed_category_count": len(displayed_rows),
                        "excluded_category_count": excluded_count,
                    },
                    data=displayed_rows,
                    explanation=(
                        f"仅展示查询结果前 {len(displayed_rows)} 类；百分比以这 "
                        f"{len(displayed_rows)} 类的合计为分母，另有 {excluded_count} 类"
                        "未进入图表，不能将该百分比解释为全量占比。"
                        if excluded_count
                        else "展示查询返回的全部类别；百分比以全部返回类别的合计为分母。"
                    ),
                )
            )

    metric = _safe_metric_column(df, plan.metric_column)
    if plan.time_column and metric and plan.time_column != metric and plan.time_column in df.columns and metric in df.columns:
        line_df = df[[plan.time_column, metric]].copy()
        line_df[metric] = pd.to_numeric(line_df[metric], errors="coerce")
        line_df = line_df.dropna(subset=[plan.time_column, metric])
        if not line_df.empty:
            line_df = (
                line_df.groupby(plan.time_column, dropna=True)[metric]
                .sum()
                .reset_index()
                .sort_values(plan.time_column)
                .head(100)
            )
            charts.append(
                ChartResponse(
                    title="趋势折线图",
                    chart_type="line",
                    spec={"x": plan.time_column, "y": metric},
                    data=tuple(_records(line_df)),
                )
            )

    if metric and metric in df.columns:
        histogram_series = pd.to_numeric(_column_series(df, metric), errors="coerce").dropna()
        if len(histogram_series) >= 10:
            charts.append(
                ChartResponse(
                    title="数值分布直方图",
                    chart_type="histogram",
                    spec={
                        "x": "label",
                        "y": "value",
                        "source_metric": metric,
                        "sample_size": len(histogram_series),
                    },
                    data=tuple(_histogram_buckets(histogram_series.tolist())),
                )
            )

        category = plan.category_column if plan.category_column != metric else None
        category = category or _first_categorical(df, exclude={metric})
        if category and category in df.columns and not histogram_series.empty:
            box_df = df[[category, metric]].copy()
            box_df[metric] = pd.to_numeric(box_df[metric], errors="coerce")
            box_df = box_df.dropna(subset=[category, metric])
            group_sizes = box_df.groupby(category, dropna=True)[metric].count()
            eligible_groups = group_sizes[group_sizes >= 5]
            box_df = box_df[box_df[category].isin(eligible_groups.index)].head(500)
            if len(eligible_groups) >= 2:
                charts.append(
                    ChartResponse(
                        title="分类箱线图",
                        chart_type="box_plot",
                        spec={
                            "x": category,
                            "y": metric,
                            "sample_size": len(box_df),
                            "group_sizes": {
                                str(key): int(value)
                                for key, value in eligible_groups.items()
                            },
                        },
                        data=tuple(_records(box_df)),
                    )
                )

    correlation_rows = int(numeric_df.dropna(how="all").shape[0])
    if len(numeric_df.columns) >= 3 and correlation_rows >= 20:
        corr = numeric_df.corr(numeric_only=True).fillna(0)
        charts.append(
            ChartResponse(
                title="相关性热力图",
                chart_type="correlation_heatmap",
                spec={
                    "columns": [str(column) for column in corr.columns],
                    "sample_size": correlation_rows,
                },
                data=tuple(
                    {
                        "source": str(source),
                        "target": str(target),
                        "value": _float_or_none(corr.loc[source, target]),
                    }
                    for source in corr.columns
                    for target in corr.columns
                ),
            )
        )
    # Text intent and grouping exclusions are user-contract decisions. Planner prose must not
    # reintroduce a text route or a dimension that the user explicitly rejected.
    text_analysis = run_text_analysis_toolbox(df, question=question)
    for text_result in text_analysis:
        for insight in text_result.insights:
            if insight and insight not in insights:
                insights.append(insight)
        charts.extend(text_result.charts)
    if text_analysis:
        statistics["text_columns"] = [text_result.text_column for text_result in text_analysis]
        statistics["text_analysis"] = [
            text_result.model_dump(mode="json") for text_result in text_analysis
        ]
    return PythonAnalysisResponse(
        statistics=_jsonable(statistics),
        insights=tuple(insights),
        charts=tuple(charts),
        text_analysis=text_analysis,
    )


def _apply_plan_filters(
    dataframe: pd.DataFrame,
    filters: tuple[AnalysisFilterResponse, ...],
) -> pd.DataFrame:
    filtered = dataframe
    for condition in filters:
        if condition.column not in filtered.columns:
            continue
        series = _column_series(filtered, condition.column)
        value = condition.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            comparable = pd.to_numeric(series, errors="coerce")
        else:
            comparable = series.astype("string").str.casefold()
            value = str(value).casefold()
        mask = {
            "=": comparable == value,
            "!=": comparable != value,
            ">": comparable > value,
            ">=": comparable >= value,
            "<": comparable < value,
            "<=": comparable <= value,
        }[condition.operator]
        filtered = filtered.loc[mask.fillna(False)]
    return filtered


def _report_markdown(
    *,
    question: str,
    profile: DatasetProfileResponse,
    plan: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse,
) -> str:
    analysis_row_count = _analysis_row_count(profile, python_result)
    findings = _build_final_insights(
        question=question,
        python_result=python_result,
        sql_result=sql_result,
    )
    charts = _format_report_charts(
        question=question,
        sql_result=sql_result,
        charts=python_result.charts,
        findings=findings,
        plan=plan,
    )
    lines = [
        "# DataMind 分析报告",
        "",
        "## Executive Summary",
        f"- 用户问题: {question}",
        f"- 数据规模: {analysis_row_count} 行, {profile.column_count} 列",
        f"- 缺失值: {profile.missing_value_count} 个, 重复行: {profile.duplicate_row_count} 行",
        f"- Planner 路由: {plan.route}",
        "",
    ]
    if sql_result is not None:
        lines.extend(
            [
                "## SQL Results",
                "```sql",
                sql_result.sql,
                "```",
                f"- 解释: {sql_result.explanation}",
                f"- 返回行数: {len(sql_result.rows)}",
                "",
            ]
        )
    if findings:
        lines.extend(["## Business Insights"])
        for finding in findings:
            lines.append(f"- **{finding.title}**: {finding.content}")
            if finding.evidence:
                lines.append(f"  - 证据: {finding.evidence}")
            if finding.recommended_action:
                lines.append(f"  - 建议: {finding.recommended_action}")
        lines.append("")
    if charts:
        lines.extend(
            [
                "## Visualizations",
                *[
                    f"- {chart.title} ({chart.chart_type}): "
                    f"{chart.explanation or '见图表数据。'}"
                    for chart in charts
                ],
                "",
            ]
        )
    return "\n".join(lines)


def _build_analysis_rounds(
    *,
    question: str,
    plan: PlannedAnalysis,
    python_result: PythonAnalysisResponse,
) -> tuple[AnalysisRoundResponse, ...]:
    insights = list(python_result.insights) or ["Completed baseline dataset analysis."]
    rounds: list[AnalysisRoundResponse] = []
    for index, insight in enumerate(insights[:3], 1):
        rounds.append(
            AnalysisRoundResponse(
                round_number=index,
                hypothesis=AnalysisHypothesisResponse(
                    statement=(
                        question
                        if index == 1
                        else f"Validate supporting pattern #{index} from the dataset."
                    ),
                    judgment_criteria="Use only uploaded dataset columns and computed results.",
                    expected_direction="unknown",
                ),
                plan=AnalysisRoundPlanResponse(
                    round_number=index,
                    analytical_step=_analytical_step_for_round(index, plan),
                    question=question,
                    route=plan.route,
                    metric_column=plan.metric_column,
                    category_column=plan.category_column,
                    time_column=plan.time_column,
                ),
                reflection=AnalysisReflectionResponse(
                    insight_text=insight,
                    impact_pct=0 if index > 1 else 100,
                    has_insight=True,
                    decision="CONTINUE" if index < min(len(insights), 3) else "STOP",
                    data_source="python_result.insights",
                ),
                execution_result={
                    "insight": insight,
                    "statistics_keys": tuple(python_result.statistics.keys()),
                },
                charts=tuple(python_result.charts[:2]),
                validation_status="passed",
            )
        )
    return tuple(rounds)


def _framework_from_profile(
    *,
    question: str,
    profile: DatasetProfileResponse,
) -> AnalysisFrameworkResponse:
    dimensions = tuple(profile.categorical_columns[:5])
    metrics = tuple(profile.numeric_columns[:5])
    routes: list[str] = []
    if dimensions and metrics:
        routes.append("sql")
    if metrics:
        routes.append("python")
    if len(routes) == 2:
        routes.append("hybrid")
    hypotheses = []
    if dimensions and metrics:
        hypotheses.append(f"{dimensions[0]} 可能解释 {metrics[0]} 的主要差异。")
    if metrics:
        hypotheses.append(f"{metrics[0]} 的分布、极值或趋势可能包含关键异常。")
    if not hypotheses:
        hypotheses.append("数据规模、缺失值和重复行可能是当前问题的主要线索。")
    risk_notes = []
    if profile.missing_value_count:
        risk_notes.append(f"存在 {profile.missing_value_count} 个缺失值，结论需要说明数据缺口。")
    if profile.duplicate_row_count:
        risk_notes.append(f"存在 {profile.duplicate_row_count} 条重复行，聚合结果可能受影响。")
    return AnalysisFrameworkResponse(
        business_question=question,
        candidate_dimensions=dimensions,
        candidate_metrics=metrics,
        likely_routes=tuple(routes or ("python",)),
        initial_hypotheses=tuple(hypotheses[:3]),
        risk_notes=tuple(risk_notes),
        dimensions=dimensions,
        key_questions=tuple(
            item
            for item in (
                f"哪些 {dimensions[0]} 的表现差异最大？" if dimensions else "",
                f"{metrics[0]} 是否存在异常分布或极值？" if metrics else "",
                "是否存在影响结论可信度的数据质量问题？",
            )
            if item
        ),
        success_criteria="所有结论必须能追溯到 SQL、Python 统计或轮次反思结果。",
    )


def _format_report_charts(
    *,
    charts: tuple[ChartResponse, ...],
    findings: tuple[InsightFindingResponse, ...],
    question: str = "",
    sql_result: SQLAnalysisResponse | None = None,
    plan: PlannedAnalysis | None = None,
) -> tuple[ChartResponse, ...]:
    finding_titles = tuple(finding.title for finding in findings[:2])
    focused_rows = _coherent_sql_rows(sql_result, question)
    preferred_fields = tuple(
        dict.fromkeys(
            field
            for field in (
                plan.metric_column if plan else None,
                plan.category_column if plan else None,
                plan.time_column if plan else None,
                *(plan.requested_dimensions if plan else ()),
            )
            if field
        )
    )
    time_column = _preferred_column(
        tuple(focused_rows[0]) if focused_rows else (),
        tuple(field for field in (plan.time_column if plan else None,) if field),
    ) or _time_column(focused_rows)
    metric_columns = _numeric_columns(focused_rows)
    primary_metric = _preferred_column(
        metric_columns,
        tuple(field for field in (plan.metric_column if plan else None,) if field),
    ) or _primary_metric(metric_columns, question)
    folded_question = question.casefold()
    trend_requested = any(
        token in folded_question for token in ("趋势", "月度", "按月", "trend", "monthly")
    )
    formatted: list[ChartResponse] = []
    if trend_requested and time_column and primary_metric:
        data = tuple(
            {
                time_column: row.get(time_column),
                primary_metric: row.get(primary_metric),
            }
            for row in focused_rows
            if row.get(time_column) is not None
            and str(row.get(time_column)).casefold() != "all"
            and _finite_number(row.get(primary_metric)) is not None
        )
        if data:
            peak = max(data, key=lambda row: float(row[primary_metric]))
            formatted.append(
                ChartResponse(
                    title=f"{_metric_label(primary_metric)}月度趋势",
                    chart_type="line",
                    spec={"x": time_column, "y": primary_metric},
                    data=data,
                    explanation=(
                        f"按 {_time_label(time_column)} 展示 {_metric_label(primary_metric)}；"
                        f"峰值出现在 {peak[time_column]}，为 "
                        f"{_format_metric_value(primary_metric, peak[primary_metric])}。"
                    ),
                    related_finding_ids=finding_titles,
                )
            )

    if not trend_requested and focused_rows and primary_metric:
        dimension_candidates = tuple(
            column
            for column in focused_rows[0]
            if column not in {"evidence_id", "query_index", *metric_columns}
        )
        dimension = _preferred_column(
            dimension_candidates,
            tuple(
                field
                for field in (
                    plan.category_column if plan else None,
                    *(plan.requested_dimensions if plan else ()),
                )
                if field
            ),
        ) or next(
            (
                column for column in dimension_candidates
            ),
            None,
        )
        if dimension:
            data = tuple(
                {
                    dimension: row.get(dimension),
                    primary_metric: row.get(primary_metric),
                }
                for row in focused_rows
                if row.get(dimension) is not None
                and _finite_number(row.get(primary_metric)) is not None
            )
            if data:
                formatted.append(
                    ChartResponse(
                        title=f"{_metric_label(primary_metric)}分组对比",
                        chart_type="bar",
                        spec={"x": dimension, "y": primary_metric},
                        data=data,
                        explanation=(
                            f"按 {dimension} 比较 {_metric_label(primary_metric)}，"
                            "图表与核心排名使用相同聚合口径。"
                        ),
                        related_finding_ids=finding_titles,
                    )
                )

    seen = {(chart.title.casefold(), chart.chart_type) for chart in formatted}
    question_fields = {
        field.casefold()
        for row in focused_rows[:1]
        for field in row
        if field not in {"evidence_id", "query_index"}
        and _field_relevant_to_question(field, folded_question)
    }
    for chart in charts:
        signature = (chart.title.casefold(), chart.chart_type)
        if formatted and chart.title in {"SQL 结果图", "占比图"}:
            continue
        if signature in seen or (formatted and trend_requested and chart.chart_type in {"line", "pie"}):
            continue
        if not _chart_is_suitable(chart):
            continue
        chart_context = f"{chart.title} {chart.spec}".casefold()
        text_chart = any(
            token in chart_context
            for token in ("review", "comment", "keyword", "文本", "评论", "关键词")
        )
        if text_chart and not any(
            token in folded_question
            for token in ("review", "comment", "text", "文本", "评论", "关键词")
        ):
            continue
        if preferred_fields and not _chart_matches_plan(chart, preferred_fields):
            continue
        if not preferred_fields and question_fields and not any(
            field in chart_context for field in question_fields
        ):
            continue
        explanation = chart.explanation or _chart_explanation(chart)
        formatted.append(
            ChartResponse(
                title=chart.title,
                chart_type=chart.chart_type,
                spec=chart.spec,
                data=chart.data,
                explanation=explanation,
                related_finding_ids=chart.related_finding_ids or finding_titles,
            )
        )
        seen.add(signature)
        if len(formatted) >= (3 if trend_requested else 6):
            break
    return tuple(formatted)


_DERIVED_CHART_FIELDS = {
    "bin",
    "binend",
    "binstart",
    "count",
    "frequency",
    "label",
    "name",
    "value",
}


def _preferred_column(candidates: tuple[str, ...], preferred: tuple[str, ...]) -> str | None:
    return next(
        (
            candidate
            for candidate in candidates
            if any(_field_names_match(candidate, field) for field in preferred)
        ),
        None,
    )


def _chart_matches_plan(chart: ChartResponse, preferred: tuple[str, ...]) -> bool:
    fields = _chart_semantic_fields(chart)
    substantive = [
        field
        for field in fields
        if _normalized_field_name(field) not in _DERIVED_CHART_FIELDS
    ]
    return bool(substantive) and all(
        any(_field_names_match(field, expected) for expected in preferred)
        for field in substantive
    )


def _chart_semantic_fields(chart: ChartResponse) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("x", "y", "field", "group_by", "category", "metric", "columns"):
        value = chart.spec.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value if isinstance(item, str))
    if not values and chart.data:
        values.extend(str(field) for field in chart.data[0])
    return tuple(dict.fromkeys(values))


def _field_names_match(left: str, right: str) -> bool:
    left_name = _normalized_field_name(left)
    right_name = _normalized_field_name(right)
    return bool(
        left_name
        and right_name
        and (
            left_name == right_name
            or left_name.endswith(right_name)
            or right_name.endswith(left_name)
        )
    )


def _normalized_field_name(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


_BUSINESS_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sales": ("销售", "销售额", "营收", "收入"),
    "amount": ("金额", "销售额", "交易额", "营收"),
    "revenue": ("营收", "收入", "销售额"),
    "profit": ("利润", "毛利"),
    "price": ("价格", "单价"),
    "quantity": ("数量", "销量"),
    "count": ("数量", "次数", "订单数"),
    "order": ("订单",),
    "customer": ("客户", "用户"),
    "region": ("地区", "区域"),
    "segment": ("细分", "客群"),
    "date": ("日期", "时间"),
    "time": ("时间", "日期"),
}


def _field_relevant_to_question(field: str, folded_question: str) -> bool:
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", field.casefold())
    normalized_question = re.sub(
        r"[^0-9a-z\u4e00-\u9fff]+", "", folded_question.casefold()
    )
    if normalized and normalized in normalized_question:
        return True
    tokens = re.findall(r"[a-z]+|[\u4e00-\u9fff]+", field.casefold().replace("_", " "))
    return any(
        alias in normalized_question
        for token in tokens
        for alias in _BUSINESS_FIELD_ALIASES.get(token, ())
    )


def _chart_is_suitable(chart: ChartResponse) -> bool:
    if not chart.data:
        return False
    if chart.chart_type == "histogram":
        sample_size = chart.spec.get("sample_size")
        if not isinstance(sample_size, int):
            sample_size = sum(
                int(value)
                for row in chart.data
                if isinstance(row, dict)
                and isinstance((value := row.get("value")), (int, float))
            )
        return sample_size >= 10
    if chart.chart_type == "box_plot":
        group_sizes = chart.spec.get("group_sizes")
        if isinstance(group_sizes, dict):
            sizes = [int(value) for value in group_sizes.values()]
        else:
            group_by = str(chart.spec.get("x") or "")
            counts: dict[str, int] = {}
            for row in chart.data:
                if isinstance(row, dict) and row.get(group_by) is not None:
                    key = str(row[group_by])
                    counts[key] = counts.get(key, 0) + 1
            sizes = list(counts.values())
        return len(sizes) >= 2 and min(sizes) >= 5
    if chart.chart_type == "correlation_heatmap":
        columns = chart.spec.get("columns")
        sample_size = chart.spec.get("sample_size")
        return (
            isinstance(columns, list)
            and len(columns) >= 3
            and isinstance(sample_size, int)
            and sample_size >= 20
        )
    return True


def _build_final_insights(
    *,
    python_result: PythonAnalysisResponse | None,
    sql_result: SQLAnalysisResponse | None,
    question: str = "",
) -> tuple[InsightFindingResponse, ...]:
    findings = list(
        _sql_findings(
            question=question,
            sql_result=sql_result,
            python_result=python_result,
        )
    )
    if python_result and python_result.insights:
        findings = findings[:4]
    for index, insight in enumerate(python_result.insights[:4] if python_result else (), 1):
        if any(
            insight == finding.evidence or insight in finding.content
            for finding in findings
        ):
            continue
        findings.append(
            InsightFindingResponse(
                title=f"发现 {index}",
                content=insight,
                data_source="python_result.insights",
                impact_pct=0,
                evidence=insight,
                confidence="medium",
                business_impact="需要结合业务上下文判断影响范围。",
                recommended_action="继续用相关字段切分验证该发现。",
            )
        )
        if len(findings) >= 6:
            break
    if sql_result and sql_result.rows:
        findings.append(
            InsightFindingResponse(
                title="SQL 结果摘要",
                content=f"SQL 查询返回 {len(sql_result.rows)} 行结果，可作为主要排序或聚合依据。",
                data_source="sql_result.rows",
                impact_pct=100,
                evidence=f"SQL returned {len(sql_result.rows)} rows.",
                confidence="high",
                business_impact="可作为当前问题的主要量化依据。",
                recommended_action="检查排名靠前记录的结构性原因。",
            )
        )
    return tuple(findings[:6])


def _sql_findings(
    *,
    question: str,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
) -> tuple[InsightFindingResponse, ...]:
    rows = _coherent_sql_rows(sql_result, question)
    numeric_columns = _numeric_columns(rows)
    if not rows or not numeric_columns:
        return ()

    time_column = _time_column(rows)
    primary_metric = _primary_metric(numeric_columns, question)
    overall_row = next(
        (
            row
            for row in rows
            if time_column
            and str(row.get(time_column) or "").casefold() in {"all", "overall", "total"}
        ),
        None,
    )
    detail_rows = tuple(row for row in rows if row is not overall_row)
    values: dict[str, float] = {}
    for column in numeric_columns:
        overall_value = (
            _finite_number(overall_row.get(column)) if overall_row is not None else None
        )
        if overall_value is not None:
            values[column] = overall_value
            continue
        numbers = [
            number
            for row in detail_rows
            if (number := _finite_number(row.get(column))) is not None
        ]
        if not numbers:
            continue
        folded = column.casefold()
        if len(rows) == 1:
            values[column] = numbers[0]
        elif _is_average_metric(column):
            continue
        elif any(
            token in folded
            for token in (
                "gmv",
                "revenue",
                "sales",
                "amount",
                "price",
                "total",
                "sum",
                "count",
            )
        ):
            values[column] = sum(numbers)

    gmv_column = next(
        (
            column
            for column in numeric_columns
            if any(
                token in column.casefold()
                for token in ("gmv", "revenue", "sales", "amount", "price")
            )
            and column in values
        ),
        None,
    )
    count_column = next(
        (
            column
            for column in numeric_columns
            if "count" in column.casefold() and column in values
        ),
        None,
    )
    average_column = next(
        (
            column
            for column in numeric_columns
            if _is_average_metric(column)
        ),
        None,
    )
    if gmv_column and count_column and average_column and values[count_column]:
        values[average_column] = values[gmv_column] / values[count_column]

    statistics = (
        python_result.statistics.get("numeric_summary") or {}
        if python_result
        else {}
    )
    if isinstance(statistics, dict):
        for column in numeric_columns:
            if "rate" not in column.casefold() and "ratio" not in column.casefold():
                continue
            candidates = (column, column.removesuffix("_rate"), column.removesuffix("_ratio"))
            summary = next(
                (
                    statistics.get(candidate)
                    for candidate in candidates
                    if isinstance(statistics.get(candidate), dict)
                ),
                None,
            )
            mean = _finite_number((summary or {}).get("mean"))
            if mean is not None:
                values[column] = mean

    evidence_ids = tuple(
        dict.fromkeys(
            str(row.get("evidence_id"))
            for row in rows
            if row.get("evidence_id")
        )
    )
    evidence_ref = "; ".join(f"evidence_id:{item}" for item in evidence_ids)
    source = "sql_result.rows"
    findings: list[InsightFindingResponse] = []
    ordered_values = [column for column in numeric_columns if column in values]
    if ordered_values:
        summary = "；".join(
            f"{_metric_label(column)}={_format_metric_value(column, values[column])}"
            for column in ordered_values[:6]
        )
        findings.append(
            InsightFindingResponse(
                title="核心指标概览",
                content=summary + "。",
                data_source=source,
                evidence=(
                    f"{evidence_ref}; 基于 {len(rows)} 个"
                    f"{_time_label(time_column) if time_column else '结果'}分组汇总"
                ).strip("; "),
                confidence="high",
                business_impact="形成经营规模、效率和履约表现的统一基线。",
                recommended_action="按相同指标口径持续监控，并对显著偏离整体水平的期间下钻分析。",
            )
        )

    if time_column and primary_metric:
        ranked = sorted(
            (
                (row.get(time_column), number)
                for row in detail_rows
                if row.get(time_column) is not None
                and (number := _finite_number(row.get(primary_metric))) is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            top = "、".join(
                f"{period}（{_format_metric_value(primary_metric, value)}）"
                for period, value in ranked[:3]
            )
            detail = f"{_metric_label(primary_metric)}最高的三个期间为 {top}"
            rate_column = next(
                (column for column in numeric_columns if "rate" in column.casefold()),
                None,
            )
            if rate_column:
                rate_rows = [
                    (row.get(time_column), number)
                    for row in detail_rows
                    if row.get(time_column) is not None
                    and (number := _finite_number(row.get(rate_column))) is not None
                ]
                if rate_rows:
                    period, rate = min(rate_rows, key=lambda item: item[1])
                    detail += (
                        f"；{_metric_label(rate_column)}最低出现在 {period}"
                        f"（{_format_metric_value(rate_column, rate)}）"
                    )
            findings.append(
                InsightFindingResponse(
                    title="月度趋势与异常期间",
                    content=detail + "。",
                    data_source=source,
                    evidence=(f"{evidence_ref}; {time_column} × {primary_metric}").strip("; "),
                    confidence="high",
                    business_impact="峰值与履约低点可用于定位需求、产能或配送压力集中的期间。",
                    recommended_action="优先复盘峰值月份及履约低点的订单结构、库存和配送环节。",
                )
            )
    elif primary_metric:
        dimension = next(
            (
                column
                for column in rows[0]
                if column not in {"evidence_id", "query_index", *numeric_columns}
            ),
            None,
        )
        if dimension:
            ranked = sorted(
                (
                    (row.get(dimension), number)
                    for row in rows
                    if row.get(dimension) is not None
                    and (number := _finite_number(row.get(primary_metric))) is not None
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if ranked:
                top = "、".join(
                    f"{category}（{_format_metric_value(primary_metric, value)}）"
                    for category, value in ranked[:3]
                )
                findings.append(
                    InsightFindingResponse(
                        title="分组表现排名",
                        content=f"按 {_metric_label(primary_metric)} 排名前三的 {dimension} 为 {top}。",
                        data_source=source,
                        evidence=(f"{evidence_ref}; {dimension} × {primary_metric}").strip("; "),
                        confidence="high",
                        business_impact="头部分组贡献可用于确定优先分析和资源投入方向。",
                        recommended_action="复核头部分组的规模、结构和持续性，并与低表现分组进行对照。",
                    )
                )
    return tuple(findings)


def _is_average_metric(column: str) -> bool:
    folded = column.casefold()
    return (
        folded in {"aov", "average_order_value", "avg_order_value"}
        or folded.startswith(("average_", "avg_", "mean_"))
    )


def _coherent_sql_rows(
    sql_result: SQLAnalysisResponse | None,
    question: str,
) -> tuple[dict[str, Any], ...]:
    if sql_result is None or not sql_result.rows:
        return ()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sql_result.rows:
        key = (str(row.get("evidence_id") or ""), str(row.get("query_index") or ""))
        groups.setdefault(key, []).append(row)
    folded_question = question.casefold()

    def score(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
        fields = [field for field in rows[0] if field not in {"evidence_id", "query_index"}]
        matches = sum(field.casefold() in folded_question for field in fields)
        return matches, len(_numeric_columns(tuple(rows))), min(len(rows), 200)

    return tuple(max(groups.values(), key=score))


def _numeric_columns(rows: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    if not rows:
        return ()
    return tuple(
        column
        for column in rows[0]
        if column not in {"evidence_id", "query_index"}
        and any(_finite_number(row.get(column)) is not None for row in rows)
    )


def _time_column(rows: tuple[dict[str, Any], ...]) -> str | None:
    if not rows:
        return None
    return next(
        (
            column
            for column in rows[0]
            if any(
                token in column.casefold()
                for token in ("month", "date", "time", "period", "year", "week", "月份", "日期", "时间")
            )
        ),
        None,
    )


def _primary_metric(columns: tuple[str, ...], question: str) -> str | None:
    folded_question = question.casefold()
    return next((column for column in columns if column.casefold() in folded_question), None) or next(
        (
            column
            for column in columns
            if any(
                token in column.casefold()
                for token in (
                    "gmv",
                    "revenue",
                    "sales",
                    "amount",
                    "price",
                    "total",
                    "sum",
                    "count",
                )
            )
        ),
        columns[0] if columns else None,
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _metric_label(column: str) -> str:
    label = {
        "gmv": "GMV",
        "total_amount": "总金额",
        "order_count": "订单数",
        "aov": "客单价",
        "average_amount": "平均金额",
        "average_order_value": "客单价",
        "avg_order_value": "客单价",
        "on_time_rate": "准时率",
    }.get(column.casefold())
    if label:
        return label
    if column.casefold().startswith("total_"):
        return f"{column[6:]} 总额"
    return column


def _time_label(column: str) -> str:
    return {
        "month": "月份",
        "date": "日期",
        "period": "期间",
        "year": "年份",
        "week": "周",
    }.get(column.casefold(), column)


def _format_metric_value(column: str, value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return str(value)
    folded = column.casefold()
    if "rate" in folded or "ratio" in folded:
        return f"{number:.2%}"
    if "count" in folded and number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}"


def _build_validation_issues(
    *,
    findings: tuple[InsightFindingResponse, ...],
    charts: tuple[ChartResponse, ...],
    extra_issues: tuple[ValidationIssueResponse, ...] = (),
) -> tuple[ValidationIssueResponse, ...]:
    return (
        *validate_findings_traceability(findings),
        *validate_chart_specs(charts),
        *extra_issues,
    )


def _structured_report(
    *,
    question: str,
    profile: DatasetProfileResponse,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
    rounds: tuple[AnalysisRoundResponse, ...],
    final_insights: tuple[InsightFindingResponse, ...],
    validation_issues: tuple[ValidationIssueResponse, ...],
    analysis_framework: AnalysisFrameworkResponse | None = None,
    charts: tuple[ChartResponse, ...] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> StructuredReportResponse:
    analysis_row_count = _analysis_row_count(profile, python_result)
    summary_findings = " ".join(
        dict.fromkeys(finding.content for finding in final_insights[:2])
    )
    report_charts = (
        charts
        if charts is not None
        else python_result.charts
        if python_result
        else ()
    )
    return StructuredReportResponse(
        executive_summary=(
            f"围绕“{question}”，DataMind 基于 {analysis_row_count} 行、"
            f"{profile.column_count} 列数据完成分析。"
            f"{summary_findings or '已完成基础数据分析。'}"
        ),
        analysis_context=(
            analysis_framework.success_criteria
            if analysis_framework
            else "基于上传数据集的可追溯分析证据生成。"
        ),
        key_findings=final_insights,
        charts=report_charts,
        chart_explanations=tuple(chart.explanation for chart in report_charts if chart.explanation),
        sql_results=sql_result.rows if sql_result else (),
        python_results=python_result.statistics if python_result else {},
        data_gaps=_data_gaps(profile),
        validation_issues=validation_issues,
        recommended_next_steps=tuple(
            dict.fromkeys(
                finding.recommended_action
                for finding in final_insights
                if finding.recommended_action
            )
        )[:5],
        analysis_trace=rounds,
        provider=provider,
        model=model,
    )


def _analysis_row_count(
    profile: DatasetProfileResponse,
    python_result: PythonAnalysisResponse | None,
) -> int:
    value = python_result.statistics.get("rows") if python_result else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    return profile.row_count


def _chart_explanation(chart: ChartResponse) -> str:
    if chart.chart_type == "bar":
        return f"{chart.title} 用于比较不同类别的指标高低。"
    if chart.chart_type == "line":
        return f"{chart.title} 用于观察指标随时间或顺序的变化趋势。"
    if chart.chart_type == "pie":
        return f"{chart.title} 用于观察前几类在整体结果中的占比。"
    if chart.chart_type == "histogram":
        return f"{chart.title} 用于判断数值分布、集中区间和异常尾部。"
    if chart.chart_type == "box_plot":
        return f"{chart.title} 用于比较分类下的中位数、离散程度和异常值。"
    if chart.chart_type == "correlation_heatmap":
        return f"{chart.title} 用于查看数值字段之间的相关性强弱。"
    return f"{chart.title} 用于辅助解释分析结果。"


def render_structured_report_html(
    report: StructuredReportResponse,
    *,
    title: str = "DataMind 分析报告",
) -> str:
    findings_html = "".join(
        (
            "<section class='card'>"
            f"<h2>{escape(finding.title)}</h2>"
            f"<p>{escape(finding.content)}</p>"
            f"<small>来源: {escape(finding.data_source)}</small>"
            "</section>"
        )
        for finding in report.key_findings
    )
    charts_html = "".join(_chart_html(chart) for chart in report.charts)
    sql_html = _rows_table_html(report.sql_results)
    gaps_html = "".join(f"<li>{escape(gap)}</li>" for gap in report.data_gaps)
    issues_html = "".join(
        (
            "<li>"
            f"<strong>{escape(issue.severity)}</strong> · "
            f"{escape(issue.finding_ref)}: {escape(issue.issue)}"
            "</li>"
        )
        for issue in report.validation_issues
    )
    trace_html = "".join(
        (
            "<section class='trace-item'>"
            f"<b>Round {round_item.round_number}</b>"
            f"<p>{escape(round_item.hypothesis.statement)}</p>"
            f"<small>{escape(round_item.reflection.insight_text)}</small>"
            "</section>"
        )
        for round_item in report.analysis_trace
    )
    next_steps_html = "".join(f"<li>{escape(step)}</li>" for step in report.recommended_next_steps)
    contract_html = ""
    if report.analysis_contract:
        contract = report.analysis_contract
        contract_html = (
            "<section class='card contract-grid'>"
            f"<div><small>目标</small><p>{escape(contract.objective)}</p></div>"
            f"<div><small>总体</small><p>{escape(contract.population)}</p></div>"
            f"<div><small>指标</small><p>{escape(contract.metric or '计数/文本分析')}</p></div>"
            f"<div><small>粒度</small><p>{escape(', '.join(contract.grain))}</p></div>"
            f"<div class='wide'><small>方法</small><p>{escape(contract.method)}</p></div>"
            "</section>"
        )
    verification_html = ""
    if report.statistical_verification:
        verification = report.statistical_verification
        checks = "".join(
            (
                f"<li class='check-{escape(check.status)}'>"
                f"<strong>{escape(check.status)}</strong> · "
                f"{escape(check.message)}</li>"
            )
            for check in verification.checks
            if check.status != "not_applicable"
        )
        verification_html = (
            f"<section class='card verification verification-{escape(verification.status)}'>"
            f"<h2>{escape(verification.summary)}</h2>"
            f"<p>数值证据覆盖率: {verification.numeric_evidence_coverage:.0%}</p>"
            f"<ul>{checks}</ul>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin:0; background:#f3f6fb; color:#111827; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    main {{ max-width:1080px; margin:0 auto; padding:40px 28px 64px; }}
    header {{ margin-bottom:28px; }}
    h1 {{ margin:0 0 10px; font-size:34px; }}
    h2 {{ margin:0 0 12px; font-size:20px; }}
    h3 {{ margin:28px 0 14px; font-size:22px; }}
    p {{ line-height:1.75; }}
    .summary {{ padding:24px 28px; background:#cfe6fb; border-radius:8px; font-size:17px; }}
    .card {{ background:#fff; border:1px solid #dbe5f0; border-radius:8px; padding:20px; margin:14px 0; }}
    .card small, .trace-item small {{ color:#4b5563; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid #e5edf5; text-align:left; font-size:14px; }}
    th {{ background:#111827; color:#fff; }}
    .trace-item {{ border-left:4px solid #0f766e; background:#fff; padding:14px 16px; margin:10px 0; border-radius:6px; }}
    .contract-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px 24px; }}
    .contract-grid p {{ margin:4px 0 0; }}
    .contract-grid .wide {{ grid-column:1 / -1; }}
    .verification-passed {{ border-left:4px solid #059669; }}
    .verification-warning {{ border-left:4px solid #d97706; }}
    .verification-failed {{ border-left:4px solid #dc2626; }}
    .check-failed {{ color:#991b1b; }}
    .check-warning {{ color:#92400e; }}
    .chart-grid {{ display:grid; grid-template-columns:280px 1fr; gap:20px; align-items:center; }}
    .legend {{ display:grid; gap:8px; font-size:14px; }}
    details {{ margin-top:12px; }}
    summary {{ cursor:pointer; font-weight:700; color:#475569; }}
    .muted {{ color:#64748b; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escape(title)}</h1>
      <div class="muted">Provider: {escape(report.provider or "-")} · Model: {escape(report.model or "-")}</div>
    </header>
    <section class="summary">{escape(report.executive_summary)}</section>
    <h3>Analysis Context</h3>
    <section class="card">{escape(report.analysis_context or "基于当前数据集完成分析。")}</section>
    <h3>Analysis Contract</h3>
    {contract_html or "<p class='muted'>暂无分析契约。</p>"}
    <h3>Statistical Verification</h3>
    {verification_html or "<p class='muted'>暂无统计审查结果。</p>"}
    <h3>Key Findings</h3>
    {findings_html or "<p class='muted'>暂无核心发现。</p>"}
    <h3>Visualizations</h3>
    {charts_html or "<p class='muted'>暂无图表。</p>"}
    <h3>SQL Results</h3>
    {sql_html or "<p class='muted'>暂无 SQL 结果。</p>"}
    <h3>Data Gaps</h3>
    <ul>{gaps_html or "<li>暂无数据盲点。</li>"}</ul>
    <h3>Validation Issues</h3>
    <ul>{issues_html or "<li>未发现阻断性校验问题。</li>"}</ul>
    <h3>Recommended Next Steps</h3>
    <ul>{next_steps_html or "<li>暂无后续建议。</li>"}</ul>
    <h3>Analysis Trace</h3>
    {trace_html or "<p class='muted'>暂无分析轨迹。</p>"}
  </main>
</body>
</html>"""


def _chart_html(chart: ChartResponse) -> str:
    chart_svg = _chart_svg_html(chart)
    rows_html = _rows_table_html(chart.data[:20])
    return (
        "<section class='card'>"
        f"<h2>{escape(chart.title)}</h2>"
        f"<p class='muted'>类型: {escape(chart.chart_type)}</p>"
        f"{chart_svg or '<p class=\"muted\">暂无可绘制图表。</p>'}"
        f"<p>{escape(chart.explanation)}</p>"
        "<details><summary>查看图表数据</summary>"
        f"{rows_html or '<p class=\"muted\">暂无图表数据。</p>'}"
        "</details>"
        "</section>"
    )


def _chart_svg_html(chart: ChartResponse) -> str:
    if not chart.data:
        return ""
    if chart.chart_type == "pie":
        return _pie_svg_html(chart)
    if chart.chart_type == "line":
        return _cartesian_svg_html(chart, mode="line")
    if chart.chart_type == "histogram":
        return _histogram_svg_html(chart)
    if chart.chart_type == "box_plot":
        return _box_svg_html(chart)
    if chart.chart_type == "correlation_heatmap":
        return _heatmap_svg_html(chart)
    return _cartesian_svg_html(chart, mode="bar")


def _cartesian_svg_html(chart: ChartResponse, *, mode: str) -> str:
    first_row = chart.data[0]
    keys = list(first_row.keys())
    x_key = str(chart.spec.get("x") or keys[0])
    y_key = str(chart.spec.get("y") or keys[-1])
    points = [
        (str(row.get(x_key, "")), _as_number(row.get(y_key)))
        for row in chart.data[:24]
    ]
    points = [(label, value) for label, value in points if value is not None]
    if not points:
        return ""
    width, height, pad = 720, 280, 58
    max_value = max(1, *(value for _, value in points))
    inner_width = width - pad * 2
    inner_height = height - pad * 2
    step = inner_width / max(len(points), 1)
    shapes: list[str] = []
    for tick in _axis_ticks(max_value):
        y = height - pad - (tick / max_value) * inner_height
        shapes.append(f"<line x1='{pad}' y1='{y:.2f}' x2='{width-pad}' y2='{y:.2f}' stroke='#e2e8f0'/>")
        shapes.append(f"<text x='{pad-10}' y='{y+4:.2f}' text-anchor='end' font-size='11' fill='#64748b'>{escape(_format_axis_value(tick))}</text>")
    shapes.extend(
        [
            f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#cbd5e1'/>",
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#cbd5e1'/>",
        ]
    )
    if mode == "line":
        poly_points = []
        for index, (_, value) in enumerate(points):
            x = pad + step * index + step / 2
            y = height - pad - (value / max_value) * inner_height
            poly_points.append(f"{x:.2f},{y:.2f}")
        shapes.append(
            f"<polyline points='{' '.join(poly_points)}' fill='none' stroke='#2563eb' "
            "stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/>"
        )
        for index, (_, value) in enumerate(points):
            x = pad + step * index + step / 2
            y = height - pad - (value / max_value) * inner_height
            shapes.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4' fill='#2563eb'/>")
    else:
        for index, (_, value) in enumerate(points):
            bar_width = max(step * 0.58, 6)
            bar_height = (value / max_value) * inner_height
            x = pad + step * index + (step - bar_width) / 2
            y = height - pad - bar_height
            shapes.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' "
                f"height='{bar_height:.2f}' rx='5' fill='#0f766e'/>"
            )
    for index, (label, _) in enumerate(points):
        x = pad + step * index + step / 2
        shapes.append(
            f"<text x='{x:.2f}' y='{height-14}' text-anchor='middle' "
            f"font-size='11' fill='#475569'>{escape(_short_label(label))}</text>"
        )
    shapes.append(f"<text x='{pad}' y='24' font-size='12' fill='#64748b'>{escape(y_key)}</text>")
    return _svg_container(width, height, "".join(shapes))


def _pie_svg_html(chart: ChartResponse) -> str:
    first_row = chart.data[0]
    keys = list(first_row.keys())
    name_key = str(chart.spec.get("names") or keys[0])
    value_key = str(chart.spec.get("values") or keys[-1])
    slices = [
        (str(row.get(name_key, "")), max(_as_number(row.get(value_key)) or 0, 0))
        for row in chart.data[:8]
    ]
    slices = [(label, value) for label, value in slices if value > 0]
    total = sum(value for _, value in slices)
    if total <= 0:
        return ""
    colors = ["#0f766e", "#2563eb", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#16a34a", "#db2777"]
    current = 0.0
    paths: list[str] = []
    legend: list[str] = []
    for index, (label, value) in enumerate(slices):
        start = current
        end = current + (value / total) * pi * 2
        current = end
        color = colors[index % len(colors)]
        paths.append(f"<path d='{_arc_path(110, 110, 82, start, end)}' fill='{color}'/>")
        legend.append(
            f"<div><span style='display:inline-block;width:11px;height:11px;background:{color};"
            f"margin-right:8px'></span>{escape(label)} <b>{value / total * 100:.1f}%</b></div>"
        )
    svg = _svg_container(220, 220, "".join(paths) + "<circle cx='110' cy='110' r='44' fill='white'/>")
    return f"<div class='chart-grid'>{svg}<div class='legend'>{''.join(legend)}</div></div>"


def _histogram_svg_html(chart: ChartResponse) -> str:
    if chart.spec.get("y"):
        return _cartesian_svg_html(chart, mode="bar")
    first_row = chart.data[0]
    x_key = str(chart.spec.get("x") or next(iter(first_row)))
    values = [_as_number(row.get(x_key)) for row in chart.data]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    minimum = min(values)
    maximum = max(values)
    bucket_count = 10
    buckets = [{"label": str(index + 1), "value": 0} for index in range(bucket_count)]
    for value in values:
        ratio = 0 if maximum == minimum else (value - minimum) / (maximum - minimum)
        buckets[min(bucket_count - 1, int(ratio * bucket_count))]["value"] += 1
    return _cartesian_svg_html(
        ChartResponse(
            title=chart.title,
            chart_type="bar",
            spec={"x": "label", "y": "value"},
            data=tuple(buckets),
        ),
        mode="bar",
    )


def _box_svg_html(chart: ChartResponse) -> str:
    first_row = chart.data[0]
    keys = list(first_row.keys())
    x_key = str(chart.spec.get("x") or keys[0])
    y_key = str(chart.spec.get("y") or keys[-1])
    groups: dict[str, list[float]] = {}
    for row in chart.data:
        value = _as_number(row.get(y_key))
        if value is None:
            continue
        groups.setdefault(str(row.get(x_key, "未分组")), []).append(value)
    summaries = [
        (label, _quartiles(values))
        for label, values in islice(groups.items(), 8)
        if values
    ]
    if not summaries:
        return ""
    width, height, pad = 720, 280, 58
    max_value = max(summary["max"] for _, summary in summaries) or 1
    step = (width - pad * 2) / len(summaries)

    def sy(value: float) -> float:
        return height - pad - (value / max_value) * (height - pad * 2)

    shapes = []
    for tick in _axis_ticks(max_value):
        y = sy(tick)
        shapes.append(f"<line x1='{pad}' y1='{y:.2f}' x2='{width-pad}' y2='{y:.2f}' stroke='#e2e8f0'/>")
        shapes.append(f"<text x='{pad-10}' y='{y+4:.2f}' text-anchor='end' font-size='11' fill='#64748b'>{escape(_format_axis_value(tick))}</text>")
    shapes.extend(
        [
            f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#cbd5e1'/>",
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#cbd5e1'/>",
        ]
    )
    for index, (label, summary) in enumerate(summaries):
        x = pad + step * index + step / 2
        q3_y = sy(summary["q3"])
        q1_y = sy(summary["q1"])
        shapes.append(f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{sy(summary['min']):.2f}' y2='{sy(summary['max']):.2f}' stroke='#0f766e' stroke-width='3'/>")
        shapes.append(f"<rect x='{x-20:.2f}' y='{q3_y:.2f}' width='40' height='{max(q1_y-q3_y, 2):.2f}' fill='#99f6e4' stroke='#0f766e'/>")
        shapes.append(f"<line x1='{x-24:.2f}' x2='{x+24:.2f}' y1='{sy(summary['median']):.2f}' y2='{sy(summary['median']):.2f}' stroke='#111827' stroke-width='3'/>")
        shapes.append(f"<text x='{x:.2f}' y='{height-14}' text-anchor='middle' font-size='11' fill='#475569'>{escape(_short_label(label))}</text>")
    return _svg_container(width, height, "".join(shapes))


def _heatmap_svg_html(chart: ChartResponse) -> str:
    labels = []
    for row in chart.data:
        for key in ("source", "target"):
            label = str(row.get(key, ""))
            if label and label not in labels:
                labels.append(label)
    labels = labels[:10]
    if not labels:
        return ""
    cell, pad = 44, 90
    size = pad + len(labels) * cell + 16
    value_map = {
        (str(row.get("source", "")), str(row.get("target", ""))): _as_number(row.get("value")) or 0
        for row in chart.data
    }
    shapes: list[str] = []
    for index, label in enumerate(labels):
        shapes.append(f"<text x='{pad + index * cell + cell / 2}' y='{pad - 12}' text-anchor='middle' font-size='10' fill='#475569'>{escape(_short_label(label))}</text>")
        shapes.append(f"<text x='{pad - 10}' y='{pad + index * cell + cell / 2 + 4}' text-anchor='end' font-size='10' fill='#475569'>{escape(_short_label(label))}</text>")
    for y_index, source in enumerate(labels):
        for x_index, target in enumerate(labels):
            value = value_map.get((source, target), 0)
            intensity = min(abs(value), 1)
            color = f"rgba(15, 118, 110, {0.12 + intensity * 0.78:.2f})" if value >= 0 else f"rgba(220, 38, 38, {0.12 + intensity * 0.78:.2f})"
            shapes.append(f"<rect x='{pad + x_index * cell}' y='{pad + y_index * cell}' width='{cell-2}' height='{cell-2}' fill='{color}'/>")
    return _svg_container(size, size, "".join(shapes))


def _svg_container(width: int, height: int, body: str) -> str:
    return (
        f"<svg viewBox='0 0 {width} {height}' style='width:100%;max-height:360px;"
        "background:#f8fafc;border-radius:8px;margin:12px 0'>"
        f"{body}</svg>"
    )


def _arc_path(cx: int, cy: int, radius: int, start: float, end: float) -> str:
    start_x = cx + radius * cos(start)
    start_y = cy + radius * sin(start)
    end_x = cx + radius * cos(end)
    end_y = cy + radius * sin(end)
    large_arc = 1 if end - start > pi else 0
    return f"M {cx} {cy} L {start_x:.2f} {start_y:.2f} A {radius} {radius} 0 {large_arc} 1 {end_x:.2f} {end_y:.2f} Z"


def _as_number(value: Any) -> float | None:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if isnan(number):
        return None
    return number


def _short_label(value: str) -> str:
    return value if len(value) <= 10 else f"{value[:9]}..."


def _axis_ticks(max_value: float) -> list[float]:
    upper = max(max_value, 1)
    return [upper * ratio for ratio in (0, 0.25, 0.5, 0.75, 1)]


def _format_axis_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def _quartiles(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "q1": _percentile(sorted_values, 0.25),
        "median": _percentile(sorted_values, 0.5),
        "q3": _percentile(sorted_values, 0.75),
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _rows_table_html(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _analytical_step_for_round(index: int, plan: PlannedAnalysis) -> str:
    if index == 1 and plan.time_column:
        return "trend_analysis"
    if index == 1 and plan.category_column:
        return "decomposition"
    if index == 2:
        return "attribution"
    return "risk_mining"


def _data_gaps(profile: DatasetProfileResponse) -> tuple[str, ...]:
    gaps: list[str] = []
    if not profile.numeric_columns:
        gaps.append("缺少数值列，无法进行金额、利润或趋势类量化分析。")
    if not profile.categorical_columns:
        gaps.append("缺少分类列，无法进行区域、品类或客群拆解。")
    if profile.missing_value_count:
        gaps.append(f"存在 {profile.missing_value_count} 个缺失值，结论需结合清洗规则复核。")
    return tuple(gaps)


def _dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    dataframe = pd.DataFrame.from_records(records).replace({pd.NA: None})
    # csv.DictReader preserves empty cells as strings. Treat empty and
    # whitespace-only cells as missing consistently with JSON/XLSX nulls.
    return dataframe.replace(r"^\s*$", pd.NA, regex=True)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.where(pd.notna(df), None)
    return [_jsonable(record) for record in clean.to_dict(orient="records")]


def _histogram_buckets(values: list[float], bucket_count: int = 10) -> list[dict[str, Any]]:
    clean_values = [float(value) for value in values if pd.notna(value)]
    if not clean_values:
        return []
    minimum = min(clean_values)
    maximum = max(clean_values)
    if minimum == maximum:
        return [{"label": _format_number(minimum), "value": len(clean_values)}]
    width = (maximum - minimum) / bucket_count
    buckets = [
        {
            "label": f"{_format_number(minimum + width * index)}-{_format_number(minimum + width * (index + 1))}",
            "value": 0,
        }
        for index in range(bucket_count)
    ]
    for value in clean_values:
        index = min(bucket_count - 1, int((value - minimum) / width))
        buckets[index]["value"] += 1
    return buckets


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def _pick_metric_column(text: str, numeric_columns: list[str]) -> str | None:
    if not numeric_columns:
        return None
    for column in numeric_columns:
        lowered = column.lower()
        if lowered in text or any(
            token in lowered
            for token in ("sales", "revenue", "profit", "amount", "销售", "收入", "利润")
        ):
            return column
    return numeric_columns[0]


def _pick_category_column(text: str, categorical_columns: list[str]) -> str | None:
    if not categorical_columns:
        return None
    for column in categorical_columns:
        lowered = column.lower()
        if lowered in text or any(
            token in lowered for token in ("region", "product", "category", "区域", "产品", "类别")
        ):
            return column
    return next(
        (
            column
            for column in categorical_columns
            if not _looks_like_identifier_column(column)
        ),
        None,
    )


def _pick_time_column(profile: DatasetProfileResponse) -> str | None:
    for column in [*profile.categorical_columns, *profile.numeric_columns]:
        lowered = column.lower()
        if any(token in lowered for token in ("date", "month", "time", "日期", "月份", "时间")):
            return column
    return None


def _first_numeric(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if _is_metric_column(df, str(column)):
            return str(column)
    return None


def _first_categorical(df: pd.DataFrame, *, exclude: set[str] | None = None) -> str | None:
    excluded = exclude or set()
    for column in df.columns:
        if str(column) not in excluded and not _is_metric_column(df, str(column)):
            return str(column)
    return next((str(column) for column in df.columns if str(column) not in excluded), None)


def _safe_metric_column(df: pd.DataFrame, candidate: str | None) -> str | None:
    if candidate and candidate in df.columns and _is_metric_column(df, candidate):
        return candidate
    return _first_numeric(df)


def _is_metric_column(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns or _looks_like_identifier_column(column):
        return False
    series = _column_series(df, column)
    non_missing = series.notna().sum()
    if not non_missing:
        return False
    numeric_series = pd.to_numeric(series, errors="coerce")
    return bool(numeric_series.notna().sum() == non_missing)


def _column_series(df: pd.DataFrame, column: str) -> pd.Series:
    selected = df.loc[:, column]
    if isinstance(selected, pd.DataFrame):
        return selected.iloc[:, 0]
    return selected


def _looks_like_identifier_column(column: str) -> bool:
    lowered = column.lower().strip()
    normalized = lowered.replace("-", "_").replace(" ", "_")
    identifier_tokens = ("id", "uuid", "guid", "hash", "code", "编号", "编码", "工号", "学号")
    if normalized in identifier_tokens:
        return True
    return normalized.endswith("_id") or any(
        token in normalized for token in ("customer_id", "user_id", "order_id", "product_id")
    )


def _alias(column: str) -> str:
    return (
        "".join(char.lower() if char.isalnum() else "_" for char in column).strip("_")
        or "metric"
    )


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if isnan(number) else number


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and isnan(value):
        return None
    return value
