from __future__ import annotations

import re
from uuid import UUID

from app.analysis.query_intent import (
    infer_query_intent,
    infer_source_aggregations,
    strip_negated_clauses,
)
from app.analysis.services import PlannedAnalysis
from app.core.settings import get_settings
from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisContractResponse,
    DatasetProfileResponse,
    MultiDatasetProfileResponse,
    PlannerMetadataResponse,
)


def build_analysis_contract(
    *,
    question: str,
    dataset_id: UUID,
    additional_dataset_ids: tuple[UUID, ...],
    profile: DatasetProfileResponse,
    plan: PlannedAnalysis,
    planner_metadata: PlannerMetadataResponse,
    multi_dataset_context: MultiDatasetProfileResponse | None,
    analysis_row_count: int | None = None,
) -> AnalysisContractResponse:
    settings = get_settings()
    analysis_type = _analysis_type(question, plan)
    intent = infer_query_intent(question, profile)
    dimensions = (
        intent.required_dimensions
        or plan.requested_dimensions
    )
    source_aggregations = (
        infer_source_aggregations(
            question,
            tuple(
                (dataset.name, dataset.columns)
                for dataset in (
                    multi_dataset_context.primary_dataset,
                    *multi_dataset_context.additional_datasets,
                )
            ),
        )
        if multi_dataset_context is not None
        else ()
    )
    aggregations = _merge_aggregations(
        (*source_aggregations, *intent.aggregations, *plan.aggregations)
    )
    filters = intent.filters or plan.filters
    grain = tuple(
        dict.fromkeys(
            value for value in (*dimensions, plan.time_column) if value
        )
    ) or ("dataset",)
    assumptions = [
        "上传记录可代表本次问题所指的数据范围。",
        "字段类型与角色使用已发布语义模型或当前数据画像。",
        "观察性数据默认只能支持描述和相关性结论，不能单独证明因果。",
    ]
    if multi_dataset_context is not None:
        assumptions.append(
            "跨表指标必须保持来源事实表粒度；存在行膨胀时先聚合再连接。"
        )
    if planner_metadata.requires_confirmation:
        assumptions.append("低置信度语义计划已由用户明确确认后执行。")
    row_count = (
        profile.row_count
        if analysis_row_count is None
        else max(0, int(analysis_row_count))
    )
    aggregation_metrics = tuple(
        dict.fromkeys(item.column for item in aggregations if item.column)
    )
    return AnalysisContractResponse(
        objective=question.strip(),
        population=(
            f"当前分析数据范围内的 {row_count} 行记录，"
            f"共 {profile.column_count} 个字段。"
        ),
        dataset_ids=tuple(
            dict.fromkeys((dataset_id, *additional_dataset_ids))
        ),
        analysis_type=analysis_type,
        metric=(
            aggregation_metrics[0]
            if len(aggregation_metrics) == 1
            else plan.metric_column
            if not aggregation_metrics
            else None
        ),
        dimensions=dimensions,
        time_field=plan.time_column,
        aggregations=aggregations,
        filters=filters,
        grain=grain,
        hypothesis=_hypothesis(question, analysis_type),
        method=_method(analysis_type, plan),
        assumptions=tuple(assumptions),
        acceptance_criteria=(
            "每个数值结论必须引用可读取的 evidence_id 或确定性执行结果。",
            "比较型结论必须披露样本量，并提供效应量或置信区间。",
            "错误数据粒度或未控制的 Join 行膨胀不得用于指标聚合。",
            "SQL/Python 结果必须覆盖用户明确要求的维度、过滤条件和聚合指标。",
            "观察性数据中的因果措辞必须降级为相关性表述。",
        ),
        stop_conditions=(
            "证据已覆盖用户问题且通过统计与数据粒度审查。",
            "达到工具调用、决策、时间或 token 预算。",
            "现有数据无法支持目标时明确报告数据缺口。",
        ),
        causal_claim_allowed=False,
        analysis_budget={
            "max_tool_calls": settings.agent_loop_max_tool_calls,
            "max_decisions": settings.agent_loop_max_decisions,
            "timeout_seconds": settings.agent_loop_timeout_seconds,
            "max_tokens": settings.agent_loop_max_tokens,
        },
    )


def _merge_aggregations(
    aggregations: tuple[AnalysisAggregationResponse, ...],
) -> tuple[AnalysisAggregationResponse, ...]:
    merged: list[AnalysisAggregationResponse] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for item in aggregations:
        source, column = _contract_column_parts(item.column)
        key = (
            item.operation,
            source,
            column,
        )
        if source is None and any(
            operation == item.operation and existing_column == column
            for operation, _, existing_column in seen
        ):
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return tuple(merged)


def _contract_column_parts(column: str | None) -> tuple[str | None, str | None]:
    if not column:
        return None, None
    if "__" not in column:
        return None, column.casefold()
    source, leaf = column.rsplit("__", 1)
    normalized_source = re.sub(
        r"(?:dataset)?(?:csv)?$", "", re.sub(r"[^a-z0-9]+", "", source.casefold())
    )
    return normalized_source or None, leaf.casefold()


def _analysis_type(question: str, plan: PlannedAnalysis) -> str:
    folded = strip_negated_clauses(question).casefold()
    if plan.time_column or any(
        token in folded for token in ("趋势", "同比", "环比", "trend", "over time")
    ):
        return "trend"
    if any(
        token in folded
        for token in ("相关", "关系", "影响", "correlation", "relationship")
    ):
        return "association"
    if any(
        token in folded
        for token in ("比较", "差异", "高于", "低于", "compare", "difference")
    ):
        return "comparison"
    if any(
        token in folded
        for token in ("分布", "异常", "histogram", "distribution", "outlier")
    ):
        return "distribution"
    if any(
        token in folded
        for token in ("文本", "评论", "关键词", "text", "review", "keyword")
    ):
        return "text"
    return "descriptive"


def _hypothesis(question: str, analysis_type: str) -> str:
    if analysis_type == "descriptive":
        return "探索性描述：识别与用户问题最相关的可验证模式。"
    return f"检验数据是否支持以下分析目标：{question.strip()}"


def _method(analysis_type: str, plan: PlannedAnalysis) -> str:
    methods = {
        "trend": "按时间粒度聚合，并检查样本覆盖与变化区间。",
        "association": "计算分组或变量关联，仅报告观察性相关关系。",
        "comparison": "执行分组比较，并报告样本量、效应量或置信区间。",
        "distribution": "使用稳健描述统计、分箱和异常值检查。",
        "text": "执行文本长度、关键词与分组画像分析。",
        "descriptive": "使用确定性 SQL/Python 汇总回答业务问题。",
    }
    route = plan.route.upper()
    return f"{methods[analysis_type]} 执行路线：{route}。"
