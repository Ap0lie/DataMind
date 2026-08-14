from __future__ import annotations

import json
from itertools import islice
from typing import Any

from app.analysis.prompt_utils import compact_prompt_columns, compact_prompt_records
from app.analysis.services import PlannedAnalysis
from app.analysis.workflow_prompt_context import (
    compact_multi_dataset_context,
    experience_context,
    prompt_system,
)
from app.schemas.analysis import (
    AnalysisContractResponse,
    AnalysisRoundResponse,
    ChartResponse,
    DatasetProfileResponse,
    InsightFindingResponse,
    MultiDatasetProfileResponse,
    MultimodalInputResponse,
    PythonAnalysisResponse,
    SQLAnalysisResponse,
    StatisticalVerificationResponse,
    StructuredReportResponse,
)


def framework_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
                "distinct_count": column.distinct_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=5),
        "multi_dataset_context": compact_multi_dataset_context(multi_dataset_context),
        "experience_context": experience_context(
            "framework", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind design_framework. Return only JSON with keys: "
                "business_question, candidate_dimensions, candidate_metrics, likely_routes, "
                "initial_hypotheses, risk_notes, key_questions, success_criteria. "
                "Return at most 3 initial_hypotheses and at most 6 items in every other list. "
                "Use only provided column names. Use experience_context only for business "
                "priority, risk framing, and analysis style; never invent evidence from it."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
def round_plan_messages(
    *,
    question: str,
    hypothesis: str,
    profile: DatasetProfileResponse,
    previous_rounds: tuple[AnalysisRoundResponse, ...],
    fallback_plan: PlannedAnalysis,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "hypothesis": truncate_text(hypothesis, 1200),
        "fallback_plan": {
            "route": fallback_plan.route,
            "metric_column": fallback_plan.metric_column,
            "category_column": fallback_plan.category_column,
            "time_column": fallback_plan.time_column,
            "steps": list(fallback_plan.steps),
        },
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "previous_rounds": [
            {
                "round_number": round_item.round_number,
                "hypothesis": round_item.hypothesis.statement,
                "route": round_item.plan.route,
                "reflection": round_item.reflection.insight_text,
            }
            for round_item in previous_rounds
        ],
        "experience_context": experience_context(
            "round_plan", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind generate_plan inside an iterative analysis loop. "
                "Return only compact JSON with keys: route, category_column, metric_column, "
                "time_column, steps. route must be sql, python, or hybrid. Use only provided "
                "column names and avoid repeating previous rounds. Use experience_context to "
                "prioritize useful checks, but keep every step executable against the dataset. "
                "For joined data, respect column_source_map and row-expansion warnings before choosing metrics."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

def reflection_messages(
    *,
    question: str,
    rounds: tuple[AnalysisRoundResponse, ...],
    python_result: PythonAnalysisResponse,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "rounds": [round_item.model_dump(mode="json") for round_item in rounds],
        "python_insights": list(python_result.insights),
        "experience_context": experience_context("reflection"),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind reflect_on_result. Return only JSON with key reflections, "
                "an array of concise Chinese reflection strings aligned to the rounds. Use "
                "experience_context for review criteria, not as evidence."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def integrate_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    rounds: tuple[AnalysisRoundResponse, ...],
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "profile": compact_profile(profile),
        "rounds": [compact_round(round_item) for round_item in rounds],
        "sql_result": compact_sql_result(sql_result),
        "python_statistics": compact_python_statistics(
            python_result.statistics if python_result else {}
        ),
        "python_insights": list(python_result.insights) if python_result else [],
        "charts": [compact_chart(chart) for chart in python_result.charts]
        if python_result
        else [],
        "multimodal_context": multimodal_payload(multimodal_inputs),
        "multi_dataset_context": compact_multi_dataset_context(multi_dataset_context),
        "experience_context": experience_context(
            "integrate", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind integrate_insights. Return only JSON with key insights. "
                "Each insight must include title, content, data_source, evidence, confidence, "
                "business_impact, recommended_action, and impact_pct. Do not invent evidence. "
                "Return at most 8 insights. "
                "Multimodal context may guide interpretation but dataset-backed claims must cite "
                "SQL, Python, chart, or round evidence. Use experience_context to rank insight "
                "importance and choose report language."
            ),
        },
        {
            "role": "user",
            "content": multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def review_messages(
    *,
    question: str,
    final_insights: tuple[InsightFindingResponse, ...],
    charts: tuple[ChartResponse, ...],
    sql_result: SQLAnalysisResponse | None,
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
    analysis_contract: AnalysisContractResponse | None = None,
    statistical_verification: StatisticalVerificationResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "final_insights": [finding.model_dump(mode="json") for finding in final_insights],
        "charts": [compact_chart(chart) for chart in charts],
        "sql_result": compact_sql_result(sql_result),
        "multimodal_context": multimodal_payload(multimodal_inputs),
        "multi_dataset_context": compact_multi_dataset_context(multi_dataset_context),
        "analysis_contract": (
            analysis_contract.model_dump(mode="json") if analysis_contract else None
        ),
        "statistical_verification": (
            statistical_verification.model_dump(mode="json")
            if statistical_verification
            else None
        ),
        "experience_context": experience_context("review"),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind adversarial_validate. Return only JSON with key issues. "
                "Each issue has severity, finding_ref, issue, suggestion. Return at most 10 issues. "
                "Flag unsupported claims, chart/text mismatch, over-attribution, data gaps, hallucinated fields, "
                "unqualified causal language, missing comparison support, unsafe analysis grain, "
                "and any misuse of multimodal context as if it were tabular evidence. Apply "
                "experience_context as additional review criteria."
            ),
        },
        {
            "role": "user",
            "content": multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def chart_refine_messages(
    *,
    question: str,
    charts: tuple[ChartResponse, ...],
    final_insights: tuple[InsightFindingResponse, ...],
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "charts": [compact_chart(chart) for chart in charts],
        "final_insights": [finding.model_dump(mode="json") for finding in final_insights],
        "multimodal_context": multimodal_payload(multimodal_inputs),
        "multi_dataset_context": compact_multi_dataset_context(multi_dataset_context),
        "experience_context": experience_context("chart_refine"),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind chart_refine. Return only JSON with key chart_explanations. "
                "Each item has title and explanation. Explain what the chart proves and the "
                "business reading. Do not invent numbers. Use experience_context for chart "
                "communication style and priority."
            ),
        },
        {
            "role": "user",
            "content": multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def report_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    structured_report: StructuredReportResponse,
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": truncate_text(question, 2000),
        "fallback_structured_report": compact_structured_report(structured_report),
        "multimodal_context": multimodal_payload(multimodal_inputs),
        "multi_dataset_context": compact_multi_dataset_context(multi_dataset_context),
        "experience_context": experience_context(
            "report", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind generate_structured_report. Return only JSON, no Markdown. "
                "Generate a high-quality Chinese structured data analysis report. Required keys: "
                "executive_summary, analysis_context, key_findings, chart_explanations, "
                "data_gaps, validation_issues, recommended_next_steps. key_findings must be an "
                "array with title, content, data_source, evidence, confidence, business_impact, "
                "recommended_action, impact_pct. Do not invent columns, rows, or metrics beyond "
                "the payload. Keep all dataset claims traceable to SQL, Python, chart, or round "
                "evidence. Findings whose data_source starts with sql_result or tool_evidence are mandatory: "
                "preserve their verified KPIs, trend periods, source-table totals, relationship direction, "
                "fact grain, cardinality risk, prevention method, and evidence_id in both the executive "
                "summary and key findings. "
                "Respect analysis_contract and statistical_verification. Do not restore findings whose "
                "verdict failed. Comparison claims must retain sample size plus effect size or confidence "
                "interval. Describe observational associations without causal language. "
                "Multimodal context can enrich explanation and data-gap notes but must "
                "not be treated as verified tabular data unless the payload provides matching evidence. "
                "Use experience_context to shape narrative quality, prioritization, and review discipline."
            ),
        },
        {
            "role": "user",
            "content": multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]



def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated]"


def compact_profile(profile: DatasetProfileResponse) -> dict[str, Any]:
    return {
        "dataset_id": str(profile.dataset_id),
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "missing_value_count": profile.missing_value_count,
        "missing_value_ratio": profile.missing_value_ratio,
        "duplicate_row_count": profile.duplicate_row_count,
        "numeric_columns": list(profile.numeric_columns[:20]),
        "categorical_columns": list(profile.categorical_columns[:20]),
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
                "distinct_count": column.distinct_count,
                "min_value": column.min_value,
                "max_value": column.max_value,
                "mean": column.mean,
            }
            for column in profile.columns[:60]
        ],
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=5),
    }


def compact_sql_result(sql_result: SQLAnalysisResponse | None) -> dict[str, Any] | None:
    if sql_result is None:
        return None
    return {
        "sql": sql_result.sql,
        "explanation": sql_result.explanation,
        "row_count": len(sql_result.rows),
        "rows_sample": list(sql_result.rows[:30]),
    }


def compact_chart(chart: ChartResponse) -> dict[str, Any]:
    return {
        "title": chart.title,
        "chart_type": chart.chart_type,
        "spec": chart.spec,
        "data_row_count": len(chart.data),
        "data_sample": list(chart.data[:30]),
        "explanation": chart.explanation,
        "related_finding_ids": list(chart.related_finding_ids),
    }


def compact_python_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in statistics.items():
        if key == "numeric_summary" and isinstance(value, dict):
            compact[key] = {
                str(column): summary
                for column, summary in islice(value.items(), 12)
                if isinstance(summary, dict)
            }
        elif key == "text_analysis" and isinstance(value, list):
            compact[key] = [
                {
                    "task": item.get("task"),
                    "text_column": item.get("text_column"),
                    "group_column": item.get("group_column"),
                    "summary": item.get("summary"),
                    "insights": item.get("insights"),
                }
                for item in value[:5]
                if isinstance(item, dict)
            ]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = value[:20]
        elif isinstance(value, dict):
            compact[key] = dict(islice(value.items(), 20))
    return compact


def compact_round(round_item: AnalysisRoundResponse) -> dict[str, Any]:
    return {
        "round_number": round_item.round_number,
        "hypothesis": round_item.hypothesis.model_dump(mode="json"),
        "plan": round_item.plan.model_dump(mode="json"),
        "reflection": round_item.reflection.model_dump(mode="json"),
        "execution_result": compact_execution_result(round_item.execution_result),
        "charts": [compact_chart(chart) for chart in round_item.charts[:5]],
        "validation_status": round_item.validation_status,
    }


def compact_execution_result(execution_result: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "sql_row_count",
        "chart_count",
        "text_analysis_count",
        "python_source",
        "python_execution_error",
        "fanout_mode",
        "fanout_group",
        "plan_validation_issues",
    }
    compact = {key: value for key, value in execution_result.items() if key in allowed_keys}
    if "insight" in execution_result:
        compact["insight"] = truncate_text(str(execution_result["insight"]), 1200)
    if "statistics_keys" in execution_result:
        compact["statistics_keys"] = execution_result["statistics_keys"]
    return compact


def compact_structured_report(report: StructuredReportResponse) -> dict[str, Any]:
    payload = report.model_dump(mode="json")
    payload["charts"] = [compact_chart(chart) for chart in report.charts[:10]]
    payload["sql_results"] = list(report.sql_results[:30])
    payload["python_results"] = compact_python_statistics(report.python_results)
    payload["analysis_trace"] = [
        compact_round(round_item) for round_item in report.analysis_trace[:6]
    ]
    return payload


def multimodal_payload(
    multimodal_inputs: tuple[MultimodalInputResponse, ...],
) -> list[dict[str, str | None]]:
    return [
        {
            "kind": item.kind,
            "title": truncate_text(item.title or "", 240) or None,
            "description": truncate_text(item.description or "", 1200) or None,
            "source_ref": truncate_text(item.source_ref or "", 500) or None,
            "media_type": item.media_type,
            "has_native_payload": "true" if item.data_url else "false",
            "processing_status": item.processing_status,
            "text_excerpt": truncate_text(item.text_excerpt or "", 1200) or None,
        }
        for item in multimodal_inputs[:8]
    ]


def multimodal_content(
    text: str,
    multimodal_inputs: tuple[MultimodalInputResponse, ...],
) -> str | list[dict[str, Any]]:
    image_parts = [
        {
            "type": "image_url",
            "image_url": {
                "url": item.data_url,
                "detail": "auto",
            },
        }
        for item in multimodal_inputs[:4]
        if item.data_url
        and item.kind in {"image", "chart", "screenshot"}
        and (item.media_type or "").startswith("image/")
    ]
    if not image_parts:
        return text
    return [{"type": "text", "text": text}, *image_parts]



def json_repair_messages(
    *,
    stage: str,
    invalid_content: str | None,
    error: str,
    contract: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You repair structured model output. Return exactly one valid JSON object "
                "and no Markdown, commentary, or reasoning. Preserve only claims already "
                "present in the supplied output; do not invent facts or numbers."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "stage": stage,
                    "validation_error": truncate_text(error, 1000),
                    "output_contract": contract,
                    "invalid_output": truncate_text(invalid_content or "", 8000),
                },
                ensure_ascii=False,
            ),
        },
    ]
