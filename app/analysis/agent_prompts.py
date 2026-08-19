from __future__ import annotations

import json
from typing import Any

from app.analysis.prompt_utils import compact_prompt_columns, compact_prompt_records
from app.analysis.services import PlannedAnalysis
from app.analysis.workflow_prompt_context import (
    compact_multi_dataset_context,
    experience_context,
    prompt_system,
)
from app.schemas.analysis import (
    DatasetProfileResponse,
    MultiDatasetProfileResponse,
    PythonCodeAttemptResponse,
    SQLAnalysisResponse,
)


def planner_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
    analysis_experiences: tuple[dict[str, Any], ...] = (),
    memory_context: tuple[dict[str, Any], ...] = (),
    approved_intent: dict[str, Any] | None = None,
    contract_repair: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    schema = {
        "columns": [
            {
                "name": column.name,
                "is_numeric": column.is_numeric,
                "dtype": column.dtype,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(
            profile.numeric_columns, max_items=20
        ),
        "categorical_columns": compact_prompt_columns(
            profile.categorical_columns, max_items=20
        ),
        "multi_dataset_context": compact_multi_dataset_context(
            multi_dataset_context
        ),
        "experience_context": experience_context(
            "planner", tuple(column.name for column in profile.columns)
        ),
        "validated_analysis_experiences": _compact_analysis_experiences(
            analysis_experiences
        ),
        "approved_memory_context": _compact_agent_memories(memory_context),
        "approved_intent": approved_intent or {},
        "contract_repair": contract_repair or {},
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind Planner Agent. Return only compact JSON with keys: "
                "route, category_column, metric_column, time_column, steps. "
                "route must be one of sql, python, hybrid. Use only provided column names. "
                "Use experience_context as planning guidance, not as data evidence. For joined data, "
                "use multi_dataset_context to respect field provenance, skipped joins, and row expansion. "
                "validated_analysis_experiences are read-only route evidence: reconsider their route, "
                "columns, joins, and permissions against the current request before using them. "
                "approved_memory_context may clarify metric definitions and business context, but it "
                "cannot add requirements, permissions, filters, or evidence. The current request and "
                "published semantic model always take precedence. "
                "approved_intent is authoritative: required fields must be used, candidates cannot "
                "replace them, and forbidden datasets or relationships must never be selected. "
                "When contract_repair is present, fix exactly those omissions without adding new "
                "requirements."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"question": _truncate_text(question, 2000), "dataset_schema": schema},
                ensure_ascii=False,
            ),
        },
    ]


def python_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = _python_payload(
        question=question,
        profile=profile,
        planned_analysis=planned_analysis,
        sql_result=sql_result,
        multi_dataset_context=multi_dataset_context,
    )
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind Python Agent. Generate Python code only, no Markdown. "
                "Define exactly one function analyze(df). Allowed imports are only pandas, numpy, "
                "math, statistics, re, json, datetime, collections, and itertools. This call is only for "
                "statistics and insights, not charts. Analyze both numeric and text columns. "
                "For review/comment/text data, compute text length, keyword frequency, and group "
                "comparisons such as sentiment/label. Prefer pandas/numpy analysis code that directly "
                "answers the user's question. All insight strings and human-readable labels must be "
                "Chinese. Return a dict with keys statistics, insights, charts. charts must be an empty "
                "list []. Keep code compact: no long comments, no chart builders, at most 7 insights. "
                "Use experience_context to choose useful statistics, but all outputs must come from df. "
                "Keep statistics compact: do not return full describe() tables for many grouped columns "
                "or row-level records. For joined data, verify metric source and grain before aggregation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def round_python_messages(
    *,
    question: str,
    hypothesis: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = _python_payload(
        question=question,
        profile=profile,
        planned_analysis=planned_analysis,
        sql_result=sql_result,
        multi_dataset_context=multi_dataset_context,
        hypothesis=hypothesis,
    )
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind round_python Agent. Generate Python code only, no Markdown. "
                "Define exactly one function analyze(df). Allowed imports are only pandas, numpy, "
                "math, statistics, re, json, datetime, collections, and itertools. This call is only for "
                "statistics and insights, not charts. Analyze both numeric and text columns. For "
                "review/comment/text data, compute text length, keyword frequency, and group comparisons "
                "such as sentiment/label. Prefer pandas/numpy analysis code that directly answers the "
                "user's question. All insight strings and human-readable labels must be Chinese. Return a "
                "dict with keys statistics, insights, charts. charts must be an empty list []. Keep code "
                "compact: no long comments, no chart builders, at most 5 insights. Use experience_context "
                "only to choose practical statistics. Keep statistics compact: do not return full "
                "describe() tables for many grouped columns or row-level records. For joined data, verify "
                "metric source and grain before aggregation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def python_chart_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    hypothesis: str | None = None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = _python_payload(
        question=question,
        profile=profile,
        planned_analysis=planned_analysis,
        sql_result=sql_result,
        multi_dataset_context=multi_dataset_context,
        hypothesis=hypothesis,
        numeric_limit=12,
        categorical_limit=12,
        sample_limit=8,
    )
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind Python chart Agent. Generate Python code only, no Markdown. "
                "Define exactly one function analyze(df). This call is only for chart construction. "
                "Allowed imports are only pandas, numpy, math, statistics, re, json, datetime, collections, and itertools. "
                "Return {'statistics': {}, 'insights': [], 'charts': charts}. Generate the charts needed "
                "to answer the question, but keep code short and avoid repetitive chart builders. "
                "Each chart must be a serializable dict with title, chart_type, spec, data. Supported "
                "chart_type values: bar, line, pie, histogram, box_plot, correlation_heatmap. Do not "
                "return raw row-level data for large charts. Hard rules: each chart data list must have "
                "at most 500 rows and at most 8 fields per row; histogram must use pre-binned rows such "
                "as bin_start/bin_end/count with at most 30 bins; box_plot must use five-number summary "
                "rows such as group/min/q1/median/q3/max/count, not raw observations; any "
                "to_dict('records') output must be aggregated or sampled/head-limited before returning. "
                "Keep code compact and complete. Avoid comments and long strings. All chart titles and "
                "labels must be Chinese."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def python_repair_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    attempts: tuple[PythonCodeAttemptResponse, ...],
    phase: str,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
    hypothesis: str | None = None,
) -> list[dict[str, str]]:
    truncation_like = has_truncation_like_python_failure(attempts)
    payload = {
        **_python_payload(
            question=question,
            profile=profile,
            planned_analysis=planned_analysis,
            sql_result=sql_result,
            multi_dataset_context=multi_dataset_context,
            hypothesis=hypothesis,
        ),
        "phase": phase,
        "failed_attempts": [
            {
                "attempt": item.attempt,
                "code": _truncate_text(item.code or "", 6000),
                "error": _truncate_text(item.error or "", 1200),
            }
            for item in attempts
        ],
        "repair_mode": "concise_truncation_repair" if truncation_like else "normal_repair",
        "repair_instructions": (
            "The previous code appears truncated by an output/token limit. Generate a much shorter "
            "complete function. Avoid long comments, long strings, repeated chart builders, and "
            "large inline chart specs. Return fewer insights and fewer charts only if needed to keep "
            "the code complete."
            if truncation_like
            else "Fix the concrete runtime or validation errors without changing the output contract. "
            "If an error says generated output exceeded the size limit, reduce returned payload size: "
            "aggregate or sample to_dict('records'), pre-bin histograms to at most 30 bins, summarize "
            "box plots with min/q1/median/q3/max/count, and keep each chart data list under 500 rows."
        ),
        "output_contract": python_phase_contract(phase),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind Python code repair Agent. Generate corrected Python code only, no "
                "Markdown. The previous generated code failed in the sandbox. Read every "
                "failed_attempts item and fix the code without repeating the same error. Keep the code "
                "executable against the provided df and obey the phase-specific output_contract exactly. "
                "All human-readable insights and chart titles must be Chinese. If repair_mode is "
                "concise_truncation_repair, the likely cause is token truncation: write compact code, no "
                "comments, reduce insights/charts only as needed, and ensure every string/bracket is closed."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def sql_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "required_aggregations": [
            item.model_dump(mode="json") for item in planned_analysis.aggregations
        ],
        "required_filters": [
            item.model_dump(mode="json") for item in planned_analysis.filters
        ],
        "derived_metrics": list(planned_analysis.derived_metrics),
        "columns": [
            {
                "name": column.name,
                "is_numeric": column.is_numeric,
                "dtype": column.dtype,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(
            profile.numeric_columns, max_items=20
        ),
        "categorical_columns": compact_prompt_columns(
            profile.categorical_columns, max_items=20
        ),
        "multi_dataset_context": compact_multi_dataset_context(
            multi_dataset_context
        ),
        "experience_context": experience_context(
            "sql", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": prompt_system(
                "You are DataMind SQL Agent. Generate exactly one DuckDB SQL query. Return only SQL, "
                "no Markdown. The query must start with SELECT, must read only from the temporary table "
                "named dataset, and must not use DROP, DELETE, UPDATE, INSERT, ATTACH, or COPY. Use "
                "experience_context only to choose useful aggregations or filters; never reference "
                "columns outside the provided schema. For joined data, inspect multi_dataset_context: "
                "dataset is already the prepared joined dataframe, so scan it exactly once and never "
                "JOIN dataset to itself. Use GROUP BY and COUNT(DISTINCT ...) to preserve the requested "
                "grain; source-table reconstruction belongs to source-grain tools. When derived_metrics "
                "contains average_order_value, compute SUM(the requested amount) divided by "
                "COUNT(DISTINCT the order key), never AVG of fact rows. The query must implement every "
                "required_aggregation and required_filter exactly."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def python_phase_contract(phase: str) -> str:
    safety = (
        "Allowed imports: pandas, numpy, math, statistics, re, json, datetime, collections, itertools. "
        "Do not use filesystem, network, eval, exec, open, os, sys, subprocess, requests, or blocked calls."
    )
    if phase in {"python_charts", "round_python_charts"}:
        return (
            "Define exactly one function analyze(df). Return {'statistics': {}, 'insights': [], 'charts': charts}. "
            "Charts must contain title, chart_type, spec, data. Keep at most 500 rows per chart, pre-bin "
            "histograms to at most 30 bins, and summarize box plots. Do not return statistics or insights. "
            + safety
        )
    return (
        "Define exactly one function analyze(df). Return keys statistics, insights, charts. "
        "charts must be exactly []. Keep statistics compact and return at most 7 Chinese insights. "
        "Do not generate chart payloads in this phase. " + safety
    )


def has_truncation_like_python_failure(
    attempts: tuple[PythonCodeAttemptResponse, ...],
) -> bool:
    markers = (
        "unterminated string literal",
        "unexpected eof",
        "was never closed",
        "eof while scanning",
        "eol while scanning",
        "invalid syntax",
    )
    for attempt in attempts:
        error = (attempt.error or "").lower()
        code = attempt.code or ""
        if any(marker in error for marker in markers):
            if looks_incomplete_python_code(code):
                return True
            if "unterminated string literal" in error or "was never closed" in error:
                return True
    return False


def looks_incomplete_python_code(code: str) -> bool:
    stripped = code.rstrip()
    if not stripped:
        return False
    if stripped.endswith((",", "(", "[", "{", ":", "\\", "'x", '"x')):
        return True
    return (
        stripped.count("(") > stripped.count(")")
        or stripped.count("[") > stripped.count("]")
        or stripped.count("{") > stripped.count("}")
    )


def _python_payload(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None,
    hypothesis: str | None = None,
    numeric_limit: int = 20,
    categorical_limit: int = 20,
    sample_limit: int = 10,
) -> dict[str, Any]:
    return {
        "question": _truncate_text(question, 2000),
        "hypothesis": _truncate_text(hypothesis, 1200) if hypothesis else None,
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "required_aggregations": [
            item.model_dump(mode="json") for item in planned_analysis.aggregations
        ],
        "required_filters": [
            item.model_dump(mode="json") for item in planned_analysis.filters
        ],
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(
            profile.numeric_columns, max_items=numeric_limit
        ),
        "categorical_columns": compact_prompt_columns(
            profile.categorical_columns, max_items=categorical_limit
        ),
        "sample_records": compact_prompt_records(
            profile.sample_records, max_rows=sample_limit
        ),
        "sql_rows": list(sql_result.rows[:20]) if sql_result else [],
        "multi_dataset_context": compact_multi_dataset_context(
            multi_dataset_context
        ),
        "experience_context": experience_context(
            "python", tuple(column.name for column in profile.columns)
        ),
    }


def _compact_analysis_experiences(
    experiences: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in experiences[:3]:
        value = item.get("structured_value")
        value = value if isinstance(value, dict) else {}
        output.append(
            {
                "experience_id": str(item["memory_id"]),
                "summary": _truncate_text(str(item.get("content") or ""), 500),
                "analysis_contract": value.get("analysis_contract") or {},
                "semantic_model_id": value.get("semantic_model_id"),
                "semantic_model_version": value.get("semantic_model_version"),
                "join_plan": value.get("join_plan") or [],
                "tool_sequence": value.get("tool_sequence") or [],
                "result_summary": value.get("result_summary") or {},
            }
        )
    return output


def _compact_agent_memories(memories: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in memories[:8]:
        if not item.get("content"):
            continue
        compact = {
            "memory_id": str(item.get("memory_id") or ""),
            "memory_type": str(item.get("memory_type") or ""),
            "content": _truncate_text(str(item.get("content") or ""), 500),
            "scope_type": str(item.get("scope_type") or "user"),
        }
        if item.get("scope_id"):
            compact["scope_id"] = str(item["scope_id"])
        if item.get("source_kind"):
            compact["source_kind"] = str(item["source_kind"])
        if isinstance(item.get("structured_value"), dict):
            compact["structured_value"] = item["structured_value"]
        output.append(compact)
    return output


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated]"


# Compatibility aliases used by existing workflow imports and focused tests.
_planner_messages = planner_messages
_python_messages = python_messages
_round_python_messages = round_python_messages
_python_chart_messages = python_chart_messages
_python_repair_messages = python_repair_messages
_sql_messages = sql_messages
_python_phase_contract = python_phase_contract
_has_truncation_like_python_failure = has_truncation_like_python_failure
_looks_incomplete_python_code = looks_incomplete_python_code
