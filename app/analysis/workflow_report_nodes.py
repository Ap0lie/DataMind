from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.analysis.lineage import build_analysis_lineage
from app.analysis.model_router import AnalysisModelRouter
from app.analysis.services import (
    _apply_plan_filters,
    _build_final_insights,
    _build_validation_issues,
    _structured_report,
    render_structured_report_html,
)
from app.analysis.statistical_verifier import (
    qualify_observational_findings,
    reportable_findings,
    statistical_validation_issues,
    verify_statistical_analysis,
)
from app.analysis.workflow_nodes import (
    ADVERSARIAL_VALIDATE_NODE,
    FORMAT_CHARTS_NODE,
    INTEGRATE_INSIGHTS_NODE,
    JOIN_PREPARE_NODE,
    PLANNER_NODE,
    PYTHON_NODE,
    REPORT_COMMIT_NODE,
    REPORT_DECIDE_NODE,
    REPORT_EXECUTE_NODE,
    REPORT_FALLBACK_NODE,
    REPORT_NODE,
    REPORT_REPAIR_NODE,
    REPORT_VERIFY_NODE,
    ROUND_REFLECT_NODE,
    SQL_NODE,
    STATISTICAL_VERIFY_NODE,
)
from app.analysis.workflow_nodes import (
    require_reportable_verification as _require_reportable_verification,
)
from app.analysis.workflow_prompts import (
    json_repair_messages as _json_repair_messages,
)
from app.analysis.workflow_prompts import (
    report_messages as _report_messages,
)
from app.analysis.workflow_prompts import (
    review_messages as _review_messages,
)
from app.analysis.workflow_support import (
    extract_json_object as _extract_json_object,
)
from app.analysis.workflow_support import (
    float_payload as _float_payload,
)
from app.analysis.workflow_support import (
    require as _require,
)
from app.analysis.workflow_support import (
    string_list as _string_list,
)
from app.core.settings import get_settings
from app.schemas.analysis import (
    AnalysisPlanResponse,
    AnalysisRunResponse,
    ChartResponse,
    DatasetProfileResponse,
    InsightFindingResponse,
    PythonAnalysisResponse,
    StatisticalVerificationResponse,
    StructuredReportResponse,
    ValidationIssueResponse,
    WorkflowTraceNodeResponse,
)
from app.storage.dataset_store import DatasetStoreRepository

WorkflowState = dict[str, Any]


@dataclass(frozen=True)
class ReportNodeRuntime:
    notify_progress: Callable[..., None]
    emit_loop_event: Callable[..., None]
    workflow_dataframe: Callable[..., Any]


def _adversarial_validate_node(
    model_router: AnalysisModelRouter | None, runtime: ReportNodeRuntime
) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        runtime.notify_progress(
            state,
            stage=ADVERSARIAL_VALIDATE_NODE,
            progress=93,
            message="Reviewing analysis quality and gaps.",
        )
        final_insights = state.get("final_insights", ())
        python_result = state.get("python_result")
        charts = state.get("report_charts", python_result.charts if python_result else ())
        validation_issues = _build_validation_issues(
            findings=final_insights,
            charts=charts,
            extra_issues=(
                *state.get("plan_validation_issues", ()),
                *state.get("statistical_validation_issues", ()),
            ),
        )
        if state.get("python_execution_error") and len(state.get("python_attempts", ())) >= 3:
            validation_issues = (
                *validation_issues,
                ValidationIssueResponse(
                    severity="warning",
                    finding_ref="python_agent",
                    issue="LLM Python code failed after 3 attempts; rule fallback was used.",
                    suggestion=str(state.get("python_execution_error") or ""),
                ),
            )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if model_router is not None and not state.get("analysis_fast_path", False):
            try:
                response = model_router.complete(
                    messages=_review_messages(
                        question=state["question"],
                        final_insights=final_insights,
                        charts=charts,
                        sql_result=state.get("sql_result"),
                        multimodal_inputs=state.get("multimodal_inputs", ()),
                        multi_dataset_context=state.get("multi_dataset_context"),
                        analysis_contract=state.get("analysis_contract"),
                        statistical_verification=state.get(
                            "statistical_verification"
                        ),
                    ),
                    temperature=0.1,
                    max_tokens=1000,
                    metadata={
                        "agent": "review",
                        "dataset_id": str(state["dataset_id"]),
                        "optional_stage": True,
                    },
                )
                validation_issues = (
                    *validation_issues,
                    *_parse_model_validation_issues(response.content),
                )
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
                validation_issues = (
                    *validation_issues,
                    ValidationIssueResponse(
                        severity="info",
                        finding_ref="adversarial_validate",
                        issue="LLM review unavailable; rule validation was used.",
                        suggestion=str(exc),
                    ),
                )
        return {
            "validation_issues": validation_issues,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), ADVERSARIAL_VALIDATE_NODE),
        }

    return run


def _statistical_verify_node(
    repository: DatasetStoreRepository, runtime: ReportNodeRuntime
) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        runtime.notify_progress(
            state,
            stage=STATISTICAL_VERIFY_NODE,
            progress=91,
            message="Checking evidence, statistical support and analysis grain.",
        )
        contract = _require(
            state.get("analysis_contract"),
            "Planner did not produce an analysis contract.",
        )
        findings = _merge_report_findings(
            _mandatory_evidence_findings(state),
            state.get("final_insights", ()),
        )
        findings = _attach_finding_evidence_ids(
            findings,
            state.get("tool_evidence", ()),
        )
        findings = qualify_observational_findings(findings, contract)
        planned_analysis = _require(
            state.get("planned_analysis"),
            "Planner did not produce an analysis plan.",
        )
        analysis_dataframe = _apply_plan_filters(
            runtime.workflow_dataframe(repository, state),
            planned_analysis.filters,
        )
        verification_kwargs = {
            "contract": contract,
            "profile": _require(state.get("profile"), "Missing dataset profile."),
            "dataframe": analysis_dataframe,
            "evidence": state.get("tool_evidence", ()),
            "sql_result": state.get("sql_result"),
            "python_result": state.get("python_result"),
            "multi_dataset_context": state.get("multi_dataset_context"),
        }
        verification = verify_statistical_analysis(
            findings=findings,
            **verification_kwargs,
        )
        verified_findings = reportable_findings(findings, verification)
        if verified_findings and len(verified_findings) < len(findings):
            sanitized_verification = verify_statistical_analysis(
                findings=verified_findings,
                **verification_kwargs,
            )
            if not sanitized_verification.requires_replan:
                findings = verified_findings
                verification = sanitized_verification
        issues = statistical_validation_issues(verification)
        lineage = build_analysis_lineage(
            contract=contract,
            planner_metadata=state.get("planner_metadata"),
            multi_dataset_context=state.get("multi_dataset_context"),
            findings=findings,
            charts=state.get("report_charts", ()),
            report_id=str(state["run_id"]),
        )
        runtime.emit_loop_event(
            state,
            event_type="statistical_validation",
            status="completed" if verification.status != "failed" else "failed",
            message=verification.summary,
            iteration=state.get("adversarial_repair_count", 0),
            payload={
                "status": verification.status,
                "requires_replan": verification.requires_replan,
                "numeric_evidence_coverage": verification.numeric_evidence_coverage,
            },
        )
        return {
            "final_insights": findings,
            "statistical_verification": verification,
            "statistical_validation_issues": issues,
            "analysis_lineage": lineage,
            "executed_nodes": (
                *state.get("executed_nodes", ()),
                STATISTICAL_VERIFY_NODE,
            ),
        }

    return run


def _remaining_report_timeout(
    state: WorkflowState,
    *,
    started_epoch: float | None = None,
) -> float:
    started = started_epoch or state.get("report_started_epoch") or time.time()
    remaining = get_settings().report_loop_timeout_seconds - (time.time() - float(started))
    if remaining <= 0:
        raise TimeoutError("Report loop time budget exhausted.")
    return remaining


def _report_decide_node(runtime: ReportNodeRuntime) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        settings = get_settings()
        count = state.get("report_decision_count", 0) + 1
        started_epoch = state.get("report_started_epoch") or time.time()
        runtime.notify_progress(
            state,
            stage=REPORT_DECIDE_NODE,
            progress=95,
            message="AI is selecting the report strategy from verified evidence.",
        )
        findings = state.get("final_insights", ())
        evidence_gaps = [
            item.title for item in findings if not item.evidence and not item.data_source
        ]
        if not findings and state.get("report_evidence_return_count", 0) < 1:
            strategy, reason = "evidence_gap", "No final findings are available."
        elif count > settings.report_loop_max_decisions:
            strategy, reason = "rules_fallback", "Report decision budget exhausted."
        elif time.time() - started_epoch >= settings.report_loop_timeout_seconds:
            strategy, reason = "rules_fallback", "Report time budget exhausted."
        elif state.get("report_used_tokens", 0) >= settings.report_loop_max_tokens:
            strategy, reason = "rules_fallback", "Report token budget exhausted."
        elif evidence_gaps and not any(
            item.evidence or item.data_source for item in findings
        ):
            strategy, reason = "evidence_gap", "Findings lack traceable evidence."
        else:
            strategy, reason = "llm", "Verified evidence is ready for report generation."
        used_tokens = state.get("report_used_tokens", 0)
        evidence_returns = state.get("report_evidence_return_count", 0)
        result: dict[str, Any] = {
            "report_decision_count": count,
            "report_strategy": strategy,
            "report_started_epoch": started_epoch,
            "report_used_tokens": used_tokens,
        }
        if strategy == "evidence_gap":
            if evidence_returns >= 1:
                result.update(
                    {
                        "report_strategy": "rules_fallback",
                        "report_terminal_reason": "evidence_gap_after_reanalysis",
                    }
                )
            else:
                result.update(
                    {
                        "report_evidence_return_count": evidence_returns + 1,
                        "loop_repair_context": {
                            "error_type": "report_evidence_gap",
                            "message": reason,
                            "requested_tools": [
                                "execute_semantic_query",
                                "execute_safe_sql",
                                "execute_python_analysis",
                            ],
                        },
                        "loop_terminal_reason": None,
                    }
                )
                runtime.emit_loop_event(
                    state,
                    event_type="evidence_request",
                    status="completed",
                    message=reason,
                    iteration=evidence_returns + 1,
                    payload={"return_count": evidence_returns + 1},
                )
        runtime.emit_loop_event(
            state,
            event_type="report_decision",
            status="completed",
            message=f"Report strategy selected: {result.get('report_strategy', strategy)}.",
            iteration=count,
            payload={"strategy": result.get("report_strategy", strategy), "reason": reason},
        )
        return result

    return run


def _native_grain_report_evidence_ids(
    state: WorkflowState,
) -> set[str]:
    evidence = state.get("tool_evidence", ())
    native_ids = {
        str(item.get("evidence_id"))
        for item in evidence
        if item.get("evidence_id")
        and (
            item.get("source_guard") is True
            or (item.get("result") or {}).get("native_grain") is True
        )
    }
    if not native_ids:
        return set()
    relationship_risk = any(
        relationship.get("relationship_type") == "many_to_many"
        for item in evidence
        if item.get("relationship_guard") is True
        for relationship in (item.get("result") or {}).get("relationships", ())
        if isinstance(relationship, dict)
    )
    context = state.get("multi_dataset_context")
    expansion = float((context.join_summary if context else {}).get("row_expansion_ratio") or 1)
    return native_ids if relationship_risk or expansion > 1 else set()


def _build_report_draft(
    state: WorkflowState,
    model_router: AnalysisModelRouter | None,
    *,
    strategy: str,
    agent: str = "report_execute",
) -> dict[str, Any]:
    question = state["question"]
    profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
    python_result = state.get("python_result")
    sql_result = state.get("sql_result")
    rounds = state.get("rounds", ())
    findings = _attach_finding_evidence_ids(
        state.get("final_insights", ()), state.get("tool_evidence", ())
    )
    native_evidence_ids = _native_grain_report_evidence_ids(state)
    if native_evidence_ids:
        findings = tuple(
            finding
            for finding in findings
            if not (
                "sql_result" in finding.data_source.casefold()
                and _numeric_claim_tokens(finding.content)
                and not any(
                    f"evidence_id:{evidence_id}" in finding.evidence
                    for evidence_id in native_evidence_ids
                )
            )
        )
        if sql_result is not None:
            sql_result = sql_result.model_copy(
                update={
                    "rows": tuple(
                        row
                        for row in sql_result.rows
                        if str(row.get("evidence_id") or "") in native_evidence_ids
                    )
                }
            )
    verification = state.get("statistical_verification")
    findings = reportable_findings(findings, verification)
    mandatory_findings = reportable_findings(
        _mandatory_evidence_findings(state),
        verification,
    )
    findings = _merge_report_findings(mandatory_findings, findings)
    issues = state.get("validation_issues", ())
    charts = state.get("report_charts", python_result.charts if python_result else ())
    framework = state.get("analysis_framework")
    baseline_structured = _structured_report(
        question=question,
        profile=profile,
        sql_result=sql_result,
        python_result=python_result,
        rounds=rounds,
        final_insights=findings,
        validation_issues=issues,
        analysis_framework=framework,
        charts=charts,
    )
    structured = baseline_structured
    source = "rules"
    provider = state.get("model_router_provider")
    model = state.get("model_router_model")
    error = state.get("model_router_error")
    augmented_model_content: str | None = None
    if strategy == "llm" and model_router is not None:
        try:
            report_messages = _report_messages(
                question=question,
                profile=profile,
                structured_report=structured,
                multimodal_inputs=state.get("multimodal_inputs", ()),
                multi_dataset_context=state.get("multi_dataset_context"),
            )
            if state.get("report_validation"):
                report_messages.append(
                    {
                        "role": "user",
                        "content": "上一版报告未通过确定性校验。只修复以下问题，不新增无证据数字或图表引用："
                        + json.dumps(state["report_validation"], ensure_ascii=False),
                    }
                )
            fast_path = bool(state.get("analysis_fast_path", False))
            response = model_router.complete(
                messages=report_messages,
                temperature=0.2,
                max_tokens=1400 if fast_path else 2200,
                metadata={
                    "agent": agent,
                    "dataset_id": str(state["dataset_id"]),
                    "revision": state.get("report_revision_count", 0) + 1,
                    "timeout_seconds": _remaining_report_timeout(state),
                    "optional_stage": fast_path,
                },
            )
            token_usage = int(response.token_usage.get("total_tokens") or 0)
            try:
                structured = _parse_model_structured_report(
                    response.content,
                    fallback=structured,
                    provider=response.provider,
                    model=response.model,
                )
                source = "model_router_structured"
            except ValueError as parse_error:
                try:
                    repaired = model_router.complete(
                        messages=_json_repair_messages(
                            stage="report",
                            invalid_content=response.content,
                            error=str(parse_error),
                            contract=(
                                "Return one JSON object with executive_summary (at least 20 "
                                "characters), analysis_context, key_findings, chart_explanations, "
                                "data_gaps, validation_issues, and recommended_next_steps."
                            ),
                        ),
                        temperature=0.0,
                        max_tokens=1400 if fast_path else 2200,
                        metadata={
                            "agent": agent,
                            "dataset_id": str(state["dataset_id"]),
                            "revision": state.get("report_revision_count", 0) + 1,
                            "structured_repair": True,
                            "timeout_seconds": _remaining_report_timeout(state),
                            "optional_stage": fast_path,
                        },
                    )
                    token_usage += int(
                        repaired.token_usage.get("total_tokens") or 0
                    )
                    structured = _parse_model_structured_report(
                        repaired.content,
                        fallback=structured,
                        provider=repaired.provider,
                        model=repaired.model,
                    )
                    response = repaired
                    source = "model_router_structured_repair"
                except Exception as repair_error:
                    narrative = str(response.content or "").strip()
                    if len(narrative) >= 40 and not narrative.startswith("[mock:"):
                        augmented_model_content = narrative
                        source = "model_router_augmented"
                    else:
                        source = "rules"
                    error = (
                        f"Structured report repair failed: {type(repair_error).__name__}: "
                        f"{repair_error}"
                    )
            provider, model = response.provider, response.model
            if source != "rules":
                error = None
        except Exception as exc:
            error = str(exc)
            source = "rules"
    structured = _preserve_mandatory_report_findings(structured, mandatory_findings)
    structured = _preserve_verified_report_findings(
        structured,
        findings,
        verification,
    )
    structured, cardinality_sanitized = _sanitize_report_cardinality_claims(
        report=structured,
        fallback=baseline_structured,
        verified_findings=findings,
    )
    if cardinality_sanitized and source != "rules":
        source = f"{source}_cardinality_sanitized"
    unsupported_summary_numbers = _unsupported_summary_numbers(
        state,
        structured,
        profile,
        findings=findings,
    )
    if unsupported_summary_numbers:
        structured = structured.model_copy(
            update={
                "executive_summary": baseline_structured.executive_summary,
                "validation_issues": (
                    *structured.validation_issues,
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="executive_summary",
                        issue=(
                            "模型摘要包含无已验证证据支持的数值："
                            f"{', '.join(unsupported_summary_numbers)}；已使用确定性摘要。"
                        ),
                        suggestion="仅从已通过统计审查的结论生成摘要。",
                    ),
                ),
            }
        )
        if source != "rules":
            source = f"{source}_numeric_sanitized"
    structured = structured.model_copy(
        update={
            "analysis_contract": state.get("analysis_contract"),
            "statistical_verification": verification,
            "analysis_lineage": state.get("analysis_lineage"),
        }
    )
    markdown = _markdown_from_structured_report(structured)
    if augmented_model_content:
        markdown = _merge_model_report(
            base_report=markdown,
            model_content=augmented_model_content,
        )
    html = render_structured_report_html(structured, title="DataMind 分析报告")
    return {
        "structured_report": structured,
        "report_markdown": markdown,
        "html_report": html,
        "report_source": source,
        "model_router_provider": provider,
        "model_router_model": model,
        "model_router_error": error,
        "report_charts": charts,
        "final_insights": findings,
        "validation_issues": issues,
        "report_draft_tokens": locals().get("token_usage", 0),
    }


def _report_execute_node(
    model_router: AnalysisModelRouter | None, runtime: ReportNodeRuntime
) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        runtime.notify_progress(
            state,
            stage=REPORT_EXECUTE_NODE,
            progress=96,
            message="Generating a traceable report draft.",
        )
        revision = state.get("report_revision_count", 0) + 1
        draft = _build_report_draft(
            state, model_router, strategy=state.get("report_strategy") or "llm"
        )
        runtime.emit_loop_event(
            state,
            event_type="report_draft",
            status="completed",
            message=f"Report draft revision {revision} generated.",
            iteration=revision,
            payload={
                "strategy": state.get("report_strategy"),
                "source": draft.get("report_source"),
            },
        )
        return {
            **draft,
            "report_revision_count": revision,
            "report_used_tokens": state.get("report_used_tokens", 0)
            + int(draft.get("report_draft_tokens") or 0),
            "report_draft_ready": True,
            "executed_nodes": (*state.get("executed_nodes", ()), REPORT_EXECUTE_NODE),
        }

    return run


_NUMERIC_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


_CARDINALITY_CLAIM_PATTERNS = {
    "1:1": re.compile(r"(?:1\s*:\s*1|一对一|one[- ]to[- ]one)", re.IGNORECASE),
    "1:N": re.compile(r"(?:1\s*:\s*[Nn]|一对多|one[- ]to[- ]many)", re.IGNORECASE),
    "N:1": re.compile(r"(?:[Nn]\s*:\s*1|多对一|many[- ]to[- ]one)", re.IGNORECASE),
    "M:N": re.compile(r"(?:[Mm]\s*:\s*[Nn]|多对多|many[- ]to[- ]many)", re.IGNORECASE),
}


def _numeric_claim_tokens(value: Any) -> set[str]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    return {match.group(0).replace(",", "") for match in _NUMERIC_CLAIM_RE.finditer(text)}


def _cardinality_claims(value: str) -> set[str]:
    return {
        label
        for label, pattern in _CARDINALITY_CLAIM_PATTERNS.items()
        if pattern.search(value)
    }


def _sanitize_report_cardinality_claims(
    *,
    report: StructuredReportResponse,
    fallback: StructuredReportResponse,
    verified_findings: tuple[InsightFindingResponse, ...],
) -> tuple[StructuredReportResponse, bool]:
    relationship_findings = tuple(
        finding
        for finding in verified_findings
        if "relationship" in finding.data_source.casefold()
        or "关系" in finding.title
        or "基数" in finding.title
    )
    trusted_claims = {
        claim
        for finding in relationship_findings
        for claim in _cardinality_claims(finding.content)
    }
    summary_claims = _cardinality_claims(report.executive_summary)
    unsupported_summary = bool(summary_claims - trusted_claims)
    filtered_findings = tuple(
        finding
        for finding in report.key_findings
        if not (
            (claims := _cardinality_claims(finding.content))
            and claims - trusted_claims
        )
    )
    changed = unsupported_summary or len(filtered_findings) != len(report.key_findings)
    if not changed:
        return report, False
    trusted_summary = fallback.executive_summary
    if relationship_findings and relationship_findings[0].content not in trusted_summary:
        trusted_summary = (
            f"{trusted_summary.strip()} {relationship_findings[0].content}"
        ).strip()
    return (
        report.model_copy(
            update={
                "executive_summary": (
                    trusted_summary if unsupported_summary else report.executive_summary
                ),
                "key_findings": filtered_findings,
                "validation_issues": (
                    *report.validation_issues,
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="join_cardinality",
                        issue=(
                            "模型报告包含未被关系画像支持的 Join 基数声明；"
                            "已恢复为确定性关系方向和基数。"
                        ),
                        suggestion="仅引用关系证据中的表方向与 1:1、1:N、N:1 或 M:N。",
                    ),
                ),
            }
        ),
        True,
    )


def _unsupported_summary_numbers(
    state: WorkflowState,
    report: StructuredReportResponse,
    profile: DatasetProfileResponse,
    *,
    findings: tuple[InsightFindingResponse, ...] | None = None,
) -> list[str]:
    verification = state.get("statistical_verification")
    verified_titles = {
        verdict.title
        for verdict in verification.finding_verdicts
        if verdict.status != "failed"
    } if verification else set()
    allowed: set[str] = set()
    for finding in (
        findings if findings is not None else state.get("final_insights", ())
    ):
        if finding.title in verified_titles:
            allowed.update(_numeric_claim_tokens(finding.content))
    summary = report.executive_summary
    question = str(state.get("question") or "")
    if question:
        summary = summary.replace(question, "", 1)
    python_result = state.get("python_result")
    row_value = (
        python_result.statistics.get("rows")
        if isinstance(python_result, PythonAnalysisResponse)
        else None
    )
    analysis_rows = (
        max(0, int(row_value))
        if isinstance(row_value, (int, float)) and not isinstance(row_value, bool)
        else profile.row_count
    )
    trusted_profile_phrase = f"基于 {analysis_rows} 行、{profile.column_count} 列数据"
    summary = summary.replace(trusted_profile_phrase, "基于数据", 1)
    return sorted(_numeric_claim_tokens(summary) - allowed)


def _report_verify_node(runtime: ReportNodeRuntime) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        settings = get_settings()
        runtime.notify_progress(
            state,
            stage=REPORT_VERIFY_NODE,
            progress=97,
            message="Validating report claims, evidence and chart references.",
        )
        structured = _require(state.get("structured_report"), "Report draft is missing.")
        unsupported: list[str] = []
        evidence_gaps: list[str] = []
        known_evidence_ids = {
            str(item.get("evidence_id"))
            for item in state.get("tool_evidence", ())
            if item.get("evidence_id")
        }
        for finding in structured.key_findings:
            has_number = bool(_numeric_claim_tokens(finding.content))
            cited_ids = {
                evidence_id for evidence_id in known_evidence_ids if evidence_id in finding.evidence
            }
            if has_number and not cited_ids:
                unsupported.append(finding.title)
            if not finding.evidence and not finding.data_source:
                evidence_gaps.append(finding.title)
        profile = _require(
            state.get("profile"),
            "Planner did not produce a dataset profile.",
        )
        unsupported_summary_numbers = _unsupported_summary_numbers(
            state,
            structured,
            profile,
        )
        chart_titles = {chart.title for chart in state.get("report_charts", ())}
        missing_chart_refs = [
            chart.title for chart in structured.charts if chart.title not in chart_titles
        ]
        revision = state.get("report_revision_count", 0)
        evidence_returns = state.get("report_evidence_return_count", 0)
        budget_exhausted = (
            time.time() - float(state.get("report_started_epoch") or time.time())
            >= settings.report_loop_timeout_seconds
            or state.get("report_used_tokens", 0) >= settings.report_loop_max_tokens
        )
        if budget_exhausted:
            outcome = "fallback"
        elif evidence_gaps and evidence_returns < 1:
            outcome = "evidence_gap"
        elif (
            unsupported or unsupported_summary_numbers or missing_chart_refs
        ) and revision < settings.report_loop_max_revisions:
            outcome = "report_issue"
        elif unsupported or unsupported_summary_numbers or missing_chart_refs:
            outcome = "fallback"
        else:
            outcome = "sufficient"
        validation = {
            "outcome": outcome,
            "unsupported_numeric_findings": unsupported,
            "unsupported_summary_numbers": unsupported_summary_numbers,
            "evidence_gaps": evidence_gaps,
            "missing_chart_references": missing_chart_refs,
        }
        update: dict[str, Any] = {"report_validation": validation}
        if outcome == "evidence_gap":
            update.update(
                {
                    "report_evidence_return_count": evidence_returns + 1,
                    "loop_repair_context": {
                        "error_type": "report_evidence_gap",
                        "message": f"Evidence missing for: {', '.join(evidence_gaps[:5])}",
                    },
                    "loop_terminal_reason": None,
                }
            )
        if outcome == "sufficient":
            update["report_terminal_reason"] = "validated"
        runtime.emit_loop_event(
            state,
            event_type="report_validation",
            status="completed" if outcome == "sufficient" else "failed",
            message=f"Report validation outcome: {outcome}.",
            iteration=revision,
            payload=validation,
        )
        return update

    return run


def _report_repair_node(runtime: ReportNodeRuntime) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        validation = state.get("report_validation") or {}
        runtime.notify_progress(
            state,
            stage=REPORT_REPAIR_NODE,
            progress=97,
            message="Repairing unsupported report claims and references.",
        )
        runtime.emit_loop_event(
            state,
            event_type="report_repair",
            status="completed",
            message="Report draft will be regenerated with validation feedback.",
            iteration=state.get("report_revision_count", 0),
            payload=validation,
        )
        return {
            "report_strategy": "llm",
            "report_draft_ready": False,
            "executed_nodes": (*state.get("executed_nodes", ()), REPORT_REPAIR_NODE),
        }

    return run


def _report_fallback_node(runtime: ReportNodeRuntime) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        runtime.notify_progress(
            state,
            stage=REPORT_FALLBACK_NODE,
            progress=98,
            message="Using deterministic report fallback with explicit data gaps.",
        )
        draft = _build_report_draft(state, None, strategy="template")
        structured = draft["structured_report"]
        gaps = tuple(
            dict.fromkeys(
                (
                    *structured.data_gaps,
                    *[
                        str(item)
                        for item in (state.get("report_validation") or {}).get("evidence_gaps", [])
                    ],
                )
            )
        )
        structured = structured.model_copy(update={"data_gaps": gaps})
        draft.update(
            {
                "structured_report": structured,
                "report_markdown": _markdown_from_structured_report(structured),
                "html_report": render_structured_report_html(structured, title="DataMind 分析报告"),
                "report_source": "rules",
            }
        )
        runtime.emit_loop_event(
            state,
            event_type="report_fallback",
            status="completed",
            message="Rule report fallback prepared.",
            iteration=state.get("report_revision_count", 0),
            payload={"data_gap_count": len(gaps)},
        )
        return {
            **draft,
            "report_strategy": "rules_fallback",
            "report_terminal_reason": state.get("report_terminal_reason") or "rules_fallback",
            "report_draft_ready": True,
            "executed_nodes": (*state.get("executed_nodes", ()), REPORT_FALLBACK_NODE),
        }

    return run


def _report_commit_node(
    repository: DatasetStoreRepository, runtime: ReportNodeRuntime
) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        _require_reportable_verification(state)
        runtime.notify_progress(
            state,
            stage=REPORT_COMMIT_NODE,
            progress=99,
            message="Committing the validated report idempotently.",
        )
        dataset_id = state["dataset_id"]
        planned = _require(
            state.get("planned_analysis"), "Planner did not produce an analysis plan."
        )
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        python_result = state.get("python_result")
        structured = _require(state.get("structured_report"), "Validated report draft is missing.")
        markdown = _require(state.get("report_markdown"), "Validated report markdown is missing.")
        html = _require(state.get("html_report"), "Validated report HTML is missing.")
        charts = state.get("report_charts", python_result.charts if python_result else ())
        executed = (*state.get("executed_nodes", ()), REPORT_COMMIT_NODE)
        report_source = state.get("report_source", "rules")
        trace = _workflow_trace(
            state=state,
            executed_nodes=executed,
            report_source=report_source,
            provider=state.get("model_router_provider"),
            model=state.get("model_router_model"),
        )
        analysis_summary = state.get("loop_summary", {})
        combined_summary = {
            **analysis_summary,
            "analysis": analysis_summary,
            "statistical_verification": (
                state["statistical_verification"].model_dump(mode="json")
                if state.get("statistical_verification")
                else None
            ),
            "report": {
                "strategy": state.get("report_strategy"),
                "revision_count": state.get("report_revision_count", 0),
                "terminal_reason": state.get("report_terminal_reason"),
                "validation": state.get("report_validation") or {},
                "evidence_return_count": state.get("report_evidence_return_count", 0),
            },
        }
        analysis_framework = state.get("analysis_framework")
        planner_metadata = state.get("planner_metadata")
        multi_dataset_context = state.get("multi_dataset_context")
        report_id = repository.save_report(
            dataset_id=dataset_id,
            title="DataMind 分析报告",
            markdown=markdown,
            job_id=state["run_id"],
            metadata={
                "question": state["question"],
                "route": planned.route,
                "workflow": "langgraph_analysis",
                "prompt_overrides": dict(state.get("prompt_overrides") or {}),
                "nodes": list(executed),
                "analysis_framework": analysis_framework.model_dump(mode="json")
                if analysis_framework
                else None,
                "multi_dataset_context": multi_dataset_context.model_dump(mode="json")
                if multi_dataset_context
                else None,
                "primary_dataset_id": str(dataset_id),
                "dataset_group_id": str(state.get("dataset_group_id"))
                if state.get("dataset_group_id")
                else None,
                "additional_dataset_ids": [
                    str(item) for item in state.get("additional_dataset_ids", ())
                ],
                "join_plan": [item.model_dump(mode="json") for item in state.get("join_plan", ())],
                "relationship_plan": [
                    item.model_dump(mode="json") for item in state.get("relationship_plan", ())
                ],
                "join_summary": multi_dataset_context.join_summary
                if multi_dataset_context
                else {},
                "multimodal_inputs": [
                    item.model_dump(mode="json") for item in state.get("multimodal_inputs", ())
                ],
                "planner_metadata": planner_metadata.model_dump(mode="json")
                if planner_metadata
                else None,
                "analysis_contract": (
                    state["analysis_contract"].model_dump(mode="json")
                    if state.get("analysis_contract")
                    else None
                ),
                "statistical_verification": (
                    state["statistical_verification"].model_dump(mode="json")
                    if state.get("statistical_verification")
                    else None
                ),
                "analysis_lineage": (
                    state["analysis_lineage"].model_dump(mode="json")
                    if state.get("analysis_lineage")
                    else None
                ),
                "planner_source": state.get("planner_source", "rules"),
                "python_source": state.get("python_source", "not_run"),
                "python_generated_code": state.get("python_generated_code"),
                "python_execution_error": state.get("python_execution_error"),
                "python_attempts": [
                    attempt.model_dump(mode="json")
                    for attempt in state.get("python_attempts", ())
                ],
                "model_router_provider": state.get("model_router_provider"),
                "model_router_model": state.get("model_router_model"),
                "model_router_error": state.get("model_router_error"),
                "sql_source": state.get("sql_source", "none"),
                "sql_validation_error": state.get("sql_validation_error"),
                "structured_report": structured.model_dump(mode="json"),
                "html_report": html,
                "validation_issue_count": len(state.get("validation_issues", ())),
                "workflow_trace": [item.model_dump(mode="json") for item in trace],
                "report_source": report_source,
                "report_strategy": state.get("report_strategy"),
                "report_revision_count": state.get("report_revision_count", 0),
                "report_terminal_reason": state.get("report_terminal_reason"),
                "agent_mode": state.get("agent_mode", "legacy"),
                "loop_summary": combined_summary,
                "loop_terminal_reason": state.get("loop_terminal_reason"),
                "semantic_model_id": str(
                    (state.get("planner_decision") or {}).get("semantic_model_id") or ""
                )
                or None,
                "semantic_model_version": (state.get("planner_decision") or {}).get(
                    "semantic_model_version"
                ),
                "evidence_ids": [
                    item.get("evidence_id") or item.get("artifact_id")
                    for item in state.get("tool_evidence", ())
                    if item.get("evidence_id") or item.get("artifact_id")
                ],
                "finding_evidence": [
                    {
                        "title": finding.title,
                        "evidence": finding.evidence,
                        "data_source": finding.data_source,
                    }
                    for finding in structured.key_findings
                ],
            },
        )
        for chart in charts:
            repository.save_chart(
                dataset_id=dataset_id,
                title=chart.title,
                chart_type=chart.chart_type,
                chart_spec=chart.spec,
                chart_data=list(chart.data),
            )
        response = AnalysisRunResponse(
            dataset_id=dataset_id,
            dataset_group_id=state.get("dataset_group_id"),
            report_id=report_id,
            question=state["question"],
            multimodal_inputs=state.get("multimodal_inputs", ()),
            plan=AnalysisPlanResponse(route=planned.route, steps=planned.steps),
            planner_metadata=state.get("planner_metadata"),
            multi_dataset_context=state.get("multi_dataset_context"),
            profile=profile,
            analysis_framework=state.get("analysis_framework"),
            analysis_contract=state.get("analysis_contract"),
            statistical_verification=state.get("statistical_verification"),
            analysis_lineage=state.get("analysis_lineage"),
            sql_result=state.get("sql_result"),
            python_result=python_result,
            rounds=state.get("rounds", ()),
            final_insights=state.get("final_insights", ()),
            validation_issues=state.get("validation_issues", ()),
            structured_report=structured,
            html_report=html,
            sql_source=state.get("sql_source", "none"),
            python_source=state.get("python_source", "not_run"),
            python_generated_code=state.get("python_generated_code"),
            python_execution_error=state.get("python_execution_error"),
            python_attempts=state.get("python_attempts", ()),
            workflow_trace=trace,
            report_markdown=markdown,
            agent_mode=state.get("agent_mode", "legacy"),
            loop_summary=combined_summary,
            loop_terminal_reason=state.get("loop_terminal_reason"),
            report_strategy=state.get("report_strategy"),
            report_revision_count=state.get("report_revision_count", 0),
            report_terminal_reason=state.get("report_terminal_reason"),
        )
        runtime.emit_loop_event(
            state,
            event_type="report_commit",
            status="completed",
            message="Validated report committed idempotently.",
            iteration=state.get("report_revision_count", 0),
            payload={"report_id": str(report_id)},
        )
        return {
            "final_response": response,
            "workflow_trace": trace,
            "executed_nodes": executed,
            "loop_summary": combined_summary,
        }

    return run


def _attach_finding_evidence_ids(
    findings: tuple[InsightFindingResponse, ...],
    evidence: tuple[dict[str, Any], ...],
) -> tuple[InsightFindingResponse, ...]:
    successful = [
        item
        for item in evidence
        if item.get("evidence_id") and item.get("status") in {"succeeded", "completed", None}
    ]
    if not successful:
        return findings
    output: list[InsightFindingResponse] = []
    for finding in findings:
        if any(str(item["evidence_id"]) in finding.evidence for item in successful):
            output.append(finding)
            continue
        source = f"{finding.data_source} {finding.evidence}".lower()
        preferred = next(
            (
                item
                for item in successful
                if (
                    (
                        "sql" in source
                        and (
                            "sql" in str(item.get("tool_name") or "")
                            or "semantic" in str(item.get("tool_name") or "")
                            or isinstance((item.get("result") or {}).get("sql_result"), dict)
                            or bool((item.get("result") or {}).get("sql"))
                        )
                    )
                    or ("python" in source and "python" in str(item.get("tool_name") or ""))
                    or ("文本" in source and "text" in str(item.get("tool_name") or ""))
                )
            ),
            successful[0],
        )
        evidence_text = finding.evidence.strip()
        reference = f"evidence_id:{preferred['evidence_id']}"
        output.append(
            finding.model_copy(
                update={"evidence": f"{evidence_text}; {reference}" if evidence_text else reference}
            )
        )
    return tuple(output)


def _mandatory_evidence_findings(
    state: WorkflowState,
) -> tuple[InsightFindingResponse, ...]:
    folded_question = state.get("question", "").casefold()
    trend_requested = any(
        token in folded_question for token in ("趋势", "月度", "按月", "trend", "monthly")
    )
    query_findings = (
        tuple(
            finding
            for finding in _build_final_insights(
                question=state.get("question", ""),
                python_result=state.get("python_result"),
                sql_result=state.get("sql_result"),
            )
            if finding.data_source == "sql_result.rows"
        )
        if trend_requested
        else ()
    )
    evidence: list[dict[str, Any]] = []
    for item in state.get("tool_evidence", ()):
        result = item.get("result")
        if not isinstance(result, dict):
            result = item.get("claim_result") or item.get("contract_result")
        if (
            item.get("evidence_id")
            and item.get("status") in {"succeeded", "completed", None}
            and isinstance(result, dict)
        ):
            evidence.append({**item, "result": result})
    aggregates = [
        item for item in evidence if (item.get("result") or {}).get("native_grain") is True
    ]
    relationship_item = next(
        (
            item
            for item in evidence
            if item.get("relationship_guard") is True
            and isinstance((item.get("result") or {}).get("relationships"), list)
        ),
        None,
    )
    relationship_requested = _relationship_analysis_requested(
        state.get("question", "")
    )
    if not aggregates and not (relationship_item and relationship_requested):
        return query_findings

    fact_sources = {
        str((item.get("result") or {}).get("source_dataset") or "")
        for item in aggregates
        if (item.get("result") or {}).get("source_dataset")
    }
    relationships = (
        list((relationship_item.get("result") or {}).get("relationships") or [])
        if relationship_item
        else []
    )
    relationship_phrases: list[str] = []
    risk_phrases: list[str] = []
    seen_relationships: set[tuple[str, str, str]] = set()
    direct_fact_risk_keys = {
        str(item.get("left_column") or item.get("right_column") or "")
        for item in relationships
        if isinstance(item, dict)
        and item.get("relationship_type") == "many_to_many"
        and {
            str(item.get("left_dataset") or ""),
            str(item.get("right_dataset") or ""),
        }.issubset(fact_sources)
    }
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        left_name = str(relationship.get("left_dataset") or "")
        right_name = str(relationship.get("right_dataset") or "")
        key = str(relationship.get("left_column") or relationship.get("right_column") or "")
        relationship_type = str(relationship.get("relationship_type") or "")
        if fact_sources and left_name not in fact_sources and right_name not in fact_sources:
            continue
        left_column = str(relationship.get("left_column") or key)
        right_column = str(relationship.get("right_column") or key)
        relationship_label = {
            "one_to_one": "1:1",
            "one_to_many": "1:N",
            "many_to_one": "N:1",
            "many_to_many": "M:N",
        }.get(relationship_type, relationship_type or "未知")
        many_rows = many_distinct = many_duplicates = 0
        if relationship_type == "one_to_many":
            many_rows = int(relationship.get("right_non_null_count") or 0)
            many_distinct = int(relationship.get("right_distinct_count") or 0)
            many_duplicates = int(relationship.get("right_duplicate_count") or 0)
        elif relationship_type == "many_to_one":
            many_rows = int(relationship.get("left_non_null_count") or 0)
            many_distinct = int(relationship.get("left_distinct_count") or 0)
            many_duplicates = int(relationship.get("left_duplicate_count") or 0)
        if (
            relationship_type in {"one_to_one", "one_to_many", "many_to_one"}
            and (not fact_sources or left_name in fact_sources or right_name in fact_sources)
            and (not direct_fact_risk_keys or key in direct_fact_risk_keys)
        ):
            signature = (left_name, right_name, f"{left_column}:{right_column}")
            if signature not in seen_relationships and len(relationship_phrases) < 8:
                detail = (
                    f"（多侧 {many_rows} 个非空键值、{many_distinct} 个唯一键，"
                    f"重复 {many_duplicates} 行）"
                    if relationship_type in {"one_to_many", "many_to_one"}
                    else ""
                )
                relationship_phrases.append(
                    f"{left_name}.{left_column} → {right_name}.{right_column} "
                    f"为 {relationship_label}{detail}"
                )
                seen_relationships.add(signature)
        if (
            relationship_type == "many_to_many"
            and (not fact_sources or {left_name, right_name}.issubset(fact_sources))
            and len(risk_phrases) < 4
        ):
            risk_phrases.append(
                f"{left_name} 与 {right_name} 的 {key} 在两侧都重复，"
                "直接逐行连接会形成多对多乘积并重复累计指标"
            )

    aggregate_phrases: list[str] = []
    evidence_ids: list[str] = []
    for item in aggregates[:12]:
        result = item.get("result") or {}
        rows = result.get("rows") or result.get("claim_rows") or []
        if not rows or not isinstance(rows[0], dict):
            continue
        metric = str(result.get("metric") or "metric")
        aggregation = str(result.get("aggregation") or "sum")
        value_key = f"{aggregation}_{metric}"
        value = rows[0].get(value_key)
        if value is None:
            continue
        aggregate_phrases.append(
            f"{result.get('source_dataset')}.{metric} 的 {aggregation.upper()}="
            f"{_format_evidence_number(value)}（源表 {int(result.get('source_row_count') or 0)} 行）"
        )
        evidence_ids.append(str(item["evidence_id"]))
    if relationship_item:
        evidence_ids.insert(0, str(relationship_item["evidence_id"]))

    sections: list[str] = []
    if relationship_phrases:
        sections.append("关系画像：" + "；".join(relationship_phrases))
    if risk_phrases:
        sections.append("基数风险：" + "；".join(risk_phrases))
    if aggregate_phrases:
        sections.append("原生粒度结果：" + "；".join(aggregate_phrases))
    if aggregates or risk_phrases:
        sections.append(
            "防重复方法：金额或数量先在各原始事实表按共同业务键分别预聚合到一行，"
            "再连接聚合结果；本次总额直接来自源表粒度，未对展开后的 join 行求和"
        )
    if not sections:
        return query_findings
    evidence_reference = "; ".join(
        f"evidence_id:{item}" for item in dict.fromkeys(evidence_ids)
    )
    return (
        *query_findings,
        InsightFindingResponse(
            title="多表关系、事实粒度与防重复口径",
            content="。".join(sections) + "。",
            data_source="tool_evidence.source_relationships_and_native_aggregates",
            evidence=evidence_reference,
            confidence="high",
            business_impact=(
                "避免把多个一对多事实表直接连接后重复放大金额、数量或其他可加指标。"
            ),
            recommended_action=(
                "跨事实表分析时保留源表聚合 SQL 和 evidence_id，并在共同业务键粒度预聚合后再连接。"
            ),
        ),
    )


def _relationship_analysis_requested(question: str) -> bool:
    folded = question.casefold()
    return any(
        token in folded
        for token in (
            "关系",
            "关联",
            "连接",
            "基数",
            "粒度",
            "重复",
            "放大",
            "一对多",
            "多对多",
            "join",
        )
    )


def _format_evidence_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.2f}"


def _merge_report_findings(
    preferred: tuple[InsightFindingResponse, ...],
    existing: tuple[InsightFindingResponse, ...],
) -> tuple[InsightFindingResponse, ...]:
    output: list[InsightFindingResponse] = []
    seen: set[str] = set()
    for finding in (*preferred, *existing):
        key = finding.title.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return tuple(output)


def _preserve_mandatory_report_findings(
    report: StructuredReportResponse,
    mandatory: tuple[InsightFindingResponse, ...],
) -> StructuredReportResponse:
    if not mandatory:
        return report
    existing_titles = {item.title.strip().casefold() for item in report.key_findings}
    missing = [
        item for item in mandatory if item.title.strip().casefold() not in existing_titles
    ]
    merged = _merge_report_findings(mandatory, report.key_findings)
    summary = report.executive_summary.strip()
    for finding in missing[:2]:
        if finding.content not in summary:
            summary = f"{summary} {finding.content}".strip()
    next_steps = tuple(
        dict.fromkeys(
            (
                *report.recommended_next_steps,
                *(
                    item.recommended_action
                    for item in mandatory
                    if item.recommended_action
                ),
            )
        )
    )
    return report.model_copy(
        update={
            "executive_summary": summary,
            "key_findings": merged,
            "recommended_next_steps": next_steps[:8],
        }
    )


def _preserve_verified_report_findings(
    report: StructuredReportResponse,
    verified_findings: tuple[InsightFindingResponse, ...],
    verification: StatisticalVerificationResponse | None,
) -> StructuredReportResponse:
    if verification is None:
        return report
    reviewed_titles = {
        verdict.title
        for verdict in verification.finding_verdicts
    }
    generated = tuple(
        finding
        for finding in report.key_findings
        if finding.title not in reviewed_titles
        and not _numeric_claim_tokens(finding.content)
    )
    return report.model_copy(
        update={
            "key_findings": _merge_report_findings(
                generated,
                verified_findings,
            )
        }
    )


def _report_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
    runtime: ReportNodeRuntime,
) -> Any:
    def run(state: WorkflowState) -> dict[str, Any]:
        runtime.notify_progress(
            state,
            stage=REPORT_NODE,
            progress=97,
            message="Generating and saving report.",
        )
        dataset_id = state["dataset_id"]
        question = state["question"]
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        planned_analysis = _require(
            state.get("planned_analysis"),
            "Planner did not produce an analysis plan.",
        )
        python_result = _require(
            state.get("python_result"),
            "Python Agent did not produce analysis results.",
        )
        sql_result = state.get("sql_result")
        rounds = state.get("rounds", ())
        analysis_framework = state.get("analysis_framework")
        draft = _build_report_draft(
            state,
            model_router,
            strategy="llm",
            agent="report",
        )
        final_insights = draft["final_insights"]
        validation_issues = draft["validation_issues"]
        report_charts = draft["report_charts"]
        structured_report = draft["structured_report"]
        report_markdown = draft["report_markdown"]
        html_report = draft["html_report"]
        report_source = draft["report_source"]
        model_router_provider = draft["model_router_provider"]
        model_router_model = draft["model_router_model"]
        model_router_error = draft["model_router_error"]
        if model_router is not None and report_source == "rules" and model_router_error:
            validation_issues = (
                *validation_issues,
                ValidationIssueResponse(
                    severity="info",
                    finding_ref="generate_structured_report",
                    issue="LLM structured report unavailable; rule report was used.",
                    suggestion=str(model_router_error),
                ),
            )
            structured_report = structured_report.model_copy(
                update={"validation_issues": validation_issues}
            )
            report_markdown = _markdown_from_structured_report(structured_report)
            html_report = render_structured_report_html(
                structured_report,
                title="DataMind 分析报告",
            )
        executed_nodes = (*state.get("executed_nodes", ()), REPORT_NODE)
        planner_metadata = state.get("planner_metadata")
        multi_dataset_context = state.get("multi_dataset_context")
        workflow_trace = _workflow_trace(
            state=state,
            executed_nodes=executed_nodes,
            report_source=report_source,
            provider=model_router_provider,
            model=model_router_model,
        )
        report_id = repository.save_report(
            dataset_id=dataset_id,
            title="DataMind 分析报告",
            markdown=report_markdown,
            job_id=state["run_id"],
            metadata={
                "question": question,
                "route": planned_analysis.route,
                "workflow": "langgraph_analysis",
                "nodes": list(executed_nodes),
                "analysis_framework": (
                    analysis_framework.model_dump(mode="json") if analysis_framework else None
                ),
                "planner_metadata": (
                    planner_metadata.model_dump(mode="json") if planner_metadata else None
                ),
                "analysis_contract": (
                    state["analysis_contract"].model_dump(mode="json")
                    if state.get("analysis_contract")
                    else None
                ),
                "statistical_verification": (
                    state["statistical_verification"].model_dump(mode="json")
                    if state.get("statistical_verification")
                    else None
                ),
                "analysis_lineage": (
                    state["analysis_lineage"].model_dump(mode="json")
                    if state.get("analysis_lineage")
                    else None
                ),
                "multi_dataset_context": (
                    multi_dataset_context.model_dump(mode="json") if multi_dataset_context else None
                ),
                "primary_dataset_id": str(dataset_id),
                "dataset_group_id": (
                    str(state.get("dataset_group_id")) if state.get("dataset_group_id") else None
                ),
                "additional_dataset_ids": [
                    str(item) for item in state.get("additional_dataset_ids", ())
                ],
                "join_plan": [item.model_dump(mode="json") for item in state.get("join_plan", ())],
                "relationship_plan": [
                    item.model_dump(mode="json") for item in state.get("relationship_plan", ())
                ],
                "join_summary": (
                    multi_dataset_context.join_summary if multi_dataset_context else {}
                ),
                "workflow_trace": [node.model_dump(mode="json") for node in workflow_trace],
                "multimodal_inputs": [
                    item.model_dump(mode="json") for item in state.get("multimodal_inputs", ())
                ],
                "planner_source": state.get("planner_source", "rules"),
                "python_source": state.get("python_source", "rules"),
                "python_generated_code": state.get("python_generated_code"),
                "python_execution_error": state.get("python_execution_error"),
                "python_attempts": [
                    attempt.model_dump(mode="json") for attempt in state.get("python_attempts", ())
                ],
                "report_source": report_source,
                "model_router_provider": model_router_provider,
                "model_router_model": model_router_model,
                "model_router_error": model_router_error,
                "sql_source": state.get("sql_source", "none"),
                "sql_validation_error": state.get("sql_validation_error"),
                "structured_report": structured_report.model_dump(mode="json"),
                "html_report": html_report,
                "validation_issue_count": len(validation_issues),
                "agent_mode": state.get("agent_mode", "legacy"),
                "loop_summary": state.get("loop_summary", {}),
                "loop_terminal_reason": state.get("loop_terminal_reason"),
            },
        )
        for chart in report_charts:
            repository.save_chart(
                dataset_id=dataset_id,
                title=chart.title,
                chart_type=chart.chart_type,
                chart_spec=chart.spec,
                chart_data=list(chart.data),
            )

        final_response = AnalysisRunResponse(
            dataset_id=dataset_id,
            dataset_group_id=state.get("dataset_group_id"),
            report_id=report_id,
            question=question,
            multimodal_inputs=state.get("multimodal_inputs", ()),
            plan=AnalysisPlanResponse(route=planned_analysis.route, steps=planned_analysis.steps),
            planner_metadata=planner_metadata,
            multi_dataset_context=multi_dataset_context,
            profile=profile,
            analysis_framework=analysis_framework,
            analysis_contract=state.get("analysis_contract"),
            statistical_verification=state.get("statistical_verification"),
            analysis_lineage=state.get("analysis_lineage"),
            sql_result=sql_result,
            python_result=python_result,
            rounds=rounds,
            final_insights=final_insights,
            validation_issues=validation_issues,
            structured_report=structured_report,
            html_report=html_report,
            sql_source=state.get("sql_source", "none"),
            python_source=state.get("python_source", "rules"),
            python_generated_code=state.get("python_generated_code"),
            python_execution_error=state.get("python_execution_error"),
            python_attempts=state.get("python_attempts", ()),
            workflow_trace=workflow_trace,
            report_markdown=report_markdown,
            agent_mode=state.get("agent_mode", "legacy"),
            loop_summary=state.get("loop_summary", {}),
            loop_terminal_reason=state.get("loop_terminal_reason"),
        )
        return {
            "rounds": rounds,
            "final_insights": final_insights,
            "validation_issues": validation_issues,
            "structured_report": structured_report,
            "html_report": html_report,
            "report_markdown": report_markdown,
            "report_charts": report_charts,
            "final_response": final_response,
            "executed_nodes": executed_nodes,
            "report_source": report_source,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
        }

    return run


def _parse_model_validation_issues(content: str) -> tuple[ValidationIssueResponse, ...]:
    payload = _extract_json_object(content)
    if payload is None:
        return ()
    raw_items = payload.get("issues")
    if not isinstance(raw_items, list):
        return ()
    issues: list[ValidationIssueResponse] = []
    for item in raw_items[:10]:
        if not isinstance(item, dict):
            continue
        issue = str(item.get("issue") or "").strip()
        if not issue:
            continue
        issues.append(
            ValidationIssueResponse(
                severity=str(item.get("severity") or "warning"),
                finding_ref=str(item.get("finding_ref") or "report"),
                issue=issue,
                suggestion=str(item.get("suggestion") or ""),
            )
        )
    return tuple(issues)


def _parse_model_structured_report(
    content: str,
    *,
    fallback: StructuredReportResponse,
    provider: str,
    model: str,
) -> StructuredReportResponse:
    payload = _extract_json_object(content)
    if payload is None:
        raise ValueError("Model Router report did not return JSON.")
    executive_summary = str(payload.get("executive_summary") or "").strip()
    if len(executive_summary) < 20:
        raise ValueError("Model Router report executive summary was too short.")

    key_findings = _parse_report_findings(payload.get("key_findings"))
    validation_issues = _parse_report_validation_issues(payload.get("validation_issues"))
    chart_explanations = _parse_chart_explanation_strings(payload.get("chart_explanations"))
    report_charts = _apply_report_chart_explanations(
        charts=fallback.charts,
        chart_explanations=chart_explanations,
    )
    return StructuredReportResponse(
        executive_summary=executive_summary,
        analysis_context=str(payload.get("analysis_context") or fallback.analysis_context),
        key_findings=key_findings or fallback.key_findings,
        charts=report_charts,
        chart_explanations=chart_explanations or fallback.chart_explanations,
        sql_results=fallback.sql_results,
        python_results=fallback.python_results,
        data_gaps=tuple(_string_list(payload.get("data_gaps"))) or fallback.data_gaps,
        validation_issues=validation_issues or fallback.validation_issues,
        recommended_next_steps=(
            tuple(_string_list(payload.get("recommended_next_steps")))
            or fallback.recommended_next_steps
        ),
        analysis_trace=fallback.analysis_trace,
        provider=provider,
        model=model,
    )


def _parse_report_findings(value: object) -> tuple[InsightFindingResponse, ...]:
    if not isinstance(value, list):
        return ()
    findings: list[InsightFindingResponse] = []
    for index, item in enumerate(value[:8], 1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        data_source = str(item.get("data_source") or item.get("evidence") or "").strip()
        if not content or not data_source:
            continue
        findings.append(
            InsightFindingResponse(
                title=str(item.get("title") or f"洞察 {index}"),
                content=content,
                data_source=data_source,
                impact_pct=_float_payload(item.get("impact_pct")),
                evidence=str(item.get("evidence") or data_source),
                confidence=str(item.get("confidence") or "medium"),
                business_impact=str(item.get("business_impact") or ""),
                recommended_action=str(item.get("recommended_action") or ""),
            )
        )
    return tuple(findings)


def _parse_report_validation_issues(value: object) -> tuple[ValidationIssueResponse, ...]:
    if not isinstance(value, list):
        return ()
    issues: list[ValidationIssueResponse] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        issue = str(item.get("issue") or "").strip()
        if not issue:
            continue
        issues.append(
            ValidationIssueResponse(
                severity=str(item.get("severity") or "info"),
                finding_ref=str(item.get("finding_ref") or "report"),
                issue=issue,
                suggestion=str(item.get("suggestion") or ""),
            )
        )
    return tuple(issues)


def _parse_chart_explanation_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    explanations: list[str] = []
    for item in value[:12]:
        if isinstance(item, dict):
            explanation = str(item.get("explanation") or "").strip()
            title = str(item.get("title") or "").strip()
            if explanation:
                explanations.append(f"{title}: {explanation}" if title else explanation)
        else:
            explanation = str(item).strip()
            if explanation:
                explanations.append(explanation)
    return tuple(explanations)


def _apply_report_chart_explanations(
    *,
    charts: tuple[ChartResponse, ...],
    chart_explanations: tuple[str, ...],
) -> tuple[ChartResponse, ...]:
    if not chart_explanations:
        return charts
    updated: list[ChartResponse] = []
    for chart in charts:
        explanation = chart.explanation
        for item in chart_explanations:
            prefix = f"{chart.title}:"
            if item.startswith(prefix):
                explanation = item.removeprefix(prefix).strip()
                break
        updated.append(
            ChartResponse(
                title=chart.title,
                chart_type=chart.chart_type,
                spec=chart.spec,
                data=chart.data,
                explanation=_preserve_chart_denominator_scope(chart, explanation),
                related_finding_ids=chart.related_finding_ids,
            )
        )
    return tuple(updated)


def _markdown_from_structured_report(report: StructuredReportResponse) -> str:
    lines = [
        "# DataMind 分析报告",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
    ]
    if report.analysis_context:
        lines.extend(["## Analysis Context", report.analysis_context, ""])
    if report.analysis_contract:
        contract = report.analysis_contract
        lines.extend(
            [
                "## Analysis Contract",
                f"- 目标: {contract.objective}",
                f"- 总体: {contract.population}",
                f"- 方法: {contract.method}",
                f"- 指标: {contract.metric or '计数/文本分析'}",
                f"- 粒度: {', '.join(contract.grain)}",
                "",
            ]
        )
    if report.statistical_verification:
        verification = report.statistical_verification
        lines.extend(
            [
                "## Statistical Verification",
                verification.summary,
                f"- 数值证据覆盖率: {verification.numeric_evidence_coverage:.0%}",
                *[
                    f"- {check.status} · {check.code}: {check.message}"
                    for check in verification.checks
                    if check.status != "not_applicable"
                ],
                "",
            ]
        )
    if report.key_findings:
        lines.append("## Key Findings")
        for finding in report.key_findings:
            lines.append(f"- **{finding.title}**: {finding.content}")
            if finding.evidence:
                lines.append(f"  - 证据: {finding.evidence}")
            if finding.recommended_action:
                lines.append(f"  - 建议: {finding.recommended_action}")
        lines.append("")
    if report.charts:
        lines.extend(
            [
                "## Visualizations",
                *[
                    f"- {chart.title} ({chart.chart_type}): {chart.explanation or '见图表数据。'}"
                    for chart in report.charts
                ],
                "",
            ]
        )
    if report.data_gaps:
        lines.extend(["## Data Gaps", *[f"- {gap}" for gap in report.data_gaps], ""])
    if report.validation_issues:
        lines.extend(
            [
                "## Validation Issues",
                *[
                    f"- {issue.severity} · {issue.finding_ref}: {issue.issue}"
                    for issue in report.validation_issues
                ],
                "",
            ]
        )
    if report.recommended_next_steps:
        lines.extend(
            [
                "## Recommended Next Steps",
                *[f"- {step}" for step in report.recommended_next_steps],
                "",
            ]
        )
    return "\n".join(lines)


def _preserve_chart_denominator_scope(
    chart: ChartResponse,
    explanation: str,
) -> str:
    scope = str(chart.spec.get("denominator_scope") or "")
    if chart.chart_type != "pie" or not scope:
        return explanation
    displayed = int(chart.spec.get("displayed_category_count") or len(chart.data))
    excluded = int(chart.spec.get("excluded_category_count") or 0)
    if scope == "displayed_top_n":
        required = (
            f"百分比以展示的前 {displayed} 类合计为分母；另有 {excluded} 类未展示，"
            "不代表全量占比。"
        )
    else:
        required = "百分比以查询返回的全部类别合计为分母。"
    base = str(explanation or "").strip()
    return base if required in base else f"{base} {required}".strip()


def _workflow_trace(
    *,
    state: WorkflowState,
    executed_nodes: tuple[str, ...],
    report_source: str,
    provider: str | None,
    model: str | None,
) -> tuple[WorkflowTraceNodeResponse, ...]:
    trace: list[WorkflowTraceNodeResponse] = []
    for node in executed_nodes:
        error = _node_error(state, node)
        trace.append(
            WorkflowTraceNodeResponse(
                node=node,
                status="fallback" if error else "completed",
                provider=None
                if node == STATISTICAL_VERIFY_NODE
                else (
                    provider
                    if node
                    in {
                        REPORT_NODE,
                        INTEGRATE_INSIGHTS_NODE,
                        ADVERSARIAL_VALIDATE_NODE,
                        FORMAT_CHARTS_NODE,
                    }
                    else state.get("model_router_provider")
                ),
                model=None
                if node == STATISTICAL_VERIFY_NODE
                else (
                    model
                    if node
                    in {
                        REPORT_NODE,
                        INTEGRATE_INSIGHTS_NODE,
                        ADVERSARIAL_VALIDATE_NODE,
                        FORMAT_CHARTS_NODE,
                    }
                    else state.get("model_router_model")
                ),
                input_summary=_node_input_summary(state, node),
                output_summary=_node_output_summary(state, node, report_source=report_source),
                fallback=_node_fallback(state, node, report_source=report_source),
                error=error,
            )
        )
    return tuple(trace)


def _node_error(state: WorkflowState, node: str) -> str | None:
    if node == JOIN_PREPARE_NODE:
        context = state.get("multi_dataset_context")
        if context and context.validation_issues:
            return "; ".join(issue.issue for issue in context.validation_issues[:2])
        return None
    if node == PLANNER_NODE:
        return state.get("model_router_error") if state.get("planner_source") == "rules" else None
    if node == SQL_NODE:
        return state.get("sql_validation_error")
    if node == PYTHON_NODE:
        error = state.get("python_execution_error")
        attempts = state.get("python_attempts", ())
        if error and len(attempts) >= 3:
            return f"LLM Python code failed after 3 attempts: {error}"
        return error
    if node == REPORT_NODE:
        return state.get("model_router_error") if state.get("report_source") == "rules" else None
    return None


def _node_fallback(
    state: WorkflowState,
    node: str,
    *,
    report_source: str,
) -> str | None:
    if node == PLANNER_NODE and state.get("planner_source") == "rules":
        return "rule_planner"
    if node == SQL_NODE and state.get("sql_source") == "rules":
        return "rule_sql"
    if node == PYTHON_NODE and state.get("python_source") == "rules":
        return "rule_python"
    if node == REPORT_NODE and report_source == "rules":
        return "rule_report"
    return None


def _node_input_summary(state: WorkflowState, node: str) -> str:
    profile = state.get("profile")
    if node == JOIN_PREPARE_NODE:
        additional_count = len(state.get("additional_dataset_ids", ()))
        join_count = len(state.get("join_plan", ()))
        return f"主数据集 + {additional_count} 个附加数据集，{join_count} 个 join 配置。"
    if node == PLANNER_NODE and profile is not None:
        return f"{profile.row_count} 行，{profile.column_count} 列，问题：{state['question']}"
    if node == SQL_NODE:
        return "使用 planner 输出和 dataframe 执行安全 SELECT。"
    if node == PYTHON_NODE:
        return "使用 dataframe、planner 输出和 SQL 结果生成 Python 分析。"
    if node == STATISTICAL_VERIFY_NODE:
        return "检查分析契约、逐条结论证据、比较支持与 Join 粒度。"
    if node == REPORT_NODE:
        return "整合 SQL、Python、图表、验证问题和多轮分析 trace。"
    return "使用上游节点输出。"


def _node_output_summary(
    state: WorkflowState,
    node: str,
    *,
    report_source: str,
) -> str:
    if node == PLANNER_NODE:
        plan = state.get("planned_analysis")
        return f"route={plan.route if plan else '-'}"
    if node == JOIN_PREPARE_NODE:
        context = state.get("multi_dataset_context")
        if not context:
            return "单数据集分析"
        summary = context.join_summary
        return (
            f"mode={summary.get('mode', '-')}, "
            f"joined_datasets={summary.get('joined_dataset_count', 1)}/{summary.get('dataset_count', 1)}, "
            f"joined_rows={summary.get('joined_row_count', '-')}, "
            f"joined_columns={summary.get('joined_column_count', '-')}, "
            f"row_expansion={summary.get('row_expansion_ratio', 1)}x, "
            f"skipped_joins={summary.get('skipped_join_count', 0)}"
        )
    if node == SQL_NODE:
        sql_result = state.get("sql_result")
        return f"{len(sql_result.rows) if sql_result else 0} 行 SQL 结果"
    if node == PYTHON_NODE:
        python_result = state.get("python_result")
        attempts = state.get("python_attempts", ())
        attempt_summary = f"，Python 代码尝试 {len(attempts)} 次" if attempts else ""
        return f"{len(python_result.insights) if python_result else 0} 条洞察，{len(python_result.charts) if python_result else 0} 个图表{attempt_summary}"
    if node == STATISTICAL_VERIFY_NODE:
        verification = state.get("statistical_verification")
        return verification.summary if verification else "未生成统计审查结果"
    if node == REPORT_NODE:
        return f"report_source={report_source}"
    if node == ROUND_REFLECT_NODE:
        return f"{len(state.get('rounds', ()))} 个分析轮次"
    return "节点完成"


def _merge_model_report(*, base_report: str, model_content: str) -> str:
    narrative = model_content.strip()
    if len(narrative) < 40:
        raise ValueError("Model Router report was too short.")
    if narrative.startswith("[mock:"):
        raise ValueError("Mock model output is not a usable report narrative.")
    return f"{base_report}\n\n## Model Router Narrative\n\n{narrative}\n"
