from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.analysis.services import render_structured_report_html
from app.schemas.analysis import ChartResponse, StructuredReportResponse

_CONCISE_TERMS = ("简", "精炼", "浓缩", "摘要", "short", "concise")
_BEAUTIFY_TERMS = ("美化", "视觉", "样式", "配色", "图表", "beaut", "style", "visual")
_REPORT_PALETTE = ("#0f766e", "#2563eb", "#d97706", "#7c3aed", "#dc2626", "#0891b2")


@dataclass(frozen=True)
class ReportRevision:
    title: str
    markdown: str
    metadata: dict[str, Any]
    source_evidence_fingerprint: str


def revise_report_snapshot(
    *, report: dict[str, Any], report_id: UUID, instruction: str
) -> ReportRevision:
    """Create a presentation-only report version without rerunning analysis.

    Narrative blocks are selected from the source report rather than regenerated, and chart
    records are copied byte-for-byte. This makes a report revision unable to silently change
    the analytical result that supports it.
    """

    source_metadata = (
        deepcopy(report.get("metadata")) if isinstance(report.get("metadata"), dict) else {}
    )
    structured_payload = source_metadata.get("structured_report")
    if not isinstance(structured_payload, dict):
        structured_payload = {
            "executive_summary": str(report.get("markdown") or "报告内容为空。"),
        }
    source = StructuredReportResponse.model_validate(structured_payload)
    normalized_instruction = instruction.strip()
    lowered = normalized_instruction.casefold()
    concise = any(term in lowered for term in _CONCISE_TERMS)
    beautify = any(term in lowered for term in _BEAUTIFY_TERMS)

    charts = tuple(_restyle_chart(chart) for chart in source.charts)
    if concise:
        charts = charts[:6]
    revised = source.model_copy(
        update={
            "executive_summary": _concise_summary(source.executive_summary)
            if concise
            else source.executive_summary,
            "key_findings": source.key_findings[:4] if concise else source.key_findings,
            "charts": charts,
            "chart_explanations": tuple(
                chart.explanation for chart in charts if chart.explanation
            ),
            "data_gaps": source.data_gaps[:4] if concise else source.data_gaps,
            "validation_issues": source.validation_issues[:6]
            if concise
            else source.validation_issues,
            "recommended_next_steps": source.recommended_next_steps[:3]
            if concise
            else source.recommended_next_steps,
            "analysis_trace": () if concise else source.analysis_trace,
        }
    )
    _assert_revision_uses_source_evidence(source=source, revised=revised)

    source_fingerprint = report_evidence_fingerprint(source)
    revision_kind = "+".join(
        value for enabled, value in ((concise, "concise"), (beautify, "visual")) if enabled
    ) or "presentation"
    title = _revision_title(str(report.get("title") or "DataMind 分析报告"), concise)
    metadata = source_metadata | {
        "structured_report": revised.model_dump(mode="json"),
        "html_report": render_structured_report_html(revised, title=title),
        "report_revision": {
            "source_report_id": str(report_id),
            "source_job_id": str(report.get("job_id")) if report.get("job_id") else None,
            "instruction": normalized_instruction,
            "kind": revision_kind,
            "analysis_rerun": False,
            "evidence_frozen": True,
            "source_evidence_fingerprint": source_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
        },
    }
    return ReportRevision(
        title=title,
        markdown=_markdown_from_report(revised, title=title),
        metadata=metadata,
        source_evidence_fingerprint=source_fingerprint,
    )


def report_evidence_fingerprint(report: StructuredReportResponse) -> str:
    payload = {
        "sql_results": report.sql_results,
        "python_results": report.python_results,
        "charts": [
            {
                "title": chart.title,
                "chart_type": chart.chart_type,
                "data": chart.data,
                "related_finding_ids": chart.related_finding_ids,
            }
            for chart in report.charts
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assert_revision_uses_source_evidence(
    *, source: StructuredReportResponse, revised: StructuredReportResponse
) -> None:
    source_charts = {
        (chart.title, chart.chart_type): chart.model_dump(mode="json") for chart in source.charts
    }
    for chart in revised.charts:
        original = source_charts.get((chart.title, chart.chart_type))
        if original is None or original["data"] != chart.model_dump(mode="json")["data"]:
            raise RuntimeError("Report revision attempted to change frozen chart evidence.")
    if revised.sql_results != source.sql_results or revised.python_results != source.python_results:
        raise RuntimeError("Report revision attempted to change frozen analytical metrics.")


def _restyle_chart(chart: ChartResponse) -> ChartResponse:
    spec = deepcopy(chart.spec)
    spec["presentation"] = {
        "theme": "datamind_report_v2",
        "palette": list(_REPORT_PALETTE),
        "background": "#ffffff",
        "grid_color": "#e2e8f0",
        "label_color": "#334155",
        "show_legend": True,
    }
    return chart.model_copy(update={"spec": spec})


def _concise_summary(value: str) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[。！？!?])", value) if item.strip()]
    return "".join(sentences[:2]) if sentences else value


def _revision_title(source_title: str, concise: bool) -> str:
    suffix = "精简版" if concise else "修订版"
    base = re.sub(r"\s*[·-]\s*(?:精简版|修订版)\s*$", "", source_title).strip()
    return f"{base} · {suffix}"


def _markdown_from_report(report: StructuredReportResponse, *, title: str) -> str:
    lines = [f"# {title}", "", "## 核心摘要", report.executive_summary, ""]
    if report.key_findings:
        lines.append("## 关键结论")
        for finding in report.key_findings:
            lines.append(f"- **{finding.title}**：{finding.content}")
            if finding.evidence:
                lines.append(f"  - 证据：{finding.evidence}")
        lines.append("")
    if report.charts:
        lines.extend(
            [
                "## 图表",
                *[
                    f"- {chart.title}：{chart.explanation or '图表数据沿用原分析证据。'}"
                    for chart in report.charts
                ],
                "",
            ]
        )
    if report.validation_issues:
        lines.extend(
            [
                "## 校验提示",
                *[
                    f"- {issue.severity} · {issue.finding_ref}：{issue.issue}"
                    for issue in report.validation_issues
                ],
                "",
            ]
        )
    if report.recommended_next_steps:
        lines.extend(
            ["## 下一步建议", *[f"- {step}" for step in report.recommended_next_steps], ""]
        )
    return "\n".join(lines)
