from __future__ import annotations

from app.schemas.analysis import (
    ChartResponse,
    InsightFindingResponse,
    ValidationIssueResponse,
)


def validate_analysis_plan(
    *,
    finding_ref: str,
    route: str,
    category_column: str | None,
    metric_column: str | None,
    time_column: str | None,
    steps: tuple[str, ...],
    available_columns: set[str],
) -> tuple[ValidationIssueResponse, ...]:
    issues: list[ValidationIssueResponse] = []
    if route not in {"sql", "python", "hybrid"}:
        issues.append(
            ValidationIssueResponse(
                severity="critical",
                finding_ref=finding_ref,
                issue=f"Unsupported plan route: {route}.",
                suggestion="Use sql, python, or hybrid.",
            )
        )
    for label, column in (
        ("category_column", category_column),
        ("metric_column", metric_column),
        ("time_column", time_column),
    ):
        if column and column not in available_columns:
            issues.append(
                ValidationIssueResponse(
                    severity="critical",
                    finding_ref=finding_ref,
                    issue=f"Plan references unknown {label}: {column}.",
                    suggestion="Use only columns from the uploaded dataset schema.",
                )
            )
    if not steps:
        issues.append(
            ValidationIssueResponse(
                severity="warning",
                finding_ref=finding_ref,
                issue="Plan has no executable steps.",
                suggestion="Provide at least one concrete SQL, Python, or chart step.",
            )
        )
    return tuple(issues)


def validate_findings_traceability(
    findings: tuple[InsightFindingResponse, ...],
) -> tuple[ValidationIssueResponse, ...]:
    return tuple(
        ValidationIssueResponse(
            severity="critical",
            finding_ref=finding.title,
            issue="Finding is missing a data source.",
            suggestion="Attach the SQL, Python, or round result that supports it.",
        )
        for finding in findings
        if not finding.data_source.strip()
    )


def validate_chart_specs(
    charts: tuple[ChartResponse, ...],
) -> tuple[ValidationIssueResponse, ...]:
    supported = {"bar", "line", "pie", "histogram", "box_plot", "correlation_heatmap"}
    issues: list[ValidationIssueResponse] = []
    for chart in charts:
        if chart.chart_type not in supported:
            issues.append(
                ValidationIssueResponse(
                    severity="warning",
                    finding_ref=chart.title,
                    issue=f"Unsupported chart type: {chart.chart_type}.",
                    suggestion="Use bar, line, pie, histogram, box_plot, or correlation_heatmap.",
                )
            )
        if not chart.data:
            issues.append(
                ValidationIssueResponse(
                    severity="warning",
                    finding_ref=chart.title,
                    issue="Chart has no data rows.",
                    suggestion="Skip the chart or provide non-empty chart data.",
                )
            )
    return tuple(issues)
