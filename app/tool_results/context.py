from __future__ import annotations

from app.tool_results.contracts import (
    DistillationAttempt,
    ToolContextBundle,
    ToolResultArtifact,
    ToolResultSummary,
)
from app.tool_results.reducers import summary_for_model


def build_tool_context_bundle(
    artifact: ToolResultArtifact,
    summary: ToolResultSummary,
    *,
    max_context_chars: int | None = None,
    attempts: tuple[DistillationAttempt, ...] = (),
) -> ToolContextBundle:
    summary = fit_tool_summary(summary, max_chars=max_context_chars)
    model_content = summary_for_model(summary)
    context_size = len(model_content.encode("utf-8"))
    ratio = context_size / artifact.size_bytes if artifact.size_bytes else 0.0
    return ToolContextBundle(
        artifact_id=artifact.artifact_id,
        summary=summary,
        model_content=model_content,
        original_size_bytes=artifact.size_bytes,
        context_size_bytes=context_size,
        reduction_ratio=min(1.0, ratio),
        distillation_attempts=attempts,
    )


def fit_tool_summary(
    summary: ToolResultSummary,
    *,
    max_chars: int | None,
) -> ToolResultSummary:
    if not max_chars or len(summary_for_model(summary)) <= max_chars:
        return summary
    plans = (
        (12, 40, 8, 10),
        (8, 24, 6, 6),
        (4, 16, 4, 3),
        (0, 12, 3, 0),
    )
    for preview, facts, findings, fields in plans:
        candidate = summary.model_copy(
            update={
                "preview": summary.preview[:preview],
                "canonical_facts": summary.canonical_facts[:facts],
                "key_findings": tuple(item[:800] for item in summary.key_findings[:findings]),
                "schema_fields": summary.schema_fields[:fields] if fields else (),
                "omitted_sections": tuple(
                    dict.fromkeys((*summary.omitted_sections, "context_budget_reduction"))
                ),
            }
        )
        if len(summary_for_model(candidate)) <= max_chars:
            return candidate
    return candidate
