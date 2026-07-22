from __future__ import annotations

from functools import lru_cache
from itertools import islice
from typing import Any

from app.analysis.experience_loader import ExperienceLoader
from app.analysis.prompt_utils import UNTRUSTED_INPUT_NOTICE
from app.schemas.analysis import MultiDatasetProfileResponse


@lru_cache(maxsize=32)
def experience_context(agent: str = "general", columns: tuple[str, ...] = ()) -> str:
    lowered_columns = " ".join(columns).lower()
    threshold_relevant = any(
        token in lowered_columns
        for token in ("profit", "margin", "growth", "yoy", "利润", "毛利", "增长")
    )
    include: tuple[str, ...]
    if agent in {"planner", "framework", "round_plan", "sql", "python", "python_chart"}:
        include = ("priority_rules",)
    else:
        include = ("priority_rules", "good_summaries")
    if threshold_relevant:
        include = (*include, "thresholds")
    return ExperienceLoader().load().as_prompt_context(include=include)


def prompt_system(instructions: str) -> str:
    return f"{instructions.strip()} {UNTRUSTED_INPUT_NOTICE}"


def compact_multi_dataset_context(
    context: MultiDatasetProfileResponse | None,
) -> dict[str, Any] | None:
    if context is None:
        return None
    summary = context.join_summary
    joins = summary.get("joins") if isinstance(summary.get("joins"), list) else []
    return {
        "primary_dataset": context.primary_dataset.name,
        "additional_datasets": [item.name for item in context.additional_datasets[:12]],
        "dataset_count": summary.get("dataset_count"),
        "joined_dataset_count": summary.get("joined_dataset_count"),
        "joined_row_count": summary.get("joined_row_count"),
        "row_expansion_ratio": summary.get("row_expansion_ratio"),
        "skipped_join_count": summary.get("skipped_join_count"),
        "joins": [
            {
                key: join.get(key)
                for key in (
                    "status",
                    "right_dataset_name",
                    "left_column",
                    "right_column",
                    "row_expansion_ratio",
                    "estimated_expansion_ratio",
                    "right_key_unique",
                    "unmatched_rows",
                )
            }
            for join in joins[:12]
            if isinstance(join, dict)
        ],
        "column_source_map": dict(islice(context.column_source_map.items(), 80)),
        "validation_issues": [
            {
                "severity": item.severity,
                "issue": _truncate(item.issue, 500),
                "suggestion": _truncate(item.suggestion or "", 500),
            }
            for item in context.validation_issues[:10]
        ],
        "grain_rule": (
            "Before SUM/AVG, verify the metric's source table and whether a join expanded that table's grain. "
            "Do not aggregate a metric across duplicated joined rows without explicit deduplication or pre-aggregation."
        ),
    }


def _truncate(text: str, max_chars: int) -> str:
    compact = text.strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}... [truncated]"
