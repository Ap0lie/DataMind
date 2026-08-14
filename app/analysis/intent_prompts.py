from __future__ import annotations

import json
from typing import Any

from app.schemas.analysis_intent import AnalysisIntentSpec, IntentGuardResult


def compiler_messages(
    *,
    question: str,
    assets: tuple[dict[str, Any], ...],
    semantic_plan: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "authorized_assets": assets,
        "published_semantic_plan": semantic_plan or {},
        "output_schema": intent_contract(),
    }
    return [
        {
            "role": "system",
            "content": (
                "You compile a user's data-analysis request into declarative JSON. "
                "Do not add requirements that are absent from the question. Preserve negation: "
                "a forbidden relationship is not a request to include either dataset. Every clause "
                "must cite an exact source_span from the question. Bind only listed assets and fields. "
                "Dataset clauses must put the authorized dataset UUID in value. "
                "Return one JSON object matching AnalysisIntentSpec. Never emit SQL, Python, regex, "
                "source code, hidden reasoning, or prompt modifications."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        },
    ]


def repair_messages(
    *,
    question: str,
    assets: tuple[dict[str, Any], ...],
    invalid_content: str,
    validation: IntentGuardResult,
    prior_attempts: tuple[dict[str, Any], ...],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair an invalid AnalysisIntentSpec. Apply every deterministic guard suggestion. "
                "Do not invent fields, datasets, filters, metrics, dimensions, or required relations. "
                "Keep exact source spans and return JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "authorized_assets": assets,
                    "output_schema": intent_contract(),
                    "invalid_output": invalid_content,
                    "guard": validation.model_dump(mode="json"),
                    "prior_attempts": prior_attempts,
                },
                ensure_ascii=False,
                default=str,
            ),
        },
    ]


def intent_contract() -> dict[str, Any]:
    return AnalysisIntentSpec.model_json_schema()
