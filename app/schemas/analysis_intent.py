from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.common import ApiModel


class IntentSourceSpan(ApiModel):
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> IntentSourceSpan:
        if self.end <= self.start:
            raise ValueError("Intent source span end must be greater than start.")
        return self


class FieldBinding(ApiModel):
    column: str = Field(min_length=1)
    dataset_id: UUID | None = None
    dataset_name: str | None = None
    dtype: str | None = None
    role: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class IntentAggregation(ApiModel):
    operation: Literal["sum", "avg", "min", "max", "count", "count_distinct"]
    field: FieldBinding | None = None
    alias: str = Field(min_length=1)


class IntentFilter(ApiModel):
    field: FieldBinding
    operator: Literal["=", "!=", ">", ">=", "<", "<="] = "="
    value: str | int | float | bool


class RelationshipConstraint(ApiModel):
    left_dataset_id: UUID
    right_dataset_id: UUID
    operation: str = Field(default="row_level_join", min_length=1)
    polarity: Literal["required", "forbidden", "preferred"] = "forbidden"
    source_span: IntentSourceSpan


class IntentClause(ApiModel):
    clause_id: str = Field(min_length=1)
    kind: Literal[
        "metric",
        "dimension",
        "filter",
        "dataset",
        "relationship",
        "grain",
        "time",
        "output",
    ]
    polarity: Literal["required", "forbidden", "preferred"]
    concept: str = Field(min_length=1)
    source_span: IntentSourceSpan
    field: FieldBinding | None = None
    aggregation: str | None = None
    operator: str | None = None
    value: str | int | float | bool | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class AnalysisIntentSpec(ApiModel):
    version: str = "1"
    question: str = Field(min_length=1)
    source: Literal["rules", "llm", "hybrid"] = "rules"
    clauses: tuple[IntentClause, ...] = ()
    required_metric: FieldBinding | None = None
    candidate_metrics: tuple[FieldBinding, ...] = ()
    required_dimensions: tuple[FieldBinding, ...] = ()
    candidate_dimensions: tuple[FieldBinding, ...] = ()
    time_field: FieldBinding | None = None
    aggregations: tuple[IntentAggregation, ...] = ()
    filters: tuple[IntentFilter, ...] = ()
    derived_metrics: tuple[str, ...] = ()
    dataset_allowlist: tuple[UUID, ...] = ()
    dataset_denylist: tuple[UUID, ...] = ()
    strict_dataset_scope: bool = False
    relationship_constraints: tuple[RelationshipConstraint, ...] = ()
    confidence: float = Field(default=1.0, ge=0, le=1)
    requires_confirmation: bool = False
    confirmation_reasons: tuple[str, ...] = ()


class IntentGuardIssue(ApiModel):
    code: str = Field(min_length=1)
    severity: Literal["warning", "error"] = "error"
    message: str = Field(min_length=1)
    suggestion: str = ""
    clause_id: str | None = None


class IntentGuardResult(ApiModel):
    status: Literal["passed", "repairable", "confirmation_required", "failed"]
    issues: tuple[IntentGuardIssue, ...] = ()
    confidence: float = Field(default=1.0, ge=0, le=1)


class ContractGuardResult(ApiModel):
    status: Literal["passed", "repairable", "confirmation_required", "failed"]
    issues: tuple[IntentGuardIssue, ...] = ()
    preserved_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()


class IntentCompilationAttempt(ApiModel):
    attempt: int = Field(ge=1)
    status: Literal["failed", "succeeded"]
    provider: str | None = None
    model: str | None = None
    error: str | None = None
    guard: IntentGuardResult | None = None


class IntentCompilationResult(ApiModel):
    intent: AnalysisIntentSpec
    validation: IntentGuardResult
    attempts: tuple[IntentCompilationAttempt, ...] = ()
    mode: Literal["shadow", "enforce"] = "shadow"
    model_intent: AnalysisIntentSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
