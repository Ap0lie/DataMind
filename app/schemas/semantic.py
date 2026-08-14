from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.schemas.analysis_intent import (
    AnalysisIntentSpec,
    ContractGuardResult,
    IntentCompilationAttempt,
    IntentGuardResult,
)
from app.schemas.common import ApiModel

ScopeType = Literal["dataset", "dataset_group"]
SemanticStatus = Literal["draft", "published", "stale", "archived"]


class SemanticModelDraftRequest(ApiModel):
    scope_type: ScopeType
    scope_id: UUID
    name: str | None = None
    source_model_id: UUID | None = None


class SemanticModelUpdateRequest(ApiModel):
    revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1)
    definition: dict[str, Any]


class SemanticModelCopyRequest(ApiModel):
    scope_type: ScopeType
    scope_id: UUID
    name: str | None = None


class SemanticValidationResponse(ApiModel):
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_fingerprint: str = ""


class SemanticModelResponse(ApiModel):
    model_id: UUID
    user_id: str
    scope_type: ScopeType
    scope_id: UUID
    name: str
    version: int = Field(ge=1)
    revision: int = Field(ge=1)
    status: SemanticStatus
    source: str
    parent_model_id: UUID | None = None
    definition: dict[str, Any] = Field(default_factory=dict)
    schema_fingerprint: str = ""
    validation: SemanticValidationResponse | None = None
    created_at: str | None = None
    updated_at: str | None = None
    published_at: str | None = None


class SemanticModelListResponse(ApiModel):
    models: tuple[SemanticModelResponse, ...]


class SemanticPlanRequest(ApiModel):
    dataset_id: UUID
    question: str = Field(min_length=1)
    dataset_group_id: UUID | None = None
    additional_dataset_ids: tuple[UUID, ...] = ()


class PlannerConfidenceBreakdown(ApiModel):
    intent: float = Field(ge=0, le=1)
    metric: float | None = Field(default=None, ge=0, le=1)
    dimension: float | None = Field(default=None, ge=0, le=1)
    time: float | None = Field(default=None, ge=0, le=1)
    join: float | None = Field(default=None, ge=0, le=1)
    data_quality: float = Field(ge=0, le=1)
    route: float = Field(ge=0, le=1)


class PlannerDecisionResponse(ApiModel):
    decision_id: UUID
    dataset_id: UUID
    dataset_group_id: UUID | None = None
    question: str
    semantic_model_id: UUID | None = None
    semantic_model_version: int | None = None
    semantic_source: str = "legacy"
    semantic_plan: dict[str, Any] = Field(default_factory=dict)
    relationship_graph: dict[str, Any] = Field(default_factory=dict)
    grain_plan: dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: PlannerConfidenceBreakdown
    raw_confidence: float = Field(ge=0, le=1)
    calibrated_confidence: float = Field(ge=0, le=1)
    confidence_level: Literal["low", "medium", "high"]
    requires_confirmation: bool = False
    ambiguities: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    intent_spec: AnalysisIntentSpec | None = None
    intent_validation: IntentGuardResult | None = None
    intent_attempts: tuple[IntentCompilationAttempt, ...] = ()
    contract_validation: ContractGuardResult | None = None
    confirmation_reasons: tuple[str, ...] = ()
    created_at: str | None = None


class PlannerFeedbackRequest(ApiModel):
    action: Literal["accepted", "edited", "rejected"]
    corrected_plan: dict[str, Any] = Field(default_factory=dict)


class PlannerFeedbackResponse(ApiModel):
    feedback_id: UUID
    decision_id: UUID
    action: str
    created_at: str
