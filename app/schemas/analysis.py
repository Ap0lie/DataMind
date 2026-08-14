from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.analysis_intent import (
    AnalysisIntentSpec,
    ContractGuardResult,
    IntentCompilationAttempt,
    IntentGuardResult,
)
from app.schemas.common import ApiModel
from app.schemas.prompt_overrides import AgentPromptOverrides


class DatasetColumnProfile(ApiModel):
    name: str
    dtype: str
    missing_count: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    is_numeric: bool
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None


class DatasetProfileResponse(ApiModel):
    dataset_id: UUID
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    missing_value_count: int = Field(ge=0)
    missing_value_ratio: float = Field(ge=0)
    duplicate_row_count: int = Field(ge=0)
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    columns: tuple[DatasetColumnProfile, ...]
    sample_records: tuple[dict[str, Any], ...]


class DatasetJoinConfig(ApiModel):
    left_dataset_id: UUID
    right_dataset_id: UUID
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)
    join_type: str = Field(default="left", pattern="^(left|inner)$")
    left_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    right_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    left_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    right_delimiter: str | None = Field(default=None, min_length=1, max_length=4)


class DatasetReferenceResponse(ApiModel):
    dataset_id: UUID
    name: str
    status: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: tuple[str, ...] = ()


class JoinSuggestionCandidateResponse(ApiModel):
    left_dataset_id: UUID
    right_dataset_id: UUID
    left_column: str
    right_column: str
    join_type: str = "left"
    score: float = Field(ge=0, le=1)
    reason: str
    left_type: str = ""
    right_type: str = ""
    left_role: str = ""
    right_role: str = ""
    estimated_match_rate: float = Field(default=0, ge=0, le=1)
    left_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    right_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    left_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    right_delimiter: str | None = Field(default=None, min_length=1, max_length=4)


class JoinSuggestionRequest(ApiModel):
    dataset_id: UUID
    additional_dataset_ids: tuple[UUID, ...] = ()


class MultiDatasetProfileResponse(ApiModel):
    primary_dataset: DatasetReferenceResponse
    additional_datasets: tuple[DatasetReferenceResponse, ...] = ()
    join_plan: tuple[DatasetJoinConfig, ...] = ()
    join_summary: dict[str, Any] = Field(default_factory=dict)
    joined_profile: DatasetProfileResponse | None = None
    column_source_map: dict[str, str] = Field(default_factory=dict)
    validation_issues: tuple[ValidationIssueResponse, ...] = ()


class JoinSuggestionResponse(ApiModel):
    primary_dataset: DatasetReferenceResponse
    additional_datasets: tuple[DatasetReferenceResponse, ...] = ()
    suggestions: tuple[JoinSuggestionCandidateResponse, ...] = ()
    validation_issues: tuple[ValidationIssueResponse, ...] = ()


class AnalysisRunRequest(ApiModel):
    dataset_id: UUID
    question: str = Field(min_length=1)
    dataset_group_id: UUID | None = None
    additional_dataset_ids: tuple[UUID, ...] = ()
    join_plan: tuple[DatasetJoinConfig, ...] = ()
    relationship_plan: tuple[DatasetJoinConfig, ...] = ()
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = ()
    planner_decision_id: UUID | None = None
    confirmed_low_confidence: bool = False
    agent_mode: str = Field(default="auto", pattern="^(auto|legacy|loop)$")
    prompt_overrides: AgentPromptOverrides = Field(default_factory=AgentPromptOverrides)


class MultimodalInputResponse(ApiModel):
    kind: str = Field(default="note", pattern="^(image|chart|pdf_page|screenshot|note)$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_ref: str | None = None
    media_type: str | None = None
    data_url: str | None = None
    processing_status: str | None = None
    text_excerpt: str | None = None


class AnalysisPlanResponse(ApiModel):
    route: str
    steps: tuple[str, ...]


class PlannerMetadataResponse(ApiModel):
    confidence: float = Field(default=0, ge=0, le=1)
    route_reason: str = ""
    candidate_metrics: tuple[str, ...] = ()
    candidate_dimensions: tuple[str, ...] = ()
    candidate_time_fields: tuple[str, ...] = ()
    candidate_text_fields: tuple[str, ...] = ()
    clarifying_questions: tuple[str, ...] = ()
    multi_dataset_summary: dict[str, Any] = Field(default_factory=dict)
    semantic_model_id: UUID | None = None
    semantic_model_version: int | None = None
    semantic_source: str = "legacy"
    semantic_plan: dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: dict[str, float | None] = Field(default_factory=dict)
    raw_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_level: str = "medium"
    requires_confirmation: bool = False


class WorkflowTraceNodeResponse(ApiModel):
    node: str
    status: str = "completed"
    provider: str | None = None
    model: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    fallback: str | None = None
    error: str | None = None


class SQLAnalysisResponse(ApiModel):
    sql: str
    rows: tuple[dict[str, Any], ...]
    explanation: str


class ChartResponse(ApiModel):
    title: str
    chart_type: str
    spec: dict[str, Any]
    data: tuple[dict[str, Any], ...]
    explanation: str = ""
    related_finding_ids: tuple[str, ...] = ()


class TextAnalysisResultResponse(ApiModel):
    task: str
    text_column: str
    group_column: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    insights: tuple[str, ...] = ()
    charts: tuple[ChartResponse, ...] = ()


class AnalysisAggregationResponse(ApiModel):
    operation: str = Field(
        pattern="^(sum|avg|min|max|count|count_distinct)$"
    )
    column: str | None = None
    alias: str


class AnalysisFilterResponse(ApiModel):
    column: str
    operator: str = Field(default="=", pattern="^(=|!=|>|>=|<|<=)$")
    value: str | int | float | bool


class PythonExecutionContextResponse(ApiModel):
    source_row_count: int = Field(ge=0)
    input_row_count: int = Field(ge=0)
    input_evidence_id: str | None = None
    applied_filters: tuple[AnalysisFilterResponse, ...] = ()
    referenced_columns: tuple[str, ...] = ()


class PythonAnalysisResponse(ApiModel):
    statistics: dict[str, Any]
    insights: tuple[str, ...]
    charts: tuple[ChartResponse, ...]
    text_analysis: tuple[TextAnalysisResultResponse, ...] = ()
    execution_context: PythonExecutionContextResponse | None = None


class PythonCodeAttemptResponse(ApiModel):
    attempt: int = Field(ge=1)
    phase: str = "python"
    status: str = Field(pattern="^(failed|succeeded)$")
    code: str | None = None
    error: str | None = None
    provider: str | None = None
    model: str | None = None


class AnalysisFrameworkResponse(ApiModel):
    business_question: str = ""
    candidate_dimensions: tuple[str, ...] = ()
    candidate_metrics: tuple[str, ...] = ()
    likely_routes: tuple[str, ...] = ()
    initial_hypotheses: tuple[str, ...] = ()
    risk_notes: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    key_questions: tuple[str, ...] = ()
    success_criteria: str = ""


class AnalysisContractResponse(ApiModel):
    contract_version: str = "1"
    objective: str
    population: str
    dataset_ids: tuple[UUID, ...] = ()
    analysis_type: str
    metric: str | None = None
    dimensions: tuple[str, ...] = ()
    time_field: str | None = None
    aggregations: tuple[AnalysisAggregationResponse, ...] = ()
    filters: tuple[AnalysisFilterResponse, ...] = ()
    grain: tuple[str, ...] = ()
    hypothesis: str | None = None
    method: str
    assumptions: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    causal_claim_allowed: bool = False
    analysis_budget: dict[str, int | float] = Field(default_factory=dict)


class StatisticalCheckResponse(ApiModel):
    code: str
    status: str = Field(pattern="^(passed|warning|failed|not_applicable)$")
    severity: str = Field(pattern="^(info|warning|error)$")
    message: str
    finding_ref: str = "analysis"
    evidence_ids: tuple[str, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)


class StatisticalFindingVerdictResponse(ApiModel):
    finding_ref: str
    title: str
    status: str = Field(pattern="^(passed|warning|failed)$")
    evidence_ids: tuple[str, ...] = ()
    sample_size: int | None = Field(default=None, ge=0)
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    notes: tuple[str, ...] = ()


class StatisticalVerificationResponse(ApiModel):
    status: str = Field(pattern="^(passed|warning|failed)$")
    summary: str
    checks: tuple[StatisticalCheckResponse, ...] = ()
    finding_verdicts: tuple[StatisticalFindingVerdictResponse, ...] = ()
    requires_replan: bool = False
    numeric_evidence_coverage: float = Field(default=1, ge=0, le=1)


class LineageNodeResponse(ApiModel):
    node_id: str
    node_type: str
    label: str
    source_ref: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LineageEdgeResponse(ApiModel):
    source_node_id: str
    target_node_id: str
    relation: str


class AnalysisLineageResponse(ApiModel):
    nodes: tuple[LineageNodeResponse, ...] = ()
    edges: tuple[LineageEdgeResponse, ...] = ()
    relationship_graph: dict[str, Any] = Field(default_factory=dict)
    grain_plan: dict[str, Any] = Field(default_factory=dict)


class AnalysisHypothesisResponse(ApiModel):
    statement: str
    judgment_criteria: str
    expected_direction: str = "unknown"


class AnalysisRoundPlanResponse(ApiModel):
    round_number: int = Field(ge=1)
    analytical_step: str
    question: str
    route: str
    metric_column: str | None = None
    category_column: str | None = None
    time_column: str | None = None


class AnalysisReflectionResponse(ApiModel):
    insight_text: str
    impact_pct: float = 0
    has_insight: bool = True
    decision: str = "CONTINUE"
    data_source: str = ""


class AnalysisRoundResponse(ApiModel):
    round_number: int = Field(ge=1)
    hypothesis: AnalysisHypothesisResponse
    plan: AnalysisRoundPlanResponse
    reflection: AnalysisReflectionResponse
    execution_result: dict[str, Any] = Field(default_factory=dict)
    charts: tuple[ChartResponse, ...] = ()
    validation_status: str = "passed"


class InsightFindingResponse(ApiModel):
    title: str
    content: str
    data_source: str
    impact_pct: float = 0
    evidence: str = ""
    confidence: str = "medium"
    business_impact: str = ""
    recommended_action: str = ""


class ValidationIssueResponse(ApiModel):
    severity: str
    finding_ref: str
    issue: str
    suggestion: str = ""


class StructuredReportResponse(ApiModel):
    executive_summary: str
    analysis_context: str = ""
    key_findings: tuple[InsightFindingResponse, ...] = ()
    charts: tuple[ChartResponse, ...] = ()
    chart_explanations: tuple[str, ...] = ()
    sql_results: tuple[dict[str, Any], ...] = ()
    python_results: dict[str, Any] = Field(default_factory=dict)
    data_gaps: tuple[str, ...] = ()
    validation_issues: tuple[ValidationIssueResponse, ...] = ()
    recommended_next_steps: tuple[str, ...] = ()
    analysis_trace: tuple[AnalysisRoundResponse, ...] = ()
    analysis_contract: AnalysisContractResponse | None = None
    statistical_verification: StatisticalVerificationResponse | None = None
    analysis_lineage: AnalysisLineageResponse | None = None
    provider: str | None = None
    model: str | None = None


class AnalysisRunResponse(ApiModel):
    dataset_id: UUID
    dataset_group_id: UUID | None = None
    report_id: UUID | None = None
    question: str
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = ()
    plan: AnalysisPlanResponse
    planner_metadata: PlannerMetadataResponse | None = None
    multi_dataset_context: MultiDatasetProfileResponse | None = None
    profile: DatasetProfileResponse
    analysis_framework: AnalysisFrameworkResponse | None = None
    analysis_contract: AnalysisContractResponse | None = None
    intent_spec: AnalysisIntentSpec | None = None
    intent_validation: IntentGuardResult | None = None
    intent_attempts: tuple[IntentCompilationAttempt, ...] = ()
    contract_validation: ContractGuardResult | None = None
    statistical_verification: StatisticalVerificationResponse | None = None
    analysis_lineage: AnalysisLineageResponse | None = None
    sql_result: SQLAnalysisResponse | None = None
    python_result: PythonAnalysisResponse | None = None
    rounds: tuple[AnalysisRoundResponse, ...] = ()
    final_insights: tuple[InsightFindingResponse, ...] = ()
    validation_issues: tuple[ValidationIssueResponse, ...] = ()
    structured_report: StructuredReportResponse | None = None
    html_report: str | None = None
    sql_source: str | None = None
    python_source: str | None = None
    python_generated_code: str | None = None
    python_execution_error: str | None = None
    python_attempts: tuple[PythonCodeAttemptResponse, ...] = ()
    workflow_trace: tuple[WorkflowTraceNodeResponse, ...] = ()
    report_markdown: str
    agent_mode: str = "legacy"
    loop_summary: dict[str, Any] = Field(default_factory=dict)
    loop_terminal_reason: str | None = None
    report_strategy: str | None = None
    report_revision_count: int = Field(default=0, ge=0)
    report_terminal_reason: str | None = None


class WorkflowEventResponse(ApiModel):
    sequence: int = Field(default=0, ge=0)
    node: str = ""
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    status: str
    attempt: int = Field(default=0, ge=0)
    duration_ms: float | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    error_code: str | None = None
    event_type: str | None = None
    iteration: int | None = Field(default=None, ge=0)
    tool_name: str | None = None
    repair_of_sequence: int | None = Field(default=None, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AnalysisJobEventResponse(WorkflowEventResponse):
    """Backward-compatible name for embedded workflow events."""


class AnalysisJobResponse(ApiModel):
    job_id: UUID
    dataset_id: UUID
    dataset_group_id: UUID | None = None
    additional_dataset_ids: tuple[UUID, ...] = ()
    join_plan: tuple[DatasetJoinConfig, ...] = ()
    relationship_plan: tuple[DatasetJoinConfig, ...] = ()
    question: str
    prompt_overrides: AgentPromptOverrides = Field(default_factory=AgentPromptOverrides)
    status: str
    progress: int = Field(ge=0, le=100)
    current_stage: str
    events: tuple[AnalysisJobEventResponse, ...] = ()
    error: str | None = None
    report_id: UUID | None = None
    retry_of: UUID | None = None
    cancel_requested: bool = False
    attempt: int = Field(default=0, ge=0)
    resumable: bool = False
    last_event_sequence: int = Field(default=0, ge=0)
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    agent_mode: str = "legacy"
    loop_summary: dict[str, Any] = Field(default_factory=dict)
    loop_terminal_reason: str | None = None
    report_strategy: str | None = None
    report_revision_count: int = Field(default=0, ge=0)
    report_terminal_reason: str | None = None


class AnalysisJobListResponse(ApiModel):
    jobs: tuple[AnalysisJobResponse, ...]
