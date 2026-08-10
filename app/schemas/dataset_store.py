from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.common import ApiModel
from app.schemas.prompt_overrides import AgentPromptOverrides


class CreateDatasetRequest(ApiModel):
    name: str = Field(min_length=1)
    source_type: str = Field(pattern="^(csv|xlsx|json|txt)$")
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class CreateDatasetResponse(ApiModel):
    dataset_id: UUID
    user_id: str = "default"
    name: str
    source_type: str
    status: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class DatasetListResponse(ApiModel):
    datasets: tuple[CreateDatasetResponse, ...]


class DatasetPreviewResponse(ApiModel):
    dataset_id: UUID
    record_source: str = "raw"
    records: tuple[dict[str, Any], ...]


class FileDatasetImportResponse(ApiModel):
    dataset: CreateDatasetResponse
    inserted: int
    preview_records: tuple[dict[str, Any], ...] = ()


class CreateDatasetGroupRequest(ApiModel):
    name: str = Field(min_length=1)
    dataset_ids: tuple[UUID, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetRelationshipPlan(ApiModel):
    relationship_id: str | None = None
    left_dataset_id: UUID
    right_dataset_id: UUID
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)
    join_type: str = Field(default="left", pattern="^(left|inner)$")
    left_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    right_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    left_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    right_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    enabled: bool = True
    confidence: float = Field(default=0, ge=0, le=1)
    source: str = "user"
    reason: str = ""
    relationship_type: str = Field(
        default="unknown",
        pattern="^(one_to_one|one_to_many|many_to_one|many_to_many|unknown)$",
    )
    risk_note: str = ""
    baseline_match_rate: float | None = Field(default=None, ge=0, le=1)
    last_match_rate: float | None = Field(default=None, ge=0, le=1)
    match_rate_drift: float = Field(default=0, ge=0, le=1)
    freshness_status: str = Field(default="fresh", pattern="^(fresh|warning|stale)$")
    stale_reason: str = ""
    last_validated_at: str | None = None
    drift_event_id: UUID | None = None


class DatasetGroupTable(ApiModel):
    dataset: CreateDatasetResponse
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: tuple[str, ...] = ()
    entity_type: str = Field(
        default="unknown", pattern="^(fact|dimension|bridge|lookup|wide|unknown)$"
    )
    sample_records: tuple[dict[str, Any], ...] = ()


class DatasetGroupResponse(ApiModel):
    group_id: UUID
    user_id: str = "default"
    name: str
    description: str = ""
    tables: tuple[DatasetGroupTable, ...] = ()
    relationships: tuple[DatasetRelationshipPlan, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class DatasetGroupListResponse(ApiModel):
    groups: tuple[DatasetGroupResponse, ...]


class DatasetRelationshipCandidate(ApiModel):
    left_dataset_id: UUID
    right_dataset_id: UUID
    left_column: str
    right_column: str
    join_type: str = Field(default="left", pattern="^(left|inner)$")
    left_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    right_value_mode: str = Field(default="scalar", pattern="^(scalar|delimited)$")
    left_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    right_delimiter: str | None = Field(default=None, min_length=1, max_length=4)
    confidence: float = Field(ge=0, le=1)
    source: str = Field(default="rules", pattern="^(rules|llm|validated_llm)$")
    reason: str
    left_type: str = ""
    right_type: str = ""
    left_role: str = ""
    right_role: str = ""
    estimated_match_rate: float = Field(default=0, ge=0, le=1)
    relationship_type: str = Field(
        default="unknown",
        pattern="^(one_to_one|one_to_many|many_to_one|many_to_many|unknown)$",
    )
    risk_note: str = ""
    embedding_score: float = Field(default=0, ge=0, le=1)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    embedding_model_revision: str | None = None


class DatasetRelationshipSuggestionResponse(ApiModel):
    group: DatasetGroupResponse
    candidates: tuple[DatasetRelationshipCandidate, ...] = ()
    llm_used: bool = False
    compact_context: dict[str, Any] = Field(default_factory=dict)
    validation_issues: tuple[str, ...] = ()


class DatasetRelationshipAutoConfigureResponse(DatasetRelationshipSuggestionResponse):
    saved_relationships: tuple[DatasetRelationshipPlan, ...] = ()
    primary_dataset_id: UUID | None = None
    unresolved_dataset_ids: tuple[UUID, ...] = ()


class UpdateDatasetGroupRelationshipsRequest(ApiModel):
    relationships: tuple[DatasetRelationshipPlan, ...] = ()


class ExcelSheetPreview(ApiModel):
    sheet_name: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    score: int = Field(ge=0)
    selected: bool = False
    preview_records: tuple[dict[str, Any], ...] = ()


class ExcelSheetPreviewResponse(ApiModel):
    sheets: tuple[ExcelSheetPreview, ...]


class DatasetReportResponse(ApiModel):
    id: UUID
    dataset_id: UUID
    title: str
    markdown: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    version: int = Field(default=1, ge=1)


class DatasetReportListResponse(ApiModel):
    reports: tuple[DatasetReportResponse, ...]


class DeleteDatasetResponse(ApiModel):
    dataset_id: UUID
    deleted: bool


class DeleteDatasetGroupResponse(ApiModel):
    group_id: UUID
    deleted: bool
    deleted_dataset_ids: tuple[UUID, ...] = ()


class DeleteReportResponse(ApiModel):
    report_id: UUID
    deleted: bool


class ReportUpdateRequest(ApiModel):
    title: str | None = Field(default=None, min_length=1)


class ReportVersionSummary(ApiModel):
    report_id: UUID
    dataset_id: UUID
    title: str
    question: str | None = None
    version: int = Field(default=1, ge=1)
    created_at: str | None = None
    updated_at: str | None = None


class ReportVersionListResponse(ApiModel):
    versions: tuple[ReportVersionSummary, ...]


class AppendRawRecordsRequest(ApiModel):
    records: list[dict[str, Any]] = Field(default_factory=list)


class AppendRawRecordsResponse(ApiModel):
    dataset_id: UUID
    inserted: int


class RunDatasetCleaningRequest(ApiModel):
    requirement: str = ""
    use_llm: bool = True


class CreateCleaningJobRequest(ApiModel):
    requirement: str = ""
    cleaning_strategy: str = Field(default="auto", pattern="^(auto|rules|llm|hybrid)$")
    prompt_overrides: AgentPromptOverrides = Field(default_factory=AgentPromptOverrides)


class CleaningJobEventResponse(ApiModel):
    sequence: int = Field(ge=0)
    stage: str
    status: str
    progress: int = Field(ge=0, le=100)
    message: str
    event_type: str | None = None
    iteration: int | None = Field(default=None, ge=0)
    strategy: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CleaningJobResponse(ApiModel):
    job_id: UUID
    dataset_id: UUID
    requirement: str
    prompt_overrides: AgentPromptOverrides = Field(default_factory=AgentPromptOverrides)
    cleaning_strategy: str
    selected_strategy: str | None = None
    status: str
    progress: int = Field(ge=0, le=100)
    current_stage: str
    events: tuple[CleaningJobEventResponse, ...] = ()
    loop_summary: dict[str, Any] = Field(default_factory=dict)
    terminal_reason: str | None = None
    error: str | None = None
    cleaning_run_id: UUID | None = None
    retry_of: UUID | None = None
    cancel_requested: bool = False
    attempt: int = Field(default=0, ge=0)
    last_event_sequence: int = Field(default=0, ge=0)
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class CleaningJobListResponse(ApiModel):
    jobs: tuple[CleaningJobResponse, ...] = ()


class DatasetCleaningRunResponse(ApiModel):
    dataset_id: UUID
    run_id: UUID
    version: int = Field(default=1, ge=1)
    provider: str
    model: str
    source: str
    raw_row_count: int = Field(ge=0)
    cleaned_row_count: int = Field(ge=0)
    cleaned_column_count: int = Field(ge=0)
    result_markdown: str
    preview_records: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()


class CleaningDiffSummary(ApiModel):
    raw_row_count: int = Field(ge=0)
    previous_row_count: int = Field(ge=0)
    current_row_count: int = Field(ge=0)
    added_rows: int = Field(ge=0)
    removed_rows: int = Field(ge=0)
    changed_rows: int = Field(ge=0)
    added_columns: tuple[str, ...] = ()
    removed_columns: tuple[str, ...] = ()
    changed_cells: int = Field(ge=0)
    raw_missing_count: int = Field(ge=0)
    previous_missing_count: int = Field(ge=0)
    current_missing_count: int = Field(ge=0)
    sample_diffs: tuple[dict[str, Any], ...] = ()


class CleaningRunDetail(ApiModel):
    id: UUID
    dataset_id: UUID
    version: int = Field(ge=1)
    is_active: bool = False
    provider: str
    model: str
    prompt: str
    result_markdown: str
    cleaned_dataset: dict[str, Any] = Field(default_factory=dict)
    raw_summary: dict[str, Any] = Field(default_factory=dict)
    previous_summary: dict[str, Any] = Field(default_factory=dict)
    current_summary: dict[str, Any] = Field(default_factory=dict)
    diff_summary: CleaningDiffSummary
    created_at: str | None = None


class DatasetCleaningRunListResponse(ApiModel):
    runs: tuple[CleaningRunDetail, ...]


class CleaningRule(ApiModel):
    rule_type: str = Field(
        pattern="^(fill_missing|drop_duplicates|rename_column|convert_type|trim_text|drop_column|filter_rows)$"
    )
    column: str | None = None
    value: Any = None
    new_name: str | None = None
    target_type: str | None = Field(
        default=None, pattern="^(text|number|integer|float|date|boolean)$"
    )
    strategy: str | None = Field(default=None, pattern="^(empty_string|zero|value|drop_row)$")
    operator: str | None = Field(
        default=None,
        pattern="^(equals|not_equals|contains|not_contains|blank|not_blank|gt|gte|lt|lte)$",
    )
    mode: str | None = Field(default=None, pattern="^(keep|delete)$")
    enabled: bool = True


class CleaningRulePreviewRequest(ApiModel):
    rules: list[CleaningRule] = Field(default_factory=list)


class CleaningRulePreviewResponse(ApiModel):
    dataset_id: UUID
    preview_records: tuple[dict[str, Any], ...] = ()
    diff_summary: CleaningDiffSummary
    validation_issues: tuple[str, ...] = ()
    applied_rules: tuple[CleaningRule, ...] = ()


class DatasetColumnMetadata(ApiModel):
    column_name: str = Field(min_length=1)
    inferred_type: str = "text"
    override_type: str | None = None
    description: str = ""
    role: str = Field(default="dimension", pattern="^(dimension|metric|id|text|date|ignore)$")
    created_at: str | None = None
    updated_at: str | None = None


class SaveDatasetColumnsRequest(ApiModel):
    columns: list[DatasetColumnMetadata] = Field(default_factory=list)


class UpdateDatasetColumnRequest(ApiModel):
    inferred_type: str | None = None
    override_type: str | None = None
    description: str | None = None
    role: str | None = Field(default=None, pattern="^(dimension|metric|id|text|date|ignore)$")


class DatasetColumnMetadataListResponse(ApiModel):
    columns: tuple[DatasetColumnMetadata, ...]


class SaveArtifactResponse(ApiModel):
    id: UUID


class SaveDatasetArtifactRequest(ApiModel):
    artifact_type: str = Field(min_length=1)
    file_name: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)


class SaveChartRequest(ApiModel):
    title: str = Field(min_length=1)
    chart_type: str = Field(min_length=1)
    chart_spec: dict[str, Any] = Field(default_factory=dict)
    chart_data: list[dict[str, Any]] = Field(default_factory=list)


class SaveReportRequest(ApiModel):
    title: str = Field(min_length=1)
    markdown: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
