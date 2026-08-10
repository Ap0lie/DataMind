from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.schemas.common import ApiModel

AssistantScopeType = Literal["auto", "dataset", "dataset_group", "report"]
AssistantExecutionMode = Literal["ask", "execute"]
AssistantCapability = Literal[
    "data_prepare",
    "relationship_manage",
    "analysis_manage",
    "report_manage",
    "semantic_manage",
    "asset_recycle",
]
AssistantAssetType = Literal["dataset", "dataset_group", "report", "semantic_model"]


class AssistantCitationReliabilityResponse(ApiModel):
    status: Literal["verified", "warning", "rejected", "unverified"] = "unverified"
    summary: str = "未提供统计审查状态。"


class AssistantCitationResponse(ApiModel):
    source_type: Literal["dataset", "analysis_job", "report"]
    source_id: UUID
    label: str
    excerpt: str = ""
    dataset_id: UUID | None = None
    artifact_role: Literal["evidence", "deliverable"] = "evidence"
    reliability: AssistantCitationReliabilityResponse = Field(
        default_factory=AssistantCitationReliabilityResponse
    )


class AssistantConversationCreateRequest(ApiModel):
    title: str | None = Field(default=None, max_length=120)
    scope_type: AssistantScopeType = "auto"
    scope_id: UUID | None = None


class AssistantConversationUpdateRequest(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    scope_type: AssistantScopeType | None = None
    scope_id: UUID | None = None


class AssistantConversationResponse(ApiModel):
    conversation_id: UUID
    title: str
    scope_type: AssistantScopeType
    scope_id: UUID | None = None
    summary: str = ""
    active_run_id: UUID | None = None
    active_run_status: str | None = None
    created_at: str
    updated_at: str
    last_message_at: str | None = None


class AssistantConversationListResponse(ApiModel):
    conversations: tuple[AssistantConversationResponse, ...]


class AssistantAttachmentResponse(ApiModel):
    attachment_id: UUID
    conversation_id: UUID
    message_id: UUID | None = None
    file_name: str
    media_type: str
    size_bytes: int
    width: int = 0
    height: int = 0
    attachment_kind: Literal["image", "data_file"] = "image"
    import_status: str | None = None
    dataset_id: UUID | None = None
    import_batch_id: UUID | None = None
    created_at: str
    content_url: str


class AssistantMessageResponse(ApiModel):
    message_id: UUID
    conversation_id: UUID
    role: Literal["user", "assistant", "tool"]
    content: str
    status: str
    provider: str | None = None
    model: str | None = None
    citations: tuple[AssistantCitationResponse, ...] = ()
    attachments: tuple[AssistantAttachmentResponse, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AssistantMessageListResponse(ApiModel):
    messages: tuple[AssistantMessageResponse, ...]


class AssistantMessageCreateRequest(ApiModel):
    content: str = Field(min_length=1, max_length=20_000)
    attachment_ids: tuple[UUID, ...] = ()
    execution_mode: AssistantExecutionMode = "ask"


class AssistantRunResponse(ApiModel):
    run_id: UUID
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    status: str
    current_stage: str
    analysis_job_id: UUID | None = None
    pending_confirmation: dict[str, Any] = Field(default_factory=dict)
    execution_mode: AssistantExecutionMode = "ask"
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    current_action_id: UUID | None = None
    required_permission: AssistantCapability | None = None
    error: str | None = None
    last_event_sequence: int = 0
    created_at: str
    updated_at: str
    completed_at: str | None = None


class AssistantRunEventResponse(ApiModel):
    sequence: int
    event_type: str
    status: str
    message: str
    tool_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AssistantRunConfirmRequest(ApiModel):
    accepted: bool


class AssistantPermissionGrantCreateRequest(ApiModel):
    asset_type: AssistantAssetType
    asset_id: UUID
    capabilities: tuple[AssistantCapability, ...]


class AssistantPermissionGrantResponse(ApiModel):
    grant_id: UUID
    asset_type: AssistantAssetType
    asset_id: UUID
    capabilities: tuple[AssistantCapability, ...]
    status: str
    created_at: str
    revoked_at: str | None = None


class AssistantPermissionGrantListResponse(ApiModel):
    grants: tuple[AssistantPermissionGrantResponse, ...]


class AssistantActionResponse(ApiModel):
    action_id: UUID
    run_id: UUID | None = None
    conversation_id: UUID | None = None
    tool_name: str
    status: str
    asset_type: AssistantAssetType | None = None
    asset_id: UUID | None = None
    reversible: bool = False
    undone_at: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class AssistantActionListResponse(ApiModel):
    actions: tuple[AssistantActionResponse, ...]


class AssistantImportBatchPreviewRequest(ApiModel):
    conversation_id: UUID
    attachment_ids: tuple[UUID, ...] = Field(min_length=1, max_length=20)


class AssistantImportBatchCommitRequest(ApiModel):
    allow_partial: bool = False
    name: str | None = Field(default=None, max_length=120)
    sheet_selections: dict[UUID, str] = Field(default_factory=dict)


class AssistantImportBatchResponse(ApiModel):
    batch_id: UUID
    conversation_id: UUID
    attachment_ids: tuple[UUID, ...]
    status: str
    preview: dict[str, Any] = Field(default_factory=dict)
    dataset_ids: tuple[UUID, ...] = ()
    dataset_group_id: UUID | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class RecycledAssetResponse(ApiModel):
    asset_type: AssistantAssetType
    asset_id: UUID
    name: str
    deleted_at: str
    purge_after: str


class RecycledAssetListResponse(ApiModel):
    assets: tuple[RecycledAssetResponse, ...]
