from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class StoredDataset:
    id: UUID
    user_id: str
    name: str
    source_type: str
    status: str
    source_metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class StoredDatasetGroup:
    id: UUID
    user_id: str
    name: str
    description: str
    dataset_ids: tuple[UUID, ...]
    relationships: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class StoredAnalysisJob:
    id: UUID
    user_id: str
    dataset_id: UUID
    dataset_group_id: UUID | None
    additional_dataset_ids: tuple[UUID, ...]
    join_plan: tuple[dict[str, Any], ...]
    relationship_plan: tuple[dict[str, Any], ...]
    question: str
    prompt_overrides: dict[str, str]
    multimodal_inputs: tuple[dict[str, Any], ...]
    status: str
    progress: int
    current_stage: str
    events: tuple[dict[str, Any], ...]
    result: dict[str, Any] | None = None
    error: str | None = None
    report_id: UUID | None = None
    retry_of: UUID | None = None
    cancel_requested: bool = False
    broker_task_id: str | None = None
    attempt_count: int = 0
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    heartbeat_at: str | None = None
    checkpoint_thread_id: str | None = None
    planner_decision_id: UUID | None = None
    semantic_model_id: UUID | None = None
    semantic_model_version: int | None = None
    agent_mode: str = "legacy"
    loop_summary: dict[str, Any] | None = None
    loop_terminal_reason: str | None = None
    report_strategy: str | None = None
    report_revision_count: int = 0
    report_terminal_reason: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class StoredCleaningJob:
    id: UUID
    user_id: str
    dataset_id: UUID
    requirement: str
    prompt_overrides: dict[str, str]
    cleaning_strategy: str
    status: str
    progress: int
    current_stage: str
    events: tuple[dict[str, Any], ...]
    selected_strategy: str | None = None
    loop_summary: dict[str, Any] | None = None
    terminal_reason: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cleaning_run_id: UUID | None = None
    retry_of: UUID | None = None
    cancel_requested: bool = False
    broker_task_id: str | None = None
    attempt_count: int = 0
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    heartbeat_at: str | None = None
    checkpoint_thread_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
