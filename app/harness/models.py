from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import HarnessKind, PlanStepStatus


class HarnessModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TraceEventLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CheckpointStatus(StrEnum):
    CREATED = "created"
    RESTORED = "restored"


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"
    PERMISSION_DENIED = "permission_denied"


class RetryPolicy(HarnessModel):
    max_retries: int = Field(default=1, ge=0, le=10)
    backoff_seconds: float = Field(default=0.1, ge=0.0)


class TimeoutPolicy(HarnessModel):
    step_timeout_seconds: float = Field(default=30.0, gt=0)
    workflow_timeout_seconds: float | None = Field(default=None, gt=0)


class TokenUsage(HarnessModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LatencyMetric(HarnessModel):
    name: str = Field(min_length=1)
    duration_ms: float = Field(ge=0.0)


class TraceEvent(HarnessModel):
    trace_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    harness: HarnessKind
    name: str = Field(min_length=1)
    level: TraceEventLevel = TraceEventLevel.INFO
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowNode(HarnessModel):
    node_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: PlanStepStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(HarnessModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str | None = None


class WorkflowVisualization(HarnessModel):
    workflow_id: UUID
    nodes: tuple[WorkflowNode, ...] = Field(default_factory=tuple)
    edges: tuple[WorkflowEdge, ...] = Field(default_factory=tuple)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HarnessContext(HarnessModel):
    task_id: UUID
    prompt: str = Field(min_length=1)
    compressed_prompt: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolRoute(HarnessModel):
    server_name: str | None = None
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(HarnessModel):
    valid: bool
    errors: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class EvaluationReport(HarnessModel):
    scores: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    notes: tuple[str, ...] = Field(default_factory=tuple)


class Checkpoint(HarnessModel):
    checkpoint_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    step_id: UUID | None = None
    status: CheckpointStatus = CheckpointStatus.CREATED
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StepExecutionResult(HarnessModel):
    step_id: UUID
    status: ExecutionStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    attempts: int = Field(default=1, ge=1)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency: LatencyMetric | None = None


class WorkflowExecutionResult(HarnessModel):
    workflow_id: UUID
    status: ExecutionStatus
    steps: tuple[StepExecutionResult, ...] = Field(default_factory=tuple)
    checkpoints: tuple[Checkpoint, ...] = Field(default_factory=tuple)
    visualization: WorkflowVisualization
    evaluation: EvaluationReport | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    trace_id: UUID
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
