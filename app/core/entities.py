from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.core.enums import AgentKind, McpCapability, PlanStepStatus, TaskStatus


class CoreModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskIntent(CoreModel):
    task_id: UUID = Field(default_factory=uuid4)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    locale: str = Field(default="en-US", min_length=2)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCallSpec(CoreModel):
    capability: McpCapability
    tool_name: str = Field(min_length=1)
    arguments_schema: dict[str, object] = Field(default_factory=dict)


class PlanStep(CoreModel):
    step_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    agent: AgentKind
    description: str = Field(min_length=1)
    required_capabilities: tuple[McpCapability, ...] = Field(default_factory=tuple)
    tool_calls: tuple[ToolCallSpec, ...] = Field(default_factory=tuple)
    depends_on: tuple[UUID, ...] = Field(default_factory=tuple)
    status: PlanStepStatus = PlanStepStatus.PENDING


class ExecutionPlan(CoreModel):
    plan_id: UUID = Field(default_factory=uuid4)
    task: TaskIntent
    objective: str = Field(min_length=1)
    steps: tuple[PlanStep, ...] = Field(default_factory=tuple)
    status: TaskStatus = TaskStatus.PLANNED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SourceRef(CoreModel):
    url: HttpUrl
    title: str | None = None
    content_hash: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Evidence(CoreModel):
    source: SourceRef
    quote: str | None = None
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
