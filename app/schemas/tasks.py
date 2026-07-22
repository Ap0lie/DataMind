from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.core.enums import AgentKind, McpCapability, PlanStepStatus, TaskStatus
from app.schemas.common import ApiModel
from app.workflows.models import WorkflowStatus


class TaskCreateRequest(ApiModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    locale: str = Field(default="en-US", min_length=2)


class ToolCallSchema(ApiModel):
    capability: McpCapability
    tool_name: str = Field(min_length=1)
    arguments_schema: dict[str, object] = Field(default_factory=dict)


class PlanStepSchema(ApiModel):
    step_id: UUID
    name: str
    agent: AgentKind
    description: str
    required_capabilities: tuple[McpCapability, ...]
    tool_calls: tuple[ToolCallSchema, ...]
    depends_on: tuple[UUID, ...]
    status: PlanStepStatus


class TaskPlanResponse(ApiModel):
    plan_id: UUID
    task_id: UUID
    objective: str
    status: TaskStatus
    created_at: datetime
    steps: tuple[PlanStepSchema, ...]


class TaskRunResponse(ApiModel):
    task_id: UUID
    workflow_status: WorkflowStatus
    plan: TaskPlanResponse | None = None
    review_passed: bool | None = None
    report_title: str | None = None
    report_markdown: str | None = None
    checkpoint_count: int = Field(ge=0)
    trace_event_count: int = Field(ge=0)
