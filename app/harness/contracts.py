from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol
from uuid import UUID

from app.core.entities import ExecutionPlan, TaskIntent, ToolCallSpec
from app.harness.models import (
    Checkpoint,
    EvaluationReport,
    HarnessContext,
    ToolRoute,
    TraceEvent,
    ValidationReport,
    WorkflowExecutionResult,
    WorkflowVisualization,
)

StepHandler = Callable[[ExecutionPlan, UUID], Awaitable[dict[str, Any]]]


class ContextHarness(Protocol):
    async def build_context(self, intent: TaskIntent) -> HarnessContext:
        """Construct compressed context for planning and execution."""


class PlannerHarness(Protocol):
    async def schedule(self, plan: ExecutionPlan) -> tuple[UUID, ...]:
        """Create an ordered route for plan execution."""


class ToolHarness(Protocol):
    async def route(self, call: ToolCallSpec) -> ToolRoute:
        """Resolve a tool call to an MCP server/tool target."""


class ExecutionHarness(Protocol):
    async def execute(self, plan: ExecutionPlan, handler: StepHandler) -> WorkflowExecutionResult:
        """Coordinate retry, timeout, checkpoint, recovery, and workflow isolation."""


class PermissionHarness(Protocol):
    async def authorize(self, intent: TaskIntent, call: ToolCallSpec | None = None) -> bool:
        """Authorize task and MCP tool access."""


class ValidationHarness(Protocol):
    async def validate(self, payload: dict[str, Any]) -> ValidationReport:
        """Validate outputs before storage, reporting, or downstream agent use."""


class MemoryHarness(Protocol):
    async def recall(self, key: str) -> dict[str, Any] | None:
        """Load task, conversation, knowledge, or entity memory."""

    async def remember(self, key: str, value: dict[str, Any]) -> None:
        """Store task, conversation, knowledge, or entity memory."""


class ObservabilityHarness(Protocol):
    async def record_event(self, event: TraceEvent) -> None:
        """Record trace, latency, token, cost, or workflow visualization event."""

    async def visualize(self, plan: ExecutionPlan) -> WorkflowVisualization:
        """Build workflow visualization data."""


class EvaluationHarness(Protocol):
    async def evaluate(self, plan: ExecutionPlan, outputs: dict[str, Any]) -> EvaluationReport:
        """Evaluate planning, extraction, knowledge graph, summary, and workflow quality."""


class CheckpointStore(Protocol):
    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist checkpoint state."""

    async def latest(self, workflow_id: UUID) -> Checkpoint | None:
        """Load latest checkpoint for recovery."""
