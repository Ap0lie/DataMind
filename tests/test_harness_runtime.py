from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from app.core.entities import ExecutionPlan, PlanStep, TaskIntent, ToolCallSpec
from app.core.enums import AgentKind, HarnessKind, McpCapability
from app.harness.models import ExecutionStatus, RetryPolicy, TimeoutPolicy
from app.harness.runtime import (
    AllowAllPermissionHarness,
    DefaultExecutionHarness,
    DefaultToolHarness,
    DependencyPlannerHarness,
    InMemoryCheckpointStore,
    InMemoryHarnessMemory,
    InMemoryObservabilityHarness,
    SimpleContextHarness,
    SimpleEvaluationHarness,
    StrictValidationHarness,
)


def make_plan() -> ExecutionPlan:
    task = TaskIntent(tenant_id="tenant", user_id="user", prompt="Analyze sales data")
    first = PlanStep(
        name="profile",
        agent=AgentKind.DATA_ANALYST,
        description="Profile dataset",
        required_capabilities=(McpCapability.DATA_ANALYSIS,),
    )
    second = PlanStep(
        name="parse",
        agent=AgentKind.PARSER,
        description="Parse sources",
        depends_on=(first.step_id,),
    )
    return ExecutionPlan(task=task, objective="Analyze sales data", steps=(first, second))


@pytest.mark.asyncio
async def test_context_harness_compresses_prompt() -> None:
    intent = TaskIntent(tenant_id="tenant", user_id="user", prompt="  hello   DataMind  ")

    context = await SimpleContextHarness().build_context(intent)

    assert context.compressed_prompt == "hello DataMind"
    assert context.metadata["tenant_id"] == "tenant"


@pytest.mark.asyncio
async def test_planner_harness_schedules_dependencies() -> None:
    plan = make_plan()

    order = await DependencyPlannerHarness().schedule(plan)

    assert order == (plan.steps[0].step_id, plan.steps[1].step_id)


@pytest.mark.asyncio
async def test_tool_harness_routes_tool_call() -> None:
    call = ToolCallSpec(
        capability=McpCapability.DATA_ANALYSIS,
        tool_name="profile_dataset",
        arguments_schema={"records": "array"},
    )

    route = await DefaultToolHarness().route(call)

    assert route.tool_name == "profile_dataset"
    assert route.arguments["schema"] == {"records": "array"}


@pytest.mark.asyncio
async def test_memory_and_permission_harnesses() -> None:
    memory = InMemoryHarnessMemory()
    intent = TaskIntent(tenant_id="tenant", user_id="user", prompt="test")

    await memory.remember("task", {"ok": True})

    assert await memory.recall("task") == {"ok": True}
    assert await AllowAllPermissionHarness().authorize(intent)


@pytest.mark.asyncio
async def test_execution_harness_runs_steps_checkpoints_traces_and_tokens() -> None:
    plan = make_plan()
    checkpoints = InMemoryCheckpointStore()
    observability = InMemoryObservabilityHarness()
    harness = make_execution_harness(checkpoints=checkpoints, observability=observability)

    async def handler(_: ExecutionPlan, step_id: UUID) -> dict[str, object]:
        return {
            "step_id": str(step_id),
            "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    result = await harness.execute(plan, handler)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert len(result.steps) == 2
    assert len(result.checkpoints) == 2
    assert result.token_usage.total_tokens == 10
    assert result.visualization.nodes
    assert result.visualization.edges
    assert {event.harness for event in observability.events} == {HarnessKind.EXECUTION}
    assert await checkpoints.latest(plan.plan_id) == checkpoints.checkpoints[-1]


@pytest.mark.asyncio
async def test_execution_harness_retries_failed_step() -> None:
    plan = make_plan()
    calls = {"count": 0}
    harness = make_execution_harness(retry_policy=RetryPolicy(max_retries=1, backoff_seconds=0))

    async def handler(_: ExecutionPlan, __: UUID) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary")
        return {"ok": True}

    result = await harness.execute(plan, handler)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.steps[0].attempts == 2


@pytest.mark.asyncio
async def test_execution_harness_returns_timeout_result() -> None:
    plan = make_plan()
    harness = make_execution_harness(
        retry_policy=RetryPolicy(max_retries=0),
        timeout_policy=TimeoutPolicy(step_timeout_seconds=0.01),
    )

    async def handler(_: ExecutionPlan, __: UUID) -> dict[str, object]:
        await asyncio.sleep(0.1)
        return {"ok": True}

    result = await harness.execute(plan, handler)

    assert result.status == ExecutionStatus.TIMEOUT
    assert result.steps[0].status == ExecutionStatus.TIMEOUT


@pytest.mark.asyncio
async def test_execution_harness_stops_on_validation_failure() -> None:
    plan = make_plan()
    harness = make_execution_harness()

    async def handler(_: ExecutionPlan, __: UUID) -> dict[str, object]:
        return {"error": "invalid output"}

    result = await harness.execute(plan, handler)

    assert result.status == ExecutionStatus.VALIDATION_FAILED
    assert result.steps[0].status == ExecutionStatus.VALIDATION_FAILED


def make_execution_harness(
    *,
    checkpoints: InMemoryCheckpointStore | None = None,
    observability: InMemoryObservabilityHarness | None = None,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
) -> DefaultExecutionHarness:
    return DefaultExecutionHarness(
        planner=DependencyPlannerHarness(),
        validation=StrictValidationHarness(),
        observability=observability or InMemoryObservabilityHarness(),
        evaluation=SimpleEvaluationHarness(),
        checkpoint_store=checkpoints or InMemoryCheckpointStore(),
        retry_policy=retry_policy,
        timeout_policy=timeout_policy,
    )
