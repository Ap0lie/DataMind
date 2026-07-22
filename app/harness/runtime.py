from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.core.entities import ExecutionPlan, TaskIntent, ToolCallSpec
from app.core.enums import HarnessKind
from app.harness.contracts import StepHandler
from app.harness.models import (
    Checkpoint,
    EvaluationReport,
    ExecutionStatus,
    HarnessContext,
    LatencyMetric,
    RetryPolicy,
    StepExecutionResult,
    TimeoutPolicy,
    TokenUsage,
    ToolRoute,
    TraceEvent,
    TraceEventLevel,
    ValidationReport,
    WorkflowEdge,
    WorkflowExecutionResult,
    WorkflowNode,
    WorkflowVisualization,
)


class SimpleContextHarness:
    async def build_context(self, intent: TaskIntent) -> HarnessContext:
        compressed = " ".join(intent.prompt.split())
        return HarnessContext(
            task_id=intent.task_id,
            prompt=intent.prompt,
            compressed_prompt=compressed[:4000],
            metadata={"tenant_id": intent.tenant_id, "locale": intent.locale},
        )


class DependencyPlannerHarness:
    async def schedule(self, plan: ExecutionPlan) -> tuple[UUID, ...]:
        scheduled: list[UUID] = []
        pending = {step.step_id: step for step in plan.steps}
        while pending:
            ready = [
                step_id
                for step_id, step in pending.items()
                if all(dependency in scheduled for dependency in step.depends_on)
            ]
            if not ready:
                raise ValueError("Execution plan contains unsatisfied or cyclic dependencies.")
            for step_id in ready:
                scheduled.append(step_id)
                pending.pop(step_id)
        return tuple(scheduled)


class DefaultToolHarness:
    async def route(self, call: ToolCallSpec) -> ToolRoute:
        return ToolRoute(tool_name=call.tool_name, arguments={"schema": call.arguments_schema})


class AllowAllPermissionHarness:
    async def authorize(self, intent: TaskIntent, call: ToolCallSpec | None = None) -> bool:
        return True


class StrictValidationHarness:
    async def validate(self, payload: dict[str, Any]) -> ValidationReport:
        if payload.get("error"):
            return ValidationReport(valid=False, errors=(str(payload["error"]),))
        return ValidationReport(valid=True)


class InMemoryHarnessMemory:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    async def recall(self, key: str) -> dict[str, Any] | None:
        return self.items.get(key)

    async def remember(self, key: str, value: dict[str, Any]) -> None:
        self.items[key] = value


class InMemoryObservabilityHarness:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def record_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def visualize(self, plan: ExecutionPlan) -> WorkflowVisualization:
        nodes = tuple(
            WorkflowNode(
                node_id=str(step.step_id),
                label=step.name,
                status=step.status,
                metadata={"agent": step.agent.value},
            )
            for step in plan.steps
        )
        edges = tuple(
            WorkflowEdge(source=str(dependency), target=str(step.step_id))
            for step in plan.steps
            for dependency in step.depends_on
        )
        return WorkflowVisualization(workflow_id=plan.plan_id, nodes=nodes, edges=edges)


class SimpleEvaluationHarness:
    async def evaluate(self, plan: ExecutionPlan, outputs: dict[str, Any]) -> EvaluationReport:
        completed = sum(1 for value in outputs.values() if isinstance(value, dict))
        total = max(len(plan.steps), 1)
        score = completed / total
        return EvaluationReport(scores={"workflow_completion": score}, passed=score == 1.0)


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self.checkpoints: list[Checkpoint] = []

    async def save(self, checkpoint: Checkpoint) -> None:
        self.checkpoints.append(checkpoint)

    async def latest(self, workflow_id: UUID) -> Checkpoint | None:
        matching = [item for item in self.checkpoints if item.workflow_id == workflow_id]
        return matching[-1] if matching else None


class DefaultExecutionHarness:
    def __init__(
        self,
        *,
        planner: DependencyPlannerHarness,
        validation: StrictValidationHarness,
        observability: InMemoryObservabilityHarness,
        evaluation: SimpleEvaluationHarness,
        checkpoint_store: InMemoryCheckpointStore,
        retry_policy: RetryPolicy | None = None,
        timeout_policy: TimeoutPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._validation = validation
        self._observability = observability
        self._evaluation = evaluation
        self._checkpoint_store = checkpoint_store
        self._retry_policy = retry_policy or RetryPolicy()
        self._timeout_policy = timeout_policy or TimeoutPolicy()

    async def execute(self, plan: ExecutionPlan, handler: StepHandler) -> WorkflowExecutionResult:
        trace_id = uuid4()
        started_at = datetime.now(UTC)
        step_order = await self._planner.schedule(plan)
        step_results: list[StepExecutionResult] = []
        outputs: dict[str, Any] = {}
        checkpoints: list[Checkpoint] = []

        await self._record(trace_id, "workflow_started", {"plan_id": str(plan.plan_id)})

        for step_id in step_order:
            result = await self._execute_step(plan, step_id, handler, trace_id)
            step_results.append(result)
            outputs[str(step_id)] = result.output
            checkpoint = Checkpoint(
                workflow_id=plan.plan_id,
                step_id=step_id,
                state={"outputs": outputs, "status": result.status.value},
            )
            await self._checkpoint_store.save(checkpoint)
            checkpoints.append(checkpoint)
            if result.status != ExecutionStatus.SUCCEEDED:
                visualization = await self._observability.visualize(plan)
                return WorkflowExecutionResult(
                    workflow_id=plan.plan_id,
                    status=result.status,
                    steps=tuple(step_results),
                    checkpoints=tuple(checkpoints),
                    visualization=visualization,
                    trace_id=trace_id,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    token_usage=_sum_tokens(step_results),
                )

        evaluation = await self._evaluation.evaluate(plan, outputs)
        visualization = await self._observability.visualize(plan)
        status = ExecutionStatus.SUCCEEDED if evaluation.passed else ExecutionStatus.FAILED
        await self._record(trace_id, "workflow_finished", {"status": status.value})
        return WorkflowExecutionResult(
            workflow_id=plan.plan_id,
            status=status,
            steps=tuple(step_results),
            checkpoints=tuple(checkpoints),
            visualization=visualization,
            evaluation=evaluation,
            token_usage=_sum_tokens(step_results),
            trace_id=trace_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    async def _execute_step(
        self,
        plan: ExecutionPlan,
        step_id: UUID,
        handler: StepHandler,
        trace_id: UUID,
    ) -> StepExecutionResult:
        attempts = 0
        started_at = datetime.now(UTC)
        last_error: str | None = None

        while attempts <= self._retry_policy.max_retries:
            attempts += 1
            await self._record(
                trace_id,
                "step_attempt",
                {"step_id": str(step_id), "attempt": attempts},
            )
            try:
                output = await asyncio.wait_for(
                    handler(plan, step_id),
                    timeout=self._timeout_policy.step_timeout_seconds,
                )
                validation = await self._validation.validate(output)
                latency = _latency("step", started_at)
                if not validation.valid:
                    return StepExecutionResult(
                        step_id=step_id,
                        status=ExecutionStatus.VALIDATION_FAILED,
                        output=output,
                        error="; ".join(validation.errors),
                        attempts=attempts,
                        latency=latency,
                        token_usage=_token_usage(output),
                    )
                return StepExecutionResult(
                    step_id=step_id,
                    status=ExecutionStatus.SUCCEEDED,
                    output=output,
                    attempts=attempts,
                    latency=latency,
                    token_usage=_token_usage(output),
                )
            except TimeoutError:
                last_error = f"Step timed out after {self._timeout_policy.step_timeout_seconds}s."
                status = ExecutionStatus.TIMEOUT
            except Exception as exc:
                last_error = str(exc)
                status = ExecutionStatus.FAILED

            if attempts > self._retry_policy.max_retries:
                return StepExecutionResult(
                    step_id=step_id,
                    status=status,
                    error=last_error,
                    attempts=attempts,
                    latency=_latency("step", started_at),
                )
            if self._retry_policy.backoff_seconds > 0:
                await asyncio.sleep(self._retry_policy.backoff_seconds)

        return StepExecutionResult(
            step_id=step_id,
            status=ExecutionStatus.FAILED,
            error=last_error or "Unknown execution failure.",
            attempts=max(attempts, 1),
            latency=_latency("step", started_at),
        )

    async def _record(self, trace_id: UUID, name: str, payload: dict[str, Any]) -> None:
        await self._observability.record_event(
            TraceEvent(
                trace_id=trace_id,
                harness=HarnessKind.EXECUTION,
                name=name,
                level=TraceEventLevel.INFO,
                payload=payload,
            )
        )


def _token_usage(output: dict[str, Any]) -> TokenUsage:
    raw = output.get("token_usage")
    if isinstance(raw, dict):
        return TokenUsage.model_validate(raw)
    return TokenUsage()


def _sum_tokens(results: list[StepExecutionResult]) -> TokenUsage:
    prompt = sum(result.token_usage.prompt_tokens for result in results)
    completion = sum(result.token_usage.completion_tokens for result in results)
    total = sum(result.token_usage.total_tokens for result in results)
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _latency(name: str, started_at: datetime) -> LatencyMetric:
    return LatencyMetric(
        name=name,
        duration_ms=(datetime.now(UTC) - started_at).total_seconds() * 1000,
    )
