from __future__ import annotations

from fastapi import APIRouter, status

from app.core.entities import ExecutionPlan, TaskIntent
from app.schemas.tasks import PlanStepSchema, TaskCreateRequest, TaskPlanResponse, TaskRunResponse
from app.workflows.examples import run_data_analysis_for_task

router = APIRouter()


@router.post(
    "",
    response_model=TaskRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_task(request: TaskCreateRequest) -> TaskRunResponse:
    task = TaskIntent(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        prompt=request.prompt,
        locale=request.locale,
    )
    state = await run_data_analysis_for_task(task)
    plan = state.get("plan")
    review = state.get("review")
    report = state.get("report")
    return TaskRunResponse(
        task_id=task.task_id,
        workflow_status=state["status"],
        plan=_plan_response(plan) if plan is not None else None,
        review_passed=review.passed if review is not None else None,
        report_title=report.title if report is not None else None,
        report_markdown=report.markdown if report is not None else None,
        checkpoint_count=len(state["checkpoints"]),
        trace_event_count=len(state["trace"]),
    )


def _plan_response(plan: ExecutionPlan) -> TaskPlanResponse:
    return TaskPlanResponse(
        plan_id=plan.plan_id,
        task_id=plan.task.task_id,
        objective=plan.objective,
        status=plan.status,
        created_at=plan.created_at,
        steps=tuple(
            PlanStepSchema(
                step_id=step.step_id,
                name=step.name,
                agent=step.agent,
                description=step.description,
                required_capabilities=step.required_capabilities,
                tool_calls=step.tool_calls,
                depends_on=step.depends_on,
                status=step.status,
            )
            for step in plan.steps
        ),
    )
