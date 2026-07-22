from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.analysis.jobs import revoke_analysis_job, start_analysis_job
from app.analysis.multidataset import suggest_dataset_joins
from app.analysis.runtime import build_analysis_runner
from app.api.v1.deps import current_user_id
from app.core.settings import get_settings
from app.schemas.analysis import (
    AnalysisJobEventResponse,
    AnalysisJobListResponse,
    AnalysisJobResponse,
    AnalysisRunRequest,
    AnalysisRunResponse,
    JoinSuggestionRequest,
    JoinSuggestionResponse,
)
from app.schemas.semantic import (
    PlannerDecisionResponse,
    PlannerFeedbackRequest,
    PlannerFeedbackResponse,
    SemanticPlanRequest,
)
from app.security.rate_limit import RateLimitExceeded, enforce_rate_limit
from app.semantic.service import SemanticLayerService
from app.storage.dataset_store import DatasetStoreRepository, StoredAnalysisJob

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/plans", response_model=PlannerDecisionResponse)
def create_semantic_plan(
    request: SemanticPlanRequest, user_id: str = Depends(current_user_id)
) -> PlannerDecisionResponse:
    try:
        decision = SemanticLayerService(_repository(user_id)).create_planner_decision(
            dataset_id=request.dataset_id,
            dataset_group_id=request.dataset_group_id,
            question=request.question,
        )
        return _planner_decision_response(decision)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/planner-decisions/{decision_id}/feedback", response_model=PlannerFeedbackResponse)
def create_planner_feedback(
    decision_id: UUID, request: PlannerFeedbackRequest, user_id: str = Depends(current_user_id)
) -> PlannerFeedbackResponse:
    try:
        repository = _repository(user_id)
        feedback = repository.save_planner_feedback(
            decision_id=decision_id,
            action=request.action,
            corrected_plan=request.corrected_plan,
        )
        SemanticLayerService(repository).rebuild_user_calibrator()
        return PlannerFeedbackResponse(
            feedback_id=feedback["id"],
            decision_id=feedback["decision_id"],
            action=feedback["action"],
            created_at=feedback["created_at"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs", response_model=AnalysisJobResponse, status_code=202)
def create_analysis_job(
    request: AnalysisRunRequest,
    user_id: str = Depends(current_user_id),
) -> AnalysisJobResponse:
    repository = _repository(user_id)
    try:
        decision = None
        if request.planner_decision_id:
            decision = repository.get_planner_decision(request.planner_decision_id)
            if bool(decision["requires_confirmation"]) and not (
                bool(decision["confirmed"]) or request.confirmed_low_confidence
            ):
                raise ValueError(
                    "Low-confidence semantic plan requires confirmation before execution."
                )
        settings = get_settings()
        agent_mode = _resolve_agent_mode(request.agent_mode)
        enforce_rate_limit(
            f"analysis-job:{user_id}",
            limit=settings.job_rate_limit,
            window_seconds=60,
        )
        job = repository.create_analysis_job(
            dataset_id=request.dataset_id,
            dataset_group_id=request.dataset_group_id,
            additional_dataset_ids=request.additional_dataset_ids,
            join_plan=tuple(item.model_dump(mode="json") for item in request.join_plan),
            relationship_plan=tuple(
                item.model_dump(mode="json") for item in request.relationship_plan
            ),
            question=request.question,
            prompt_overrides=request.prompt_overrides.as_dict(),
            multimodal_inputs=tuple(
                item.model_dump(mode="json") for item in request.multimodal_inputs
            ),
            agent_mode=agent_mode,
        )
        if decision is not None:
            repository.attach_planner_decision_to_job(job_id=job.id, decision=decision)
        start_analysis_job(
            job_id=job.id,
            user_id=user_id,
            dataset_store_path=get_settings().dataset_store_path,
        )
        return _job_response(job)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs", response_model=AnalysisJobListResponse)
def list_analysis_jobs(
    dataset_id: UUID | None = None,
    limit: int = 50,
    user_id: str = Depends(current_user_id),
) -> AnalysisJobListResponse:
    try:
        jobs = _repository(user_id).list_analysis_jobs(dataset_id=dataset_id, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AnalysisJobListResponse(jobs=tuple(_job_response(job) for job in jobs))


@router.get("/jobs/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job(
    job_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AnalysisJobResponse:
    try:
        return _job_response(_repository(user_id).get_analysis_job(job_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/events")
async def stream_analysis_job_events(
    job_id: UUID,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    user_id: str = Depends(current_user_id),
) -> StreamingResponse:
    repository = _repository(user_id)
    try:
        repository.get_analysis_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        cursor = max(after_sequence, int(last_event_id or 0))
    except ValueError:
        cursor = after_sequence

    async def event_stream():
        nonlocal cursor
        idle_ticks = 0
        while True:
            events = repository.list_analysis_job_events(job_id, after_sequence=cursor)
            for event in events:
                cursor = int(event.get("sequence") or cursor)
                yield (
                    f"id: {cursor}\n"
                    "event: workflow\n"
                    f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                )
                idle_ticks = 0
            job = repository.get_analysis_job(job_id)
            if job.status not in {"queued", "running", "cancel_requested"} and not events:
                yield f"event: end\ndata: {json.dumps({'status': job.status})}\n\n"
                return
            idle_ticks += 1
            if idle_ticks % 15 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/jobs/{job_id}/result", response_model=AnalysisRunResponse)
def get_analysis_job_result(
    job_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AnalysisRunResponse:
    try:
        job = _repository(user_id).get_analysis_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis job is {job.status}; result is not available yet.",
        )
    if job.result is None:
        raise HTTPException(status_code=500, detail="Analysis job completed without a result.")
    return AnalysisRunResponse.model_validate(job.result)


@router.post("/jobs/{job_id}/cancel", response_model=AnalysisJobResponse)
def cancel_analysis_job(
    job_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AnalysisJobResponse:
    try:
        job = _repository(user_id).request_analysis_job_cancel(job_id)
        if job.status == "canceled":
            revoke_analysis_job(job_id)
        return _job_response(job)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/retry", response_model=AnalysisJobResponse, status_code=202)
def retry_analysis_job(
    job_id: UUID,
    user_id: str = Depends(current_user_id),
) -> AnalysisJobResponse:
    repository = _repository(user_id)
    try:
        original = repository.get_analysis_job(job_id)
        job = repository.create_analysis_job(
            dataset_id=original.dataset_id,
            dataset_group_id=original.dataset_group_id,
            additional_dataset_ids=original.additional_dataset_ids,
            join_plan=original.join_plan,
            relationship_plan=original.relationship_plan,
            question=original.question,
            prompt_overrides=original.prompt_overrides,
            multimodal_inputs=original.multimodal_inputs,
            retry_of=original.id,
            agent_mode=original.agent_mode,
        )
        start_analysis_job(
            job_id=job.id,
            user_id=user_id,
            dataset_store_path=get_settings().dataset_store_path,
        )
        return _job_response(job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/run", response_model=AnalysisRunResponse)
def run_analysis(
    request: AnalysisRunRequest,
    user_id: str = Depends(current_user_id),
) -> AnalysisRunResponse:
    try:
        agent_mode = _resolve_agent_mode(request.agent_mode)
        return build_analysis_runner(
            _repository(user_id),
            prompt_overrides=request.prompt_overrides.as_dict(),
        ).run(
            dataset_id=request.dataset_id,
            dataset_group_id=request.dataset_group_id,
            additional_dataset_ids=request.additional_dataset_ids,
            join_plan=tuple(item.model_dump(mode="json") for item in request.join_plan),
            relationship_plan=tuple(
                item.model_dump(mode="json") for item in request.relationship_plan
            ),
            question=request.question,
            prompt_overrides=request.prompt_overrides.as_dict(),
            multimodal_inputs=request.multimodal_inputs,
            agent_mode=agent_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis run failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis run failed: {type(exc).__name__}: {exc}",
        ) from exc


@router.post("/join-suggestions", response_model=JoinSuggestionResponse)
def join_suggestions(
    request: JoinSuggestionRequest,
    user_id: str = Depends(current_user_id),
) -> JoinSuggestionResponse:
    try:
        return suggest_dataset_joins(
            _repository(user_id),
            dataset_id=request.dataset_id,
            additional_dataset_ids=request.additional_dataset_ids,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _repository(user_id: str = "default") -> DatasetStoreRepository:
    return DatasetStoreRepository(get_settings().dataset_store_path, user_id=user_id)


def _resolve_agent_mode(requested: str) -> str:
    settings = get_settings()
    if requested == "auto":
        return settings.agent_loop_default_mode if settings.agent_loop_enabled else "legacy"
    if requested == "loop":
        if not settings.agent_loop_enabled:
            raise ValueError("Autonomous analysis loop is disabled by deployment policy.")
        if (
            settings.environment.lower() == "production"
            and not settings.agent_loop_allow_request_override
            and settings.agent_loop_default_mode != "loop"
        ):
            raise ValueError("Production policy does not allow clients to force loop mode.")
    return requested


def _planner_decision_response(decision: dict[str, object]) -> PlannerDecisionResponse:
    plan = decision.get("semantic_plan") if isinstance(decision.get("semantic_plan"), dict) else {}
    scores = (
        decision.get("component_scores")
        if isinstance(decision.get("component_scores"), dict)
        else {}
    )
    return PlannerDecisionResponse(
        decision_id=UUID(str(decision["id"])),
        dataset_id=UUID(str(decision["dataset_id"])),
        dataset_group_id=UUID(str(decision["dataset_group_id"]))
        if decision.get("dataset_group_id")
        else None,
        question=str(decision["question"]),
        semantic_model_id=UUID(str(decision["semantic_model_id"]))
        if decision.get("semantic_model_id")
        else None,
        semantic_model_version=int(decision["semantic_model_version"])
        if decision.get("semantic_model_version") is not None
        else None,
        semantic_source=str(decision.get("semantic_source") or "legacy"),
        semantic_plan=plan,
        confidence_breakdown=scores,
        raw_confidence=float(decision["raw_confidence"]),
        calibrated_confidence=float(decision["calibrated_confidence"]),
        confidence_level=str(decision["confidence_level"]),
        requires_confirmation=bool(decision["requires_confirmation"]),
        ambiguities=tuple(str(item) for item in plan.get("ambiguities") or ()),
        evidence=tuple(str(item) for item in plan.get("evidence") or ()),
        created_at=str(decision.get("created_at") or ""),
    )


def _job_response(job: StoredAnalysisJob) -> AnalysisJobResponse:
    return AnalysisJobResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        dataset_group_id=job.dataset_group_id,
        additional_dataset_ids=job.additional_dataset_ids,
        join_plan=job.join_plan,
        relationship_plan=job.relationship_plan,
        question=job.question,
        prompt_overrides=job.prompt_overrides,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        events=tuple(
            AnalysisJobEventResponse(
                sequence=int(event.get("sequence") or 0),
                node=str(event.get("node") or event.get("stage") or ""),
                stage=str(event.get("stage") or ""),
                progress=int(event.get("progress") or 0),
                message=str(event.get("message") or ""),
                status=str(event.get("status") or job.status),
                attempt=int(event.get("attempt") or 0),
                duration_ms=(
                    float(event["duration_ms"]) if event.get("duration_ms") is not None else None
                ),
                provider=str(event["provider"]) if event.get("provider") else None,
                model=str(event["model"]) if event.get("model") else None,
                token_usage={
                    str(key): int(value)
                    for key, value in dict(event.get("token_usage") or {}).items()
                    if isinstance(value, (int, float))
                },
                error_code=str(event["error_code"]) if event.get("error_code") else None,
                event_type=str(event["event_type"]) if event.get("event_type") else None,
                iteration=int(event["iteration"]) if event.get("iteration") is not None else None,
                tool_name=str(event["tool_name"]) if event.get("tool_name") else None,
                repair_of_sequence=int(event["repair_of_sequence"])
                if event.get("repair_of_sequence") is not None
                else None,
                payload=dict(event.get("payload") or {}),
                created_at=str(event.get("created_at") or ""),
            )
            for event in job.events
        ),
        error=job.error,
        report_id=job.report_id,
        retry_of=job.retry_of,
        cancel_requested=job.cancel_requested,
        attempt=job.attempt_count,
        resumable=job.status in {"queued", "running", "interrupted", "failed"},
        last_event_sequence=max(
            (int(event.get("sequence") or 0) for event in job.events),
            default=0,
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        agent_mode=job.agent_mode,
        loop_summary=job.loop_summary or {},
        loop_terminal_reason=job.loop_terminal_reason,
        report_strategy=job.report_strategy,
        report_revision_count=job.report_revision_count,
        report_terminal_reason=job.report_terminal_reason,
    )
