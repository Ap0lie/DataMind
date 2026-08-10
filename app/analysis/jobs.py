from __future__ import annotations

import logging
import socket
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from uuid import UUID

from app.analysis.runtime import build_analysis_runner
from app.core.settings import get_settings
from app.schemas.analysis import AnalysisRunResponse, MultimodalInputResponse
from app.storage.dataset_store import DatasetStoreRepository

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="datamind-analysis")
_futures: dict[UUID, Future[None]] = {}
_lock = Lock()


class AnalysisJobCanceled(RuntimeError):
    pass


def start_analysis_job(
    *,
    job_id: UUID,
    user_id: str,
    dataset_store_path: str,
) -> None:
    settings = get_settings()
    if settings.execution_backend.lower() == "celery":
        _enqueue_celery_job(
            job_id=job_id,
            user_id=user_id,
            dataset_store_path=dataset_store_path,
        )
        return
    with _lock:
        future = _futures.get(job_id)
        if future is not None and not future.done():
            return
        _futures[job_id] = _executor.submit(
            run_analysis_job,
            job_id,
            user_id,
            dataset_store_path,
        )


def _enqueue_celery_job(*, job_id: UUID, user_id: str, dataset_store_path: str) -> None:
    from app.task_queue import celery_app

    if celery_app is None:
        raise RuntimeError(
            "DATAMIND_EXECUTION_BACKEND=celery requires the celery and redis packages."
        )
    result = celery_app.send_task(
        "datamind.analysis.run",
        args=(str(job_id), user_id, dataset_store_path),
        task_id=str(job_id),
    )
    DatasetStoreRepository(dataset_store_path, user_id=user_id).set_analysis_job_broker_task(
        job_id,
        str(result.id),
    )


def revoke_analysis_job(job_id: UUID, *, terminate: bool = False) -> None:
    if get_settings().execution_backend.lower() != "celery":
        return
    from app.task_queue import celery_app

    if celery_app is not None:
        celery_app.control.revoke(str(job_id), terminate=terminate)


def recover_queued_analysis_jobs(dataset_store_path: str) -> int:
    settings = get_settings()
    if settings.execution_backend.lower() != "celery":
        return 0
    repository = DatasetStoreRepository(dataset_store_path, user_id="default")
    recovered = 0
    for job in repository.list_all_recoverable_analysis_jobs():
        if job.status == "queued" and job.broker_task_id and job.updated_at:
            updated_at = datetime.fromisoformat(job.updated_at)
            if datetime.now(UTC) - updated_at < timedelta(seconds=settings.worker_lease_seconds):
                continue
        _enqueue_celery_job(
            job_id=job.id,
            user_id=job.user_id,
            dataset_store_path=dataset_store_path,
        )
        recovered += 1
    return recovered


def run_analysis_job(
    job_id: UUID,
    user_id: str,
    dataset_store_path: str,
    *,
    worker_id: str | None = None,
) -> None:
    repository = DatasetStoreRepository(dataset_store_path, user_id=user_id)
    settings = get_settings()
    resolved_worker_id = worker_id or f"{socket.gethostname()}:{job_id}"
    heartbeat_stop = Event()
    heartbeat_thread: Thread | None = None
    try:
        job = repository.get_analysis_job(job_id)
        if job.status == "canceled" or job.cancel_requested:
            repository.update_analysis_job(
                job_id,
                status="canceled",
                current_stage="canceled",
                event_message="Analysis job canceled before it started.",
                completed=True,
            )
            return
        claimed = repository.claim_analysis_job(
            job_id,
            worker_id=resolved_worker_id,
            lease_seconds=settings.worker_lease_seconds,
        )
        if claimed is None:
            return
        heartbeat_thread = Thread(
            target=_maintain_analysis_job_lease,
            kwargs={
                "repository": repository,
                "job_id": job_id,
                "worker_id": resolved_worker_id,
                "lease_seconds": settings.worker_lease_seconds,
                "stop": heartbeat_stop,
            },
            name=f"analysis-heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        def progress_callback(stage: str, progress: int, message: str) -> None:
            current = repository.get_analysis_job(job_id)
            if current.cancel_requested:
                raise AnalysisJobCanceled("Analysis job was canceled.")
            repository.update_analysis_job(
                job_id,
                status="running",
                progress=progress,
                current_stage=stage,
                event_message=message,
                lease_owner=resolved_worker_id,
                lease_expires_at=(
                    datetime.now(UTC) + timedelta(seconds=settings.worker_lease_seconds)
                ).isoformat(),
                heartbeat_at=datetime.now(UTC).isoformat(),
            )

        def cancel_checker() -> bool:
            return repository.get_analysis_job(job_id).cancel_requested

        def node_event_callback(event: dict[str, object]) -> None:
            raw_token_usage = event.get("token_usage")
            token_usage = {
                str(key): int(value)
                for key, value in (
                    raw_token_usage.items()
                    if isinstance(raw_token_usage, dict)
                    else ()
                )
                if isinstance(value, (int, float))
            }
            repository.append_analysis_job_event(
                job_id,
                node=str(event.get("node") or "unknown"),
                status=str(event.get("status") or "unknown"),
                message=str(event.get("message") or ""),
                attempt=int(event.get("attempt") or 0),
                duration_ms=(
                    float(event["duration_ms"]) if event.get("duration_ms") is not None else None
                ),
                provider=str(event["provider"]) if event.get("provider") else None,
                model=str(event["model"]) if event.get("model") else None,
                token_usage=token_usage,
                error_code=str(event["error_code"]) if event.get("error_code") else None,
                event_type=str(event["event_type"]) if event.get("event_type") else None,
                iteration=int(event["iteration"]) if event.get("iteration") is not None else None,
                tool_name=str(event["tool_name"]) if event.get("tool_name") else None,
                repair_of_sequence=int(event["repair_of_sequence"])
                if event.get("repair_of_sequence") is not None
                else None,
                payload=dict(event.get("payload") or {}),
            )

        claimed_job = repository.get_analysis_job(job_id)
        result = build_analysis_runner(
            repository,
            prompt_overrides=job.prompt_overrides,
        ).run(
            dataset_id=job.dataset_id,
            dataset_group_id=job.dataset_group_id,
            additional_dataset_ids=job.additional_dataset_ids,
            join_plan=job.join_plan,
            relationship_plan=job.relationship_plan,
            planner_decision=(
                repository.get_planner_decision(job.planner_decision_id)
                if job.planner_decision_id
                else None
            ),
            question=job.question,
            prompt_overrides=job.prompt_overrides,
            multimodal_inputs=tuple(
                MultimodalInputResponse.model_validate(item) for item in job.multimodal_inputs
            ),
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
            workflow_id=job_id,
            resume=claimed_job.attempt_count > 1,
            node_event_callback=node_event_callback,
            agent_mode=job.agent_mode,
        )
        _complete_job(repository, job_id, result)
    except AnalysisJobCanceled:
        repository.update_analysis_job(
            job_id,
            status="canceled",
            current_stage="canceled",
            event_message="Analysis job canceled.",
            completed=True,
        )
    except RuntimeError as exc:
        if "canceled" in str(exc).lower():
            repository.update_analysis_job(
                job_id,
                status="canceled",
                current_stage="canceled",
                event_message="Analysis job canceled.",
                completed=True,
            )
            return
        logger.exception("Analysis job failed.")
        repository.update_analysis_job(
            job_id,
            status="failed",
            current_stage="failed",
            event_message="Analysis job failed.",
            error=str(exc),
            completed=True,
        )
    except Exception as exc:
        logger.exception("Analysis job failed.")
        repository.update_analysis_job(
            job_id,
            status="failed",
            current_stage="failed",
            event_message="Analysis job failed.",
            error=f"{type(exc).__name__}: {exc}",
            completed=True,
        )
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        with _lock:
            _futures.pop(job_id, None)


def _maintain_analysis_job_lease(
    *,
    repository: DatasetStoreRepository,
    job_id: UUID,
    worker_id: str,
    lease_seconds: int,
    stop: Event,
) -> None:
    interval = max(5.0, min(30.0, lease_seconds / 3))
    while not stop.wait(interval):
        try:
            current = repository.get_analysis_job(job_id)
            if current.status != "running" or current.lease_owner != worker_id:
                return
            repository.heartbeat_analysis_job(
                job_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        except Exception:
            logger.exception("Analysis job heartbeat failed.")


def _complete_job(
    repository: DatasetStoreRepository,
    job_id: UUID,
    result: AnalysisRunResponse,
) -> None:
    repository.update_analysis_job(
        job_id,
        status="completed",
        progress=100,
        current_stage="complete",
        event_message="Analysis job completed.",
        result=result.model_dump(mode="json"),
        report_id=result.report_id,
        completed=True,
        loop_summary=result.loop_summary,
        loop_terminal_reason=result.loop_terminal_reason,
        report_strategy=result.report_strategy,
        report_revision_count=result.report_revision_count,
        report_terminal_reason=result.report_terminal_reason,
    )
