from __future__ import annotations

import logging
import socket
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Lock
from uuid import UUID

from app.analysis.cleaning_workflow import CleaningWorkflowRunner
from app.analysis.model_router import MCPAnalysisModelRouter
from app.analysis.prompt_override_router import PromptOverrideModelRouter
from app.core.settings import get_settings
from app.storage.dataset_store import DatasetStoreRepository

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="datamind-cleaning")
_futures: dict[UUID, Future[None]] = {}
_lock = Lock()


def start_cleaning_job(*, job_id: UUID, user_id: str, dataset_store_path: str) -> None:
    settings = get_settings()
    if settings.execution_backend.lower() == "celery":
        _enqueue_celery_job(job_id, user_id, dataset_store_path)
        return
    with _lock:
        future = _futures.get(job_id)
        if future is not None and not future.done():
            return
        _futures[job_id] = _executor.submit(run_cleaning_job, job_id, user_id, dataset_store_path)


def _enqueue_celery_job(job_id: UUID, user_id: str, dataset_store_path: str) -> None:
    from app.task_queue import celery_app

    if celery_app is None:
        raise RuntimeError("Celery execution requires celery and Redis packages.")
    result = celery_app.send_task(
        "datamind.cleaning.run",
        args=(str(job_id), user_id, dataset_store_path),
        task_id=str(job_id),
    )
    DatasetStoreRepository(dataset_store_path, user_id=user_id).update_cleaning_job(
        job_id, broker_task_id=str(result.id)
    )


def recover_queued_cleaning_jobs(dataset_store_path: str) -> int:
    if get_settings().execution_backend.lower() != "celery":
        return 0
    repository = DatasetStoreRepository(dataset_store_path, user_id="default")
    count = 0
    for job in repository.list_all_recoverable_cleaning_jobs():
        _enqueue_celery_job(job.id, job.user_id, dataset_store_path)
        count += 1
    return count


def run_cleaning_job(
    job_id: UUID, user_id: str, dataset_store_path: str, *, worker_id: str | None = None
) -> None:
    repository = DatasetStoreRepository(dataset_store_path, user_id=user_id)
    settings = get_settings()
    resolved_worker = worker_id or f"{socket.gethostname()}:{job_id}"
    try:
        job = repository.get_cleaning_job(job_id)
        if job.cancel_requested or job.status == "canceled":
            return
        claimed = repository.claim_cleaning_job(
            job_id, worker_id=resolved_worker, lease_seconds=settings.worker_lease_seconds
        )
        if claimed is None:
            return

        def progress(stage: str, value: int, message: str) -> None:
            current = repository.get_cleaning_job(job_id)
            if current.cancel_requested:
                raise RuntimeError("Cleaning job canceled.")
            repository.update_cleaning_job(
                job_id,
                status="running",
                progress=value,
                current_stage=stage,
                event_message=message,
                lease_owner=resolved_worker,
                lease_expires_at=(
                    datetime.now(UTC) + timedelta(seconds=settings.worker_lease_seconds)
                ).isoformat(),
                heartbeat_at=datetime.now(UTC).isoformat(),
            )

        def emit(event: dict[str, object]) -> None:
            repository.append_cleaning_job_event(
                job_id,
                stage=str(event.get("stage") or "cleaning_loop"),
                status=str(event.get("status") or "completed"),
                message=str(event.get("message") or ""),
                event_type=str(event["event_type"]) if event.get("event_type") else None,
                iteration=int(event["iteration"]) if event.get("iteration") is not None else None,
                strategy=str(event["strategy"]) if event.get("strategy") else None,
                payload=dict(event.get("payload") or {}),
            )

        model_router = PromptOverrideModelRouter(
            MCPAnalysisModelRouter(),
            job.prompt_overrides,
        )
        result = CleaningWorkflowRunner(repository, model_router=model_router).run(
            job_id=job_id,
            progress_callback=progress,
            event_callback=emit,
            cancel_checker=lambda: repository.get_cleaning_job(job_id).cancel_requested,
            resume=claimed.attempt_count > 1,
        )
        repository.update_cleaning_job(
            job_id,
            status="completed",
            progress=100,
            current_stage="complete",
            event_message="Cleaning job completed.",
            selected_strategy=str(result.get("selected_strategy") or "rules"),
            loop_summary={
                "decisions": claimed.attempt_count,
                "quality": result.get("quality") or {},
                "failures": result.get("failures") or [],
            },
            terminal_reason=str(result.get("terminal_reason") or "validated"),
            result=result,
            cleaning_run_id=UUID(str(result["cleaning_run_id"])),
            completed=True,
        )
    except Exception as exc:
        canceled = "canceled" in str(exc).lower()
        logger.exception("Cleaning job stopped.")
        repository.update_cleaning_job(
            job_id,
            status="canceled" if canceled else "failed",
            current_stage="canceled" if canceled else "failed",
            event_message="Cleaning job canceled; active version was preserved."
            if canceled
            else "Cleaning job failed; active version was preserved.",
            error=None if canceled else f"{type(exc).__name__}: {exc}",
            completed=True,
        )
    finally:
        with _lock:
            _futures.pop(job_id, None)
