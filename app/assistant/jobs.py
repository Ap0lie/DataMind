from __future__ import annotations

import logging
import socket
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from uuid import UUID, uuid4

from app.assistant.control import AssistantRunCanceled, AssistantRunPaused
from app.assistant.workflow import AssistantWorkflowRunner
from app.core.settings import get_settings
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="datamind-assistant")
_futures: dict[UUID, Future[None]] = {}
_lock = Lock()


def start_assistant_run(*, run_id: UUID, user_id: str, dataset_store_path: str) -> None:
    if get_settings().execution_backend.lower() == "celery":
        _enqueue_celery_run(run_id, user_id, dataset_store_path)
        return
    with _lock:
        future = _futures.get(run_id)
        if future is not None and not future.done():
            return
        _futures[run_id] = _executor.submit(run_assistant_run, run_id, user_id, dataset_store_path)


def _enqueue_celery_run(run_id: UUID, user_id: str, dataset_store_path: str) -> None:
    from app.task_queue import celery_app

    if celery_app is None:
        raise RuntimeError("Celery execution backend is unavailable.")
    result = celery_app.send_task(
        "datamind.assistant.run",
        args=(str(run_id), user_id, dataset_store_path),
        task_id=str(uuid4()),
    )
    AssistantRepository(dataset_store_path, user_id=user_id).set_broker_task(
        run_id, str(result.id)
    )


def recover_queued_assistant_runs(dataset_store_path: str) -> int:
    settings = get_settings()
    if settings.execution_backend.lower() != "celery":
        return 0
    repository = AssistantRepository(dataset_store_path, user_id="default")
    recovered = 0
    for item in repository.list_all_recoverable_runs():
        if item["status"] == "queued" and item["broker_task_id"]:
            updated_at = datetime.fromisoformat(str(item["updated_at"]))
            if datetime.now(UTC) - updated_at < timedelta(
                seconds=settings.worker_lease_seconds
            ):
                continue
        try:
            _enqueue_celery_run(
                item["run_id"],
                str(item["user_id"]),
                dataset_store_path,
            )
            recovered += 1
        except Exception:
            logger.exception("Unable to recover queued Assistant run %s.", item["run_id"])
    return recovered


def run_assistant_run(
    run_id: UUID,
    user_id: str,
    dataset_store_path: str,
    *,
    worker_id: str | None = None,
) -> None:
    assistant_store = AssistantRepository(dataset_store_path, user_id=user_id)
    store = DatasetStoreRepository(dataset_store_path, user_id=user_id)
    settings = get_settings()
    resolved_worker = worker_id or f"{socket.gethostname()}:{run_id}"
    stop_heartbeat = Event()
    heartbeat_thread: Thread | None = None
    try:
        current = assistant_store.get_run(run_id)
        if current.cancel_requested or current.status in {
            "canceled",
            "pause_requested",
            "paused",
        }:
            return
        claimed = assistant_store.claim_run(
            run_id,
            worker_id=resolved_worker,
            lease_seconds=settings.worker_lease_seconds,
        )
        if claimed is None:
            return

        def heartbeat() -> None:
            interval = max(5.0, settings.worker_lease_seconds / 3)
            while not stop_heartbeat.wait(interval):
                try:
                    if not assistant_store.heartbeat_run(
                        run_id,
                        worker_id=resolved_worker,
                        lease_seconds=settings.worker_lease_seconds,
                    ):
                        return
                except Exception:
                    logger.exception("Assistant run heartbeat failed for %s.", run_id)

        heartbeat_thread = Thread(
            target=heartbeat,
            name=f"datamind-assistant-heartbeat-{run_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        AssistantWorkflowRunner(store=store, assistant_store=assistant_store).run(run_id)
    except AssistantRunPaused:
        assistant_store.mark_paused(run_id)
    except AssistantRunCanceled:
        assistant_store.request_cancel(run_id)
    except Exception as exc:
        if "canceled" in str(exc).lower() or assistant_store.cancel_requested(run_id):
            assistant_store.request_cancel(run_id)
        else:
            logger.exception("Assistant run failed.")
            run = assistant_store.get_run(run_id)
            assistant_store.update_run(run_id, status="failed", current_stage="failed", error=f"{type(exc).__name__}: {exc}", completed=True)
            assistant_store.update_message(run.assistant_message_id, content="Kimi 暂时无法完成这次回答，请稍后重试。", status="failed", metadata={"error": str(exc)})
            assistant_store.append_event(
                run_id,
                event_type="run.failed",
                status="failed",
                message=f"Kimi 回答失败：{exc}",
                payload={"error": str(exc)},
            )
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        assistant_store.release_run_lease(run_id, worker_id=resolved_worker)
        with _lock:
            _futures.pop(run_id, None)
