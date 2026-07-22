from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from uuid import UUID

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
        from app.task_queue import celery_app

        if celery_app is None:
            raise RuntimeError("Celery execution backend is unavailable.")
        result = celery_app.send_task("datamind.assistant.run", args=(str(run_id), user_id, dataset_store_path), task_id=str(run_id))
        AssistantRepository(dataset_store_path, user_id=user_id).set_broker_task(run_id, str(result.id))
        return
    with _lock:
        future = _futures.get(run_id)
        if future is not None and not future.done():
            return
        _futures[run_id] = _executor.submit(run_assistant_run, run_id, user_id, dataset_store_path)


def run_assistant_run(run_id: UUID, user_id: str, dataset_store_path: str) -> None:
    assistant_store = AssistantRepository(dataset_store_path, user_id=user_id)
    store = DatasetStoreRepository(dataset_store_path, user_id=user_id)
    try:
        AssistantWorkflowRunner(store=store, assistant_store=assistant_store).run(run_id)
    except Exception as exc:
        logger.exception("Assistant run failed.")
        run = assistant_store.get_run(run_id)
        if "canceled" in str(exc).lower() or assistant_store.cancel_requested(run_id):
            assistant_store.update_run(run_id, status="canceled", current_stage="canceled", error=str(exc), completed=True)
            status, message = "canceled", "Kimi 回答已取消。"
        else:
            assistant_store.update_run(run_id, status="failed", current_stage="failed", error=f"{type(exc).__name__}: {exc}", completed=True)
            assistant_store.update_message(run.assistant_message_id, content="Kimi 暂时无法完成这次回答，请稍后重试。", status="failed", metadata={"error": str(exc)})
            status, message = "failed", f"Kimi 回答失败：{exc}"
        assistant_store.append_event(run_id, event_type="run.failed", status=status, message=message, payload={"error": str(exc)})
    finally:
        with _lock:
            _futures.pop(run_id, None)
