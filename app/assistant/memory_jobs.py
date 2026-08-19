from __future__ import annotations

import logging
import socket
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from uuid import UUID, uuid4

from app.assistant.memory import (
    AssistantMemoryService,
    MemoryCandidate,
    extract_memory_candidates,
    should_use_model_memory_extractor,
)
from app.core.settings import get_settings
from app.memory.guards import DataMindMemoryGuard
from app.memory.manager import LangMemFormationManager
from app.memory.models import GuardedMemoryRecord
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="datamind-memory")
_futures: dict[UUID, Future[None]] = {}
_lock = Lock()


@dataclass(frozen=True)
class MemoryFormationResult:
    candidates: tuple[MemoryCandidate, ...]
    rejected_codes: tuple[str, ...] = ()
    source: str = "deterministic"


def schedule_memory_maintenance(
    *,
    run_id: UUID,
    user_id: str,
    dataset_store_path: str,
) -> UUID:
    assistant_store = AssistantRepository(dataset_store_path, user_id=user_id)
    run = assistant_store.get_run(run_id)
    repository = AssistantMemoryRepository(dataset_store_path, user_id=user_id)
    job = repository.create_maintenance_job(
        run_id=run_id,
        conversation_id=run.conversation_id,
        user_message_id=run.user_message_id,
        assistant_message_id=run.assistant_message_id,
        analysis_job_id=run.analysis_job_id,
    )
    if job["status"] == "queued":
        _dispatch(job["job_id"], user_id, dataset_store_path)
    return UUID(str(job["job_id"]))


def _dispatch(job_id: UUID, user_id: str, dataset_store_path: str) -> None:
    if get_settings().execution_backend.lower() == "celery":
        from app.task_queue import celery_app

        if celery_app is None:
            raise RuntimeError("Celery execution backend is unavailable.")
        result = celery_app.send_task(
            "datamind.assistant.memory.maintain",
            args=(str(job_id), user_id, dataset_store_path),
            task_id=str(uuid4()),
        )
        AssistantMemoryRepository(
            dataset_store_path, user_id=user_id
        ).set_maintenance_broker_task(job_id, str(result.id))
        return
    with _lock:
        future = _futures.get(job_id)
        if future is None or future.done():
            _futures[job_id] = _executor.submit(
                run_memory_maintenance,
                job_id,
                user_id,
                dataset_store_path,
            )


def run_memory_maintenance(
    job_id: UUID,
    user_id: str,
    dataset_store_path: str,
    *,
    worker_id: str | None = None,
) -> None:
    settings = get_settings()
    repository = AssistantMemoryRepository(dataset_store_path, user_id=user_id)
    assistant_store = AssistantRepository(dataset_store_path, user_id=user_id)
    store = DatasetStoreRepository(dataset_store_path, user_id=user_id)
    resolved_worker = worker_id or f"{socket.gethostname()}:{job_id}"
    claimed = repository.claim_maintenance_job(
        job_id,
        worker_id=resolved_worker,
        lease_seconds=settings.worker_lease_seconds,
    )
    if claimed is None:
        return
    events: list[dict[str, object]] = []
    try:
        service = AssistantMemoryService(
            repository=repository,
            store=store,
            settings=settings,
        )
        conversation = assistant_store.get_conversation(claimed["conversation_id"])
        user_message = assistant_store.get_message(claimed["user_message_id"])
        formation = _model_memory_candidates(
            source_message=user_message,
            source_message_id=claimed["user_message_id"],
        )
        if formation.rejected_codes:
            events.append(
                {
                    "event_type": "memory.rejected",
                    "status": "warning",
                    "message": "部分候选记忆未通过来源或安全校验。",
                    "payload": {
                        "source": formation.source,
                        "reason_codes": list(formation.rejected_codes),
                    },
                }
            )
        events.extend(
            service.capture_user_memories(
                conversation=conversation,
                user_message=user_message,
                model_candidates=formation.candidates,
            )
        )
        summary = service.update_conversation_summary(
            assistant_store=assistant_store,
            conversation_id=claimed["conversation_id"],
        )
        if summary is not None:
            events.append(
                {
                    "event_type": "memory.summary_updated",
                    "status": "completed",
                    "message": "较早对话已压缩为结构化摘要。",
                    "payload": {"summary_version": summary["summary_version"]},
                }
            )
        if claimed["analysis_job_id"] is not None:
            experience = service.save_analysis_experience(claimed["analysis_job_id"])
            if experience is not None:
                events.append(
                    {
                        "event_type": "memory.validated",
                        "status": "completed",
                        "message": "已保存一条通过统计审查的分析经验。",
                        "payload": {
                            "memory_id": str(experience["memory_id"]),
                            "memory_kind": "episodic",
                        },
                    }
                )
                validated_reuses = repository.record_validated_reuse(
                    run_id=claimed["run_id"]
                )
                if validated_reuses:
                    events.append(
                        {
                            "event_type": "memory.validated",
                            "status": "completed",
                            "message": "本轮采用的长期记忆已通过结果验证。",
                            "payload": {"usage_count": validated_reuses},
                        }
                    )
        for event in events:
            assistant_store.append_event(claimed["run_id"], **event)
        _merge_message_memory_metadata(
            assistant_store,
            claimed["assistant_message_id"],
            events,
        )
        repository.finish_maintenance_job(job_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:1_000]
        logger.exception("Assistant memory maintenance failed for %s.", job_id)
        repository.finish_maintenance_job(job_id, error=error)
        try:
            assistant_store.append_event(
                claimed["run_id"],
                event_type="memory.maintenance_failed",
                status="warning",
                message="记忆维护暂时失败，不影响本次回答。",
                payload={"error": error},
            )
        except Exception:
            logger.exception("Unable to record memory maintenance failure.")
    finally:
        with _lock:
            _futures.pop(job_id, None)


def _model_memory_candidates(
    *, source_message: dict[str, object], source_message_id: UUID
) -> MemoryFormationResult:
    settings = get_settings()
    text = str(source_message.get("content") or "")
    deterministic = extract_memory_candidates(text)
    if (
        not settings.assistant_memory_model_extraction_enabled
        or not should_use_model_memory_extractor(text, deterministic)
    ):
        return MemoryFormationResult(deterministic)
    try:
        formed = LangMemFormationManager(settings).extract(
            text=text,
            source_message_id=str(source_message_id),
        )
        guarded = DataMindMemoryGuard().validate(
            formed,
            source_message=source_message,
            source_message_id=source_message_id,
        )
        candidates = tuple(_guarded_candidate(item) for item in guarded.accepted)
        return MemoryFormationResult(
            candidates or deterministic,
            tuple(sorted(set(guarded.rejected_codes))),
            "langmem",
        )
    except Exception:
        logger.warning(
            "LangMem memory formation failed; using deterministic extraction.",
            exc_info=True,
        )
        return MemoryFormationResult(
            deterministic,
            ("formation_unavailable",),
            "deterministic_fallback",
        )


def _guarded_candidate(item: GuardedMemoryRecord) -> MemoryCandidate:
    memory_type = item.memory_type
    entity_key = item.entity_key
    predicate = item.predicate
    explicit = item.explicit
    subject = f"{memory_type}:{entity_key}:{predicate}"[:160]
    evidence = item.evidence
    typed_value = dict(item.typed_value)
    return MemoryCandidate(
        memory_type=memory_type,
        normalized_key=subject,
        subject_key=subject,
        entity_key=entity_key,
        predicate=predicate,
        content=item.content,
        explicit=explicit,
        confidence=item.confidence,
        structured_value={"value": typed_value.get("value"), "evidence": evidence},
        typed_value=typed_value,
        unit=item.unit,
        correction=item.correction,
        application_policy=(
            "always"
            if explicit and memory_type in {"preference", "workflow_preference"}
            else "relevant"
        ),
    )


def recover_memory_maintenance_jobs(dataset_store_path: str) -> int:
    bootstrap = AssistantMemoryRepository(dataset_store_path, user_id="default")
    recovered = 0
    for job in bootstrap.list_all_recoverable_maintenance_jobs():
        try:
            _dispatch(job["job_id"], job["user_id"], dataset_store_path)
            recovered += 1
        except Exception:
            logger.exception("Unable to recover memory maintenance job %s.", job["job_id"])
    return recovered


def _merge_message_memory_metadata(
    repository: AssistantRepository,
    message_id: UUID,
    events: list[dict[str, object]],
) -> None:
    message = repository.get_message(message_id)
    metadata = dict(message.get("metadata") or {})
    updates: list[dict[str, object]] = []
    for event in events:
        raw_payload = event.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        updates.append(
            {
                "event_type": event["event_type"],
                "message": event["message"],
                **payload,
            }
        )
    metadata["memory_updates"] = updates
    repository.update_message(
        message_id,
        content=message["content"],
        status=message["status"],
        provider=message.get("provider"),
        model=message.get("model"),
        citations=tuple(message.get("citations") or ()),
        token_usage=dict(message.get("token_usage") or {}),
        metadata=metadata,
    )
