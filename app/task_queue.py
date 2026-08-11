from __future__ import annotations

from typing import Any

from app.core.settings import get_settings

try:
    from celery import Celery
except ImportError:  # pragma: no cover - optional in the local-only profile
    Celery = None  # type: ignore[assignment,misc]


def create_celery_app() -> Any:
    if Celery is None:
        return None
    settings = get_settings()
    app = Celery("datamind", broker=settings.redis_url, backend=settings.redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=("json",),
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        task_soft_time_limit=3600,
        task_time_limit=3660,
        beat_schedule={
            "purge-expired-assets-daily": {
                "task": "datamind.assets.purge_expired",
                "schedule": 24 * 60 * 60,
            },
        },
    )
    return app


celery_app = create_celery_app()


if celery_app is not None:

    @celery_app.task(name="datamind.analysis.run", bind=True)
    def run_analysis_job_task(
        self: Any,
        job_id: str,
        user_id: str,
        dataset_store_path: str,
    ) -> None:
        from uuid import UUID

        from app.analysis.jobs import run_analysis_job
        from app.observability import configure_observability

        configure_observability("datamind-worker")

        run_analysis_job(
            UUID(job_id),
            user_id,
            dataset_store_path,
            worker_id=str(self.request.id or "celery-worker"),
        )

    @celery_app.task(name="datamind.cleaning.run", bind=True)
    def run_cleaning_job_task(
        self: Any,
        job_id: str,
        user_id: str,
        dataset_store_path: str,
    ) -> None:
        from uuid import UUID

        from app.analysis.cleaning_jobs import run_cleaning_job
        from app.observability import configure_observability

        configure_observability("datamind-worker")
        run_cleaning_job(
            UUID(job_id), user_id, dataset_store_path,
            worker_id=str(self.request.id or "celery-worker"),
        )

    @celery_app.task(name="datamind.assistant.run", bind=True)
    def run_assistant_job_task(
        self: Any,
        run_id: str,
        user_id: str,
        dataset_store_path: str,
    ) -> None:
        from uuid import UUID

        from app.assistant.jobs import run_assistant_run
        from app.observability import configure_observability

        configure_observability("datamind-worker")
        run_assistant_run(
            UUID(run_id),
            user_id,
            dataset_store_path,
            worker_id=str(self.request.id or "celery-worker"),
        )

    @celery_app.task(name="datamind.assistant.memory.maintain", bind=True)
    def maintain_assistant_memory_task(
        self: Any,
        job_id: str,
        user_id: str,
        dataset_store_path: str,
    ) -> None:
        from uuid import UUID

        from app.assistant.memory_jobs import run_memory_maintenance
        from app.observability import configure_observability

        configure_observability("datamind-worker")
        run_memory_maintenance(
            UUID(job_id),
            user_id,
            dataset_store_path,
            worker_id=str(self.request.id or "celery-worker"),
        )

    @celery_app.task(name="datamind.assets.purge_expired")
    def purge_expired_assets_task() -> int:
        from app.main import _purge_expired_assets
        from app.observability import configure_observability

        configure_observability("datamind-worker")
        return _purge_expired_assets(get_settings())
