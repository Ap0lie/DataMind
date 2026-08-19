from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.settings import Settings, get_settings
from app.mcp.bootstrap import build_mcp_runtime
from app.observability import configure_observability
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.dataset_store import DatasetStoreRepository
from app.storage.tool_result_repository import ToolResultRepository

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_observability("datamind-api", resolved_settings)
        app.state.settings = resolved_settings
        app.state.mcp_runtime = await build_mcp_runtime(resolved_settings)
        recovery_task: asyncio.Task[None] | None = None
        recycle_task: asyncio.Task[None] | None = None
        try:
            await asyncio.to_thread(_purge_expired_assets, resolved_settings)
        except Exception:
            logger.exception("Expired asset cleanup will be retried after startup.")
        try:
            from app.assistant.memory_jobs import recover_memory_maintenance_jobs

            recover_memory_maintenance_jobs(resolved_settings.dataset_store_path)
        except Exception:
            logger.exception("Assistant memory maintenance recovery will be retried.")
        recycle_task = asyncio.create_task(
            _asset_recycle_loop(resolved_settings),
            name="datamind-asset-recycle",
        )
        if resolved_settings.execution_backend.lower() == "celery":
            try:
                from app.analysis.cleaning_jobs import recover_queued_cleaning_jobs
                from app.analysis.jobs import recover_queued_analysis_jobs
                from app.assistant.jobs import recover_queued_assistant_runs

                recover_queued_analysis_jobs(resolved_settings.dataset_store_path)
                recover_queued_cleaning_jobs(resolved_settings.dataset_store_path)
                recover_queued_assistant_runs(resolved_settings.dataset_store_path)
            except Exception:
                logger.exception("Queued analysis job recovery will be retried after startup.")
            recovery_task = asyncio.create_task(
                _job_recovery_loop(resolved_settings),
                name="datamind-job-recovery",
            )
        try:
            yield
        finally:
            if recovery_task is not None:
                recovery_task.cancel()
                with suppress(asyncio.CancelledError):
                    await recovery_task
            if recycle_task is not None:
                recycle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await recycle_task

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=tuple(
            item.strip().rstrip("/")
            for item in resolved_settings.cors_origins.split(",")
            if item.strip()
        ),
        allow_origin_regex=r"^http://(127\.0\.0\.1|localhost):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if resolved_settings.execution_backend.lower() == "local":
        repository = DatasetStoreRepository(
            resolved_settings.dataset_store_path,
            user_id="default",
        )
        repository.mark_interrupted_analysis_jobs()
        repository.mark_interrupted_cleaning_jobs()
    app.include_router(api_router, prefix=resolved_settings.api_prefix)
    return app


app = create_app()


async def _job_recovery_loop(settings: Settings) -> None:
    from app.analysis.cleaning_jobs import recover_queued_cleaning_jobs
    from app.analysis.jobs import recover_queued_analysis_jobs
    from app.assistant.jobs import recover_queued_assistant_runs
    from app.assistant.memory_jobs import recover_memory_maintenance_jobs

    while True:
        await asyncio.sleep(max(30, settings.worker_lease_seconds // 2))
        try:
            await asyncio.to_thread(
                recover_queued_analysis_jobs,
                settings.dataset_store_path,
            )
            await asyncio.to_thread(
                recover_queued_cleaning_jobs,
                settings.dataset_store_path,
            )
            await asyncio.to_thread(
                recover_queued_assistant_runs,
                settings.dataset_store_path,
            )
            await asyncio.to_thread(
                recover_memory_maintenance_jobs,
                settings.dataset_store_path,
            )
        except Exception:
            logger.exception("Periodic analysis job recovery failed.")


def _purge_expired_assets(settings: Settings) -> int:
    bootstrap = DatasetStoreRepository(settings.dataset_store_path, user_id="default")
    memory_bootstrap = AssistantMemoryRepository(
        settings.dataset_store_path,
        user_id="default",
    )
    tool_bootstrap = ToolResultRepository(
        settings.dataset_store_path,
        user_id="default",
    )
    user_ids = (
        set(bootstrap.list_asset_user_ids())
        | set(memory_bootstrap.list_user_ids())
        | set(tool_bootstrap.list_user_ids())
    )
    total = 0
    for user_id in user_ids:
        total += DatasetStoreRepository(
            settings.dataset_store_path,
            user_id=user_id,
        ).purge_expired_assets()
        memory_repository = AssistantMemoryRepository(
            settings.dataset_store_path,
            user_id=user_id,
        )
        total += memory_repository.recycle_stale(
            active_days=settings.assistant_memory_ttl_days,
            pending_days=settings.assistant_memory_recycle_days,
            retention_days=settings.assistant_memory_recycle_days,
        )
        total += memory_repository.purge_expired()
        total += ToolResultRepository(
            settings.dataset_store_path,
            user_id=user_id,
        ).purge_expired()
    orphan_files = tool_bootstrap.purge_orphan_files()
    total += orphan_files
    logger.info(
        "asset_cleanup.completed total=%s tool_orphan_files=%s",
        total,
        orphan_files,
    )
    return total


async def _asset_recycle_loop(settings: Settings) -> None:
    while True:
        await asyncio.sleep(24 * 60 * 60)
        try:
            await asyncio.to_thread(_purge_expired_assets, settings)
        except Exception:
            logger.exception("Periodic expired asset cleanup failed.")
