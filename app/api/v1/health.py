from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.settings import get_settings
from app.schemas.common import HealthResponse, ReadinessResponse
from app.semantic.embedding import get_semantic_embedding_provider
from app.storage.dataset_store import DatasetStoreRepository

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request) -> ReadinessResponse:
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        repository = DatasetStoreRepository(settings.dataset_store_path, user_id="default")
        checks["database"] = "ok" if repository.ping() else "failed"
    except Exception as exc:
        checks["database"] = f"failed:{type(exc).__name__}"

    try:
        runtime = request.app.state.mcp_runtime
        catalog = await runtime.catalog()
        checks["mcp_registry"] = "ok" if catalog.tools else "degraded"
    except Exception as exc:
        checks["mcp_registry"] = f"failed:{type(exc).__name__}"

    if settings.execution_backend.lower() == "celery":
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url, socket_timeout=1)
            checks["redis"] = "ok" if client.ping() else "failed"
        except Exception as exc:
            checks["redis"] = f"failed:{type(exc).__name__}"
    else:
        checks["worker"] = "local"

    embedding_status = get_semantic_embedding_provider(settings).status()
    checks["semantic_embedding"] = embedding_status
    if settings.semantic_embedding_required and embedding_status != "ready":
        checks["semantic_embedding"] = f"failed:{embedding_status}"

    if not settings.assistant_enabled:
        checks["assistant_model"] = "disabled"
    elif settings.assistant_llm_provider.lower() == "kimi" and not settings.kimi_api_key:
        checks["assistant_model"] = "not_configured"
    else:
        checks["assistant_model"] = "ready"

    ready = all(not value.startswith("failed") for value in checks.values())
    return ReadinessResponse(status="ok" if ready else "not_ready", checks=checks)
