from __future__ import annotations

import json
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import APIRouter, Request, Response, status

from app.core.settings import get_settings
from app.schemas.common import HealthResponse, ReadinessResponse
from app.semantic.embedding import get_semantic_embedding_provider
from app.storage.dataset_store import DatasetStoreRepository

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok", version=settings.app_version, build_sha=settings.build_sha
    )


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok", version=settings.app_version, build_sha=settings.build_sha
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    settings = getattr(request.app.state, "settings", None) or get_settings()
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
        checks["worker"] = _celery_worker_status()
    else:
        checks["worker"] = "local"

    checks["python_runner"] = _python_runner_status(settings)

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
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if ready else "not_ready",
        checks=checks,
        version=settings.app_version,
        build_sha=settings.build_sha,
    )


def _celery_worker_status() -> str:
    try:
        from app.task_queue import celery_app

        if celery_app is None:
            return "failed:unavailable"
        replies = celery_app.control.inspect(timeout=1.5).ping() or {}
        return "ok" if any(
            isinstance(value, dict) and value.get("ok") == "pong"
            for value in replies.values()
        ) else "failed:no_heartbeat"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"


def _python_runner_status(settings: object) -> str:
    runner_url = str(getattr(settings, "python_runner_url", "") or "").rstrip("/")
    if not runner_url:
        return (
            "failed:not_configured"
            if str(getattr(settings, "environment", "")).lower() == "production"
            else "local"
        )
    try:
        request = UrlRequest(f"{runner_url}/health", method="GET")
        with urlopen(request, timeout=1.5) as runner_response:
            payload = json.loads(runner_response.read(4096).decode("utf-8"))
        return "ok" if payload.get("status") == "ok" else "failed:unhealthy"
    except Exception as exc:
        return f"failed:{type(exc).__name__}"
