from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import analysis, assistant, auth, dataset_store, health, mcp, tasks

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(dataset_store.router, prefix="/store", tags=["dataset-store"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(assistant.router, prefix="/assistant", tags=["assistant"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
