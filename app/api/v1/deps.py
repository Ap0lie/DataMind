from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException, Request

from app.core.settings import get_settings
from app.storage.dataset_store import DatasetStoreRepository


def current_user_id(
    request: Request,
    x_datamind_user: Annotated[str | None, Header()] = None,
) -> str:
    settings = get_settings()
    if settings.auth_mode.lower() == "legacy":
        return x_datamind_user or "default"

    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    csrf_token = request.headers.get(settings.csrf_header_name)
    origin = request.headers.get("origin")
    allowed_origins = {
        item.strip().rstrip("/")
        for item in settings.cors_origins.split(",")
        if item.strip()
    }
    if (
        request.method.upper() not in {"GET", "HEAD", "OPTIONS"}
        and origin
        and origin.rstrip("/") not in allowed_origins
    ):
        raise HTTPException(status_code=403, detail="Origin is not allowed.")
    try:
        session = DatasetStoreRepository(settings.dataset_store_path).validate_user_session(
            token,
            csrf_token=csrf_token,
            require_csrf=request.method.upper() not in {"GET", "HEAD", "OPTIONS"},
            ttl_seconds=settings.session_ttl_seconds,
        )
    except RuntimeError as exc:
        status_code = 403 if "CSRF" in str(exc) else 401
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return str(session["user_id"])
