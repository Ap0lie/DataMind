from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Header, HTTPException, Request

from app.core.settings import get_settings
from app.storage.dataset_store import DatasetStoreRepository

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _first_header_value(value: str | None) -> str:
    return (value or "").split(",", maxsplit=1)[0].strip()


def _origin_matches_request(request: Request, origin: str) -> bool:
    try:
        parsed_origin = urlsplit(origin.rstrip("/"))
        forwarded_proto = _first_header_value(request.headers.get("x-forwarded-proto"))
        request_scheme = (forwarded_proto or request.url.scheme).lower()
        if parsed_origin.scheme.lower() != request_scheme or not parsed_origin.hostname:
            return False

        default_port = 443 if request_scheme == "https" else 80
        origin_port = parsed_origin.port or default_port
        authorities = {
            _first_header_value(request.headers.get("host")),
            _first_header_value(request.headers.get("x-forwarded-host")),
        }
        for authority in authorities - {""}:
            parsed_authority = urlsplit(f"//{authority}")
            if (
                parsed_authority.hostname
                and parsed_authority.hostname.lower() == parsed_origin.hostname.lower()
                and (parsed_authority.port or default_port) == origin_port
            ):
                return True
    except ValueError:
        return False
    return False


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
        request.method.upper() not in _SAFE_METHODS
        and origin
        and origin.rstrip("/") not in allowed_origins
        and not _origin_matches_request(request, origin)
    ):
        raise HTTPException(status_code=403, detail="Origin is not allowed.")
    try:
        session = DatasetStoreRepository(settings.dataset_store_path).validate_user_session(
            token,
            csrf_token=csrf_token,
            require_csrf=request.method.upper() not in _SAFE_METHODS,
            ttl_seconds=settings.session_ttl_seconds,
        )
    except RuntimeError as exc:
        if "CSRF" in str(exc):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "csrf_validation_failed",
                    "message": str(exc),
                },
            ) from exc
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return str(session["user_id"])
