from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.v1.deps import current_user_id
from app.core.settings import get_settings
from app.schemas.auth import CurrentUserResponse, LoginRequest, LoginResponse
from app.security.rate_limit import RateLimitExceeded, enforce_rate_limit
from app.storage.dataset_store import DatasetStoreRepository

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, response: Response, http_request: Request) -> LoginResponse:
    settings = get_settings()
    repository = DatasetStoreRepository(settings.dataset_store_path)
    try:
        client_host = http_request.client.host if http_request.client else "unknown"
        enforce_rate_limit(
            f"login:{client_host}:{request.username.strip().lower()}",
            limit=settings.login_rate_limit,
            window_seconds=60,
        )
        user = repository.login_or_create_user(
            username=request.username,
            password=request.password,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    session = repository.create_user_session(
        user_id=str(user["user_id"]),
        ttl_seconds=settings.session_ttl_seconds,
        absolute_ttl_seconds=settings.session_absolute_ttl_seconds,
    )
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session["token"],
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )
    return LoginResponse(
        user_id=str(user["user_id"]),
        display_name=str(user["display_name"]),
        created=bool(user["created"]),
        csrf_token=session["csrf_token"],
        expires_at=session["expires_at"],
    )


@router.get("/me", response_model=CurrentUserResponse)
def me(user_id: str = Depends(current_user_id)) -> CurrentUserResponse:
    user = DatasetStoreRepository(get_settings().dataset_store_path).get_user(user_id)
    return CurrentUserResponse(**user)


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    _user_id: str = Depends(current_user_id),
) -> Response:
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        DatasetStoreRepository(settings.dataset_store_path).revoke_user_session(token)
    response.delete_cookie(settings.session_cookie_name, path="/")
    response.status_code = 204
    return response
