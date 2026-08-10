from __future__ import annotations

from pydantic import Field, field_validator

from app.schemas.common import ApiModel


class LoginRequest(ApiModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=160)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Username cannot be blank.")
        return normalized


class LoginResponse(ApiModel):
    user_id: str
    display_name: str
    created: bool = False
    csrf_token: str | None = None
    expires_at: str | None = None


class CurrentUserResponse(ApiModel):
    user_id: str
    display_name: str
    expires_at: str | None = None
