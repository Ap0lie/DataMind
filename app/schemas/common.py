from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str = Field(min_length=1)


class ReadinessResponse(ApiModel):
    status: str = Field(min_length=1)
    checks: dict[str, str] = Field(default_factory=dict)
