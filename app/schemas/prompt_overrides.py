from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from app.schemas.common import ApiModel


class AgentPromptOverrides(ApiModel):
    """User-authored task preferences injected below immutable system prompts."""

    all: str | None = Field(default=None, max_length=4000)
    cleaning: str | None = Field(default=None, max_length=4000)
    planner: str | None = Field(default=None, max_length=4000)
    sql: str | None = Field(default=None, max_length=4000)
    python: str | None = Field(default=None, max_length=4000)
    visualization: str | None = Field(default=None, max_length=4000)
    review: str | None = Field(default=None, max_length=4000)
    report: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_total_size(self) -> AgentPromptOverrides:
        if sum(len(value) for value in self.as_dict().values()) > 12_000:
            raise ValueError("Prompt overrides exceed the 12000 character task limit.")
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            name: value.strip()
            for name in (
                "all",
                "cleaning",
                "planner",
                "sql",
                "python",
                "visualization",
                "review",
                "report",
            )
            if (value := getattr(self, name)) and value.strip()
        }

    @classmethod
    def from_value(cls, value: Any) -> AgentPromptOverrides:
        if isinstance(value, cls):
            return value
        return cls.model_validate(value if isinstance(value, dict) else {})
