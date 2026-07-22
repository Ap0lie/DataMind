from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.tasks import TaskCreateRequest


def test_task_create_request_accepts_minimal_valid_payload() -> None:
    request = TaskCreateRequest(tenant_id="tenant-a", user_id="user-a", prompt="Monitor MCP news")

    assert request.locale == "en-US"


def test_task_create_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        TaskCreateRequest(tenant_id="tenant-a", user_id="user-a", prompt="")
