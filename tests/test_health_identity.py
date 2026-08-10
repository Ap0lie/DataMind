from __future__ import annotations

import pytest

from app.api.v1 import health as health_api
from app.core.settings import Settings


@pytest.mark.asyncio
async def test_health_endpoints_expose_baked_build_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(app_version="1.2.3", build_sha="build-abc123", environment="test")
    monkeypatch.setattr(health_api, "get_settings", lambda: settings)

    health = await health_api.health_check()
    live = await health_api.liveness()

    assert health.model_dump() == {
        "status": "ok",
        "version": "1.2.3",
        "build_sha": "build-abc123",
    }
    assert live.model_dump() == health.model_dump()
