from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio
async def test_create_task_runs_demo_workflow() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/tasks",
            json={
                "tenant_id": "demo",
                "user_id": "system",
                "prompt": "Analyze regional sales performance.",
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["workflow_status"] == "completed"
    assert payload["plan"]["objective"] == (
        "Analyze regional sales performance and unusual revenue patterns."
    )
    assert "销售表现分析" in payload["report_markdown"]
