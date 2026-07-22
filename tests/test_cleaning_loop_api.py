from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.settings import get_settings
from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_async_cleaning_job_api_returns_traceable_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_CHECKPOINT_BACKEND", "none")
    monkeypatch.setenv("DATAMIND_EXECUTION_BACKEND", "local")
    get_settings.cache_clear()
    transport = ASGITransport(app=create_app(get_settings()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/store/datasets",
            json={"name": "中文订单.txt", "source_type": "txt", "source_metadata": {}},
        )
        assert created.status_code == 200
        dataset_id = created.json()["dataset_id"]
        raw = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/raw-records",
            json={"records": [{"客户": " 张三 ", "金额": "10"}, {"客户": " 张三 ", "金额": "10"}]},
        )
        assert raw.status_code == 200
        queued = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-jobs",
            json={"requirement": "去空格并去重", "cleaning_strategy": "rules"},
        )
        assert queued.status_code == 200, queued.text
        job_id = queued.json()["job_id"]
        job = queued.json()
        for _ in range(100):
            response = await client.get(
                f"/api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}"
            )
            assert response.status_code == 200
            job = response.json()
            if job["status"] not in {"queued", "running", "cancel_requested"}:
                break
            await asyncio.sleep(0.05)
        assert job["status"] == "completed", job
        assert job["selected_strategy"] == "rules"
        assert any(event.get("event_type") == "cleaning_commit" for event in job["events"])
        result = await client.get(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}/result"
        )
        assert result.status_code == 200
        assert result.json()["preview_records"] == [{"客户": "张三", "金额": 10}]
    get_settings.cache_clear()
