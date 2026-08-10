from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.settings import get_settings
from app.main import create_app
from app.storage.dataset_store import DatasetStoreRepository

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_semantic_model_api_publish_plan_and_low_confidence_gate(tmp_path, monkeypatch) -> None:
    store = tmp_path / "store"
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(store))
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(store), user_id="default")
    dataset = repository.create_dataset(name="orders", source_type="csv", source_metadata={})
    repository.append_raw_records(dataset_id=dataset.id, records=[{"region": "N", "revenue": 12}])
    repository.save_column_metadata(
        dataset_id=dataset.id,
        columns=[
            {"column_name": "region", "role": "dimension", "inferred_type": "text"},
            {"column_name": "revenue", "role": "metric", "inferred_type": "number"},
        ],
    )
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        draft_response = await client.post(
            "/api/v1/store/semantic-models/drafts",
            json={"scope_type": "dataset", "scope_id": str(dataset.id), "name": "Sales"},
        )
        assert draft_response.status_code == 200
        draft = draft_response.json()
        validate_response = await client.post(f"/api/v1/store/semantic-models/{draft['model_id']}/validate", json={})
        assert validate_response.json()["valid"]
        publish_response = await client.post(f"/api/v1/store/semantic-models/{draft['model_id']}/publish", json={})
        assert publish_response.status_code == 200

        plan_response = await client.post(
            "/api/v1/analysis/plans",
            json={"dataset_id": str(dataset.id), "question": "分析 completely_unknown_concept"},
        )
        assert plan_response.status_code == 200
        plan = plan_response.json()
        assert plan["confidence_level"] == "low"
        blocked = await client.post(
            "/api/v1/analysis/jobs",
            json={"dataset_id": str(dataset.id), "question": "分析 completely_unknown_concept", "planner_decision_id": plan["decision_id"]},
        )
        assert blocked.status_code == 400
        assert "requires confirmation" in blocked.json()["detail"]
    get_settings.cache_clear()


async def test_data_drift_api_exposes_latest_event_and_history(
    tmp_path,
    monkeypatch,
) -> None:
    store = tmp_path / "store"
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(store))
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(store), user_id="default")
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"amount": 10}, {"amount": 20}],
    )
    repository.replace_raw_record_batches(
        dataset_id=dataset.id,
        batches=iter(([{"sales_amount": 10}, {"sales_amount": 20}],)),
    )

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        latest = await client.get(
            f"/api/v1/store/datasets/{dataset.id}/drift"
        )
        history = await client.get(
            f"/api/v1/store/datasets/{dataset.id}/drift/history"
        )

    assert latest.status_code == 200
    assert latest.json()["status"] == "critical"
    assert latest.json()["event_id"]
    assert history.status_code == 200
    assert len(history.json()["events"]) == 1
    get_settings.cache_clear()
