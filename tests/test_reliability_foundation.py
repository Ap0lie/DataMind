from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.settings import Settings, get_settings
from app.harness.node import NodeExecutionHarness, NodeHarnessPolicy
from app.main import create_app
from app.mcp.bootstrap import build_mcp_runtime, reset_mcp_runtime
from app.python_runner.main import create_runner_app
from app.storage.dataset_store import DatasetStoreRepository


def test_node_harness_retries_transient_errors_only() -> None:
    calls = 0

    def transient(_state: object) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("provider timed out")
        return {"ok": True}

    harness = NodeExecutionHarness(NodeHarnessPolicy(transient_retries=1, backoff_seconds=0))
    assert harness.wrap("test", transient)({}) == {"ok": True}
    assert calls == 2

    def semantic(_state: object) -> dict[str, bool]:
        raise ValueError("invalid generated plan")

    with pytest.raises(ValueError, match="invalid generated plan"):
        harness.wrap("semantic", semantic)({})


def test_analysis_job_events_are_ordered_and_persisted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    job = repository.create_analysis_job(dataset_id=dataset.id, question="Summarize sales")
    repository.update_analysis_job(
        job.id,
        status="running",
        progress=25,
        current_stage="planner",
        event_message="Planner started.",
    )
    repository.update_analysis_job(
        job.id,
        status="completed",
        progress=100,
        current_stage="complete",
        event_message="Complete.",
        completed=True,
    )
    repository.append_analysis_job_event(
        job.id,
        node="report_agent",
        status="completed",
        message="report_agent completed.",
        attempt=1,
        duration_ms=42.5,
        provider="mock",
        model="fake-router",
    )

    events = repository.list_analysis_job_events(job.id)
    stored = repository.get_analysis_job(job.id)
    assert [event["sequence"] for event in events] == [1, 2, 3, 4]
    assert events[-1]["duration_ms"] == 42.5
    assert events[-1]["provider"] == "mock"
    assert stored.events == events
    assert stored.checkpoint_thread_id == str(job.id)
    get_settings.cache_clear()


def test_analysis_job_database_lease_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    job = repository.create_analysis_job(dataset_id=dataset.id, question="Summarize sales")

    first = repository.claim_analysis_job(job.id, worker_id="worker-a", lease_seconds=60)
    duplicate = repository.claim_analysis_job(job.id, worker_id="worker-b", lease_seconds=60)
    first_report = repository.save_report(
        dataset_id=dataset.id,
        title="Report",
        markdown="first",
        metadata={},
        job_id=job.id,
    )
    duplicate_report = repository.save_report(
        dataset_id=dataset.id,
        title="Report",
        markdown="duplicate",
        metadata={},
        job_id=job.id,
    )

    assert first is not None
    assert first.attempt_count == 1
    assert duplicate is None
    assert duplicate_report == first_report
    assert len(repository.list_reports(dataset.id)) == 1
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_mcp_runtime_reuses_matching_configuration(tmp_path: Path) -> None:
    reset_mcp_runtime()
    settings = Settings(dataset_store_path=str(tmp_path), llm_provider="mock")
    first = await build_mcp_runtime(settings)
    second = await build_mcp_runtime(settings)
    changed = await build_mcp_runtime(
        Settings(dataset_store_path=str(tmp_path), llm_provider="mock", llm_max_tokens=1024)
    )
    assert first is second
    assert changed is not first
    reset_mcp_runtime()


@pytest.mark.asyncio
async def test_cookie_session_requires_csrf_and_ignores_spoofed_user_header(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_AUTH_MODE", "session")
    monkeypatch.setenv("DATAMIND_SESSION_COOKIE_SECURE", "false")
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        login = await client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "strong-password"},
        )
        csrf = login.json()["csrf_token"]
        rejected = await client.post(
            "/api/v1/store/datasets",
            json={"name": "sales.csv", "source_type": "csv", "source_metadata": {}},
        )
        created = await client.post(
            "/api/v1/store/datasets",
            headers={"X-CSRF-Token": csrf, "X-DataMind-User": "bob"},
            json={"name": "sales.csv", "source_type": "csv", "source_metadata": {}},
        )
        current = await client.get("/api/v1/auth/me")
        logout = await client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
        after_logout = await client.get("/api/v1/auth/me")

    assert login.status_code == 200
    assert "datamind_session" in login.cookies
    assert rejected.status_code == 403
    assert created.status_code == 200
    assert current.json()["user_id"] == "alice"
    assert logout.status_code == 204
    assert after_logout.status_code == 401
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_terminal_job_sse_stream_returns_ordered_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_AUTH_MODE", "legacy")
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    app = create_app()
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(name="events.csv", source_type="csv", source_metadata={})
    job = repository.create_analysis_job(dataset_id=dataset.id, question="Stream events")
    repository.request_analysis_job_cancel(job.id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/analysis/jobs/{job.id}/events",
            headers={"X-DataMind-User": "alice"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 1" in response.text
    assert "event: workflow" in response.text
    assert "event: end" in response.text
    get_settings.cache_clear()


def test_sqlalchemy_sqlite_repository_compatibility(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "sqlalchemy.db"
    monkeypatch.setenv("DATAMIND_DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "files"))
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path / "files"), user_id="alice")
    dataset = repository.create_dataset(name="data.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(dataset_id=dataset.id, records=[{"amount": 10}])
    assert repository.read_raw_records(dataset.id) == [{"amount": 10}]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_python_runner_validates_code_and_returns_container_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.python_runner.main._execute_in_container",
        lambda _payload: {"statistics": {"rows": 1}, "insights": [], "charts": []},
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_runner_app()),
        base_url="http://runner",
    ) as client:
        accepted = await client.post(
            "/execute",
            json={
                "code": "def analyze(df):\n    return {'statistics': {}, 'insights': [], 'charts': []}",
                "records": [{"value": 1}],
            },
        )
        rejected = await client.post(
            "/execute",
            json={"code": "import socket\ndef analyze(df):\n    return {}", "records": []},
        )
    assert accepted.status_code == 200
    assert accepted.json()["result"]["statistics"]["rows"] == 1
    assert rejected.status_code == 422
