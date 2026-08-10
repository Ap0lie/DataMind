from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from requests.exceptions import Timeout as RequestsTimeout

from app.api.v1 import health as health_api
from app.core.settings import Settings, get_settings
from app.harness.node import (
    NodeExecutionHarness,
    NodeExecutionTimeout,
    NodeHarnessPolicy,
    remaining_node_timeout,
)
from app.main import create_app
from app.mcp.bootstrap import build_mcp_runtime, reset_mcp_runtime
from app.python_runner.main import _execute_in_container, create_runner_app
from app.schemas.auth import LoginRequest
from app.storage.assistant_repository import AssistantRepository
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


def test_node_harness_exposes_and_enforces_shared_deadline() -> None:
    observed: list[float] = []

    def slow(_state: object) -> dict[str, bool]:
        observed.append(remaining_node_timeout(10) or 0)
        time.sleep(0.02)
        return {"ok": True}

    harness = NodeExecutionHarness(
        NodeHarnessPolicy(
            transient_retries=2,
            backoff_seconds=0,
            timeout_seconds=0.01,
        )
    )
    with pytest.raises(NodeExecutionTimeout, match="exceeded its deadline"):
        harness.wrap("slow", slow)({})

    assert 0 < observed[0] <= 0.01


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


def test_assistant_repository_initializes_schema_once_per_process_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "datasets"))
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    AssistantRepository._initialized_stores.clear()
    original = AssistantRepository._initialize
    calls = 0

    def counted_initialize(repository: AssistantRepository) -> None:
        nonlocal calls
        calls += 1
        original(repository)

    monkeypatch.setattr(AssistantRepository, "_initialize", counted_initialize)
    try:
        AssistantRepository(str(tmp_path / "datasets"), user_id="alice")
        AssistantRepository(str(tmp_path / "datasets"), user_id="bob")
    finally:
        AssistantRepository._initialized_stores.clear()
        get_settings.cache_clear()

    assert calls == 1


def test_assistant_run_lease_and_event_sequence_are_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "datasets"))
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    repository = AssistantRepository(str(tmp_path / "datasets"), user_id="alice")
    conversation = repository.create_conversation(
        title="Lease",
        scope_type="auto",
        scope_id=None,
    )
    user_message = repository.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="Analyze",
    )
    assistant_message = repository.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="streaming",
    )
    run = repository.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user_message["message_id"],
        assistant_message_id=assistant_message["message_id"],
    )

    first = repository.claim_run(run.id, worker_id="worker-a", lease_seconds=60)
    duplicate = repository.claim_run(run.id, worker_id="worker-b", lease_seconds=60)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: repository.append_event(
                    run.id,
                    event_type="tool.completed",
                    status="completed",
                    message=f"event-{index}",
                ),
                range(24),
            )
        )
    events = repository.list_events(run.id)

    assert first is not None and first.attempt_count == 1
    assert duplicate is None
    assert [event["sequence"] for event in events] == list(range(1, 26))
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


def test_analysis_job_lease_is_renewed_between_long_nodes() -> None:
    from app.analysis.jobs import _maintain_analysis_job_lease

    class OneHeartbeat:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _timeout: float) -> bool:
            self.calls += 1
            return self.calls > 1

    class Repository:
        def __init__(self) -> None:
            self.heartbeats = 0

        def get_analysis_job(self, _job_id: UUID) -> SimpleNamespace:
            return SimpleNamespace(status="running", lease_owner="worker-a")

        def heartbeat_analysis_job(
            self,
            _job_id: UUID,
            *,
            worker_id: str,
            lease_seconds: int,
        ) -> None:
            assert worker_id == "worker-a"
            assert lease_seconds == 60
            self.heartbeats += 1

    repository = Repository()
    _maintain_analysis_job_lease(
        repository=repository,  # type: ignore[arg-type]
        job_id=UUID(int=1),
        worker_id="worker-a",
        lease_seconds=60,
        stop=OneHeartbeat(),  # type: ignore[arg-type]
    )

    assert repository.heartbeats == 1


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
    assert rejected.json() == {
        "detail": {
            "code": "csrf_validation_failed",
            "message": "CSRF validation failed.",
        }
    }
    assert created.status_code == 200
    UUID(current.json()["user_id"])
    assert current.json()["display_name"] == "alice"
    assert logout.status_code == 204
    assert after_logout.status_code == 401
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_cookie_session_accepts_only_matching_reverse_proxy_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_AUTH_MODE", "session")
    monkeypatch.setenv("DATAMIND_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("DATAMIND_CORS_ORIGINS", "http://127.0.0.1:5173")
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    app = create_app()

    proxy_headers = {
        "Host": "datamind.example.test",
        "Origin": "https://datamind.example.test",
        "X-Forwarded-Proto": "https",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        login = await client.post(
            "/api/v1/auth/login",
            headers=proxy_headers,
            json={"username": "proxy-user", "password": "strong-password"},
        )
        csrf = login.json()["csrf_token"]
        accepted = await client.post(
            "/api/v1/store/datasets",
            headers={**proxy_headers, "X-CSRF-Token": csrf},
            json={"name": "sales.csv", "source_type": "csv", "source_metadata": {}},
        )
        rejected = await client.post(
            "/api/v1/store/datasets",
            headers={
                **proxy_headers,
                "Origin": "https://other.trycloudflare.com",
                "X-CSRF-Token": csrf,
            },
            json={"name": "blocked.csv", "source_type": "csv", "source_metadata": {}},
        )

    assert login.status_code == 200
    assert accepted.status_code == 200
    assert rejected.status_code == 403
    assert rejected.json()["detail"] == "Origin is not allowed."
    get_settings.cache_clear()


def test_login_names_do_not_collapse_into_the_same_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.delenv("DATAMIND_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path))

    email_user = repository.login_or_create_user(
        username="alice@example.com",
        password="secret-a",
    )
    underscore_user = repository.login_or_create_user(
        username="alice_example_com",
        password="secret-b",
    )
    repeated = repository.login_or_create_user(
        username="ALICE@EXAMPLE.COM",
        password="secret-a",
    )

    assert UUID(str(email_user["user_id"]))
    assert UUID(str(underscore_user["user_id"]))
    assert email_user["user_id"] != underscore_user["user_id"]
    assert repeated["user_id"] == email_user["user_id"]
    with pytest.raises(ValidationError):
        LoginRequest(username="   ", password="secret")
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


def test_python_runner_kills_and_removes_timed_out_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeContainer:
        status = "running"
        killed = False
        removed = False

        def start(self) -> None:
            return None

        def wait(self, *, timeout: float) -> dict[str, int]:
            assert timeout == 0.25
            raise RequestsTimeout("timed out")

        def reload(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True
            self.status = "exited"

        def remove(self, *, force: bool) -> None:
            assert force is True
            self.removed = True

    container = FakeContainer()
    client = SimpleNamespace(
        containers=SimpleNamespace(create=lambda *_args, **_kwargs: container),
        close=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "docker",
        SimpleNamespace(from_env=lambda: client),
    )
    monkeypatch.setenv("DATAMIND_PYTHON_RUNNER_TEMP_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_PYTHON_RUNNER_CONTAINER_TIMEOUT_SECONDS", "0.25")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="timed out in the container sandbox"):
        _execute_in_container(
            {
                "code": "def analyze(df):\n    return {}",
                "records": [],
                "execution_kind": "analysis",
            }
        )

    assert container.killed is True
    assert container.removed is True
    get_settings.cache_clear()


def test_production_requires_authenticated_python_runner() -> None:
    common = {
        "environment": "production",
        "execution_backend": "celery",
        "auth_mode": "session",
        "database_url": "postgresql+psycopg://user:pass@db/datamind",
        "session_cookie_secure": True,
    }
    with pytest.raises(ValidationError, match="PYTHON_RUNNER_URL"):
        Settings(**common)
    with pytest.raises(ValidationError, match="PYTHON_RUNNER_SHARED_SECRET"):
        Settings(**common, python_runner_url="http://python-runner:8020")
    settings = Settings(
        **common,
        python_runner_url="http://python-runner:8020",
        python_runner_shared_secret="runner-secret",
    )
    assert settings.python_runner_url == "http://python-runner:8020"


@pytest.mark.asyncio
async def test_readiness_returns_503_for_failed_critical_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        dataset_store_path=str(tmp_path),
        execution_backend="local",
        python_runner_url="http://runner.invalid",
        assistant_enabled=False,
    )
    app = create_app(settings)
    app.state.settings = settings
    app.state.mcp_runtime = SimpleNamespace(
        catalog=_async_value(SimpleNamespace(tools=(object(),)))
    )
    monkeypatch.setattr(
        health_api,
        "_python_runner_status",
        lambda _settings: "failed:unavailable",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["python_runner"] == "failed:unavailable"


def _async_value(value: object):
    async def resolve() -> object:
        return value

    return resolve
