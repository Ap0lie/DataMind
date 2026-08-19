from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.assistant.tools import ASSISTANT_READ_TOOLS
from app.assistant.workflow import _tool_message_content
from app.core.settings import Settings
from app.storage.dataset_store import DatasetStoreRepository
from app.storage.tool_result_repository import ToolResultRepository
from app.tool_results.artifacts import ToolArtifactTooLarge
from app.tool_results.context import build_tool_context_bundle
from app.tool_results.contracts import ToolResultEnvelope, ToolResultKind, ToolResultStatus
from app.tool_results.distiller import (
    SmallModelToolResultDistiller,
    ToolDistillationPolicy,
)
from app.tool_results.reducers import reduce_tool_result

pytestmark = pytest.mark.unit


def test_sql_reducer_preserves_exact_facts_schema_and_evidence() -> None:
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="execute_safe_sql",
        action_hash="action",
        payload={
            "sql": "SELECT state, SUM(amount) AS total FROM dataset GROUP BY state",
            "total_rows": 2_000,
            "rows": [
                {"state": f"S{index}", "total": index + 0.25, "evidence_id": "ev_sql"}
                for index in range(100)
            ],
            "validation_issues": ["Join grain requires review."],
        },
        evidence_ids=("ev_contract",),
    )

    summary = reduce_tool_result(envelope)

    assert summary.kind == ToolResultKind.SQL
    assert summary.row_count == 2_000
    assert summary.schema_fields == ("state", "total", "evidence_id")
    assert len(summary.preview) == 20
    assert summary.preview[0]["total"] == 0.25
    assert summary.evidence_ids == ("ev_contract", "ev_sql")
    assert summary.validation_issues == ("Join grain requires review.",)
    assert "rows_after_20" in summary.omitted_sections


def test_error_reducer_retains_failure_details() -> None:
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="execute_python_analysis",
        action_hash="failed-action",
        status=ToolResultStatus.FAILED,
        payload={
            "error": "ValueError: missing metric at line 17",
            "attempt": 2,
        },
    )

    summary = reduce_tool_result(envelope)

    assert summary.kind == ToolResultKind.ERROR
    assert summary.error == "ValueError: missing metric at line 17"
    assert any(fact.path == "attempt" and fact.value == 2 for fact in summary.canonical_facts)


def test_repository_archives_gzip_payload_idempotently(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
    )
    monkeypatch.setattr(
        "app.storage.tool_result_repository.get_settings",
        lambda: settings,
    )
    repository = ToolResultRepository(
        settings.dataset_store_path,
        user_id="alice",
    )
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="get_report",
        action_hash="read-report",
        payload={"report_id": str(uuid4()), "markdown": "结论" * 10_000},
        metadata={"retention_policy": "report_evidence"},
    )

    first = repository.archive_and_summarize(envelope)
    second = repository.archive_and_summarize(envelope)

    assert first.artifact_id == second.artifact_id
    assert first.original_size_bytes > first.context_size_bytes
    assert repository.load_payload(first.artifact_id) == envelope.payload
    assert repository.model_context(first.artifact_id)["summary"]["verified"] is True
    assert repository.get(first.artifact_id).expires_at is None
    stored_files = tuple((tmp_path / "tool-artifacts").rglob("*.json.gz"))
    assert len(stored_files) == 1


def test_failed_artifact_uses_short_ttl_and_is_user_scoped(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
        tool_artifact_failed_ttl_days=7,
    )
    monkeypatch.setattr(
        "app.storage.tool_result_repository.get_settings",
        lambda: settings,
    )
    alice = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="execute_python_analysis",
        action_hash="failure",
        status=ToolResultStatus.FAILED,
        payload={"error": "timeout"},
    )
    artifact = alice.archive(envelope)
    expiry = datetime.fromisoformat(str(artifact.expires_at))
    assert timedelta(days=6, hours=23) < expiry - datetime.now(UTC) <= timedelta(days=7)

    bob = ToolResultRepository(settings.dataset_store_path, user_id="bob")
    with pytest.raises(RuntimeError, match="not found"):
        bob.get(artifact.artifact_id)

    assert alice.purge_expired(at=datetime.now(UTC) + timedelta(days=8)) == 1
    with pytest.raises(RuntimeError, match="not found"):
        alice.get(artifact.artifact_id)


def test_repository_rejects_payload_above_hard_limit(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
    )
    monkeypatch.setattr(
        "app.storage.tool_result_repository.get_settings",
        lambda: settings,
    )
    repository = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    repository.max_bytes = 64

    with pytest.raises(ToolArtifactTooLarge):
        repository.archive(
            ToolResultEnvelope(
                run_id=uuid4(),
                tool_name="large_tool",
                action_hash="large",
                payload={"value": "x" * 1000},
            )
        )

    assert not tuple((tmp_path / "tool-artifacts").rglob("*.tmp"))


class _DistillationRouter:
    def __init__(self, *, unsupported_number: bool = False) -> None:
        self.unsupported_number = unsupported_number
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        payload = json.loads(kwargs["messages"][-1]["content"])
        if "chunks" in payload:
            chunks = []
            for item in payload["chunks"]:
                quote = str(item["content"])[10:80]
                chunks.append(
                    {
                        "chunk_index": item["chunk_index"],
                        "summary": (
                            "该分片声称包含 999999 条记录"
                            if self.unsupported_number
                            else "该分片包含与当前工具结果相关的内容"
                        ),
                        "source_quotes": [quote],
                    }
                )
            content = json.dumps({"chunks": chunks}, ensure_ascii=False)
        else:
            quote = payload["verified_chunk_summaries"][0]["source_quotes"][0]
            content = json.dumps(
                {
                    "headline": "工具结果已完成语义蒸馏",
                    "key_findings": ["保留了与当前问题相关的主要内容"],
                    "source_quotes": [quote],
                },
                ensure_ascii=False,
            )
        return SimpleNamespace(
            provider="mock",
            model="small-distiller",
            content=content,
            finish_reason="stop",
            token_usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def _model_distiller(router: _DistillationRouter) -> SmallModelToolResultDistiller:
    return SmallModelToolResultDistiller(
        router,
        ToolDistillationPolicy(
            provider="mock",
            model="small-distiller",
            min_source_chars=1,
            chunk_chars=2_000,
            max_chunks=4,
            batch_size=2,
            max_attempts=1,
        ),
    )


def test_small_model_map_reduce_keeps_deterministic_facts_and_quotes() -> None:
    router = _DistillationRouter()
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="get_report",
        action_hash="distill-report",
        payload={
            "report_id": str(uuid4()),
            "executive_summary": "已验证结论。" * 1_000,
            "evidence_ids": ["ev-report"],
        },
    )

    result = _model_distiller(router).distill(envelope, artifact_id=uuid4())

    assert result.summary.verified is True
    assert result.summary.deterministic is False
    assert result.summary.summary_version == 2
    assert result.summary.evidence_ids == ("ev-report",)
    assert result.chunks and all(item.verified for item in result.chunks)
    assert result.input_tokens > 0
    assert router.calls >= 2


def test_small_model_unsupported_number_falls_back_to_deterministic_summary() -> None:
    router = _DistillationRouter(unsupported_number=True)
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="get_report",
        action_hash="reject-hallucination",
        payload={"markdown": "原始报告内容。" * 1_000},
    )

    result = _model_distiller(router).distill(envelope, artifact_id=uuid4())

    assert result.summary.deterministic is True
    assert result.summary.verified is True
    assert "model_distillation_unavailable" in result.summary.warnings
    assert all(not item.verified for item in result.chunks)


def test_repository_persists_verified_chunk_summaries(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
    )
    monkeypatch.setattr("app.storage.tool_result_repository.get_settings", lambda: settings)
    repository = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="get_report",
        action_hash="persist-map-reduce",
        payload={"markdown": "可核验的报告正文。" * 1_000},
    )

    router = _DistillationRouter()
    distiller = _model_distiller(router)
    bundle = repository.archive_and_summarize(
        envelope,
        distiller=distiller,
    )
    calls = router.calls
    repeated = repository.archive_and_summarize(envelope, distiller=distiller)
    chunks = repository.list_chunk_summaries(bundle.artifact_id)

    assert chunks
    assert all(chunk.verified for chunk in chunks)
    assert repository.get_summary(bundle.artifact_id).summary_version == 2
    assert repeated.artifact_id == bundle.artifact_id
    assert router.calls == calls


def test_dynamic_tool_context_keeps_required_fields_within_limit() -> None:
    envelope = ToolResultEnvelope(
        run_id=uuid4(),
        tool_name="execute_safe_sql",
        action_hash="bounded-context",
        payload={
            "rows": [{"dimension": str(index), "metric": index} for index in range(100)],
            "evidence_ids": ["ev-context"],
        },
    )
    summary = reduce_tool_result(envelope).model_copy(
        update={"artifact_id": uuid4(), "verified": True}
    )
    artifact = SimpleNamespace(
        artifact_id=summary.artifact_id,
        size_bytes=1_000_000,
    )

    bundle = build_tool_context_bundle(artifact, summary, max_context_chars=2_000)

    assert bundle.context_size_bytes <= 2_000
    assert bundle.summary.evidence_ids == ("ev-context",)


def test_repository_projects_exact_late_value_with_run_scope_and_call_limit(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
        tool_continuation_max_calls=1,
        tool_continuation_max_chars=4_000,
        tool_continuation_scan_max_bytes=1_048_576,
    )
    monkeypatch.setattr("app.storage.tool_result_repository.get_settings", lambda: settings)
    repository = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    run_id = uuid4()
    bundle = repository.archive_and_summarize(
        ToolResultEnvelope(
            run_id=run_id,
            tool_name="execute_safe_sql",
            action_hash="late-value",
            payload={
                "rows": [
                    {"customer_state": f"state_{index}", "payment_value": index + 0.25}
                    for index in range(250)
                ],
                "evidence_ids": ["ev_late"],
            },
            evidence_ids=("ev_late",),
        )
    )

    projection = repository.project_context(
        bundle.artifact_id,
        run_id=run_id,
        query="customer_state state_249",
    )
    replay = repository.project_context(
        bundle.artifact_id,
        run_id=run_id,
        query="customer_state state_249",
    )

    assert projection.projection_id == replay.projection_id
    assert projection.context_size_bytes <= settings.tool_continuation_max_chars
    assert any("state_249" in item.text for item in projection.excerpts)
    assert projection.evidence_ids == ("ev_late",)
    assert repository.metrics()["projections"] == 1
    with pytest.raises(RuntimeError, match="limit reached"):
        repository.project_context(
            bundle.artifact_id,
            run_id=run_id,
            query="payment_value 200.25",
        )
    with pytest.raises(RuntimeError, match="outside the current run"):
        repository.project_context(
            bundle.artifact_id,
            run_id=uuid4(),
            query="state_249",
        )


def test_continuation_stops_decompressed_reads_at_scan_limit(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
        tool_continuation_max_chars=2_000,
        tool_continuation_scan_max_bytes=1_048_576,
    )
    monkeypatch.setattr("app.storage.tool_result_repository.get_settings", lambda: settings)
    repository = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    run_id = uuid4()
    bundle = repository.archive_and_summarize(
        ToolResultEnvelope(
            run_id=run_id,
            tool_name="execute_safe_sql",
            action_hash="bounded-scan",
            payload={
                "rows": [
                    {"label": f"ordinary_{index}", "text": "x" * 800}
                    for index in range(2_000)
                ]
                + [{"label": "needle_after_limit", "text": "target"}],
            },
        )
    )

    projection = repository.project_context(
        bundle.artifact_id,
        run_id=run_id,
        query="needle_after_limit",
    )

    assert projection.scanned_bytes <= settings.tool_continuation_scan_max_bytes
    assert projection.truncated is True
    assert projection.more_available is True
    assert all("needle_after_limit" not in item.text for item in projection.excerpts)


def test_tool_projection_is_exposed_as_bounded_read_tool() -> None:
    names = {item["function"]["name"] for item in ASSISTANT_READ_TOOLS}
    assert "inspect_tool_result" in names
    content = _tool_message_content(
        {
            "tool_result_artifact_id": str(uuid4()),
            "projection": {"excerpts": [{"text": "x" * 10_000}]},
            "continuation_available": True,
        },
        max_chars=1_000,
    )
    parsed = json.loads(content)
    assert parsed["context_truncated"] is True
    assert len(content) <= 1_000


def test_repository_removes_unreferenced_artifact_file(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
    )
    monkeypatch.setattr("app.storage.tool_result_repository.get_settings", lambda: settings)
    repository = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    orphan = repository.artifact_root / "ff" / "orphan.json.gz"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    old = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    os.utime(orphan, (old, old))

    assert repository.purge_orphan_files() == 1
    assert not orphan.exists()


def test_report_evidence_follows_report_lifecycle(tmp_path, monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        dataset_store_path=str(tmp_path / "datasets"),
        tool_artifact_path=str(tmp_path / "tool-artifacts"),
    )
    monkeypatch.setattr("app.storage.tool_result_repository.get_settings", lambda: settings)
    store = DatasetStoreRepository(settings.dataset_store_path, user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="Validated report",
        markdown="# Report",
        metadata={"question": "total sales"},
    )
    repository = ToolResultRepository(settings.dataset_store_path, user_id="alice")
    bundle = repository.archive_and_summarize(
        ToolResultEnvelope(
            run_id=uuid4(),
            tool_name="execute_safe_sql",
            action_hash="report-evidence",
            payload={"rows": [{"total": 42}]},
        )
    )

    assert repository.retain_for_report(
        (bundle.artifact_id,), report_id=report_id
    ) == 1
    retained = repository.get(bundle.artifact_id)
    assert retained.expires_at is None
    assert retained.metadata["report_id"] == str(report_id)

    with store._connect() as connection:
        connection.execute(
            "DELETE FROM reports WHERE id=? AND user_id=?",
            (str(report_id), "alice"),
        )
    assert repository.purge_expired() == 1
    with pytest.raises(RuntimeError, match="not found"):
        repository.get(bundle.artifact_id)
