from __future__ import annotations

import json
import sqlite3
from datetime import UTC
from uuid import uuid4

from app.evaluation.benchmarking import (
    BenchmarkObservation,
    BenchmarkRunner,
    calibrate_runs,
    compare_runs,
    load_run,
    write_run_artifacts,
)
from app.evaluation.corpus import corpus_checksum, dirty_customer_records
from app.evaluation.history import aggregate_sqlite_history
from app.evaluation.models import (
    BenchmarkCase,
    BenchmarkEnvironment,
    BenchmarkMetricStatus,
    BenchmarkSuiteManifest,
    BenchmarkThreshold,
    BenchmarkThresholdDirection,
)
from app.evaluation.operations import optional_executors, release_executors
from scripts.production_smoke import _parse_datetime, _percentile


def _environment() -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        environment_id="test-local",
        python_version="3.12",
        platform="test",
        settings_fingerprint="settings-v1",
    )


def _manifest(*, minimum: float = 1.0) -> BenchmarkSuiteManifest:
    return BenchmarkSuiteManifest(
        suite_id="unit-benchmark",
        version="1.0.0",
        mode="deterministic",
        corpus_checksum="corpus-v1",
        cases=(
            BenchmarkCase(
                case_id="exact",
                name="Exact validator",
                subsystem="unit",
                operation="observe",
                expected={"value": 7},
            ),
        ),
        thresholds=(
            BenchmarkThreshold(
                metric="task_success_rate",
                direction=BenchmarkThresholdDirection.MINIMUM,
                value=minimum,
            ),
        ),
    )


def test_benchmark_runner_validates_cases_and_marks_missing_token_metric() -> None:
    run = BenchmarkRunner(
        {"observe": lambda _case: BenchmarkObservation(actual={"value": 7})}
    ).run(_manifest(), environment=_environment())

    assert run.passed
    assert run.aggregate_metrics["task_success_rate"] == 1.0
    assert run.aggregate_metrics["total_tokens"] is None
    assert run.metric_status["total_tokens"] == BenchmarkMetricStatus.UNAVAILABLE


def test_benchmark_hard_gate_fails_when_metric_is_unavailable() -> None:
    manifest = _manifest().model_copy(
        update={
            "thresholds": (
                BenchmarkThreshold(
                    metric="total_tokens",
                    direction=BenchmarkThresholdDirection.MAXIMUM,
                    value=100,
                ),
            )
        }
    )
    run = BenchmarkRunner(
        {"observe": lambda _case: BenchmarkObservation(actual={"value": 7})}
    ).run(manifest, environment=_environment())

    assert not run.passed
    assert run.hard_gate_failures == ("total_tokens: metric_unavailable",)


def test_benchmark_duration_budget_never_passes_with_skipped_cases(monkeypatch) -> None:
    manifest = _manifest().model_copy(
        update={
            "max_duration_seconds": 1.0,
            "cases": (
                *_manifest().cases,
                BenchmarkCase(
                    case_id="second",
                    name="Second case",
                    subsystem="unit",
                    operation="observe",
                    expected={"value": 7},
                ),
            ),
        }
    )
    monotonic_values = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(
        "app.evaluation.benchmarking.time.monotonic",
        lambda: next(monotonic_values),
    )

    run = BenchmarkRunner(
        {"observe": lambda _case: BenchmarkObservation(actual={"value": 7})}
    ).run(manifest, environment=_environment())

    assert not run.passed
    assert run.cases[0].status == "passed"
    assert run.cases[1].status == "skipped"
    assert "suite_duration_budget_exhausted" in run.hard_gate_failures


def test_benchmark_artifacts_round_trip_and_comparison_detects_regression(tmp_path) -> None:
    baseline = BenchmarkRunner(
        {"observe": lambda _case: BenchmarkObservation(actual={"value": 7})}
    ).run(_manifest(), environment=_environment())
    output = tmp_path / "run"
    write_run_artifacts(baseline, output)
    restored = load_run(output / "benchmark-run.json")
    assert restored.run_id == baseline.run_id
    assert (output / "benchmark-report.md").is_file()
    assert (output / "benchmark-cases.jsonl").is_file()
    assert (output / "junit.xml").is_file()

    candidate = baseline.model_copy(
        update={
            "run_id": uuid4(),
            "aggregate_metrics": baseline.aggregate_metrics | {"task_success_rate": 0.8},
        }
    )
    comparison = compare_runs(baseline, candidate)
    assert comparison.compatible
    assert not comparison.passed
    assert any("task_success_rate" in item for item in comparison.regressions)


def test_calibration_requires_five_compatible_passing_runs() -> None:
    runs = tuple(
        BenchmarkRunner(
            {"observe": lambda _case: BenchmarkObservation(actual={"value": 7})}
        ).run(_manifest(), environment=_environment())
        for _index in range(5)
    )

    baseline = calibrate_runs(runs)

    assert baseline.repeats == 5
    assert baseline.aggregate_metrics["task_success_rate"] == 1.0


def test_frozen_corpus_is_deterministic() -> None:
    first = dirty_customer_records()
    second = dirty_customer_records()
    assert first == second
    assert len(corpus_checksum()) == 64


def test_report_benchmark_rejects_numeric_claim_that_disagrees_with_evidence() -> None:
    manifest = BenchmarkSuiteManifest(
        suite_id="report-evidence",
        version="1",
        mode="deterministic",
        corpus_checksum="corpus-v1",
        cases=(
            BenchmarkCase(
                case_id="wrong-number",
                name="Wrong number",
                subsystem="report",
                operation="report.evidence",
                input={
                    "evidence": [
                        {"evidence_id": "ev_1", "values": {"revenue": 300}}
                    ],
                    "findings": [
                        {
                            "content": "Revenue is 999.",
                            "evidence_ids": ["ev_1"],
                            "claims": [{"name": "revenue", "value": 999}],
                        }
                    ],
                },
                validators=(
                    {"type": "exact", "path": "numeric_accuracy", "expected": 1.0},
                ),
            ),
        ),
    )

    run = BenchmarkRunner(release_executors()).run(
        manifest, environment=_environment()
    )

    assert not run.passed
    assert run.cases[0].metrics["report_numeric_accuracy"] == 0.0


def test_history_aggregation_reads_only_status_and_timing_fields(tmp_path) -> None:
    database = tmp_path / "history.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE analysis_jobs (
            status TEXT, question TEXT, created_at TEXT, started_at TEXT, completed_at TEXT
        );
        CREATE TABLE analysis_job_events (
            duration_ms REAL, token_usage TEXT, status TEXT, event_type TEXT, payload TEXT
        );
        CREATE TABLE assistant_runs (
            status TEXT, created_at TEXT, completed_at TEXT
        );
        CREATE TABLE assistant_run_events (
            event_type TEXT, payload TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO analysis_jobs VALUES (?,?,?,?,?)",
        ("completed", "private question", "2026-01-01T00:00:00", "2026-01-01T00:00:01", "2026-01-01T00:00:03"),
    )
    connection.execute(
        "INSERT INTO analysis_job_events VALUES (?,?,?,?,?)",
        (20.0, json.dumps({"total_tokens": 8}), "completed", "node", "private payload"),
    )
    connection.execute(
        "INSERT INTO assistant_runs VALUES (?,?,?)",
        ("completed", "2026-01-01T00:00:00", "2026-01-01T00:00:02"),
    )
    connection.execute(
        "INSERT INTO assistant_run_events VALUES (?,?)",
        (
            "message.completed",
            json.dumps(
                {
                    "latency": {
                        "retrieval_ms": 40,
                        "tool_routing_ms": 0,
                        "model_first_token_ms": 300,
                        "first_answer_ms": 380,
                        "total_ms": 900,
                        "fast_path": True,
                    },
                    "token_usage": {"total_tokens": 12},
                }
            ),
        ),
    )
    connection.commit()
    connection.close()

    summary = aggregate_sqlite_history(database)

    encoded = json.dumps(summary)
    assert summary["analysis_jobs"]["statuses"] == {"completed": 1}
    assert summary["analysis_events"]["token_usage_status"] == "available"
    assert summary["assistant_events"]["fast_path_rate"] == 1.0
    assert summary["assistant_events"]["latency_ms"]["first_answer_ms"]["median"] == 380
    assert summary["assistant_events"]["total_tokens"] == 12
    assert "private question" not in encoded
    assert "private payload" not in encoded


def test_production_smoke_time_helpers_handle_utc_and_nearest_rank() -> None:
    parsed = _parse_datetime("2026-07-16T10:20:30Z")

    assert parsed.tzinfo == UTC
    assert _percentile([50.0, 10.0, 40.0, 20.0, 30.0], 0.95) == 50.0


def test_production_smoke_metrics_preserve_unavailable_values(tmp_path) -> None:
    artifact = tmp_path / "production-smoke.json"
    artifact.write_text(
        json.dumps(
            {
                "success": True,
                "sse_event_count": 3,
                "sse_delivery_p95_ms": None,
                "analysis_duration_seconds": None,
            }
        ),
        encoding="utf-8",
    )
    manifest = BenchmarkSuiteManifest(
        suite_id="production-smoke-metrics",
        version="1",
        mode="resilience",
        corpus_checksum="corpus-v1",
        cases=(
            BenchmarkCase(
                case_id="smoke",
                name="Smoke metrics",
                subsystem="resilience",
                operation="production.smoke_metrics",
                input={"path": str(artifact)},
                validators=(
                    {"type": "truthy", "path": "success"},
                    {"type": "minimum", "path": "sse_event_count", "expected": 1},
                ),
            ),
        ),
    )

    run = BenchmarkRunner(
        release_executors() | optional_executors(backend="local")
    ).run(manifest, environment=_environment())

    assert run.cases[0].passed
    assert run.aggregate_metrics["sse_delivery_p95_ms"] is None
    assert run.metric_status["sse_delivery_p95_ms"] == BenchmarkMetricStatus.UNAVAILABLE
    assert run.aggregate_metrics["production_analysis_duration_seconds"] is None
