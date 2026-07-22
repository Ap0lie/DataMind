from __future__ import annotations

from app.evaluation.benchmarking import BenchmarkRunner
from app.evaluation.models import BenchmarkEnvironment
from app.evaluation.operations import release_executors
from app.evaluation.suites import load_suite_manifest


def test_release_benchmark_passes_all_hard_gates() -> None:
    manifest = load_suite_manifest("release")
    environment = BenchmarkEnvironment(
        environment_id="pytest-release",
        python_version="3.12",
        platform="pytest",
        settings_fingerprint="release-v1",
    )

    run = BenchmarkRunner(release_executors()).run(
        manifest, environment=environment
    )

    assert run.passed, (run.hard_gate_failures, [(case.case_id, case.error) for case in run.cases])
