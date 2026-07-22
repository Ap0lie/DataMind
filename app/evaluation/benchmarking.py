from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol
from xml.etree.ElementTree import Element, ElementTree, SubElement

from app.evaluation.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkComparison,
    BenchmarkEnvironment,
    BenchmarkMetricStatus,
    BenchmarkRun,
    BenchmarkSuiteManifest,
    BenchmarkThresholdDirection,
)
from app.harness.models import TokenUsage


@dataclass(frozen=True)
class BenchmarkObservation:
    actual: dict[str, Any]
    metrics: dict[str, float | None] = field(default_factory=dict)
    token_usage: TokenUsage | None = None
    repair_count: int = 0
    fallback_count: int = 0
    tool_calls: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class BenchmarkExecutor(Protocol):
    def __call__(self, case: BenchmarkCase) -> BenchmarkObservation: ...


class BenchmarkRunner:
    def __init__(self, executors: Mapping[str, BenchmarkExecutor]) -> None:
        self._executors = dict(executors)

    def run(
        self,
        manifest: BenchmarkSuiteManifest,
        *,
        environment: BenchmarkEnvironment,
        repeats: int = 1,
    ) -> BenchmarkRun:
        started_at = datetime.now(UTC)
        deadline = (
            time.monotonic() + manifest.max_duration_seconds
            if manifest.max_duration_seconds is not None
            else None
        )
        results: list[BenchmarkCaseResult] = []
        used_tokens = 0
        budget_failures: set[str] = set()
        for _repeat in range(repeats):
            for case in manifest.cases:
                if deadline is not None and time.monotonic() >= deadline:
                    results.append(_skipped(case, "Suite duration budget exhausted."))
                    budget_failures.add("suite_duration_budget_exhausted")
                    continue
                if manifest.max_total_tokens is not None and used_tokens >= manifest.max_total_tokens:
                    results.append(_skipped(case, "Suite token budget exhausted."))
                    budget_failures.add("suite_token_budget_exhausted")
                    continue
                result = self._run_case(case)
                results.append(result)
                if result.token_usage is not None:
                    used_tokens += result.token_usage.total_tokens

        aggregate, metric_status = aggregate_case_results(tuple(results))
        failures = (
            *evaluate_thresholds(manifest, aggregate, metric_status),
            *sorted(budget_failures),
        )
        return BenchmarkRun(
            suite_id=manifest.suite_id,
            suite_version=manifest.version,
            corpus_checksum=manifest.corpus_checksum,
            environment=environment,
            repeats=repeats,
            cases=tuple(results),
            aggregate_metrics=aggregate,
            metric_status=metric_status,
            hard_gate_failures=failures,
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )

    def _run_case(self, case: BenchmarkCase) -> BenchmarkCaseResult:
        executor = self._executors.get(case.operation)
        if executor is None:
            return _error(case, 0.0, f"No benchmark executor registered for {case.operation}.")
        started = time.perf_counter()
        try:
            observation = executor(case)
            duration_ms = (time.perf_counter() - started) * 1000.0
            if duration_ms > case.timeout_seconds * 1000.0:
                return _error(case, duration_ms, "Case exceeded its duration budget.")
            failures = validate_observation(case, observation.actual)
            metrics = dict(observation.metrics)
            metrics.setdefault("case_success", 0.0 if failures else 1.0)
            metric_status = {
                name: (
                    BenchmarkMetricStatus.AVAILABLE
                    if value is not None
                    else BenchmarkMetricStatus.UNAVAILABLE
                )
                for name, value in metrics.items()
            }
            if observation.token_usage is None:
                metrics["total_tokens"] = None
                metric_status["total_tokens"] = BenchmarkMetricStatus.UNAVAILABLE
            else:
                metrics["total_tokens"] = float(observation.token_usage.total_tokens)
                metric_status["total_tokens"] = BenchmarkMetricStatus.AVAILABLE
            return BenchmarkCaseResult(
                case_id=case.case_id,
                subsystem=case.subsystem,
                status="failed" if failures else "passed",
                passed=not failures,
                duration_ms=duration_ms,
                metrics=metrics,
                metric_status=metric_status,
                token_usage=observation.token_usage,
                repair_count=observation.repair_count,
                fallback_count=observation.fallback_count,
                tool_calls=observation.tool_calls,
                error="; ".join(failures) if failures else None,
                details=observation.details,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            return _error(case, duration_ms, f"{type(exc).__name__}: {exc}")


def validate_observation(case: BenchmarkCase, actual: dict[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    validators = case.validators or tuple(
        {"type": "exact", "path": key, "expected": value}
        for key, value in case.expected.items()
    )
    for validator in validators:
        kind = str(validator.get("type") or "exact")
        path = str(validator.get("path") or "")
        try:
            value = _path_value(actual, path)
        except KeyError:
            failures.append(f"Missing observed value: {path}")
            continue
        expected = validator.get("expected")
        if kind == "exact" and value != expected:
            failures.append(f"{path}: expected {expected!r}, observed {value!r}")
        elif kind == "approx":
            tolerance = float(validator.get("tolerance") or 1e-6)
            if not math.isclose(float(value), float(expected), abs_tol=tolerance, rel_tol=tolerance):
                failures.append(f"{path}: expected approximately {expected!r}, observed {value!r}")
        elif kind == "contains" and expected not in value:
            failures.append(f"{path}: expected to contain {expected!r}")
        elif kind == "set_equal" and set(value) != set(expected or []):
            failures.append(f"{path}: expected set {expected!r}, observed {value!r}")
        elif kind == "truthy" and not value:
            failures.append(f"{path}: expected truthy value")
        elif kind == "falsey" and value:
            failures.append(f"{path}: expected falsey value")
        elif kind == "minimum" and float(value) < float(expected):
            failures.append(f"{path}: expected >= {expected!r}, observed {value!r}")
        elif kind == "maximum" and float(value) > float(expected):
            failures.append(f"{path}: expected <= {expected!r}, observed {value!r}")
    return tuple(failures)


def aggregate_case_results(
    results: tuple[BenchmarkCaseResult, ...],
) -> tuple[dict[str, float | None], dict[str, BenchmarkMetricStatus]]:
    executed = tuple(result for result in results if result.status != "skipped")
    latencies = [result.duration_ms for result in executed]
    aggregate: dict[str, float | None] = {
        "task_success_rate": (
            sum(1 for result in executed if result.passed) / len(executed) if executed else None
        ),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_tool_calls": mean(result.tool_calls for result in executed) if executed else None,
        "duplicate_successful_actions": float(
            sum(
                int(result.metrics.get("duplicate_successful_actions") or 0)
                for result in executed
            )
        ),
    }
    values: dict[str, list[float]] = defaultdict(list)
    unavailable: set[str] = set()
    for result in executed:
        for name, value in result.metrics.items():
            if value is None:
                unavailable.add(name)
            else:
                values[name].append(float(value))
    for name, items in values.items():
        aggregate[name] = mean(items)
    for name in unavailable:
        aggregate.setdefault(name, None)
    if executed and all(result.token_usage is not None for result in executed):
        aggregate["total_tokens"] = float(
            sum(result.token_usage.total_tokens for result in executed if result.token_usage)
        )
    else:
        aggregate["total_tokens"] = None
        unavailable.add("total_tokens")
    status = {
        name: (
            BenchmarkMetricStatus.UNAVAILABLE
            if value is None or name in unavailable
            else BenchmarkMetricStatus.AVAILABLE
        )
        for name, value in aggregate.items()
    }
    return aggregate, status


def evaluate_thresholds(
    manifest: BenchmarkSuiteManifest,
    metrics: Mapping[str, float | None],
    statuses: Mapping[str, BenchmarkMetricStatus],
) -> tuple[str, ...]:
    failures: list[str] = []
    for threshold in manifest.thresholds:
        if not threshold.hard_gate:
            continue
        value = metrics.get(threshold.metric)
        if value is None or statuses.get(threshold.metric) == BenchmarkMetricStatus.UNAVAILABLE:
            failures.append(f"{threshold.metric}: metric_unavailable")
            continue
        passed = {
            BenchmarkThresholdDirection.MINIMUM: value >= threshold.value,
            BenchmarkThresholdDirection.MAXIMUM: value <= threshold.value,
            BenchmarkThresholdDirection.EXACT: value == threshold.value,
        }[threshold.direction]
        if not passed:
            failures.append(
                f"{threshold.metric}: observed {value:.6g}, expected "
                f"{threshold.direction.value} {threshold.value:.6g}"
            )
    return tuple(failures)


def compare_runs(baseline: BenchmarkRun, candidate: BenchmarkRun) -> BenchmarkComparison:
    compatible = (
        baseline.suite_id == candidate.suite_id
        and baseline.suite_version == candidate.suite_version
        and baseline.corpus_checksum == candidate.corpus_checksum
        and baseline.environment.environment_id == candidate.environment.environment_id
        and baseline.environment.provider == candidate.environment.provider
        and baseline.environment.model == candidate.environment.model
    )
    regressions: list[str] = []
    improvements: list[str] = []
    changes: dict[str, float | None] = {}
    if compatible:
        for name, baseline_value in baseline.aggregate_metrics.items():
            candidate_value = candidate.aggregate_metrics.get(name)
            if baseline_value is None or candidate_value is None:
                changes[name] = None
                continue
            if baseline_value == 0:
                changes[name] = 0.0 if candidate_value == 0 else None
                continue
            change = (candidate_value - baseline_value) / abs(baseline_value)
            changes[name] = change
            if name in {"p95_latency_ms", "mean_tool_calls", "peak_memory_mb"}:
                if change > 0.20:
                    regressions.append(f"{name} increased by {change:.1%}")
                elif change < -0.10:
                    improvements.append(f"{name} decreased by {-change:.1%}")
            elif name == "total_tokens":
                if change > 0.15:
                    regressions.append(f"{name} increased by {change:.1%}")
                elif change < -0.10:
                    improvements.append(f"{name} decreased by {-change:.1%}")
            elif name == "task_success_rate":
                point_change = candidate_value - baseline_value
                changes[name] = point_change
                if point_change < -0.05:
                    regressions.append(f"{name} decreased by {-point_change:.1%} points")
                elif point_change > 0.05:
                    improvements.append(f"{name} increased by {point_change:.1%} points")
            elif name == "throughput":
                if change < -0.15:
                    regressions.append(f"{name} decreased by {-change:.1%}")
                elif change > 0.05:
                    improvements.append(f"{name} increased by {change:.1%}")
    else:
        regressions.append("Baseline and candidate environment or corpus are incompatible.")
    return BenchmarkComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        compatible=compatible,
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        metric_changes=changes,
        passed=compatible and not regressions and candidate.passed,
    )


def calibrate_runs(runs: tuple[BenchmarkRun, ...]) -> BenchmarkRun:
    if len(runs) < 5:
        raise ValueError("Benchmark calibration requires at least five valid runs.")
    reference = runs[0]
    if any(not run.passed for run in runs):
        raise ValueError("Only passing benchmark runs can establish a baseline.")
    if any(
        run.suite_id != reference.suite_id
        or run.suite_version != reference.suite_version
        or run.corpus_checksum != reference.corpus_checksum
        or run.environment.environment_id != reference.environment.environment_id
        or run.environment.provider != reference.environment.provider
        or run.environment.model != reference.environment.model
        or run.environment.settings_fingerprint
        != reference.environment.settings_fingerprint
        for run in runs[1:]
    ):
        raise ValueError("Calibration runs must use the same suite, corpus, and environment.")
    metric_names = set().union(*(run.aggregate_metrics for run in runs))
    metrics: dict[str, float | None] = {}
    statuses: dict[str, BenchmarkMetricStatus] = {}
    for name in metric_names:
        values = [run.aggregate_metrics.get(name) for run in runs]
        if any(value is None for value in values):
            metrics[name] = None
            statuses[name] = BenchmarkMetricStatus.UNAVAILABLE
        else:
            metrics[name] = float(median(float(value) for value in values if value is not None))
            statuses[name] = BenchmarkMetricStatus.AVAILABLE
    return BenchmarkRun(
        suite_id=reference.suite_id,
        suite_version=reference.suite_version,
        corpus_checksum=reference.corpus_checksum,
        environment=reference.environment,
        repeats=sum(run.repeats for run in runs),
        cases=(),
        aggregate_metrics=metrics,
        metric_status=statuses,
        hard_gate_failures=(),
        started_at=min(run.started_at for run in runs),
        completed_at=max(run.completed_at for run in runs),
    )


def write_run_artifacts(run: BenchmarkRun, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump(mode="json") | {"passed": run.passed}
    (output_dir / "benchmark-run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "benchmark-cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in run.cases:
            handle.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")
    (output_dir / "benchmark-report.md").write_text(_markdown_report(run), encoding="utf-8")
    _write_junit(run, output_dir / "junit.xml")


def environment_fingerprint(
    *, provider: str | None, model: str | None, backend: str, settings: Mapping[str, Any]
) -> BenchmarkEnvironment:
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    environment_id = f"{platform.system().lower()}-{platform.machine().lower()}-{backend}"
    return BenchmarkEnvironment(
        environment_id=environment_id,
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_sha=_git_sha(),
        provider=provider,
        model=model,
        backend=backend,
        settings_fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
    )


def load_run(path: Path) -> BenchmarkRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("passed", None)
    return BenchmarkRun.model_validate(payload)


def _path_value(payload: Any, path: str) -> Any:
    current = payload
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return float(ordered[index])


def _error(case: BenchmarkCase, duration_ms: float, error: str) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=case.case_id,
        subsystem=case.subsystem,
        status="error",
        passed=False,
        duration_ms=max(duration_ms, 0.0),
        metrics={"case_success": 0.0, "total_tokens": None},
        metric_status={
            "case_success": BenchmarkMetricStatus.AVAILABLE,
            "total_tokens": BenchmarkMetricStatus.UNAVAILABLE,
        },
        error=error[:2000],
    )


def _skipped(case: BenchmarkCase, reason: str) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=case.case_id,
        subsystem=case.subsystem,
        status="skipped",
        passed=False,
        duration_ms=0.0,
        error=reason,
    )


def _markdown_report(run: BenchmarkRun) -> str:
    outcome = "PASS" if run.passed else "FAIL"
    lines = [
        f"# DataMind Benchmark: {run.suite_id}",
        "",
        f"- Result: **{outcome}**",
        f"- Suite version: `{run.suite_version}`",
        f"- Environment: `{run.environment.environment_id}`",
        f"- Provider/model: `{run.environment.provider or 'none'}` / `{run.environment.model or 'none'}`",
        f"- Corpus checksum: `{run.corpus_checksum}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Status |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| {name} | {'unavailable' if value is None else f'{value:.6g}'} | "
        f"{run.metric_status.get(name, BenchmarkMetricStatus.UNAVAILABLE).value} |"
        for name, value in sorted(run.aggregate_metrics.items())
    )
    lines.extend(["", "## Cases", "", "| Case | Subsystem | Result | Duration ms |", "| --- | --- | --- | ---: |"])
    lines.extend(
        f"| {case.case_id} | {case.subsystem} | {case.status} | {case.duration_ms:.1f} |"
        for case in run.cases
    )
    if run.hard_gate_failures:
        lines.extend(("", "## Gate Failures", "", *(f"- {item}" for item in run.hard_gate_failures)))
    return "\n".join(lines) + "\n"


def _write_junit(run: BenchmarkRun, path: Path) -> None:
    suite = Element(
        "testsuite",
        name=f"benchmark:{run.suite_id}",
        tests=str(len(run.cases)),
        failures=str(sum(case.status == "failed" for case in run.cases)),
        errors=str(sum(case.status == "error" for case in run.cases)),
        skipped=str(sum(case.status == "skipped" for case in run.cases)),
    )
    for case in run.cases:
        node = SubElement(
            suite,
            "testcase",
            name=case.case_id,
            classname=f"benchmark.{case.subsystem}",
            time=f"{case.duration_ms / 1000.0:.6f}",
        )
        if case.status == "failed":
            SubElement(node, "failure", message=case.error or "Benchmark validation failed.")
        elif case.status == "error":
            SubElement(node, "error", message=case.error or "Benchmark execution failed.")
        elif case.status == "skipped":
            SubElement(node, "skipped", message=case.error or "Skipped.")
    ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
