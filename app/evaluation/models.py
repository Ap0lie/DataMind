from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.harness.models import TokenUsage


class EvaluationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BenchmarkMetricName(StrEnum):
    PLANNING_QUALITY = "planning_quality"
    EXTRACTION_QUALITY = "extraction_quality"
    SUMMARY_QUALITY = "summary_quality"
    WORKFLOW_LATENCY = "workflow_latency"
    COST_TOKEN_USAGE = "cost_token_usage"


class BenchmarkSample(EvaluationModel):
    sample_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuite(EvaluationModel):
    suite_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    samples: tuple[BenchmarkSample, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkMetric(EvaluationModel):
    name: BenchmarkMetricName
    score: float = Field(ge=0.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkReport(EvaluationModel):
    report_id: UUID = Field(default_factory=uuid4)
    suite_id: UUID
    metrics: tuple[BenchmarkMetric, ...]
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    workflow_latency_ms: float = Field(default=0.0, ge=0.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)


class BenchmarkMetricStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "metric_unavailable"


class BenchmarkThresholdDirection(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXACT = "exact"


class BenchmarkCase(EvaluationModel):
    case_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    subsystem: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    validators: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    tags: tuple[str, ...] = Field(default_factory=tuple)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    requires_provider: bool = False
    requires_compose: bool = False


class BenchmarkThreshold(EvaluationModel):
    metric: str = Field(min_length=1)
    direction: BenchmarkThresholdDirection
    value: float
    hard_gate: bool = True
    relative_regression_limit: float | None = Field(default=None, ge=0.0)


class BenchmarkSuiteManifest(EvaluationModel):
    suite_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = ""
    mode: Literal["deterministic", "provider", "performance", "resilience", "frontend"]
    cases: tuple[BenchmarkCase, ...] = Field(default_factory=tuple)
    thresholds: tuple[BenchmarkThreshold, ...] = Field(default_factory=tuple)
    corpus_checksum: str = Field(min_length=1)
    max_total_tokens: int | None = Field(default=None, gt=0)
    max_duration_seconds: float | None = Field(default=None, gt=0.0)


class BenchmarkEnvironment(EvaluationModel):
    environment_id: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    git_sha: str | None = None
    provider: str | None = None
    model: str | None = None
    backend: str = "local"
    settings_fingerprint: str = Field(min_length=1)


class BenchmarkCaseResult(EvaluationModel):
    case_id: str = Field(min_length=1)
    subsystem: str = Field(min_length=1)
    status: Literal["passed", "failed", "error", "skipped"]
    passed: bool
    duration_ms: float = Field(ge=0.0)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_status: dict[str, BenchmarkMetricStatus] = Field(default_factory=dict)
    token_usage: TokenUsage | None = None
    repair_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRun(EvaluationModel):
    run_id: UUID = Field(default_factory=uuid4)
    suite_id: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    corpus_checksum: str = Field(min_length=1)
    environment: BenchmarkEnvironment
    repeats: int = Field(default=1, ge=1)
    cases: tuple[BenchmarkCaseResult, ...] = Field(default_factory=tuple)
    aggregate_metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_status: dict[str, BenchmarkMetricStatus] = Field(default_factory=dict)
    hard_gate_failures: tuple[str, ...] = Field(default_factory=tuple)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        return not self.hard_gate_failures and all(
            case.passed for case in self.cases if case.status != "skipped"
        )


class BenchmarkComparison(EvaluationModel):
    baseline_run_id: UUID
    candidate_run_id: UUID
    compatible: bool
    regressions: tuple[str, ...] = Field(default_factory=tuple)
    improvements: tuple[str, ...] = Field(default_factory=tuple)
    metric_changes: dict[str, float | None] = Field(default_factory=dict)
    passed: bool
