from __future__ import annotations

from app.evaluation.models import (
    BenchmarkMetric,
    BenchmarkMetricName,
    BenchmarkReport,
    BenchmarkSuite,
)
from app.harness.models import TokenUsage


class BenchmarkEvaluator:
    def build_report(
        self,
        suite: BenchmarkSuite,
        *,
        planning_quality: float,
        extraction_quality: float,
        summary_quality: float,
        workflow_latency_ms: float,
        token_usage: TokenUsage,
        pass_threshold: float = 0.7,
    ) -> BenchmarkReport:
        metrics = (
            _quality_metric(
                BenchmarkMetricName.PLANNING_QUALITY,
                planning_quality,
                pass_threshold,
            ),
            _quality_metric(
                BenchmarkMetricName.EXTRACTION_QUALITY,
                extraction_quality,
                pass_threshold,
            ),
            _quality_metric(
                BenchmarkMetricName.SUMMARY_QUALITY,
                summary_quality,
                pass_threshold,
            ),
            BenchmarkMetric(
                name=BenchmarkMetricName.WORKFLOW_LATENCY,
                score=workflow_latency_ms,
                passed=workflow_latency_ms >= 0.0,
                details={"unit": "ms"},
            ),
            BenchmarkMetric(
                name=BenchmarkMetricName.COST_TOKEN_USAGE,
                score=float(token_usage.total_tokens),
                passed=token_usage.total_tokens >= 0,
                details={
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                },
            ),
        )
        return BenchmarkReport(
            suite_id=suite.suite_id,
            metrics=metrics,
            token_usage=token_usage,
            workflow_latency_ms=workflow_latency_ms,
        )


def _quality_metric(
    name: BenchmarkMetricName,
    score: float,
    pass_threshold: float,
) -> BenchmarkMetric:
    return BenchmarkMetric(
        name=name,
        score=score,
        passed=score >= pass_threshold,
        details={"threshold": pass_threshold},
    )
