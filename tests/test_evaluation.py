from __future__ import annotations

from app.evaluation.models import BenchmarkMetricName, BenchmarkSample, BenchmarkSuite
from app.evaluation.services import BenchmarkEvaluator
from app.harness.models import TokenUsage


def test_benchmark_evaluator_builds_required_metrics() -> None:
    suite = BenchmarkSuite(
        name="daily-mcp-monitoring",
        samples=(BenchmarkSample(name="mcp", prompt="Monitor MCP ecosystem"),),
    )

    report = BenchmarkEvaluator().build_report(
        suite,
        planning_quality=0.8,
        extraction_quality=0.75,
        summary_quality=0.9,
        workflow_latency_ms=120.0,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
    )

    assert report.passed
    assert {metric.name for metric in report.metrics} == {
        BenchmarkMetricName.PLANNING_QUALITY,
        BenchmarkMetricName.EXTRACTION_QUALITY,
        BenchmarkMetricName.SUMMARY_QUALITY,
        BenchmarkMetricName.WORKFLOW_LATENCY,
        BenchmarkMetricName.COST_TOKEN_USAGE,
    }
    assert report.token_usage.total_tokens == 25
