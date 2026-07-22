from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.settings import get_settings
from app.evaluation.benchmarking import (
    BenchmarkRunner,
    calibrate_runs,
    compare_runs,
    environment_fingerprint,
    load_run,
    write_run_artifacts,
)
from app.evaluation.history import aggregate_sqlite_history
from app.evaluation.operations import optional_executors, release_executors
from app.evaluation.suites import available_suites, load_suite_manifest


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "history":
        return _history(args)
    if args.command == "calibrate":
        return _calibrate(args)
    parser.error("A benchmark command is required.")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DataMind project benchmark harness")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="Run a benchmark suite")
    run.add_argument("--suite", required=True, choices=available_suites())
    run.add_argument("--repeats", type=int)
    run.add_argument("--backend", choices=("local", "compose"), default="local")
    run.add_argument("--output", type=Path)
    compare = subparsers.add_parser("compare", help="Compare a candidate run with a baseline")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    history = subparsers.add_parser("history", help="Create a privacy-safe execution summary")
    history.add_argument("--database", type=Path, default=Path("data/datamind.db"))
    history.add_argument("--output", type=Path)
    calibrate = subparsers.add_parser("calibrate", help="Build a baseline from five or more runs")
    calibrate.add_argument("--runs", type=Path, nargs="+", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    manifest = load_suite_manifest(args.suite)
    repeats = args.repeats or (3 if manifest.mode == "provider" else 1)
    if repeats < 1:
        raise ValueError("Benchmark repeats must be positive.")
    settings = get_settings()
    provider = _suite_provider(manifest.mode)
    model = _suite_model(manifest.mode, settings)
    selected_settings = {
        "suite": manifest.suite_id,
        "suite_version": manifest.version,
        "agent_loop_max_tool_calls": settings.agent_loop_max_tool_calls,
        "agent_loop_max_decisions": settings.agent_loop_max_decisions,
        "agent_loop_max_tokens": settings.agent_loop_max_tokens,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
    }
    environment = environment_fingerprint(
        provider=provider,
        model=model,
        backend=args.backend,
        settings=selected_settings,
    )
    executors = release_executors()
    executors.update(optional_executors(backend=args.backend))
    run = BenchmarkRunner(executors).run(
        manifest, environment=environment, repeats=repeats
    )
    output = args.output or Path("artifacts") / "benchmarks" / manifest.suite_id
    write_run_artifacts(run, output)
    print(json.dumps({"run_id": str(run.run_id), "passed": run.passed, "output": str(output), "failures": run.hard_gate_failures}, ensure_ascii=False))
    return 0 if run.passed else 1


def _compare(args: argparse.Namespace) -> int:
    comparison = compare_runs(load_run(args.baseline), load_run(args.candidate))
    payload = comparison.model_dump(mode="json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if comparison.passed else 1


def _history(args: argparse.Namespace) -> int:
    payload = aggregate_sqlite_history(args.database)
    payload["generated_at"] = datetime.now(UTC).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    baseline = calibrate_runs(tuple(load_run(path) for path in args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(baseline.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"baseline_run_id": str(baseline.run_id), "output": str(args.output)}, ensure_ascii=False))
    return 0


def _suite_provider(mode: str) -> str | None:
    if mode != "provider":
        return None
    return os.getenv("DATAMIND_BENCHMARK_PROVIDER") or "deepseek+kimi"


def _suite_model(mode: str, settings: Any) -> str | None:
    if mode != "provider":
        return None
    return os.getenv("DATAMIND_BENCHMARK_MODEL") or (
        f"{settings.deepseek_model}+{settings.assistant_llm_model}"
    )


if __name__ == "__main__":
    sys.exit(main())
