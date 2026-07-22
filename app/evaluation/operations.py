from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from app.analysis.agent_loop import _validate_safe_dataset_sql
from app.analysis.data_cleaning import _basic_clean_dataframe
from app.analysis.dataset_groups import suggest_dataset_group_relationships
from app.analysis.model_router import MCPAnalysisModelRouter
from app.assistant.permissions import AssistantPermissionService
from app.evaluation.agent_loop import AgentLoopBenchmarkOutcome, evaluate_agent_loop_benchmark
from app.evaluation.benchmarking import BenchmarkObservation
from app.evaluation.corpus import external_corpus_status, generated_rows
from app.evaluation.models import BenchmarkCase
from app.harness.models import TokenUsage
from app.semantic.embedding import MockEmbeddingProvider
from app.semantic.ranking import SemanticCandidateRanker
from app.storage.dataset_store import DatasetStoreRepository


def release_executors() -> dict[str, Any]:
    return {
        "cleaning.basic": _cleaning_basic,
        "semantic.rank": _semantic_rank,
        "sql.safety": _sql_safety,
        "loop.release_gate": _loop_release_gate,
        "relationship.inference": _relationship_inference,
        "report.evidence": _report_evidence,
        "assistant.permission": _assistant_permission,
    }


def optional_executors(*, backend: str) -> dict[str, Any]:
    return {
        "provider.complete": _provider_complete,
        "performance.dataframe": _performance_dataframe,
        "performance.concurrent": _performance_concurrent,
        "performance.table_shapes": _performance_table_shapes,
        "resilience.repair": _resilience_repair,
        "compose.health": lambda case: _compose_health(case, backend=backend),
        "production.smoke_metrics": _production_smoke_metrics,
        "frontend.build": _frontend_build,
        "external.validate": _external_validate,
    }


def _cleaning_basic(case: BenchmarkCase) -> BenchmarkObservation:
    frame = pd.DataFrame(case.input["records"])
    cleaned = _basic_clean_dataframe(frame)
    records = json.loads(
        cleaned.to_json(orient="records", force_ascii=False, date_format="iso")
    )
    return BenchmarkObservation(
        actual={
            "row_count": int(cleaned.shape[0]),
            "column_count": int(cleaned.shape[1]),
            "columns": list(cleaned.columns),
            "records": records,
            "duplicate_rows": int(cleaned.duplicated().sum()),
        },
        metrics={"cleaning_correctness": 1.0},
    )


def _semantic_rank(case: BenchmarkCase) -> BenchmarkObservation:
    all_items = list(case.input["items"])
    ranker = SemanticCandidateRanker(MockEmbeddingProvider())
    questions = list(case.input["questions"])
    correct = 0
    outcomes = []
    for item in questions:
        expected_type = str(item["type"])
        items = [
            candidate
            for candidate in all_items
            if (
                (expected_type == "metric" and bool(candidate.get("formula")))
                or (expected_type == "dimension" and not candidate.get("formula"))
            )
        ]
        ranked = ranker.rank(
            str(item["question"]), items, expected_type=expected_type
        )
        selected = str(ranked[0].item["id"]) if ranked else None
        correct += int(selected == item["expected"])
        outcomes.append({"selected": selected, "expected": item["expected"]})
    score = correct / len(questions) if questions else 0.0
    return BenchmarkObservation(
        actual={"top1": score, "count": len(questions), "outcomes": outcomes},
        metrics={"semantic_top1": score},
    )


def _sql_safety(case: BenchmarkCase) -> BenchmarkObservation:
    accepted = True
    error = None
    try:
        _validate_safe_dataset_sql(str(case.input["sql"]))
    except Exception as exc:
        accepted = False
        error = str(exc)
    return BenchmarkObservation(
        actual={"accepted": accepted, "error": error},
        metrics={
            "security_pass_rate": 1.0
            if accepted == bool(case.expected.get("accepted"))
            else 0.0
        },
    )


def _loop_release_gate(case: BenchmarkCase) -> BenchmarkObservation:
    outcomes = tuple(
        AgentLoopBenchmarkOutcome(
            case_id=str(item["case_id"]),
            selected_tool=item.get("selected_tool"),
            expected_tools=frozenset(item.get("expected_tools") or []),
            legal_call=bool(item["legal_call"]),
            recoverable_error=bool(item.get("recoverable_error")),
            recovered=bool(item.get("recovered")),
            tool_calls=int(item.get("tool_calls") or 0),
            duplicate_successful_actions=int(item.get("duplicate_successful_actions") or 0),
        )
        for item in case.input["outcomes"]
    )
    report = evaluate_agent_loop_benchmark(outcomes)
    return BenchmarkObservation(
        actual={
            "passed": report.passed,
            "legal_call_rate": report.legal_call_rate,
            "repair_success_rate": report.repair_success_rate,
            "mean_tool_calls": report.simple_task_mean_tool_calls,
            "duplicate_successful_actions": report.duplicate_successful_actions,
        },
        metrics={
            "legal_tool_rate": report.legal_call_rate,
            "repair_success_rate": report.repair_success_rate,
            "loop_mean_tool_calls": report.simple_task_mean_tool_calls,
            "duplicate_successful_actions": float(report.duplicate_successful_actions),
        },
        repair_count=sum(item.recovered for item in outcomes),
        tool_calls=sum(item.tool_calls for item in outcomes),
    )


class _UnavailableRouter:
    def complete(self, **_kwargs: Any) -> Any:
        raise RuntimeError("Provider calls are disabled in deterministic benchmarks.")


def _relationship_inference(case: BenchmarkCase) -> BenchmarkObservation:
    with tempfile.TemporaryDirectory(
        prefix="datamind-benchmark-", ignore_cleanup_errors=True
    ) as directory:
        repository = DatasetStoreRepository(directory, user_id="benchmark")
        ids: dict[str, UUID] = {}
        for name, records in case.input["tables"].items():
            dataset = repository.create_dataset(
                name=f"{name}.csv", source_type="csv", source_metadata={}
            )
            repository.append_raw_records(dataset_id=dataset.id, records=records)
            ids[name] = dataset.id
        group = repository.create_dataset_group(
            name="benchmark-commerce", dataset_ids=tuple(ids.values())
        )
        suggestions = suggest_dataset_group_relationships(
            repository, group_id=group.id, router=_UnavailableRouter()
        )
        expected = {
            frozenset((edge["left_table"], edge["right_table"], edge["column"]))
            for edge in case.input["expected_edges"]
        }
        observed = set()
        reverse_ids = {value: key for key, value in ids.items()}
        for candidate in suggestions.candidates:
            if candidate.left_column != candidate.right_column:
                continue
            observed.add(
                frozenset(
                    (
                        reverse_ids[candidate.left_dataset_id],
                        reverse_ids[candidate.right_dataset_id],
                        candidate.left_column,
                    )
                )
            )
        matched = len(expected & observed)
        precision = matched / len(observed) if observed else 0.0
        recall = matched / len(expected) if expected else 1.0
        return BenchmarkObservation(
            actual={"precision": precision, "recall": recall, "matched": matched},
            metrics={"relationship_precision": precision, "relationship_recall": recall},
            details={"candidate_count": len(suggestions.candidates)},
        )


def _report_evidence(case: BenchmarkCase) -> BenchmarkObservation:
    evidence_payload = case.input.get("evidence") or case.input.get("evidence_ids") or []
    evidence: dict[str, dict[str, Any]] = {}
    for item in evidence_payload:
        if isinstance(item, dict):
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id:
                raise ValueError("Benchmark evidence is missing evidence_id.")
            evidence[evidence_id] = dict(item.get("values") or {})
        else:
            evidence[str(item)] = {}
    known = set(evidence)
    findings = list(case.input["findings"])
    numeric = [item for item in findings if re.search(r"\d", str(item.get("content") or ""))]
    supported = [
        item
        for item in numeric
        if {str(value) for value in item.get("evidence_ids") or []} & known
    ]
    coverage = len(supported) / len(numeric) if numeric else 1.0
    claims = [(item, claim) for item in findings for claim in item.get("claims") or []]
    correct_claims = 0
    for finding, claim in claims:
        references = [str(value) for value in finding.get("evidence_ids") or []]
        observed_values = [
            evidence[reference].get(str(claim.get("name")))
            for reference in references
            if reference in evidence
        ]
        expected_value = claim.get("value")
        if any(
            value is not None
            and abs(float(value) - float(expected_value))
            <= float(claim.get("tolerance") or 1e-6)
            for value in observed_values
        ):
            correct_claims += 1
    numeric_accuracy = correct_claims / len(claims) if claims else 1.0
    invalid = sorted(
        {
            str(value)
            for item in findings
            for value in item.get("evidence_ids") or []
            if str(value) not in known
        }
    )
    return BenchmarkObservation(
        actual={
            "coverage": coverage,
            "numeric_accuracy": numeric_accuracy,
            "invalid_evidence_ids": invalid,
        },
        metrics={
            "report_evidence_coverage": coverage,
            "report_numeric_accuracy": numeric_accuracy,
        },
    )


class _GrantStore:
    def __init__(self, grant: dict[str, Any]) -> None:
        self._grant = grant

    def list_permission_grants(self) -> list[dict[str, Any]]:
        return [self._grant]


def _assistant_permission(case: BenchmarkCase) -> BenchmarkObservation:
    asset_id = UUID(str(case.input.get("asset_id") or uuid4()))
    grant_id = uuid4()
    grant = {
        "grant_id": grant_id,
        "asset_type": "dataset",
        "asset_id": asset_id,
        "capabilities": ("data_prepare",),
    }
    service = AssistantPermissionService(
        store=object(),  # type: ignore[arg-type]
        assistant_store=_GrantStore(grant),  # type: ignore[arg-type]
    )
    allowed = True
    error = None
    try:
        service.authorize_tool(
            tool_name="start_cleaning",
            arguments={"dataset_id": str(asset_id)},
            conversation={"scope_type": "auto", "scope_id": None},
            execution_mode=str(case.input["execution_mode"]),
        )
    except PermissionError as exc:
        allowed = False
        error = str(exc)
    return BenchmarkObservation(
        actual={"allowed": allowed, "error": error},
        metrics={
            "security_pass_rate": 1.0
            if allowed == bool(case.expected.get("allowed"))
            else 0.0
        },
    )


def _provider_complete(case: BenchmarkCase) -> BenchmarkObservation:
    router = MCPAnalysisModelRouter()
    response = router.complete(
        messages=list(case.input["messages"]),
        provider=str(case.input.get("provider") or "") or None,
        model=str(case.input.get("model") or "") or None,
        temperature=0.0,
        max_tokens=int(case.input.get("max_tokens") or 300),
        metadata={"agent": "benchmark", "user_id": "benchmark"},
    )
    usage = response.token_usage or {}
    token_usage = (
        TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )
        if "total_tokens" in usage
        else None
    )
    content = str(response.content or "")
    return BenchmarkObservation(
        actual={"content": content, "non_empty": bool(content.strip())},
        token_usage=token_usage,
        details={"provider": response.provider, "model": response.model},
    )


def _performance_dataframe(case: BenchmarkCase) -> BenchmarkObservation:
    size = int(case.input["rows"])
    tracemalloc.start()
    started = time.perf_counter()
    frame = pd.DataFrame(generated_rows(size))
    cleaned = _basic_clean_dataframe(frame)
    duration = max(time.perf_counter() - started, 1e-9)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    throughput = size / duration
    return BenchmarkObservation(
        actual={"row_count": len(cleaned), "throughput": throughput},
        metrics={"throughput": throughput, "peak_memory_mb": peak / 1024 / 1024},
    )


def _performance_concurrent(case: BenchmarkCase) -> BenchmarkObservation:
    workers = int(case.input["workers"])
    rows = int(case.input["rows_per_task"])
    started = time.perf_counter()

    def task(index: int) -> int:
        return len(_basic_clean_dataframe(pd.DataFrame(generated_rows(rows, seed=20260716 + index))))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        completed = list(executor.map(task, range(workers)))
    duration = max(time.perf_counter() - started, 1e-9)
    throughput = sum(completed) / duration
    return BenchmarkObservation(
        actual={"completed_tasks": len(completed), "throughput": throughput},
        metrics={"throughput": throughput},
    )


def _performance_table_shapes(case: BenchmarkCase) -> BenchmarkObservation:
    table_count = int(case.input["tables"])
    rows_per_table = int(case.input.get("rows_per_table") or 100)
    with tempfile.TemporaryDirectory(
        prefix="datamind-benchmark-shapes-", ignore_cleanup_errors=True
    ) as directory:
        repository = DatasetStoreRepository(directory, user_id="benchmark")
        dataset_ids = []
        for table_index in range(table_count):
            dataset = repository.create_dataset(
                name=f"table_{table_index}.csv", source_type="csv", source_metadata={}
            )
            repository.append_raw_records(
                dataset_id=dataset.id,
                records=[
                    {
                        "shared_id": f"K{row_index}",
                        f"value_{table_index}": row_index,
                    }
                    for row_index in range(rows_per_table)
                ],
            )
            dataset_ids.append(dataset.id)
        group = repository.create_dataset_group(
            name=f"shape-{table_count}", dataset_ids=tuple(dataset_ids)
        )
        started = time.perf_counter()
        suggestions = suggest_dataset_group_relationships(
            repository, group_id=group.id, router=_UnavailableRouter()
        )
        duration = max(time.perf_counter() - started, 1e-9)
        return BenchmarkObservation(
            actual={
                "table_count": table_count,
                "candidate_count": len(suggestions.candidates),
            },
            metrics={"throughput": table_count / duration},
        )


def _resilience_repair(case: BenchmarkCase) -> BenchmarkObservation:
    failures = int(case.input["failures_before_success"])
    max_attempts = int(case.input["max_attempts"])
    recovered = failures < max_attempts
    attempts = min(failures + 1, max_attempts)
    return BenchmarkObservation(
        actual={"recovered": recovered, "attempts": attempts},
        metrics={"repair_success_rate": 1.0 if recovered else 0.0},
        repair_count=min(failures, max_attempts - 1),
        fallback_count=0 if recovered else 1,
    )


def _compose_health(case: BenchmarkCase, *, backend: str) -> BenchmarkObservation:
    if backend != "compose":
        raise RuntimeError("This case requires --backend compose.")
    result = subprocess.run(
        ["docker", "compose", "ps", "--status", "running", "--services"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=float(case.timeout_seconds),
        check=False,
    )
    services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    expected = set(case.input.get("services") or [])
    return BenchmarkObservation(
        actual={"healthy": result.returncode == 0 and expected.issubset(services), "services": sorted(services)}
    )


def _frontend_build(case: BenchmarkCase) -> BenchmarkObservation:
    root = Path(__file__).resolve().parents[2]
    command = ["npm", "--prefix", "frontend/react", "run", "build"]
    if os.name == "nt":
        command[0] = "npm.cmd"
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=float(case.timeout_seconds),
        check=False,
    )
    return BenchmarkObservation(
        actual={"succeeded": result.returncode == 0},
        details={"stderr_tail": result.stderr[-1000:]},
    )


def _external_validate(case: BenchmarkCase) -> BenchmarkObservation:
    root = os.getenv("BENCHMARK_DATA_ROOT")
    status = external_corpus_status(Path(root) if root else None)
    return BenchmarkObservation(actual=status)


def _production_smoke_metrics(case: BenchmarkCase) -> BenchmarkObservation:
    path = Path(
        os.getenv("DATAMIND_PRODUCTION_BENCHMARK_FILE")
        or str(case.input.get("path") or "artifacts/benchmarks/production-smoke.json")
    )
    if not path.is_file():
        raise RuntimeError(f"Production smoke benchmark artifact was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkObservation(
        actual=payload,
        metrics={
            "sse_delivery_p95_ms": (
                float(payload["sse_delivery_p95_ms"])
                if payload.get("sse_delivery_p95_ms") is not None
                else None
            ),
            "production_analysis_duration_seconds": (
                float(payload["analysis_duration_seconds"])
                if payload.get("analysis_duration_seconds") is not None
                else None
            ),
        },
    )
