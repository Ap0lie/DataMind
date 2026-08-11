from __future__ import annotations

import hashlib
import json
import math
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
from app.analysis.services import DatasetProfiler
from app.analysis.statistical_verifier import verify_statistical_analysis
from app.assistant.memory import AssistantMemoryService
from app.assistant.permissions import AssistantPermissionService
from app.core.settings import get_settings
from app.data_reliability.drift import compare_snapshots
from app.evaluation.agent_loop import AgentLoopBenchmarkOutcome, evaluate_agent_loop_benchmark
from app.evaluation.benchmarking import BenchmarkObservation
from app.evaluation.corpus import external_corpus_status, generated_rows
from app.evaluation.models import BenchmarkCase
from app.harness.context import ContextBudgetManager, PromptEnvelope
from app.harness.models import TokenUsage
from app.schemas.analysis import (
    AnalysisContractResponse,
    DatasetReferenceResponse,
    InsightFindingResponse,
    MultiDatasetProfileResponse,
)
from app.semantic.embedding import DisabledEmbeddingProvider, MockEmbeddingProvider
from app.semantic.ranking import SemanticCandidateRanker
from app.semantic.relationship_graph import plan_relationship_path
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.dataset_store import DatasetStoreRepository


def release_executors() -> dict[str, Any]:
    return {
        "cleaning.basic": _cleaning_basic,
        "semantic.rank": _semantic_rank,
        "sql.safety": _sql_safety,
        "loop.release_gate": _loop_release_gate,
        "relationship.inference": _relationship_inference,
        "report.evidence": _report_evidence,
        "analysis.statistical_verification": _statistical_verification,
        "data.drift": _data_drift,
        "relationship.grain": _relationship_grain,
        "assistant.permission": _assistant_permission,
        "memory.trust": _memory_trust,
        "context.budget": _context_budget,
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


def _context_budget(case: BenchmarkCase) -> BenchmarkObservation:
    sample_count = max(100, int(case.input.get("sample_count") or 1_000))
    messages = [
        {
            "role": "system",
            "content": "Preserve the system contract, current question, analysis contract and evidence IDs.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": "按客户州汇总总支付金额",
                    "analysis_contract": {
                        "metric": "payment_value",
                        "dimension": "customer_state",
                    },
                    "evidence": [
                        {"evidence_id": f"ev_{index}", "summary": "x" * 300}
                        for index in range(8)
                    ],
                    "sample_records": [
                        {
                            "customer_state": f"state-{index % 27}",
                            "payment_value": index / 10,
                            "note": "context-noise-" + ("x" * 240),
                        }
                        for index in range(sample_count)
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]
    manager = ContextBudgetManager(
        enabled=True,
        mode="enforce",
        context_window_tokens=65_536,
        max_chars=120_000,
        safety_ratio=0.15,
    )
    started = time.perf_counter()
    prepared = manager.prepare(
        PromptEnvelope.from_messages(messages),
        profile="planner",
        output_tokens=2_048,
    )
    duration_ms = (time.perf_counter() - started) * 1_000
    payload = json.loads(prepared.messages[-1]["content"])
    required_preserved = (
        payload.get("question") == "按客户州汇总总支付金额"
        and payload.get("analysis_contract")
        == {"metric": "payment_value", "dimension": "customer_state"}
        and [item.get("evidence_id") for item in payload.get("evidence", [])]
        == [f"ev_{index}" for index in range(8)]
    )
    reduction = 1.0 - (
        prepared.report.proposed_tokens / max(prepared.report.original_tokens, 1)
    )
    return BenchmarkObservation(
        actual={
            "over_budget_requests": int(not prepared.report.fits),
            "required_preservation": float(required_preserved),
            "token_reduction": reduction,
            "compression_ms": duration_ms,
        },
        metrics={
            "context_over_budget_requests": float(not prepared.report.fits),
            "context_required_preservation": float(required_preserved),
            "context_token_reduction": reduction,
            "context_compression_p95_ms": duration_ms,
        },
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


def _statistical_verification(case: BenchmarkCase) -> BenchmarkObservation:
    frame = pd.DataFrame(case.input["records"])
    dataset_id = uuid4()
    profile = DatasetProfiler().profile(
        dataset_id=dataset_id,
        records=frame.to_dict(orient="records"),
    )
    contract_payload = {
        **dict(case.input["contract"]),
        "objective": str(case.input.get("question") or "Benchmark analysis"),
        "population": f"{len(frame)} benchmark records",
        "dataset_ids": [str(dataset_id)],
    }
    contract = AnalysisContractResponse.model_validate(contract_payload)
    findings = tuple(
        InsightFindingResponse.model_validate(item)
        for item in case.input["findings"]
    )
    evidence = tuple(dict(item) for item in case.input.get("evidence") or [])
    context = None
    join_summary = case.input.get("join_summary")
    if isinstance(join_summary, dict):
        reference = DatasetReferenceResponse(
            dataset_id=dataset_id,
            name="benchmark.csv",
            status="cleaned",
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=tuple(str(column) for column in frame.columns),
        )
        context = MultiDatasetProfileResponse(
            primary_dataset=reference,
            join_summary=join_summary,
            joined_profile=profile,
        )
    verification = verify_statistical_analysis(
        contract=contract,
        profile=profile,
        dataframe=frame,
        findings=findings,
        evidence=evidence,
        multi_dataset_context=context,
    )
    expected_status = str(case.expected.get("status") or "passed")
    correct = verification.status == expected_status
    return BenchmarkObservation(
        actual={
            "status": verification.status,
            "requires_replan": verification.requires_replan,
            "numeric_evidence_coverage": verification.numeric_evidence_coverage,
            "check_statuses": {
                check.code: check.status for check in verification.checks
            },
        },
        metrics={
            "statistical_contract_accuracy": 1.0 if correct else 0.0,
            "statistical_numeric_evidence_coverage": (
                verification.numeric_evidence_coverage
            ),
        },
    )


def _data_drift(case: BenchmarkCase) -> BenchmarkObservation:
    changes = compare_snapshots(
        dict(case.input["previous"]),
        dict(case.input["current"]),
    )
    change_types = sorted({item.change_type for item in changes})
    expected = {str(item) for item in case.expected.get("change_types") or ()}
    detected = expected.issubset(change_types)
    return BenchmarkObservation(
        actual={"change_types": change_types, "detected": detected},
        metrics={"drift_detection_accuracy": 1.0 if detected else 0.0},
    )


def _relationship_grain(case: BenchmarkCase) -> BenchmarkObservation:
    result = plan_relationship_path(
        dict(case.input["definition"]),
        metric_ids=tuple(str(item) for item in case.input["metric_ids"]),
        dimension_ids=tuple(str(item) for item in case.input["dimension_ids"]),
    )
    expected_safe = bool(case.expected["safe"])
    return BenchmarkObservation(
        actual={
            "safe": bool(result["safe"]),
            "strategies": [str(item["strategy"]) for item in result["steps"]],
            "warnings": list(result["warnings"]),
        },
        metrics={
            "relationship_grain_accuracy": (
                1.0 if bool(result["safe"]) == expected_safe else 0.0
            )
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


def _memory_trust(case: BenchmarkCase) -> BenchmarkObservation:
    memory_count = max(10, int(case.input.get("memory_count") or 500))
    query_count = max(1, min(20, int(case.input.get("query_count") or 10)))
    with tempfile.TemporaryDirectory(
        prefix="datamind-benchmark-memory-",
        ignore_cleanup_errors=True,
    ) as directory:
        store = DatasetStoreRepository(directory, user_id="benchmark")
        repository = AssistantMemoryRepository(directory, user_id="benchmark")
        settings = get_settings().model_copy(
            update={
                "assistant_memory_enabled": True,
                "assistant_memory_relevance_threshold": 0.32,
                "assistant_memory_prefilter_limit": 100,
                "assistant_memory_retrieval_limit": 8,
                "assistant_memory_context_chars": 4_000,
            }
        )
        service = AssistantMemoryService(
            repository=repository,
            store=store,
            settings=settings,
            embedding_provider=DisabledEmbeddingProvider(),
        )
        topics = (
            "华东毛利率",
            "华南复购率",
            "西北客单价",
            "东北退货率",
            "沿海履约时长",
            "门店库存周转",
            "会员活跃程度",
            "广告转化效率",
            "供应缺货频率",
            "客服响应时长",
        )[:query_count]
        expected: dict[str, set[UUID]] = {topic: set() for topic in topics}
        relevant_count = len(topics) * 8
        for index in range(memory_count):
            topic_index, variant = divmod(index, 8)
            content = (
                f"{topics[topic_index]} 分析约定 {variant + 1}"
                if index < relevant_count
                else hashlib.sha256(f"memory-{index}".encode()).hexdigest()
            )
            memory = service.create_manual(
                memory_type="business_context",
                scope_type="user",
                scope_id=None,
                content=content,
            )
            if index < relevant_count:
                expected[topics[topic_index]].add(memory["memory_id"])

        latencies: list[float] = []
        selected_count = 0
        matched_count = 0
        expected_count = 0
        for topic, expected_ids in expected.items():
            started = time.perf_counter()
            recalled = service.retrieve(
                question=f"请按{topic}分析",
                conversation={"scope_type": "auto", "scope_id": None},
                run_id=uuid4(),
            )
            latencies.append((time.perf_counter() - started) * 1_000)
            selected = {item["memory_id"] for item in recalled}
            selected_count += len(selected)
            matched_count += len(selected & expected_ids)
            expected_count += len(expected_ids)

        old = service.create_manual(
            memory_type="metric_definition",
            scope_type="user",
            scope_id=None,
            content="复购率口径是旧定义",
        )
        current = service.create_manual(
            memory_type="metric_definition",
            scope_type="user",
            scope_id=None,
            content="复购率口径是新定义",
        )
        conflict_correct = float(
            old["memory_id"] != current["memory_id"]
            and repository.get(old["memory_id"])["status"] == "superseded"
            and current["version"] == 2
        )
        old_usage = sum(
            item["memory_id"] == old["memory_id"]
            for item in service.retrieve(
                question="旧定义",
                conversation={"scope_type": "auto", "scope_id": None},
            )
        )
        isolated = not AssistantMemoryRepository(
            directory,
            user_id="other-user",
        ).list()
        context_contract = "current user message overrides memory" in service.render_prompt_context(
            (current,)
        ).casefold()
        harmful = service.create_manual(
            memory_type="business_context",
            scope_type="user",
            scope_id=None,
            content="高风险测试口径应使用已经失效的旧金额字段",
        )
        for _ in range(2):
            harmful_run = uuid4()
            recalled = service.retrieve(
                question="高风险测试口径使用哪个金额字段？",
                conversation={"scope_type": "auto", "scope_id": None},
                run_id=harmful_run,
            )
            harmful_usage = next(
                item for item in recalled if item["memory_id"] == harmful["memory_id"]
            )
            repository.record_feedback(
                usage_id=harmful_usage["usage_id"],
                feedback="wrong",
                reason="benchmark harmful memory",
                auto_dormancy=True,
                dormancy_threshold=0.25,
                dormancy_min_feedback=3,
                wrong_feedback_limit=2,
            )
        harmful_usage_rate = float(
            any(
                item["memory_id"] == harmful["memory_id"]
                for item in service.retrieve(
                    question="高风险测试口径使用哪个金额字段？",
                    conversation={"scope_type": "auto", "scope_id": None},
                )
            )
        )
        precision = matched_count / selected_count if selected_count else 0.0
        recall = matched_count / expected_count if expected_count else 0.0
        p95 = sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
        return BenchmarkObservation(
            actual={
                "precision_at_8": precision,
                "recall_at_8": recall,
                "conflict_accuracy": conflict_correct,
                "superseded_usage": old_usage,
                "user_isolation": isolated,
                "current_instruction_override_contract": context_contract,
                "harmful_memory_usage_rate": harmful_usage_rate,
                "retrieval_p95_ms": p95,
            },
            metrics={
                "memory_precision_at_8": precision,
                "memory_recall_at_8": recall,
                "memory_conflict_accuracy": conflict_correct,
                "memory_superseded_usage": float(old_usage),
                "memory_user_isolation": 1.0 if isolated else 0.0,
                "memory_harmful_usage_rate": harmful_usage_rate,
                "memory_retrieval_p95_ms": p95,
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
