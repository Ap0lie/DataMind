from __future__ import annotations

import base64
import binascii
import io
import json
import math
import operator
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from threading import Lock
from typing import Annotated, Any, NotRequired, TypedDict, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.analysis import agent_prompts, workflow_prompts, workflow_report_nodes
from app.analysis.agent_loop import (
    TOOL_DEFINITIONS,
    AgentToolRuntime,
    LoopErrorType,
    canonical_action_hash,
    canonical_result_hash,
    error_fingerprint,
    load_evidence_result,
)
from app.analysis.analysis_contract import build_analysis_contract
from app.analysis.checkpoints import get_analysis_checkpointer
from app.analysis.dataset_scope import resolve_analysis_dataset_scope
from app.analysis.model_router import AnalysisModelRouter, MCPAnalysisModelRouter
from app.analysis.multidataset import prepare_multi_dataset_context
from app.analysis.python_execution import PythonAnalysisExecutor
from app.analysis.python_sandbox import run_generated_python_analysis
from app.analysis.services import (
    PlannedAnalysis,
    _apply_plan_filters,
    _build_final_insights,
    _build_intent_query,
    _dataframe,
    _format_report_charts,
    _framework_from_profile,
    _looks_like_identifier_column,
    _plan,
    _records,
    _run_python,
    _run_sql,
)
from app.analysis.statistical_verifier import (
    analysis_contract_gaps,
    verify_statistical_analysis,
)
from app.analysis.validators import validate_analysis_plan
from app.analysis.workflow_nodes import (
    ADVERSARIAL_VALIDATE_NODE,
    DESIGN_FRAMEWORK_NODE,
    FORMAT_CHARTS_NODE,
    INTEGRATE_INSIGHTS_NODE,
    JOIN_PREPARE_NODE,
    LOOP_ADVERSARIAL_REPAIR_NODE,
    LOOP_BOOTSTRAP_NODE,
    LOOP_DECIDE_NODE,
    LOOP_EXECUTE_NODE,
    LOOP_FALLBACK_NODE,
    LOOP_FINALIZE_NODE,
    LOOP_OBSERVE_NODE,
    LOOP_REPAIR_NODE,
    LOOP_VERIFY_NODE,
    PLANNER_NODE,
    PYTHON_NODE,
    REPORT_COMMIT_NODE,
    REPORT_DECIDE_NODE,
    REPORT_EXECUTE_NODE,
    REPORT_FALLBACK_NODE,
    REPORT_NODE,
    REPORT_REPAIR_NODE,
    REPORT_VERIFY_NODE,
    ROUND_FANOUT_NODE,
    ROUND_FOUNDATION_NODE,
    ROUND_PREPARE_NODE,
    ROUND_REFLECT_NODE,
    SQL_NODE,
    STATISTICAL_VERIFY_NODE,
)
from app.analysis.workflow_nodes import (
    loop_budget_exhaustion as _loop_budget_exhaustion,
)
from app.analysis.workflow_nodes import (
    route_after_adversarial_validate as _route_after_adversarial_validate,
)
from app.analysis.workflow_nodes import (
    route_after_framework as _route_after_framework,
)
from app.analysis.workflow_nodes import (
    route_after_loop_decide as _route_after_loop_decide,
)
from app.analysis.workflow_nodes import (
    route_after_loop_preflight as _route_after_loop_preflight,
)
from app.analysis.workflow_nodes import (
    route_after_loop_verify as _route_after_loop_verify,
)
from app.analysis.workflow_nodes import (
    route_after_report_decide as _route_after_report_decide,
)
from app.analysis.workflow_nodes import (
    route_after_report_verify as _route_after_report_verify,
)
from app.analysis.workflow_report_nodes import (
    _NUMERIC_CLAIM_RE,
    ReportNodeRuntime,
    _adversarial_validate_node,
    _mandatory_evidence_findings,
    _merge_report_findings,
    _preserve_chart_denominator_scope,
    _relationship_analysis_requested,
    _report_commit_node,
    _report_decide_node,
    _report_execute_node,
    _report_fallback_node,
    _report_node,
    _report_repair_node,
    _report_verify_node,
    _statistical_verify_node,
)
from app.analysis.workflow_support import (
    extract_json_object as _extract_json_object,
)
from app.analysis.workflow_support import (
    float_payload as _float_payload,
)
from app.analysis.workflow_support import (
    require as _require,
)
from app.analysis.workflow_support import (
    string_list as _string_list,
)
from app.assistant.memory import AssistantMemoryService
from app.core.settings import get_settings
from app.harness.node import NodeExecutionHarness, NodeHarnessPolicy
from app.schemas.analysis import (
    AnalysisContractResponse,
    AnalysisFrameworkResponse,
    AnalysisHypothesisResponse,
    AnalysisLineageResponse,
    AnalysisReflectionResponse,
    AnalysisRoundPlanResponse,
    AnalysisRoundResponse,
    AnalysisRunResponse,
    ChartResponse,
    DatasetJoinConfig,
    DatasetProfileResponse,
    InsightFindingResponse,
    MultiDatasetProfileResponse,
    MultimodalInputResponse,
    PlannerMetadataResponse,
    PythonAnalysisResponse,
    PythonCodeAttemptResponse,
    SQLAnalysisResponse,
    StatisticalVerificationResponse,
    StructuredReportResponse,
    ValidationIssueResponse,
    WorkflowTraceNodeResponse,
)
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.dataset_store import DatasetStoreRepository

_planner_messages = agent_prompts.planner_messages
_python_chart_messages = agent_prompts.python_chart_messages
_python_messages = agent_prompts.python_messages
_python_repair_messages = agent_prompts.python_repair_messages
_round_python_messages = agent_prompts.round_python_messages
_framework_messages = workflow_prompts.framework_messages
_round_plan_messages = workflow_prompts.round_plan_messages
_reflection_messages = workflow_prompts.reflection_messages
_integrate_messages = workflow_prompts.integrate_messages
_review_messages = workflow_prompts.review_messages
_chart_refine_messages = workflow_prompts.chart_refine_messages
_report_messages = workflow_prompts.report_messages
_json_repair_messages = workflow_prompts.json_repair_messages
_truncate_text = workflow_prompts.truncate_text
_compact_profile = workflow_prompts.compact_profile
_compact_sql_result = workflow_prompts.compact_sql_result
_compact_chart = workflow_prompts.compact_chart
_compact_python_statistics = workflow_prompts.compact_python_statistics
_compact_round = workflow_prompts.compact_round
_compact_execution_result = workflow_prompts.compact_execution_result
_compact_structured_report = workflow_prompts.compact_structured_report
_multimodal_payload = workflow_prompts.multimodal_payload
_multimodal_content = workflow_prompts.multimodal_content
_compact_multi_dataset_context = workflow_prompts.compact_multi_dataset_context
_sql_messages = agent_prompts.sql_messages

# Compatibility exports for focused tests and existing internal callers.
_preserve_verified_report_findings = (
    workflow_report_nodes._preserve_verified_report_findings
)
_sanitize_report_cardinality_claims = (
    workflow_report_nodes._sanitize_report_cardinality_claims
)
_unsupported_summary_numbers = workflow_report_nodes._unsupported_summary_numbers

ProgressCallback = Callable[[str, int, str], None]
CancelChecker = Callable[[], bool]
NodeEventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class WorkflowRunHooks:
    progress_callback: ProgressCallback | None = None
    cancel_checker: CancelChecker | None = None
    node_event_callback: NodeEventCallback | None = None


_RUN_HOOKS: dict[UUID, WorkflowRunHooks] = {}
_RUN_HOOKS_LOCK = Lock()


class _UsageTrackingModelRouter:
    """Aggregate provider-reported usage across every model call in one run."""

    def __init__(self, delegate: AnalysisModelRouter) -> None:
        self._delegate = delegate
        self._lock = Lock()
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset(self) -> None:
        with self._lock:
            self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._usage)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "metadata": metadata,
        }
        # Keep routers that implement the legacy completion contract working.
        # Tool-capable calls still receive both fields when they are requested.
        if tools is not None:
            call_kwargs["tools"] = tools
        if tool_choice is not None:
            call_kwargs["tool_choice"] = tool_choice
        response = self._delegate.complete(**call_kwargs)
        usage = response.token_usage or {}
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        total = int(usage.get("total_tokens") or prompt + completion)
        with self._lock:
            self._usage["prompt_tokens"] += prompt
            self._usage["completion_tokens"] += completion
            self._usage["total_tokens"] += total
        return response


class AnalysisWorkflowState(TypedDict):
    run_id: UUID
    dataset_id: UUID
    dataset_group_id: NotRequired[UUID | None]
    additional_dataset_ids: NotRequired[tuple[UUID, ...]]
    join_plan: NotRequired[tuple[DatasetJoinConfig, ...]]
    relationship_plan: NotRequired[tuple[DatasetJoinConfig, ...]]
    planner_decision: NotRequired[dict[str, Any] | None]
    question: str
    prompt_overrides: NotRequired[dict[str, str]]
    multimodal_inputs: NotRequired[tuple[MultimodalInputResponse, ...]]
    dataframe_artifact_id: NotRequired[UUID]
    profile: NotRequired[DatasetProfileResponse]
    analysis_framework: NotRequired[AnalysisFrameworkResponse]
    analysis_contract: NotRequired[AnalysisContractResponse]
    analysis_experiences: NotRequired[tuple[dict[str, Any], ...]]
    statistical_verification: NotRequired[StatisticalVerificationResponse]
    statistical_validation_issues: NotRequired[tuple[ValidationIssueResponse, ...]]
    analysis_lineage: NotRequired[AnalysisLineageResponse]
    planned_analysis: NotRequired[PlannedAnalysis]
    planner_metadata: NotRequired[PlannerMetadataResponse]
    multi_dataset_context: NotRequired[MultiDatasetProfileResponse | None]
    sql_result: NotRequired[SQLAnalysisResponse | None]
    python_result: NotRequired[PythonAnalysisResponse | None]
    round_hypotheses: NotRequired[tuple[str, ...]]
    current_round_number: NotRequired[int]
    current_round_hypothesis: NotRequired[str]
    current_previous_rounds: NotRequired[tuple[AnalysisRoundResponse, ...]]
    current_total_rounds: NotRequired[int]
    round_outputs: Annotated[list[dict[str, Any]], operator.add]
    rounds: NotRequired[tuple[AnalysisRoundResponse, ...]]
    final_insights: NotRequired[tuple[InsightFindingResponse, ...]]
    plan_validation_issues: NotRequired[tuple[ValidationIssueResponse, ...]]
    validation_issues: NotRequired[tuple[ValidationIssueResponse, ...]]
    report_charts: NotRequired[tuple[ChartResponse, ...]]
    structured_report: NotRequired[StructuredReportResponse | None]
    report_markdown: NotRequired[str]
    final_response: NotRequired[AnalysisRunResponse]
    workflow_trace: NotRequired[tuple[WorkflowTraceNodeResponse, ...]]
    executed_nodes: NotRequired[tuple[str, ...]]
    planner_source: NotRequired[str]
    analysis_fast_path: NotRequired[bool]
    report_source: NotRequired[str]
    python_source: NotRequired[str]
    python_generated_code: NotRequired[str | None]
    python_execution_error: NotRequired[str | None]
    python_attempts: NotRequired[tuple[PythonCodeAttemptResponse, ...]]
    model_router_provider: NotRequired[str | None]
    model_router_model: NotRequired[str | None]
    model_router_error: NotRequired[str | None]
    sql_source: NotRequired[str]
    sql_validation_error: NotRequired[str | None]
    agent_mode: NotRequired[str]
    loop_iteration: NotRequired[int]
    loop_decision_count: NotRequired[int]
    tool_call_count: NotRequired[int]
    tool_evidence: NotRequired[tuple[dict[str, Any], ...]]
    tool_attempts: NotRequired[dict[str, int]]
    failure_fingerprints: NotRequired[dict[str, int]]
    loop_budget: NotRequired[dict[str, Any]]
    loop_terminal_reason: NotRequired[str | None]
    loop_pending_call: NotRequired[dict[str, Any] | None]
    loop_last_execution: NotRequired[dict[str, Any] | None]
    loop_duplicate_action: NotRequired[dict[str, Any] | None]
    loop_duplicate_decision_count: NotRequired[int]
    loop_action_counts: NotRequired[dict[str, int]]
    loop_deferred_calls: NotRequired[tuple[dict[str, Any], ...]]
    loop_repair_context: NotRequired[dict[str, Any] | None]
    loop_preflight_verification: NotRequired[StatisticalVerificationResponse]
    loop_summary: NotRequired[dict[str, Any]]
    adversarial_repair_count: NotRequired[int]
    report_strategy: NotRequired[str]
    report_decision_count: NotRequired[int]
    report_revision_count: NotRequired[int]
    report_evidence_return_count: NotRequired[int]
    report_terminal_reason: NotRequired[str | None]
    report_validation: NotRequired[dict[str, Any]]
    report_draft_ready: NotRequired[bool]
    html_report: NotRequired[str]
    report_started_epoch: NotRequired[float]
    report_used_tokens: NotRequired[int]


class AnalysisWorkflowRunner:
    def __init__(
        self,
        repository: DatasetStoreRepository,
        model_router: AnalysisModelRouter | None = None,
        checkpointer: Any | None = None,
        python_executor: PythonAnalysisExecutor | None = None,
    ) -> None:
        self._repository = repository
        self._checkpointer = (
            checkpointer
            if checkpointer is not None
            else get_analysis_checkpointer(dataset_store_path=repository.root_path)
        )
        self._usage_tracker = _UsageTrackingModelRouter(
            model_router or MCPAnalysisModelRouter()
        )
        self._workflow = build_analysis_workflow(
            repository,
            model_router=self._usage_tracker,
            checkpointer=self._checkpointer,
            python_executor=python_executor or run_generated_python_analysis,
        )

    def run(
        self,
        *,
        dataset_id: UUID,
        dataset_group_id: UUID | None = None,
        additional_dataset_ids: tuple[UUID, ...] = (),
        join_plan: tuple[DatasetJoinConfig | dict[str, Any], ...] = (),
        relationship_plan: tuple[DatasetJoinConfig | dict[str, Any], ...] = (),
        planner_decision: dict[str, Any] | None = None,
        question: str,
        prompt_overrides: dict[str, str] | None = None,
        multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
        progress_callback: ProgressCallback | None = None,
        cancel_checker: CancelChecker | None = None,
        workflow_id: UUID | None = None,
        resume: bool = False,
        node_event_callback: NodeEventCallback | None = None,
        agent_mode: str = "legacy",
    ) -> AnalysisRunResponse:
        run_id = workflow_id or uuid4()
        prepared_multimodal_inputs = _prepare_multimodal_inputs(multimodal_inputs)
        prepared_join_plan = tuple(
            item if isinstance(item, DatasetJoinConfig) else DatasetJoinConfig.model_validate(item)
            for item in join_plan
        )
        prepared_relationship_plan = tuple(
            item if isinstance(item, DatasetJoinConfig) else DatasetJoinConfig.model_validate(item)
            for item in relationship_plan
        )
        effective_join_plan = prepared_join_plan or prepared_relationship_plan
        effective_additional_dataset_ids = _merge_additional_dataset_ids(
            additional_dataset_ids,
            effective_join_plan,
            primary_dataset_id=dataset_id,
        )
        dataset_scope = resolve_analysis_dataset_scope(
            self._repository,
            question=question,
            dataset_id=dataset_id,
            additional_dataset_ids=effective_additional_dataset_ids,
            join_plan=effective_join_plan,
        )
        dataset_id = dataset_scope.dataset_id
        effective_join_plan = dataset_scope.join_plan
        effective_additional_dataset_ids = dataset_scope.additional_dataset_ids
        prepared_relationship_plan = (
            effective_join_plan if prepared_relationship_plan else ()
        )
        if progress_callback is not None:
            progress_callback("queued", 0, "Analysis request accepted.")
        input_state: dict[str, Any] | None = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_group_id": dataset_group_id,
            "additional_dataset_ids": effective_additional_dataset_ids,
            "join_plan": effective_join_plan,
            "relationship_plan": prepared_relationship_plan,
            "planner_decision": planner_decision,
            "question": question,
            "prompt_overrides": dict(prompt_overrides or {}),
            "multimodal_inputs": prepared_multimodal_inputs,
            "executed_nodes": (),
            "round_outputs": [],
            "agent_mode": agent_mode,
            "adversarial_repair_count": 0,
            "report_decision_count": 0,
            "report_revision_count": 0,
            "report_evidence_return_count": 0,
            "report_used_tokens": 0,
        }
        workflow_config = {"configurable": {"thread_id": str(run_id)}}
        if resume and self._checkpointer is not None:
            try:
                snapshot = self._workflow.get_state(workflow_config)
                if snapshot.values:
                    input_state = None
            except Exception:
                pass
        with _RUN_HOOKS_LOCK:
            _RUN_HOOKS[run_id] = WorkflowRunHooks(
                progress_callback,
                cancel_checker,
                node_event_callback,
            )
        self._usage_tracker.reset()
        try:
            state = cast(
                AnalysisWorkflowState,
                self._workflow.invoke(
                    input_state,
                    config=workflow_config,
                ),
            )
        finally:
            usage = self._usage_tracker.snapshot()
            try:
                if node_event_callback is not None and usage["total_tokens"]:
                    node_event_callback(
                        {
                            "node": "model_usage",
                            "status": "completed",
                            "message": "Aggregated provider-reported model token usage.",
                            "attempt": 0,
                            "event_type": "model_usage",
                            "token_usage": usage,
                        }
                    )
            finally:
                with _RUN_HOOKS_LOCK:
                    _RUN_HOOKS.pop(run_id, None)
        final_response = state.get("final_response")
        if final_response is None:
            raise RuntimeError("Analysis workflow did not produce a final response.")
        if progress_callback is not None:
            progress_callback("complete", 100, "Analysis complete.")
        return final_response


def build_analysis_workflow(
    repository: DatasetStoreRepository,
    *,
    model_router: AnalysisModelRouter | None = None,
    checkpointer: Any | None = None,
    python_executor: PythonAnalysisExecutor | None = None,
) -> Any:
    resolved_python_executor = python_executor or run_generated_python_analysis
    settings = get_settings()
    graph: Any = StateGraph(AnalysisWorkflowState)
    harness = NodeExecutionHarness(
        NodeHarnessPolicy(timeout_seconds=settings.agent_loop_timeout_seconds),
        event_callback=_notify_node_event,
    )
    report_harness = NodeExecutionHarness(
        NodeHarnessPolicy(timeout_seconds=settings.report_loop_timeout_seconds + 5.0),
        event_callback=_notify_node_event,
    )
    report_runtime = ReportNodeRuntime(
        notify_progress=_notify_progress,
        emit_loop_event=_emit_loop_event,
        workflow_dataframe=_workflow_dataframe,
    )
    nodes = {
        PLANNER_NODE: _planner_node(repository, model_router),
        DESIGN_FRAMEWORK_NODE: _design_framework_node(model_router),
        SQL_NODE: _sql_node(repository, model_router),
        PYTHON_NODE: _python_node(repository, model_router, resolved_python_executor),
        ROUND_PREPARE_NODE: _round_prepare_node(),
        ROUND_FOUNDATION_NODE: _round_foundation_node(
            repository, model_router, resolved_python_executor
        ),
        ROUND_FANOUT_NODE: _round_fanout_node(repository, model_router, resolved_python_executor),
        ROUND_REFLECT_NODE: _round_reflect_node(model_router),
        INTEGRATE_INSIGHTS_NODE: _integrate_insights_node(model_router),
        ADVERSARIAL_VALIDATE_NODE: _adversarial_validate_node(model_router, report_runtime),
        FORMAT_CHARTS_NODE: _format_charts_node(model_router),
        STATISTICAL_VERIFY_NODE: _statistical_verify_node(repository, report_runtime),
        REPORT_NODE: _report_node(repository, model_router, report_runtime),
        REPORT_DECIDE_NODE: _report_decide_node(report_runtime),
        REPORT_EXECUTE_NODE: _report_execute_node(model_router, report_runtime),
        REPORT_VERIFY_NODE: _report_verify_node(report_runtime),
        REPORT_REPAIR_NODE: _report_repair_node(report_runtime),
        REPORT_FALLBACK_NODE: _report_fallback_node(report_runtime),
        REPORT_COMMIT_NODE: _report_commit_node(repository, report_runtime),
        LOOP_BOOTSTRAP_NODE: _loop_bootstrap_node(),
        LOOP_DECIDE_NODE: _loop_decide_node(repository, model_router),
        LOOP_EXECUTE_NODE: _loop_execute_node(repository, resolved_python_executor),
        LOOP_OBSERVE_NODE: _loop_observe_node(repository),
        LOOP_VERIFY_NODE: _loop_verify_node(),
        LOOP_REPAIR_NODE: _loop_repair_node(),
        LOOP_FALLBACK_NODE: _loop_fallback_node(repository, resolved_python_executor),
        LOOP_FINALIZE_NODE: _loop_finalize_node(repository),
        LOOP_ADVERSARIAL_REPAIR_NODE: _loop_adversarial_repair_node(),
    }
    report_nodes = {
        REPORT_NODE,
        REPORT_DECIDE_NODE,
        REPORT_EXECUTE_NODE,
        REPORT_VERIFY_NODE,
        REPORT_REPAIR_NODE,
        REPORT_FALLBACK_NODE,
        REPORT_COMMIT_NODE,
    }
    for node_name, handler in nodes.items():
        node_harness = report_harness if node_name in report_nodes else harness
        graph.add_node(node_name, node_harness.wrap(node_name, handler))

    graph.add_edge(START, PLANNER_NODE)
    graph.add_edge(PLANNER_NODE, DESIGN_FRAMEWORK_NODE)
    graph.add_conditional_edges(
        DESIGN_FRAMEWORK_NODE,
        _route_after_framework,
        {
            SQL_NODE: SQL_NODE,
            PYTHON_NODE: PYTHON_NODE,
            LOOP_BOOTSTRAP_NODE: LOOP_BOOTSTRAP_NODE,
        },
    )
    graph.add_edge(LOOP_BOOTSTRAP_NODE, LOOP_DECIDE_NODE)
    graph.add_conditional_edges(
        LOOP_DECIDE_NODE,
        _route_after_loop_decide,
        {
            LOOP_DECIDE_NODE: LOOP_DECIDE_NODE,
            LOOP_EXECUTE_NODE: LOOP_EXECUTE_NODE,
            LOOP_FALLBACK_NODE: LOOP_FALLBACK_NODE,
            LOOP_FINALIZE_NODE: LOOP_FINALIZE_NODE,
        },
    )
    graph.add_edge(LOOP_EXECUTE_NODE, LOOP_OBSERVE_NODE)
    graph.add_edge(LOOP_OBSERVE_NODE, LOOP_VERIFY_NODE)
    graph.add_conditional_edges(
        LOOP_VERIFY_NODE,
        _route_after_loop_verify,
        {
            LOOP_DECIDE_NODE: LOOP_DECIDE_NODE,
            LOOP_REPAIR_NODE: LOOP_REPAIR_NODE,
            LOOP_FALLBACK_NODE: LOOP_FALLBACK_NODE,
            LOOP_FINALIZE_NODE: LOOP_FINALIZE_NODE,
        },
    )
    graph.add_edge(LOOP_REPAIR_NODE, LOOP_DECIDE_NODE)
    graph.add_edge(LOOP_FALLBACK_NODE, LOOP_FINALIZE_NODE)
    graph.add_conditional_edges(
        LOOP_FINALIZE_NODE,
        _route_after_loop_preflight,
        {
            INTEGRATE_INSIGHTS_NODE: INTEGRATE_INSIGHTS_NODE,
            LOOP_ADVERSARIAL_REPAIR_NODE: LOOP_ADVERSARIAL_REPAIR_NODE,
        },
    )
    graph.add_edge(LOOP_ADVERSARIAL_REPAIR_NODE, LOOP_DECIDE_NODE)
    graph.add_edge(SQL_NODE, PYTHON_NODE)
    graph.add_edge(PYTHON_NODE, ROUND_PREPARE_NODE)
    graph.add_edge(ROUND_PREPARE_NODE, ROUND_FOUNDATION_NODE)
    graph.add_conditional_edges(
        ROUND_FOUNDATION_NODE,
        _fanout_after_foundation,
        [ROUND_FANOUT_NODE, ROUND_REFLECT_NODE],
    )
    graph.add_edge(ROUND_FANOUT_NODE, ROUND_REFLECT_NODE)
    graph.add_edge(ROUND_REFLECT_NODE, INTEGRATE_INSIGHTS_NODE)
    graph.add_edge(INTEGRATE_INSIGHTS_NODE, FORMAT_CHARTS_NODE)
    graph.add_edge(FORMAT_CHARTS_NODE, STATISTICAL_VERIFY_NODE)
    graph.add_edge(STATISTICAL_VERIFY_NODE, ADVERSARIAL_VALIDATE_NODE)
    graph.add_conditional_edges(
        ADVERSARIAL_VALIDATE_NODE,
        _route_after_adversarial_validate,
        {
            REPORT_NODE: REPORT_NODE,
            REPORT_DECIDE_NODE: REPORT_DECIDE_NODE,
            LOOP_ADVERSARIAL_REPAIR_NODE: LOOP_ADVERSARIAL_REPAIR_NODE,
        },
    )
    graph.add_conditional_edges(
        REPORT_DECIDE_NODE,
        _route_after_report_decide,
        {
            REPORT_EXECUTE_NODE: REPORT_EXECUTE_NODE,
            REPORT_FALLBACK_NODE: REPORT_FALLBACK_NODE,
            LOOP_BOOTSTRAP_NODE: LOOP_BOOTSTRAP_NODE,
        },
    )
    graph.add_edge(REPORT_EXECUTE_NODE, REPORT_VERIFY_NODE)
    graph.add_conditional_edges(
        REPORT_VERIFY_NODE,
        _route_after_report_verify,
        {
            REPORT_COMMIT_NODE: REPORT_COMMIT_NODE,
            REPORT_REPAIR_NODE: REPORT_REPAIR_NODE,
            REPORT_FALLBACK_NODE: REPORT_FALLBACK_NODE,
            LOOP_BOOTSTRAP_NODE: LOOP_BOOTSTRAP_NODE,
        },
    )
    graph.add_edge(REPORT_REPAIR_NODE, REPORT_EXECUTE_NODE)
    graph.add_edge(REPORT_FALLBACK_NODE, REPORT_COMMIT_NODE)
    graph.add_edge(REPORT_COMMIT_NODE, END)
    graph.add_edge(REPORT_NODE, END)
    return graph.compile(name="datamind_analysis", checkpointer=checkpointer)


def _notify_progress(
    state: AnalysisWorkflowState,
    *,
    stage: str,
    progress: int,
    message: str,
) -> None:
    with _RUN_HOOKS_LOCK:
        hooks = _RUN_HOOKS.get(state["run_id"], WorkflowRunHooks())
    cancel_checker = hooks.cancel_checker
    if cancel_checker is not None and cancel_checker():
        raise RuntimeError("Analysis job was canceled.")
    progress_callback = hooks.progress_callback
    if progress_callback is not None:
        progress_callback(stage, progress, message)


def _notify_node_event(state: Any, event: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    run_id = state.get("run_id")
    if not isinstance(run_id, UUID):
        return
    with _RUN_HOOKS_LOCK:
        hooks = _RUN_HOOKS.get(run_id)
    if hooks and hooks.node_event_callback:
        hooks.node_event_callback(event)


def _workflow_dataframe(
    repository: DatasetStoreRepository,
    state: AnalysisWorkflowState,
) -> pd.DataFrame:
    artifact_id = _require(
        state.get("dataframe_artifact_id"),
        "Planner did not persist the analysis dataframe.",
    )
    artifact = repository.get_artifact(state["dataset_id"], artifact_id)
    records = artifact.get("content", {}).get("records", [])
    if not isinstance(records, list):
        raise RuntimeError("Analysis dataframe artifact is invalid.")
    return _dataframe([record for record in records if isinstance(record, dict)])


def _planner_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=PLANNER_NODE,
            progress=8,
            message="Profiling dataset and planning analysis route.",
        )
        dataset_id = state["dataset_id"]
        multi_dataset_preparation = prepare_multi_dataset_context(
            repository,
            dataset_id=dataset_id,
            additional_dataset_ids=state.get("additional_dataset_ids", ()),
            join_plan=state.get("join_plan", ()),
        )
        records = multi_dataset_preparation.records
        profile = multi_dataset_preparation.profile
        multi_dataset_context = multi_dataset_preparation.response
        memory_service = AssistantMemoryService(
            repository=AssistantMemoryRepository(
                repository.root_path,
                user_id=repository.user_id,
            ),
            store=repository,
        )
        analysis_experiences = memory_service.retrieve_analysis_experiences(
            question=state["question"],
            dataset_id=dataset_id,
            dataset_group_id=state.get("dataset_group_id"),
            additional_dataset_ids=state.get("additional_dataset_ids", ()),
            run_id=state["run_id"],
        )
        planned_analysis = _plan(state["question"], profile)
        planner_source = "rules"
        model_router_provider: str | None = None
        model_router_model: str | None = None
        model_router_error: str | None = None
        plan_validation_issues: tuple[ValidationIssueResponse, ...] = (
            multi_dataset_preparation.validation_issues
        )
        if model_router is not None:
            try:
                response = model_router.complete(
                    messages=_planner_messages(
                        question=state["question"],
                        profile=profile,
                        multi_dataset_context=multi_dataset_context,
                        analysis_experiences=analysis_experiences,
                    ),
                    temperature=0.1,
                    max_tokens=700,
                    metadata={
                        "agent": "planner",
                        "dataset_id": str(dataset_id),
                    },
                )
                try:
                    model_plan = _parse_model_plan(
                        response.content, fallback=planned_analysis
                    )
                except ValueError as parse_error:
                    response = model_router.complete(
                        messages=_json_repair_messages(
                            stage="planner",
                            invalid_content=response.content,
                            error=str(parse_error),
                            contract=(
                                "Return one JSON object with route, category_column, "
                                "metric_column, time_column, and steps."
                            ),
                        ),
                        temperature=0.0,
                        max_tokens=700,
                        metadata={
                            "agent": "planner",
                            "dataset_id": str(dataset_id),
                            "structured_repair": True,
                        },
                    )
                    model_plan = _parse_model_plan(
                        response.content, fallback=planned_analysis
                    )
                plan_validation_issues = _validate_plan_harness(
                    planned_analysis=model_plan,
                    profile=profile,
                    finding_ref="planner",
                )
                if plan_validation_issues:
                    raise ValueError(_plan_validation_summary(plan_validation_issues))
                planned_analysis = _sanitize_model_plan(
                    model_plan,
                    profile,
                    allow_time=planned_analysis.time_column is not None,
                )
                planner_source = "model_router"
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
        planner_metadata = _planner_metadata(
            question=state["question"],
            profile=profile,
            planned_analysis=planned_analysis,
            source=planner_source,
            error=model_router_error,
            multi_dataset_context=multi_dataset_context,
        )
        semantic_decision = state.get("planner_decision")
        if semantic_decision:
            if semantic_decision.get("semantic_source") == "published":
                from app.semantic.service import SemanticLayerService

                semantic_decision = SemanticLayerService(
                    repository
                ).reconcile_planner_decision(
                    semantic_decision,
                    question=state["question"],
                )
            semantic_plan = (
                semantic_decision.get("semantic_plan")
                if isinstance(semantic_decision.get("semantic_plan"), dict)
                else {}
            )
            planner_metadata = planner_metadata.model_copy(
                update={
                    "confidence": float(
                        semantic_decision.get("calibrated_confidence")
                        or planner_metadata.confidence
                    ),
                    "semantic_model_id": UUID(str(semantic_decision["semantic_model_id"]))
                    if semantic_decision.get("semantic_model_id")
                    else None,
                    "semantic_model_version": semantic_decision.get("semantic_model_version"),
                    "semantic_source": str(semantic_decision.get("semantic_source") or "legacy"),
                    "semantic_plan": semantic_plan,
                    "confidence_breakdown": semantic_decision.get("component_scores") or {},
                    "raw_confidence": float(semantic_decision.get("raw_confidence") or 0),
                    "confidence_level": str(semantic_decision.get("confidence_level") or "medium"),
                    "requires_confirmation": bool(semantic_decision.get("requires_confirmation")),
                }
            )
        analysis_dataframe = _apply_plan_filters(
            _dataframe(records),
            planned_analysis.filters,
        )
        analysis_contract = build_analysis_contract(
            question=state["question"],
            dataset_id=dataset_id,
            additional_dataset_ids=state.get("additional_dataset_ids", ()),
            profile=profile,
            plan=planned_analysis,
            planner_metadata=planner_metadata,
            multi_dataset_context=multi_dataset_context,
            analysis_row_count=len(analysis_dataframe),
        )
        executed_nodes = (*state.get("executed_nodes", ()),)
        if multi_dataset_context is not None:
            executed_nodes = (*executed_nodes, JOIN_PREPARE_NODE)
        dataframe_artifact_id = repository.save_artifact(
            dataset_id=dataset_id,
            artifact_type="analysis_dataframe",
            content={"records": records},
            file_name=f"{state['run_id']}.json",
        )

        output: dict[str, Any] = {
            "dataframe_artifact_id": dataframe_artifact_id,
            "profile": profile,
            "multi_dataset_context": multi_dataset_context,
            "planned_analysis": planned_analysis,
            "planner_metadata": planner_metadata,
            "planner_decision": semantic_decision,
            "analysis_contract": analysis_contract,
            "analysis_experiences": analysis_experiences,
            "plan_validation_issues": plan_validation_issues,
            "planner_source": planner_source,
            "analysis_fast_path": _analysis_fast_path_eligible(
                profile=profile,
                planned_analysis=planned_analysis,
                multi_dataset_context=multi_dataset_context,
                multimodal_inputs=state.get("multimodal_inputs", ()),
            ),
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*executed_nodes, PLANNER_NODE),
        }
        return output

    return run


def _design_framework_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=DESIGN_FRAMEWORK_NODE,
            progress=18,
            message="Designing analysis framework.",
        )
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        framework = _framework_from_profile(question=state["question"], profile=profile)
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if model_router is not None and not state.get("analysis_fast_path", False):
            try:
                response = model_router.complete(
                    messages=_framework_messages(
                        question=state["question"],
                        profile=profile,
                        multi_dataset_context=state.get("multi_dataset_context"),
                    ),
                    temperature=0.1,
                    max_tokens=900,
                    metadata={
                        "agent": "design_framework",
                        "dataset_id": str(state["dataset_id"]),
                    },
                )
                framework = _parse_model_framework(
                    response.content,
                    fallback=framework,
                    profile=profile,
                    question=state["question"],
                )
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
        return {
            "analysis_framework": framework,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), DESIGN_FRAMEWORK_NODE),
        }

    return run


def _sql_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=SQL_NODE,
            progress=30,
            message="Running safe SQL analysis.",
        )
        dataframe = _workflow_dataframe(repository, state)
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        planned_analysis = _require(
            state.get("planned_analysis"),
            "Planner did not produce an analysis plan.",
        )
        sql_source = "rules"
        sql_validation_error: str | None = None
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")

        planner_decision = state.get("planner_decision")
        if (
            planner_decision
            and planner_decision.get("semantic_source") == "published"
            and planner_decision.get("semantic_plan", {}).get("metric_ids")
        ):
            from app.semantic.service import SemanticLayerService

            semantic_result = SemanticLayerService(repository).execute_semantic_plan(
                planner_decision
            )
            return {
                "sql_result": SQLAnalysisResponse.model_validate(semantic_result),
                "sql_source": "semantic_compiler",
                "sql_validation_error": None,
                "model_router_provider": model_router_provider,
                "model_router_model": model_router_model,
                "model_router_error": model_router_error,
                "executed_nodes": (*state.get("executed_nodes", ()), SQL_NODE),
            }

        if model_router is not None and not state.get("analysis_fast_path", False):
            try:
                response = model_router.complete(
                    messages=_sql_messages(
                        question=state["question"],
                        profile=profile,
                        planned_analysis=planned_analysis,
                        multi_dataset_context=state.get("multi_dataset_context"),
                    ),
                    temperature=0.0,
                    max_tokens=700,
                    metadata={
                        "agent": "sql",
                        "dataset_id": str(state["dataset_id"]),
                    },
                )
                candidate_sql = _extract_sql(response.content)
                validation = _validate_dataset_select_sql(candidate_sql)
                if not validation["ok"]:
                    raise ValueError(validation["message"])
                sql_result = _execute_dataset_sql(
                    dataframe,
                    validation["sql"],
                    explanation="DeepSeek generated this safe SELECT query for the dataset table.",
                )
                sql_source = "model_router"
                model_router_provider = response.provider
                model_router_model = response.model
                return {
                    "sql_result": sql_result,
                    "sql_source": sql_source,
                    "sql_validation_error": None,
                    "model_router_provider": model_router_provider,
                    "model_router_model": model_router_model,
                    "model_router_error": model_router_error,
                    "executed_nodes": (*state.get("executed_nodes", ()), SQL_NODE),
                }
            except Exception as exc:
                sql_validation_error = str(exc)
                model_router_error = str(exc)

        return {
            "sql_result": _run_sql(dataframe, planned_analysis),
            "sql_source": sql_source,
            "sql_validation_error": sql_validation_error,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), SQL_NODE),
        }

    return run


def _python_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
    python_executor: PythonAnalysisExecutor,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=PYTHON_NODE,
            progress=43,
            message="Running Python analysis.",
        )
        dataframe = _workflow_dataframe(repository, state)
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        planned_analysis = _require(
            state.get("planned_analysis"),
            "Planner did not produce an analysis plan.",
        )
        baseline_result = _run_python(
            dataframe,
            planned_analysis,
            state.get("sql_result"),
            question=state["question"],
        )
        python_result = baseline_result
        python_source = "rules"
        python_generated_code: str | None = None
        python_execution_error: str | None = None
        python_attempts: tuple[PythonCodeAttemptResponse, ...] = ()
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")

        if model_router is not None:
            execution = _execute_generated_python_with_repairs(
                model_router=model_router,
                agent="python",
                question=state["question"],
                dataframe=dataframe,
                profile=profile,
                planned_analysis=planned_analysis,
                sql_result=state.get("sql_result"),
                multi_dataset_context=state.get("multi_dataset_context"),
                metadata={"dataset_id": str(state["dataset_id"])},
                python_executor=python_executor,
            )
            chart_execution: PythonCodeExecutionResult | None = None
            if execution.result is not None:
                chart_execution = _execute_generated_python_with_repairs(
                    model_router=model_router,
                    agent="python_charts",
                    question=state["question"],
                    dataframe=dataframe,
                    profile=profile,
                    planned_analysis=planned_analysis,
                    sql_result=state.get("sql_result"),
                    multi_dataset_context=state.get("multi_dataset_context"),
                    metadata={"dataset_id": str(state["dataset_id"])},
                    python_executor=python_executor,
                )
            python_attempts = (
                *execution.attempts,
                *(chart_execution.attempts if chart_execution else ()),
            )
            python_generated_code = _combine_python_generated_code(
                statistics_code=execution.code,
                chart_code=chart_execution.code if chart_execution else None,
            )
            python_execution_error = execution.error
            model_router_provider = (
                (chart_execution.provider if chart_execution else None)
                or execution.provider
                or model_router_provider
            )
            model_router_model = (
                (chart_execution.model if chart_execution else None)
                or execution.model
                or model_router_model
            )
            if execution.result is not None:
                generated_result = execution.result
                if chart_execution and chart_execution.result is not None:
                    generated_result = PythonAnalysisResponse(
                        statistics=execution.result.statistics,
                        insights=execution.result.insights,
                        charts=chart_execution.result.charts,
                        text_analysis=execution.result.text_analysis,
                    )
                elif chart_execution and chart_execution.error:
                    model_router_error = chart_execution.error
                python_result = _merge_python_results(
                    generated_result=generated_result,
                    baseline_result=baseline_result,
                )
                python_source = "model_router"
                python_execution_error = None
            elif python_execution_error:
                model_router_error = python_execution_error

        return {
            "python_result": python_result,
            "python_source": python_source,
            "python_generated_code": python_generated_code,
            "python_execution_error": python_execution_error,
            "python_attempts": python_attempts,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), PYTHON_NODE),
        }

    return run


def _round_prepare_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=ROUND_PREPARE_NODE,
            progress=52,
            message="Preparing iterative analysis rounds.",
        )
        baseline_python_result = _require(
            state.get("python_result"),
            "Python Agent did not produce analysis results.",
        )
        hypotheses = _round_hypotheses(
            question=state["question"],
            framework=state.get("analysis_framework"),
            python_result=baseline_python_result,
        )
        return {
            "round_hypotheses": hypotheses,
            "executed_nodes": (*state.get("executed_nodes", ()), ROUND_PREPARE_NODE),
        }

    return run


def _round_foundation_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
    python_executor: PythonAnalysisExecutor,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=ROUND_FOUNDATION_NODE,
            progress=60,
            message="Running foundation analysis round.",
        )
        dataframe = _workflow_dataframe(repository, state)
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        hypotheses = state.get("round_hypotheses", ()) or (state["question"],)
        first_round, first_python_result, provider, model, error, first_plan_issues = (
            _execute_analysis_round(
                dataset_id=state["dataset_id"],
                question=state["question"],
                dataframe=dataframe,
                profile=profile,
                round_number=1,
                hypothesis=hypotheses[0],
                previous_rounds=(),
                total_rounds=len(hypotheses),
                model_router=model_router,
                multi_dataset_context=state.get("multi_dataset_context"),
                fanout_mode="serial_foundation",
                python_executor=python_executor,
            )
        )
        return {
            "round_outputs": [
                _round_output(
                    round_item=first_round,
                    python_result=first_python_result,
                    provider=provider,
                    model=model,
                    error=error,
                    plan_issues=first_plan_issues,
                )
            ],
            "model_router_provider": provider or state.get("model_router_provider"),
            "model_router_model": model or state.get("model_router_model"),
            "model_router_error": error or state.get("model_router_error"),
            "executed_nodes": (*state.get("executed_nodes", ()), ROUND_FOUNDATION_NODE),
        }

    return run


def _fanout_after_foundation(state: AnalysisWorkflowState) -> str | list[Send]:
    hypotheses = state.get("round_hypotheses", ())
    foundation_rounds = tuple(
        output["round"]
        for output in state.get("round_outputs", [])
        if isinstance(output.get("round"), AnalysisRoundResponse)
    )
    sends: list[Send] = []
    for round_number, hypothesis in enumerate(hypotheses[1:3], 2):
        branch_state = dict(state)
        branch_state.update(
            {
                "current_round_number": round_number,
                "current_round_hypothesis": hypothesis,
                "current_previous_rounds": foundation_rounds,
                "current_total_rounds": len(hypotheses),
            }
        )
        sends.append(Send(ROUND_FANOUT_NODE, branch_state))
    return sends or ROUND_REFLECT_NODE


def _round_fanout_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
    python_executor: PythonAnalysisExecutor,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=ROUND_FANOUT_NODE,
            progress=68,
            message="Running parallel exploration round.",
        )
        dataframe = _workflow_dataframe(repository, state)
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        round_number = int(state.get("current_round_number", 2))
        hypothesis = state.get("current_round_hypothesis", state["question"])
        previous_rounds = state.get("current_previous_rounds", ())
        total_rounds = int(state.get("current_total_rounds", round_number))
        round_item, python_result, provider, model, error, plan_issues = _execute_analysis_round(
            dataset_id=state["dataset_id"],
            question=state["question"],
            dataframe=dataframe,
            profile=profile,
            round_number=round_number,
            hypothesis=hypothesis,
            previous_rounds=previous_rounds,
            total_rounds=total_rounds,
            model_router=model_router,
            multi_dataset_context=state.get("multi_dataset_context"),
            fanout_mode="langgraph_send_fanout",
            python_executor=python_executor,
        )
        return {
            "round_outputs": [
                _round_output(
                    round_item=round_item,
                    python_result=python_result,
                    provider=provider,
                    model=model,
                    error=error,
                    plan_issues=plan_issues,
                )
            ],
        }

    return run


def _round_reflect_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=ROUND_REFLECT_NODE,
            progress=76,
            message="Reflecting on iterative analysis results.",
        )
        outputs = sorted(
            state.get("round_outputs", []),
            key=lambda output: output["round"].round_number,
        )
        rounds_tuple = tuple(
            output["round"]
            for output in outputs
            if isinstance(output.get("round"), AnalysisRoundResponse)
        )
        round_python_results = tuple(
            output["python_result"]
            for output in outputs
            if isinstance(output.get("python_result"), PythonAnalysisResponse)
        )
        plan_validation_issues: list[ValidationIssueResponse] = list(
            state.get("plan_validation_issues", ())
        )
        for output in outputs:
            plan_validation_issues.extend(
                issue
                for issue in output.get("plan_issues", ())
                if isinstance(issue, ValidationIssueResponse)
            )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        for output in outputs:
            model_router_provider = output.get("provider") or model_router_provider
            model_router_model = output.get("model") or model_router_model
            model_router_error = output.get("error") or model_router_error

        if model_router is not None:
            try:
                response = model_router.complete(
                    messages=_reflection_messages(
                        question=state["question"],
                        rounds=rounds_tuple,
                        python_result=_require(
                            state.get("python_result"),
                            "Python Agent did not produce analysis results.",
                        ),
                    ),
                    temperature=0.1,
                    max_tokens=900,
                    metadata={
                        "agent": "reflection",
                        "dataset_id": str(state["dataset_id"]),
                    },
                )
                rounds_tuple = _apply_model_reflections(response.content, rounds_tuple)
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
        merged_python_result = _merge_round_python_results(
            baseline_result=_require(
                state.get("python_result"),
                "Python Agent did not produce analysis results.",
            ),
            round_results=round_python_results,
        )
        executed_nodes = (*state.get("executed_nodes", ()),)
        if any(
            output["round"].execution_result.get("fanout_mode") == "langgraph_send_fanout"
            for output in outputs
            if isinstance(output.get("round"), AnalysisRoundResponse)
        ):
            executed_nodes = (*executed_nodes, ROUND_FANOUT_NODE)
        return {
            "rounds": rounds_tuple,
            "python_result": merged_python_result,
            "plan_validation_issues": tuple(plan_validation_issues),
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*executed_nodes, ROUND_REFLECT_NODE),
        }

    return run


def _integrate_insights_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=INTEGRATE_INSIGHTS_NODE,
            progress=82,
            message="Integrating final insights.",
        )
        python_result = state.get("python_result")
        final_insights = _merge_report_findings(
            _mandatory_evidence_findings(state),
            _build_final_insights(
                question=state["question"],
                python_result=python_result,
                sql_result=state.get("sql_result"),
            ),
        )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if model_router is not None and not state.get("analysis_fast_path", False):
            try:
                response = model_router.complete(
                    messages=_integrate_messages(
                        question=state["question"],
                        profile=_require(
                            state.get("profile"),
                            "Planner did not produce a dataset profile.",
                        ),
                        rounds=state.get("rounds", ()),
                        sql_result=state.get("sql_result"),
                        python_result=python_result,
                        multimodal_inputs=state.get("multimodal_inputs", ()),
                        multi_dataset_context=state.get("multi_dataset_context"),
                    ),
                    temperature=0.2,
                    max_tokens=1400,
                    metadata={
                        "agent": "integrate",
                        "dataset_id": str(state["dataset_id"]),
                        "optional_stage": True,
                    },
                )
                try:
                    final_insights = _parse_model_insights(
                        response.content, fallback=final_insights
                    )
                except ValueError as parse_error:
                    response = model_router.complete(
                        messages=_json_repair_messages(
                            stage="integrate_insights",
                            invalid_content=response.content,
                            error=str(parse_error),
                            contract=(
                                "Return one JSON object with an insights array. Each insight "
                                "must contain title, content, data_source, evidence, confidence, "
                                "business_impact, and recommended_action."
                            ),
                        ),
                        temperature=0.0,
                        max_tokens=1400,
                        metadata={
                            "agent": "integrate",
                            "dataset_id": str(state["dataset_id"]),
                            "structured_repair": True,
                        },
                    )
                    final_insights = _parse_model_insights(
                        response.content, fallback=final_insights
                    )
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
        return {
            "final_insights": final_insights,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), INTEGRATE_INSIGHTS_NODE),
        }

    return run


def _format_charts_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=FORMAT_CHARTS_NODE,
            progress=88,
            message="Formatting report charts.",
        )
        python_result = state.get("python_result")
        charts = (
            *(python_result.charts if python_result else ()),
            *state.get("report_charts", ()),
        )
        report_charts = _format_report_charts(
            question=state["question"],
            sql_result=state.get("sql_result"),
            charts=charts,
            findings=state.get("final_insights", ()),
            plan=state.get("planned_analysis"),
        )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if (
            model_router is not None
            and report_charts
            and not state.get("analysis_fast_path", False)
        ):
            try:
                response = model_router.complete(
                    messages=_chart_refine_messages(
                        question=state["question"],
                        charts=report_charts,
                        final_insights=state.get("final_insights", ()),
                        multimodal_inputs=state.get("multimodal_inputs", ()),
                        multi_dataset_context=state.get("multi_dataset_context"),
                    ),
                    temperature=0.2,
                    max_tokens=1000,
                    metadata={
                        "agent": "chart_refine",
                        "dataset_id": str(state["dataset_id"]),
                        "optional_stage": True,
                    },
                )
                report_charts = _apply_model_chart_explanations(response.content, report_charts)
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
        return {
            "report_charts": report_charts,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), FORMAT_CHARTS_NODE),
        }

    return run


def _loop_runtime(
    repository: DatasetStoreRepository,
    state: AnalysisWorkflowState,
    python_executor: PythonAnalysisExecutor,
) -> AgentToolRuntime:
    additional = state.get("additional_dataset_ids", ())
    allowed = tuple(dict.fromkeys((state["dataset_id"], *additional)))
    return AgentToolRuntime(
        repository=repository,
        job_id=state["run_id"],
        dataset_id=state["dataset_id"],
        allowed_dataset_ids=allowed,
        dataframe=_workflow_dataframe(repository, state),
        question=state["question"],
        profile=_require(state.get("profile"), "Planner did not produce a dataset profile."),
        plan=_require(state.get("planned_analysis"), "Planner did not produce an analysis plan."),
        planner_decision=state.get("planner_decision"),
        python_executor=python_executor,
        evidence=state.get("tool_evidence", ()),
    )


def _loop_prompt_columns(state: AnalysisWorkflowState) -> tuple[Any, ...]:
    profile = _require(state.get("profile"), "Missing profile.")
    contract = state.get("analysis_contract")
    plan = state.get("planned_analysis")
    requested_values: list[str | None] = []
    if contract is not None:
        requested_values.extend((contract.metric, *contract.dimensions, contract.time_field))
        requested_values.extend(item.column for item in contract.aggregations if item.column)
        requested_values.extend(item.column for item in contract.filters)
    if plan is not None:
        requested_values.extend((plan.metric_column, plan.category_column, plan.time_column))
    requested = {str(value) for value in requested_values if value}
    selected = [
        column
        for column in profile.columns
        if any(
            column.name == value
            or column.name.endswith(f"__{value}")
            or value.endswith(f"__{column.name}")
            for value in requested
        )
    ]
    selected_names = {column.name for column in selected}
    selected.extend(
        column
        for column in profile.columns
        if column.name not in selected_names
        and len(selected) < 24
    )
    return tuple(selected[:24])


def _compact_loop_multi_dataset_context(
    state: AnalysisWorkflowState,
) -> dict[str, Any] | None:
    compact = _compact_multi_dataset_context(state.get("multi_dataset_context"))
    if compact is None:
        return None
    relevant = {column.name for column in _loop_prompt_columns(state)}
    source_map = compact.get("column_source_map")
    if isinstance(source_map, dict):
        compact["column_source_map"] = {
            key: value for key, value in source_map.items() if key in relevant
        }
    compact["joins"] = list(compact.get("joins") or [])[:8]
    return compact


def _loop_evidence_prompt(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "tool_name": item.get("tool_name"),
        "status": item.get("status"),
        "summary": item.get("summary"),
        "action_hash": str(item.get("action_hash") or "")[:16],
        "arguments": _compact_loop_arguments(item),
        "output_fields": list(item.get("output_fields") or ())[:20],
        "contract_covered": bool(item.get("contract_covered")),
        "coverage_gaps": item.get("coverage_gaps") or {},
        "error_type": item.get("error_type"),
    }


def _compact_loop_arguments(item: dict[str, Any]) -> dict[str, Any]:
    arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
    if item.get("tool_name") == "execute_safe_sql":
        return {"sql": _truncate_text(str(arguments.get("sql") or ""), 600)}
    return {
        str(key): value
        for key, value in islice(arguments.items(), 8)
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _contract_result(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    compact: dict[str, Any] = {}
    if result.get("native_grain") is True:
        compact.update(
            {
                key: result.get(key)
                for key in (
                    "native_grain",
                    "source_dataset_id",
                    "source_dataset",
                    "metric",
                    "aggregation",
                    "group_by",
                    "grain",
                    "filters",
                    "sql",
                    "explanation",
                )
                if result.get(key) is not None
            }
        )
        compact["rows"] = []
    elif isinstance(result.get("sql_result"), dict):
        sql_result = result["sql_result"]
        compact["sql_result"] = {
            "sql": sql_result.get("sql") or "",
            "rows": [],
            "explanation": sql_result.get("explanation") or "",
        }
    elif result.get("sql"):
        compact.update(
            {
                "sql": result.get("sql"),
                "rows": [],
                "explanation": result.get("explanation") or "",
            }
        )
    if isinstance(result.get("python_result"), dict):
        python_result = result["python_result"]
        compact["python_result"] = {
            "statistics": python_result.get("statistics") or {},
            "insights": [],
            "charts": [],
            "execution_context": python_result.get("execution_context"),
        }
    return compact or None


def _claim_result(result: Any) -> dict[str, Any] | None:
    """Keep bounded deterministic claim values when the full result is artifact-backed."""

    if not isinstance(result, dict):
        return None
    compact = dict(_contract_result(result) or {})
    values = _claim_result_numeric_values(result)
    if values:
        compact["claim_values"] = sorted(values)
    claim_rows = _claim_result_rows(result)
    if claim_rows:
        compact["claim_rows"] = claim_rows
    return compact or None


def _claim_result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: Any = result.get("rows")
    if not isinstance(rows, (list, tuple)) and isinstance(result.get("sql_result"), dict):
        rows = result["sql_result"].get("rows")
    if not isinstance(rows, (list, tuple)):
        return []

    def magnitude(row: Any) -> float:
        if not isinstance(row, dict):
            return -math.inf
        numbers: list[float] = []
        for value in row.values():
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numbers.append(abs(number))
        return max(numbers, default=-math.inf)

    selected = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=magnitude,
        reverse=True,
    )[:200]
    return [
        {
            str(key): value[:160] if isinstance(value, str) else value
            for key, value in row.items()
        }
        for row in selected
    ]


def _claim_result_numeric_values(value: Any, *, key: str = "") -> set[float]:
    ignored_keys = {
        "sql",
        "code",
        "explanation",
        "evidence_id",
        "source_dataset_id",
        "dataset_id",
        "action_hash",
        "result_hash",
        "filters",
        "arguments",
    }
    if key in ignored_keys or value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        number = float(value)
        return {number} if math.isfinite(number) else set()
    if isinstance(value, str):
        return {
            float(token.replace(",", ""))
            for token in _NUMERIC_CLAIM_RE.findall(value)
            if "%" not in token
        }
    if isinstance(value, dict):
        values: set[float] = set()
        for child_key, child in value.items():
            values.update(_claim_result_numeric_values(child, key=str(child_key)))
            if child_key == "rows" and isinstance(child, (list, tuple)):
                values.update(_claim_result_row_totals(child))
        return values
    if isinstance(value, (list, tuple)):
        values = {float(len(value))} if key == "rows" else set()
        for child in value:
            values.update(_claim_result_numeric_values(child))
        return values
    return set()


def _claim_result_row_totals(rows: list[Any] | tuple[Any, ...]) -> set[float]:
    columns: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for column, value in row.items():
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                columns.setdefault(str(column), []).append(number)
    return {
        float(sum(numbers))
        for numbers in columns.values()
        if len(numbers) > 1
    }


def _loop_output_fields(result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        return ()
    rows = result.get("rows")
    if not isinstance(rows, list):
        sql_result = result.get("sql_result")
        rows = sql_result.get("rows") if isinstance(sql_result, dict) else None
    return tuple(str(key) for key in rows[0]) if isinstance(rows, list) and rows else ()


def _annotate_loop_evidence(
    evidence: tuple[dict[str, Any], ...],
    execution: dict[str, Any],
    *,
    contract_covered: bool,
    coverage_gaps: dict[str, tuple[str, ...]],
) -> tuple[dict[str, Any], ...]:
    evidence_id = execution.get("evidence_id")
    action_hash = execution.get("action_hash")
    updated: list[dict[str, Any]] = []
    for item in evidence:
        matches = (
            evidence_id and item.get("evidence_id") == evidence_id
        ) or (
            not evidence_id and action_hash and item.get("action_hash") == action_hash
        )
        updated.append(
            {
                **item,
                "contract_covered": contract_covered,
                "coverage_gaps": coverage_gaps,
            }
            if matches
            else item
        )
    return tuple(updated)


def _loop_bootstrap_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        settings = get_settings()
        _notify_progress(
            state,
            stage=LOOP_BOOTSTRAP_NODE,
            progress=25,
            message="Preparing the bounded autonomous analysis loop.",
        )
        now = time.time()
        budget = state.get("loop_budget") or {
            "started_epoch": now,
            "deadline_epoch": now + settings.agent_loop_timeout_seconds,
            "max_tool_calls": settings.agent_loop_max_tool_calls,
            "max_decisions": settings.agent_loop_max_decisions,
            "max_tool_attempts": settings.agent_loop_max_tool_attempts,
            "max_tokens": settings.agent_loop_max_tokens,
            "used_tokens": 0,
        }
        _emit_loop_event(
            state,
            event_type="loop_bootstrap",
            status="completed",
            message="Loop scope and budgets fixed by the server.",
            iteration=state.get("loop_iteration", 0),
            payload={
                "allowed_tools": [item["function"]["name"] for item in TOOL_DEFINITIONS],
                "budget": _public_loop_budget(budget, state),
            },
        )
        return {
            "loop_iteration": state.get("loop_iteration", 0),
            "loop_decision_count": state.get("loop_decision_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "tool_evidence": state.get("tool_evidence", ()),
            "tool_attempts": state.get("tool_attempts", {}),
            "failure_fingerprints": state.get("failure_fingerprints", {}),
            "loop_budget": budget,
            "loop_terminal_reason": None,
            "loop_pending_call": None,
            "loop_last_execution": None,
            "loop_duplicate_action": None,
            "loop_duplicate_decision_count": 0,
            "loop_action_counts": state.get("loop_action_counts", {}),
            "loop_deferred_calls": state.get("loop_deferred_calls", ()),
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_BOOTSTRAP_NODE),
        }

    return run


def _loop_decide_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=LOOP_DECIDE_NODE,
            progress=min(68, 28 + state.get("tool_call_count", 0) * 3),
            message="AI is selecting the next safe analysis tool.",
        )
        exhausted = _loop_budget_exhaustion(state)
        if exhausted:
            return {"loop_pending_call": None, "loop_terminal_reason": exhausted}
        repair_context = state.get("loop_repair_context")
        required_tool = str((repair_context or {}).get("required_tool") or "")
        deferred_calls = state.get("loop_deferred_calls", ())
        if deferred_calls and not repair_context:
            pending = dict(deferred_calls[0])
            _emit_loop_event(
                state,
                event_type="deferred_decision",
                status="completed",
                message=f"Continuing the planned tool sequence with {pending['tool_name']}.",
                iteration=state.get("loop_iteration", 0) + 1,
                tool_name=str(pending["tool_name"]),
                payload={
                    "arguments_hash": canonical_action_hash(
                        str(pending["tool_name"]), dict(pending.get("arguments") or {})
                    ),
                    "remaining_deferred_calls": len(deferred_calls) - 1,
                },
            )
            return {
                "loop_pending_call": pending,
                "loop_deferred_calls": tuple(deferred_calls[1:]),
            }
        if (
            required_tool == "execute_safe_sql"
            and (repair_context or {}).get("deterministic_sql") is True
            and not state.get("additional_dataset_ids")
        ):
            sql, explanation = _build_intent_query(
                _workflow_dataframe(repository, state),
                _require(state.get("planned_analysis"), "Missing plan."),
            )
            decision_count = state.get("loop_decision_count", 0) + 1
            pending = {
                "action": "tool_call",
                "tool_call_id": f"contract_sql_{decision_count}",
                "tool_name": "execute_safe_sql",
                "arguments": {"sql": sql},
                "reason": explanation,
            }
            _emit_loop_event(
                state,
                event_type="decision",
                status="completed",
                message="Generated deterministic SQL for the remaining contract requirements.",
                iteration=state.get("loop_iteration", 0) + 1,
                tool_name="execute_safe_sql",
                payload={"arguments_hash": canonical_action_hash("execute_safe_sql", {"sql": sql})},
            )
            return {
                "loop_decision_count": decision_count,
                "loop_pending_call": pending,
                "loop_deferred_calls": (),
                "loop_repair_context": None,
            }
        if model_router is None:
            return {"loop_pending_call": None, "loop_terminal_reason": "provider_unavailable"}
        settings = get_settings()
        decision_count = state.get("loop_decision_count", 0) + 1
        evidence = [_loop_evidence_prompt(item) for item in state.get("tool_evidence", ())[-6:]]
        available_tools = [
            item
            for item in TOOL_DEFINITIONS
            if not required_tool or item["function"]["name"] == required_tool
        ]
        tool_choice = "required" if required_tool else "auto"
        decision_messages = [
            {
                "role": "system",
                "content": (
                    "You are DataMind's bounded analysis controller. Select at most one provided tool per turn. "
                    "Use only known columns and evidence IDs. Never request writes, files, network access, identity, or scope. "
                    "For multi-table questions that name original fact tables, first inspect source datasets and use "
                    "aggregate_source_dataset separately for every requested monetary metric. Never finish or fallback "
                    "while an explicitly requested source-table metric lacks native-grain evidence. "
                    "When repair.required_tool is present, call exactly that tool and cover every field listed in "
                    "repair.contract_gaps; do not inspect context again. When repair.required_arguments is present, "
                    "use those arguments as the fixed starting point. "
                    'When evidence is sufficient, reply with JSON {"action":"finish","reason":"..."}. '
                    'When safe analysis is impossible, reply {"action":"fallback","reason":"..."}. '
                    'If native tool calls are unavailable, reply {"action":"tool_call","tool_name":"...","arguments":{...},"reason":"..."}. '
                    "Do not reveal hidden reasoning; provide only a short reason in tool arguments or final JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state["question"],
                        "columns": [
                            {
                                "name": column.name,
                                "dtype": column.dtype,
                                "is_numeric": column.is_numeric,
                                "missing_count": column.missing_count,
                                "distinct_count": column.distinct_count,
                            }
                            for column in _loop_prompt_columns(state)
                        ],
                        "plan": _planned_analysis_payload(
                            _require(state.get("planned_analysis"), "Missing plan.")
                        ),
                        "analysis_contract": _require(
                            state.get("analysis_contract"), "Missing analysis contract."
                        ).model_dump(mode="json"),
                        "multi_dataset_context": _compact_loop_multi_dataset_context(state),
                        "evidence": evidence,
                        "attempted_action_hashes": list(state.get("loop_action_counts", {}))[-12:],
                        "repair": repair_context,
                        "remaining_budget": _public_loop_budget(
                            state.get("loop_budget", {}), state
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        try:
            try:
                response = model_router.complete(
                    messages=decision_messages,
                    provider=settings.agent_loop_provider,
                    model=settings.agent_loop_model,
                    temperature=0.0,
                    max_tokens=min(1200, settings.llm_max_tokens),
                    metadata={
                        "agent": "agent_loop",
                        "dataset_id": str(state["dataset_id"]),
                        "job_id": str(state["run_id"]),
                        "allow_provider_fallback": False,
                    },
                    tools=available_tools,
                    tool_choice=tool_choice,
                )
            except Exception as exc:
                if not _is_tool_capability_error(exc):
                    raise
                response = model_router.complete(
                    messages=decision_messages,
                    provider=settings.agent_loop_provider,
                    model=settings.agent_loop_model,
                    temperature=0.0,
                    max_tokens=min(1200, settings.llm_max_tokens),
                    metadata={
                        "agent": "agent_loop",
                        "dataset_id": str(state["dataset_id"]),
                        "job_id": str(state["run_id"]),
                        "tool_adapter": "structured_json",
                        "allow_provider_fallback": False,
                    },
                )
            used_tokens = int(response.token_usage.get("total_tokens") or 0)
            budget = {
                **state.get("loop_budget", {}),
                "used_tokens": int(state.get("loop_budget", {}).get("used_tokens") or 0)
                + used_tokens,
            }
            normalized_calls = tuple(
                pending
                for index, raw in enumerate(response.tool_calls)
                if (pending := _pending_from_tool_call(raw, decision_count, index)) is not None
            )
            if len(normalized_calls) > 1:
                tool_names = [
                    str(call.get("tool_name") or "") for call in normalized_calls
                ]
                _emit_loop_event(
                    state,
                    event_type="normalized_decision",
                    status="completed",
                    message="Model proposed multiple tools; executing one safe call this turn.",
                    iteration=state.get("loop_iteration", 0),
                    payload={
                        "tool_call_count": len(response.tool_calls),
                        "selected_tool": tool_names[0],
                        "deferred_tools": tool_names[1:],
                    },
                )
            if response.tool_calls:
                if not normalized_calls:
                    return {
                        "loop_decision_count": decision_count,
                        "loop_budget": budget,
                        "loop_pending_call": {"action": "retry_decision"},
                        "loop_deferred_calls": (),
                        "loop_repair_context": {
                            "error_type": "invalid_arguments",
                            "message": "Tool arguments were not valid JSON.",
                        },
                    }
                pending = normalized_calls[0]
                name = str(pending["tool_name"])
                arguments = dict(pending.get("arguments") or {})
                argument_error = _tool_argument_error(name, arguments)
                if argument_error:
                    return {
                        "loop_decision_count": decision_count,
                        "loop_budget": budget,
                        "loop_pending_call": {"action": "retry_decision"},
                        "loop_deferred_calls": (),
                        "loop_repair_context": _tool_argument_repair_context(
                            state, repair_context, name, argument_error
                        ),
                    }
                if required_tool and name != required_tool:
                    return _invalid_required_tool_decision(
                        decision_count=decision_count,
                        budget=budget,
                        required_tool=required_tool,
                        selected_tool=name,
                        repair_context=repair_context,
                    )
                _emit_loop_event(
                    state,
                    event_type="decision",
                    status="completed",
                    message=f"Selected tool: {name}",
                    iteration=state.get("loop_iteration", 0) + 1,
                    tool_name=name,
                    payload={
                        "arguments_hash": canonical_action_hash(name, arguments),
                        "reason": pending["reason"],
                        "remaining_budget": _public_loop_budget(budget, state),
                    },
                )
                return {
                    "loop_decision_count": decision_count,
                    "loop_budget": budget,
                    "loop_pending_call": pending,
                    "loop_deferred_calls": normalized_calls[1:] if not required_tool else (),
                    "loop_repair_context": None,
                    "model_router_provider": response.provider,
                    "model_router_model": response.model,
                }
            decision = _parse_loop_text_decision(response.content)
            if decision["action"] == "tool_call":
                name = str(decision.get("tool_name") or "")
                arguments = dict(decision.get("arguments") or {})
                argument_error = _tool_argument_error(name, arguments)
                if argument_error:
                    return {
                        "loop_decision_count": decision_count,
                        "loop_budget": budget,
                        "loop_pending_call": {"action": "retry_decision"},
                        "loop_deferred_calls": (),
                        "loop_repair_context": _tool_argument_repair_context(
                            state, repair_context, name, argument_error
                        ),
                    }
                pending = {
                    "action": "tool_call",
                    "tool_call_id": f"json_call_{decision_count}",
                    "tool_name": name,
                    "arguments": arguments,
                    "reason": decision.get("reason"),
                }
                if required_tool and name != required_tool:
                    return _invalid_required_tool_decision(
                        decision_count=decision_count,
                        budget=budget,
                        required_tool=required_tool,
                        selected_tool=name,
                        repair_context=repair_context,
                    )
                _emit_loop_event(
                    state,
                    event_type="decision",
                    status="completed",
                    message=f"Selected tool through structured adapter: {name}",
                    iteration=state.get("loop_iteration", 0) + 1,
                    tool_name=name,
                    payload={
                        "arguments_hash": canonical_action_hash(name, arguments),
                        "reason": decision.get("reason"),
                        "adapter": "structured_json",
                    },
                )
                return {
                    "loop_decision_count": decision_count,
                    "loop_budget": budget,
                    "loop_pending_call": pending,
                    "loop_deferred_calls": (),
                    "loop_repair_context": None,
                    "model_router_provider": response.provider,
                    "model_router_model": response.model,
                }
            if required_tool:
                return _invalid_required_tool_decision(
                    decision_count=decision_count,
                    budget=budget,
                    required_tool=required_tool,
                    selected_tool="",
                    repair_context=repair_context,
                )
            terminal = (
                "model_finished" if decision["action"] == "finish" else "model_requested_fallback"
            )
            _emit_loop_event(
                state,
                event_type="decision",
                status="completed",
                message=str(decision.get("reason") or terminal),
                iteration=state.get("loop_iteration", 0),
                payload={"action": decision["action"]},
            )
            return {
                "loop_decision_count": decision_count,
                "loop_budget": budget,
                "loop_pending_call": decision,
                "loop_deferred_calls": (),
                "loop_terminal_reason": terminal,
                "model_router_provider": response.provider,
                "model_router_model": response.model,
            }
        except Exception as exc:
            _emit_loop_event(
                state,
                event_type="provider_error",
                status="failed",
                message="Loop provider unavailable; using deterministic fallback.",
                iteration=state.get("loop_iteration", 0),
                payload={"error_type": "provider_error", "error": str(exc)[:500]},
            )
            return {
                "loop_decision_count": decision_count,
                "loop_pending_call": None,
                "loop_terminal_reason": "provider_error",
                "model_router_error": str(exc),
            }

    return run


def _pending_from_tool_call(
    raw: dict[str, Any],
    decision_count: int,
    index: int,
) -> dict[str, Any] | None:
    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments") or {}
    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else dict(raw_arguments)
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if name not in {item["function"]["name"] for item in TOOL_DEFINITIONS}:
        return None
    return {
        "action": "tool_call",
        "tool_call_id": str(raw.get("id") or f"call_{decision_count}_{index}"),
        "tool_name": name,
        "arguments": arguments,
        "reason": str(function.get("reason") or "")[:240],
    }


def _tool_argument_error(tool_name: str, arguments: dict[str, Any]) -> str | None:
    definition = next(
        (
            item
            for item in TOOL_DEFINITIONS
            if item["function"]["name"] == tool_name
        ),
        None,
    )
    if definition is None:
        return f"Unknown tool: {tool_name}."
    required = definition["function"].get("parameters", {}).get("required", ())
    missing = [
        name
        for name in required
        if name not in arguments
        or arguments[name] is None
        or (isinstance(arguments[name], str) and not arguments[name].strip())
    ]
    return f"Missing required tool arguments: {', '.join(missing)}." if missing else None


def _tool_argument_repair_context(
    state: AnalysisWorkflowState,
    repair_context: dict[str, Any] | None,
    tool_name: str,
    message: str,
) -> dict[str, Any]:
    context = {
        **(repair_context or {}),
        "error_type": "invalid_arguments",
        "message": message,
    }
    if tool_name == "execute_safe_sql" and not state.get("additional_dataset_ids"):
        context.update(_contract_gap_guidance(state))
        context["deterministic_sql"] = True
    return context


def _invalid_required_tool_decision(
    *,
    decision_count: int,
    budget: dict[str, Any],
    required_tool: str,
    selected_tool: str,
    repair_context: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "loop_decision_count": decision_count,
        "loop_budget": budget,
        "loop_pending_call": {"action": "fallback"},
        "loop_terminal_reason": "contract_repair_rejected",
        "loop_repair_context": {
            **(repair_context or {}),
            "error_type": "invalid_tool_selection",
            "message": (
                f"The contract repair requires {required_tool}; "
                f"{selected_tool or 'no tool'} does not address the remaining gaps."
            ),
        },
    }


def _loop_execute_node(
    repository: DatasetStoreRepository, python_executor: PythonAnalysisExecutor
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        pending = state.get("loop_pending_call") or {}
        tool_name = str(pending.get("tool_name") or "")
        arguments = dict(pending.get("arguments") or {})
        action_hash = canonical_action_hash(tool_name, arguments)
        action_counts = dict(state.get("loop_action_counts", {}))
        action_counts[action_hash] = action_counts.get(action_hash, 0) + 1
        next_iteration = state.get("loop_iteration", 0) + 1
        idempotency_key = canonical_action_hash(
            str(state["run_id"]),
            {"iteration": next_iteration, "tool_name": tool_name, "arguments_hash": action_hash},
        )
        idempotency_artifact_id = uuid5(NAMESPACE_URL, f"datamind-agent-action:{idempotency_key}")
        for evidence in state.get("tool_evidence", ()):
            if evidence.get("action_hash") == action_hash and evidence.get("status") == "succeeded":
                result_hash = evidence.get("result_hash") or canonical_result_hash(
                    evidence.get("result")
                    if isinstance(evidence.get("result"), dict)
                    else None
                )
                duplicate = {
                    "kind": "action",
                    "action_hash": action_hash,
                    "result_hash": result_hash,
                    "evidence_id": evidence.get("evidence_id"),
                    "tool_name": tool_name,
                }
                _emit_loop_event(
                    state,
                    event_type="duplicate_action",
                    status="completed",
                    message="Repeated tool decision reused existing evidence.",
                    iteration=next_iteration,
                    tool_name=tool_name,
                    payload=duplicate,
                )
                return {
                    "loop_last_execution": {
                        **evidence,
                        "cached": True,
                        "duplicate_action": True,
                        "result_hash": result_hash,
                    },
                    "loop_duplicate_action": duplicate,
                    "loop_action_counts": action_counts,
                    "loop_iteration": next_iteration,
                }
        try:
            restored = repository.get_artifact(state["dataset_id"], idempotency_artifact_id)
        except RuntimeError:
            restored = None
        if restored is not None:
            content = restored.get("content") if isinstance(restored.get("content"), dict) else {}
            execution = {
                "tool_name": tool_name,
                "arguments": arguments,
                "action_hash": action_hash,
                "idempotency_key": idempotency_key,
                "idempotency_artifact_id": str(idempotency_artifact_id),
                "status": "succeeded",
                "result": content.get("result"),
                "result_hash": content.get("result_hash"),
                "cached": True,
            }
            return {
                "loop_iteration": next_iteration,
                "loop_last_execution": execution,
                "loop_action_counts": action_counts,
                "executed_nodes": (*state.get("executed_nodes", ()), LOOP_EXECUTE_NODE),
            }
        attempts = dict(state.get("tool_attempts", {}))
        attempts[tool_name] = attempts.get(tool_name, 0) + 1
        if attempts[tool_name] > int(state.get("loop_budget", {}).get("max_tool_attempts") or 3):
            execution = {
                "tool_name": tool_name,
                "arguments": arguments,
                "action_hash": action_hash,
                "idempotency_key": idempotency_key,
                "status": "failed",
                "error_type": "policy_error",
                "error": "Per-tool attempt budget exhausted.",
            }
        else:
            result = _loop_runtime(repository, state, python_executor).execute(tool_name, arguments)
            execution = {
                "tool_name": tool_name,
                "arguments": arguments,
                "action_hash": result.action_hash,
                "idempotency_key": idempotency_key,
                "status": "succeeded" if result.succeeded else "failed",
                "result": result.result,
                "error_type": str(result.error_type) if result.error_type else None,
                "error": result.error,
            }
            if result.succeeded:
                result_hash = canonical_result_hash(result.result)
                execution["result_hash"] = result_hash
                matching_evidence = next(
                    (
                        item
                        for item in state.get("tool_evidence", ())
                        if result_hash
                        and item.get("status") == "succeeded"
                        and (
                            item.get("result_hash")
                            or canonical_result_hash(
                                item.get("result")
                                if isinstance(item.get("result"), dict)
                                else None
                            )
                        )
                        == result_hash
                    ),
                    None,
                )
                if matching_evidence is not None:
                    execution.update(
                        {
                            "duplicate_action": True,
                            "duplicate_kind": "result",
                            "duplicate_evidence_id": matching_evidence.get("evidence_id"),
                        }
                    )
                repository.save_artifact(
                    dataset_id=state["dataset_id"],
                    artifact_type="agent_loop_action",
                    content={
                        "idempotency_key": idempotency_key,
                        "tool_name": tool_name,
                        "result": result.result,
                        "result_hash": result_hash,
                    },
                    file_name=f"{idempotency_key}.json",
                    artifact_id=idempotency_artifact_id,
                    if_absent=True,
                )
                execution["idempotency_artifact_id"] = str(idempotency_artifact_id)
        duplicate = bool(execution.get("duplicate_action"))
        _emit_loop_event(
            state,
            event_type="duplicate_action" if duplicate else "tool_execution",
            status=execution["status"],
            message=(
                "Tool result duplicated existing evidence."
                if duplicate
                else f"{tool_name} {execution['status']}."
            ),
            iteration=next_iteration,
            tool_name=tool_name,
            payload={
                "arguments_hash": action_hash,
                "idempotency_key": idempotency_key,
                "error_type": execution.get("error_type"),
                "result_hash": execution.get("result_hash"),
                "duplicate_kind": execution.get("duplicate_kind"),
                "result_summary": _loop_result_summary(execution.get("result")),
            },
        )
        return {
            "loop_iteration": next_iteration,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "tool_attempts": attempts,
            "loop_action_counts": action_counts,
            "loop_last_execution": execution,
            "loop_duplicate_action": (
                {
                    "kind": execution.get("duplicate_kind"),
                    "action_hash": action_hash,
                    "result_hash": execution.get("result_hash"),
                    "evidence_id": execution.get("duplicate_evidence_id"),
                    "tool_name": tool_name,
                }
                if duplicate
                else None
            ),
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_EXECUTE_NODE),
        }

    return run


def _loop_observe_node(repository: DatasetStoreRepository) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        execution = dict(state.get("loop_last_execution") or {})
        if execution.get("duplicate_action"):
            return {"loop_pending_call": None}
        evidence = list(state.get("tool_evidence", ()))
        evidence_id = f"ev_{len(evidence) + 1}"
        result = execution.get("result")
        artifact_id: str | None = None
        if result and len(json.dumps(result, ensure_ascii=False, default=str)) > 32_000:
            artifact_id = str(
                execution.get("idempotency_artifact_id")
                or repository.save_artifact(
                    dataset_id=state["dataset_id"],
                    artifact_type="agent_loop_evidence",
                    content=result,
                    file_name=f"{state['run_id']}-{evidence_id}.json",
                )
            )
        item = {
            **execution,
            "result": None if artifact_id else result,
            "contract_result": _contract_result(result),
            "claim_result": _claim_result(result),
            "output_fields": _loop_output_fields(result),
            "evidence_id": evidence_id,
            "artifact_id": artifact_id,
            "result_hash": execution.get("result_hash") or canonical_result_hash(result),
            "summary": _loop_result_summary(result),
        }
        evidence.append(item)
        _emit_loop_event(
            state,
            event_type="observation",
            status=str(execution.get("status") or "completed"),
            message=item["summary"],
            iteration=state.get("loop_iteration", 0),
            tool_name=str(execution.get("tool_name") or "") or None,
            payload={
                "evidence_id": evidence_id,
                "artifact_id": artifact_id,
                "summary": item["summary"],
            },
        )
        return {
            "tool_evidence": tuple(evidence),
            "loop_last_execution": item,
            "loop_pending_call": None,
            "loop_duplicate_action": None,
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_OBSERVE_NODE),
        }

    return run


def _loop_verify_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        execution = state.get("loop_last_execution") or {}
        if execution.get("status") == "succeeded":
            result = execution.get("result")
            valid = (isinstance(result, dict) and bool(result)) or bool(
                execution.get("artifact_id")
            )
            contract_covered, coverage_gaps = _loop_execution_contract_status(
                state,
                execution,
            )
            contract_covered = valid and contract_covered
            continue_sequence = (
                bool(state.get("loop_deferred_calls"))
                and valid
                and _loop_budget_exhaustion(state) is None
            )
            duplicate = bool(execution.get("duplicate_action"))
            action_hash = str(execution.get("action_hash") or "")
            duplicate_count = max(
                0,
                int(state.get("loop_action_counts", {}).get(action_hash) or 0) - 1,
            ) if duplicate else 0
            outcome = (
                "sequence_continues"
                if continue_sequence
                else "sufficient"
                if contract_covered
                else "need_more_evidence"
            )
            repeated_duplicate = duplicate and not contract_covered and duplicate_count >= 2
            action = (
                "continue_sequence"
                if continue_sequence
                else "fallback"
                if repeated_duplicate
                else "verified"
                if contract_covered
                else "repairable"
            )
            _emit_loop_event(
                state,
                event_type="verification",
                status="completed" if valid else "failed",
                message=f"Evidence verification: {outcome}.",
                iteration=state.get("loop_iteration", 0),
                tool_name=str(execution.get("tool_name") or "") or None,
                payload={
                    "outcome": outcome,
                    "valid": valid,
                    "contract_covered": contract_covered,
                    "duplicate_action": duplicate,
                    "duplicate_count": duplicate_count,
                    "result_hash": execution.get("result_hash"),
                    "coverage_gaps": coverage_gaps,
                },
            )
            evidence = _annotate_loop_evidence(
                state.get("tool_evidence", ()),
                execution,
                contract_covered=contract_covered,
                coverage_gaps=coverage_gaps,
            )
            return {
                "tool_evidence": evidence,
                "loop_pending_call": {"action": action, "outcome": outcome},
                "loop_duplicate_decision_count": duplicate_count,
                "loop_terminal_reason": (
                    "repeated_duplicate_decision" if repeated_duplicate else None
                ),
                "loop_repair_context": (
                    {
                        "error_type": "duplicate_action" if duplicate else "contract_gap",
                        "message": (
                            "Execute the deterministic contract-completing action; do not inspect "
                            "context or repeat an existing action."
                        ),
                        **(state.get("loop_duplicate_action") or {}),
                        **_contract_gap_guidance(state, coverage_gaps=coverage_gaps),
                    }
                    if not continue_sequence and not contract_covered and not repeated_duplicate
                    else None
                ),
            }
        error_type = str(execution.get("error_type") or LoopErrorType.FATAL_STATE_ERROR)
        fingerprint = error_fingerprint(
            LoopErrorType(error_type), str(execution.get("error") or "")
        )
        failures = dict(state.get("failure_fingerprints", {}))
        key = f"{execution.get('action_hash')}:{fingerprint}"
        failures[key] = failures.get(key, 0) + 1
        repairable = (
            error_type
            in {
                "transient",
                "invalid_arguments",
                "sql_error",
                "python_error",
                "chart_error",
                "validation_error",
                "data_insufficient",
                # The policy boundary has already blocked the unsafe call. Give
                # the model one bounded opportunity to replace it with a legal
                # query instead of discarding all previously verified evidence.
                "policy_error",
            }
            and failures[key] < 2
            and "attempt budget exhausted" not in str(execution.get("error") or "").lower()
        )
        outcome = (
            "repairable"
            if repairable
            else "budget_exhausted"
            if _loop_budget_exhaustion(state)
            else "fallback"
        )
        _emit_loop_event(
            state,
            event_type="verification",
            status="failed",
            message=f"Tool error classified as {error_type}; outcome={outcome}.",
            iteration=state.get("loop_iteration", 0),
            tool_name=str(execution.get("tool_name") or "") or None,
            payload={
                "outcome": outcome,
                "error_type": error_type,
                "fingerprint": fingerprint,
                "repeat_count": failures[key],
            },
        )
        return {
            "failure_fingerprints": failures,
            "loop_pending_call": {"action": outcome},
            "loop_repair_context": {
                "tool_name": execution.get("tool_name"),
                "arguments_hash": execution.get("action_hash"),
                "error_type": error_type,
                "error": execution.get("error"),
                "repeat_count": failures[key],
                **(_contract_gap_guidance(state) if repairable else {}),
            },
        }

    return run


def _loop_execution_contract_status(
    state: AnalysisWorkflowState,
    execution: dict[str, Any],
) -> tuple[bool, dict[str, tuple[str, ...]]]:
    contract = state.get("analysis_contract")
    if contract is None:
        return False, {}
    evidence = list(state.get("tool_evidence", ()))
    if execution and not any(item is execution for item in evidence):
        evidence.append(execution)
    candidates: list[dict[str, tuple[str, ...]]] = [
        analysis_contract_gaps(
            contract=contract,
            sql_result=None,
            python_result=None,
            evidence=tuple(evidence),
        )
    ]
    for item in evidence:
        if item.get("status") != "succeeded":
            continue
        result = item.get("contract_result") or item.get("result")
        if not isinstance(result, dict):
            continue
        sql_result, python_result = _loop_analysis_results(result)
        if sql_result is None and python_result is None:
            continue
        candidates.append(
            analysis_contract_gaps(
                contract=contract,
                sql_result=sql_result,
                python_result=python_result,
            )
        )
    gaps = min(candidates, key=lambda item: sum(len(values) for values in item.values()))
    return not any(gaps.values()), gaps


def _loop_analysis_results(
    result: dict[str, Any],
) -> tuple[SQLAnalysisResponse | None, PythonAnalysisResponse | None]:
    sql_result: SQLAnalysisResponse | None = None
    python_result: PythonAnalysisResponse | None = None
    try:
        if isinstance(result.get("sql_result"), dict):
            sql_result = SQLAnalysisResponse.model_validate(result["sql_result"])
        elif result.get("sql") and isinstance(result.get("rows"), list):
            sql_result = SQLAnalysisResponse.model_validate(
                {
                    "sql": result["sql"],
                    "rows": result["rows"],
                    "explanation": result.get("explanation") or "",
                }
            )
        if isinstance(result.get("python_result"), dict):
            python_result = PythonAnalysisResponse.model_validate(result["python_result"])
    except (TypeError, ValueError):
        return None, None
    return sql_result, python_result


def _contract_gap_guidance(
    state: AnalysisWorkflowState,
    *,
    coverage_gaps: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    contract = state.get("analysis_contract")
    plan = state.get("planned_analysis")
    if contract is None or plan is None:
        return {}
    if plan.route not in {"sql", "hybrid"} and not contract.aggregations:
        return {}
    multi_dataset = bool(state.get("additional_dataset_ids"))
    return {
        "required_tool": "execute_safe_sql",
        "deterministic_sql": (
            plan.route not in {"sql", "hybrid"} and bool(contract.aggregations)
        ),
        "contract_gaps": {
            "metric": contract.metric,
            "dimensions": list(
                (coverage_gaps or {}).get("dimensions") or contract.dimensions
            ),
            "time_field": contract.time_field,
            "aggregations": [
                aggregation.model_dump(mode="json")
                for aggregation in contract.aggregations
                if not coverage_gaps
                or not coverage_gaps.get("aggregations")
                or f"{aggregation.operation}({aggregation.column or '*'})"
                in coverage_gaps["aggregations"]
            ],
            "filters": [
                item.model_dump(mode="json")
                for item in contract.filters
                if not coverage_gaps
                or not coverage_gaps.get("filters")
                or f"{item.column}{item.operator}{item.value}" in coverage_gaps["filters"]
            ],
        },
        "instruction": (
            "Generate one read-only SELECT against dataset that returns the requested "
            "metric at the requested dimension/time grain and applies every contract filter. "
            + (
                "The joined result will be checked against source-grain evidence before reporting."
                if multi_dataset
                else ""
            )
        ),
    }


def _source_grain_repair_guidance(state: AnalysisWorkflowState) -> dict[str, Any]:
    contract = state.get("analysis_contract")
    context = state.get("multi_dataset_context")
    if contract is None or context is None or not contract.metric:
        return {}
    metric = contract.metric
    source_metric = metric.rsplit("__", 1)[-1]
    datasets = (context.primary_dataset, *context.additional_datasets)
    matches = [
        (dataset, column)
        for dataset in datasets
        for column in dataset.columns
        if column == metric or column == source_metric or metric.endswith(f"__{column}")
    ]
    if len(matches) != 1:
        return {}
    dataset, column = matches[0]
    aggregation = next(
        (
            item.operation
            for item in contract.aggregations
            if item.column in {None, metric, column} and item.operation != "count"
        ),
        "sum",
    )
    same_source_dimension = next(
        (
            dimension.rsplit("__", 1)[-1]
            for dimension in contract.dimensions
            if dimension.rsplit("__", 1)[-1] in dataset.columns
        ),
        None,
    )
    arguments: dict[str, Any] = {
        "dataset": dataset.name,
        "metric": column,
        "aggregation": aggregation,
    }
    if same_source_dimension:
        arguments["group_by"] = same_source_dimension
    return {
        "required_tool": "aggregate_source_dataset",
        "required_arguments": arguments,
        "contract_gaps": {
            "metric": metric,
            "dimensions": list(contract.dimensions),
            "grain": list(contract.grain),
        },
        "instruction": (
            f"Recompute {column} from {dataset.name} at its native grain before using "
            "joined rows. Preserve the requested dimension only when it exists on that source."
        ),
    }


def _loop_repair_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        context = state.get("loop_repair_context") or {}
        _emit_loop_event(
            state,
            event_type="repair",
            status="running",
            message=f"Repairing {context.get('error_type') or 'tool'} failure; the next decision must change arguments or tool.",
            iteration=state.get("loop_iteration", 0),
            tool_name=str(context.get("tool_name") or "") or None,
            payload=context,
        )
        return {
            "loop_pending_call": None,
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_REPAIR_NODE),
        }

    return run


def _loop_fallback_node(
    repository: DatasetStoreRepository, python_executor: PythonAnalysisExecutor
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=LOOP_FALLBACK_NODE,
            progress=70,
            message="Using the deterministic legacy fallback.",
        )
        fallback = _loop_runtime(repository, state, python_executor).legacy_fallback()
        components = tuple(
            component
            for component, key in (("sql", "sql_result"), ("python", "python_result"))
            if isinstance(fallback.get(key), dict)
        )
        component_label = " + ".join(item.upper() for item in components) or "evidence"
        evidence = (
            *state.get("tool_evidence", ()),
            {
                "evidence_id": f"ev_{len(state.get('tool_evidence', ())) + 1}",
                "tool_name": "legacy_fallback",
                "action_hash": "legacy_fallback",
                "status": "succeeded",
                "result": fallback,
                "summary": f"Deterministic {component_label} fallback completed.",
            },
        )
        _emit_loop_event(
            state,
            event_type="fallback",
            status="completed",
            message=f"Deterministic {component_label} fallback completed.",
            iteration=state.get("loop_iteration", 0),
            tool_name="legacy_fallback",
            payload={
                "reason": state.get("loop_terminal_reason") or "verification_fallback",
                "analysis_components": components,
            },
        )
        return {
            "tool_evidence": evidence,
            "loop_terminal_reason": state.get("loop_terminal_reason") or "legacy_fallback",
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_FALLBACK_NODE),
        }

    return run


def _loop_finalize_node(repository: DatasetStoreRepository) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        dataframe = _workflow_dataframe(repository, state)
        plan = _require(state.get("planned_analysis"), "Missing analysis plan.")
        evidence_items = list(state.get("tool_evidence", ()))
        resolved_results: dict[int, dict[str, Any]] = {}

        def resolved_result(item: dict[str, Any]) -> dict[str, Any]:
            key = id(item)
            if key not in resolved_results:
                resolved_results[key] = (
                    load_evidence_result(repository, state["dataset_id"], item) or {}
                )
            return resolved_results[key]

        covered_source_aggregates = {
            (
                str(resolved_result(item).get("source_dataset_id") or ""),
                str(resolved_result(item).get("metric") or ""),
                str(resolved_result(item).get("aggregation") or ""),
            )
            for item in evidence_items
            if resolved_result(item).get("native_grain") is True
        }
        source_guard_count = sum(
            1
            for item in evidence_items
            if resolved_result(item).get("native_grain") is True
        )
        runtime = _loop_runtime(repository, state, run_generated_python_analysis)
        for source_result in runtime.required_source_aggregates(
            excluded=covered_source_aggregates
        ):
            signature = (
                str(source_result.get("source_dataset_id") or ""),
                str(source_result.get("metric") or ""),
                str(source_result.get("aggregation") or ""),
            )
            if signature in covered_source_aggregates:
                continue
            evidence_items.append(
                {
                    "evidence_id": f"source_ev_{source_guard_count + 1}",
                    "tool_name": "aggregate_source_dataset",
                    "arguments": {
                        "dataset": source_result.get("source_dataset"),
                        "metric": source_result.get("metric"),
                        "aggregation": source_result.get("aggregation"),
                    },
                    "action_hash": canonical_action_hash(
                        "aggregate_source_dataset",
                        {
                            "dataset": source_result.get("source_dataset_id"),
                            "metric": source_result.get("metric"),
                            "aggregation": source_result.get("aggregation"),
                        },
                    ),
                    "status": "succeeded",
                    "result": source_result,
                    "summary": _loop_result_summary(source_result),
                    "source_guard": True,
                }
            )
            covered_source_aggregates.add(signature)
            source_guard_count += 1
        relationship_items = [
            item
            for item in evidence_items
            if isinstance(resolved_result(item).get("relationships"), list)
        ]
        relationship_guard_count = len(relationship_items)
        relationship_profile: dict[str, Any] = {}
        if relationship_guard_count == 0 and _relationship_analysis_requested(
            state.get("question", "")
        ):
            relationship_profile = runtime.source_relationships()
            if relationship_profile.get("relationships"):
                evidence_items.append(
                    {
                        "evidence_id": "relationship_ev_1",
                        "tool_name": "inspect_source_relationships",
                        "arguments": {},
                        "action_hash": canonical_action_hash(
                            "inspect_source_relationships",
                            {
                                "dataset_ids": sorted(
                                    str(item) for item in runtime.allowed_dataset_ids
                                )
                            },
                        ),
                        "status": "succeeded",
                        "result": relationship_profile,
                        "summary": (
                            "Profiled "
                            f"{len(relationship_profile['relationships'])} source-key relationship(s)."
                        ),
                        "relationship_guard": True,
                    }
                )
                relationship_guard_count = 1
        else:
            existing_relationship = next(
                (
                    resolved_result(item)
                    for item in relationship_items
                ),
                {},
            )
            relationship_profile = (
                existing_relationship if isinstance(existing_relationship, dict) else {}
            )
        sql_evidence: list[tuple[str, SQLAnalysisResponse]] = []
        python_evidence: list[tuple[str, PythonAnalysisResponse]] = []
        charts: list[ChartResponse] = []
        for evidence in evidence_items:
            result = resolved_result(evidence)
            if result.get("sql_result"):
                sql_evidence.append(
                    (
                        str(evidence.get("evidence_id") or f"ev_{len(sql_evidence) + 1}"),
                        SQLAnalysisResponse.model_validate(result["sql_result"]),
                    )
                )
            elif result.get("sql") and isinstance(result.get("rows"), list):
                sql_evidence.append(
                    (
                        str(evidence.get("evidence_id") or f"ev_{len(sql_evidence) + 1}"),
                        SQLAnalysisResponse.model_validate(
                            {
                                "sql": result["sql"],
                                "rows": result["rows"],
                                "explanation": result.get("explanation") or "",
                            }
                        ),
                    )
                )
            if result.get("python_result"):
                python_evidence.append(
                    (
                        str(evidence.get("evidence_id") or f"ev_{len(python_evidence) + 1}"),
                        PythonAnalysisResponse.model_validate(result["python_result"]),
                    )
                )
            if result.get("chart"):
                charts.append(ChartResponse.model_validate(result["chart"]))
        sql_result = _combined_loop_sql_result(sql_evidence)
        python_result = _combined_loop_python_result(python_evidence)
        contract = _require(
            state.get("analysis_contract"),
            "Planner did not produce an analysis contract.",
        )
        analysis_dataframe = _apply_plan_filters(dataframe, plan.filters)
        preflight = verify_statistical_analysis(
            contract=contract,
            profile=_require(state.get("profile"), "Missing dataset profile."),
            dataframe=analysis_dataframe,
            findings=(),
            evidence=tuple(evidence_items),
            sql_result=sql_result,
            python_result=python_result,
            multi_dataset_context=state.get("multi_dataset_context"),
        )
        terminal = state.get("loop_terminal_reason") or "evidence_sufficient"
        executed_tools = tuple(
            dict.fromkeys(
                str(item.get("tool_name"))
                for item in evidence_items
                if item.get("status") == "succeeded" and item.get("tool_name")
            )
        )
        fallback_sql = any(
            item.get("tool_name") == "legacy_fallback"
            and isinstance(resolved_result(item).get("sql_result"), dict)
            for item in evidence_items
        )
        fallback_python = any(
            item.get("tool_name") == "legacy_fallback"
            and isinstance(resolved_result(item).get("python_result"), dict)
            for item in evidence_items
        )
        components = tuple(
            component
            for component, executed in (
                (
                    "sql",
                    "execute_safe_sql" in executed_tools
                    or "execute_semantic_query" in executed_tools
                    or fallback_sql,
                ),
                (
                    "python",
                    "execute_python_analysis" in executed_tools or fallback_python,
                ),
            )
            if executed
        )
        summary = {
            "iterations": state.get("loop_iteration", 0),
            "decisions": state.get("loop_decision_count", 0),
            "tool_calls": state.get("tool_call_count", 0),
            "successful_tools": sum(
                1 for item in evidence_items if item.get("status") == "succeeded"
            ),
            "failed_tools": sum(
                1 for item in evidence_items if item.get("status") == "failed"
            ),
            "terminal_reason": terminal,
            "adversarial_repairs": state.get("adversarial_repair_count", 0),
            "source_aggregate_guards": source_guard_count,
            "source_relationship_guards": relationship_guard_count,
            "source_relationship_count": len(
                relationship_profile.get("relationships") or []
            ),
            "source_relationship_risk_count": int(
                relationship_profile.get("risk_count") or 0
            ),
            "executed_tools": executed_tools,
            "analysis_components": components,
        }
        _emit_loop_event(
            state,
            event_type="statistical_preflight",
            status="failed" if preflight.requires_replan else "completed",
            message=preflight.summary,
            iteration=state.get("loop_iteration", 0),
            payload={
                "failed_checks": [
                    check.code for check in preflight.checks if check.status == "failed"
                ]
            },
        )
        _emit_loop_event(
            state,
            event_type="loop_finalize",
            status="failed" if preflight.requires_replan else "completed",
            message=(
                preflight.summary
                if preflight.requires_replan
                else f"Autonomous loop finalized: {terminal}."
            ),
            iteration=state.get("loop_iteration", 0),
            payload={
                **summary,
                "preflight_status": preflight.status,
                "failed_checks": [
                    check.code for check in preflight.checks if check.status == "failed"
                ],
                "analysis_snapshot": {
                    "plan": _planned_analysis_payload(plan),
                    "contract": contract.model_dump(mode="json"),
                    "sql": (
                        {
                            "sql": sql_result.sql,
                            "row_count": len(sql_result.rows),
                            "columns": list(sql_result.rows[0]) if sql_result.rows else [],
                            "result_hash": canonical_result_hash(
                                {"rows": list(sql_result.rows[:5])}
                            ),
                            "explanation": sql_result.explanation,
                        }
                        if sql_result is not None
                        else None
                    ),
                    "charts": [
                        chart.model_dump(mode="json", exclude={"data"})
                        for chart in charts[:10]
                    ],
                    "verification": preflight.model_dump(mode="json"),
                    "evidence": [
                        {
                            "evidence_id": item.get("evidence_id"),
                            "tool_name": item.get("tool_name"),
                            "status": item.get("status"),
                            "summary": item.get("summary"),
                        }
                        for item in evidence_items[:24]
                    ],
                },
            },
        )
        return {
            "sql_result": sql_result,
            "python_result": python_result,
            "rounds": (),
            "report_charts": tuple(charts),
            "tool_evidence": tuple(evidence_items),
            "loop_summary": summary,
            "loop_preflight_verification": preflight,
            "statistical_verification": preflight,
            "loop_terminal_reason": terminal,
            "sql_source": _loop_component_source(evidence_items, "sql") if sql_result else "not_run",
            "python_source": _loop_component_source(evidence_items, "python") if python_result else "not_run",
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_FINALIZE_NODE),
        }

    return run


def _combined_loop_sql_result(
    evidence: list[tuple[str, SQLAnalysisResponse]],
) -> SQLAnalysisResponse | None:
    if not evidence:
        return None
    if len(evidence) == 1:
        return evidence[0][1]
    rows: list[dict[str, Any]] = []
    statements: list[str] = []
    for query_index, (evidence_id, result) in enumerate(evidence, start=1):
        statements.append(f"-- {evidence_id}\n{result.sql.rstrip(';')};")
        for row in result.rows:
            rows.append(
                {
                    **row,
                    "evidence_id": evidence_id,
                    "query_index": query_index,
                }
            )
            if len(rows) >= 1000:
                break
        if len(rows) >= 1000:
            break
    return SQLAnalysisResponse(
        sql="\n\n".join(statements),
        rows=tuple(rows),
        explanation=(
            f"Combined {len(evidence)} verified agent-loop SQL results; "
            "evidence_id and query_index preserve their provenance."
        ),
    )


def _combined_loop_python_result(
    evidence: list[tuple[str, PythonAnalysisResponse]],
) -> PythonAnalysisResponse | None:
    if not evidence:
        return None
    if len(evidence) == 1:
        return evidence[0][1]

    statistics: dict[str, Any] = {}
    evidence_statistics: list[dict[str, Any]] = []
    insights: list[str] = []
    charts: list[ChartResponse] = []
    chart_keys: set[tuple[str, str, str]] = set()
    text_analysis = []
    for evidence_id, result in evidence:
        statistics.update(result.statistics)
        evidence_statistics.append(
            {"evidence_id": evidence_id, "statistics": result.statistics}
        )
        for insight in result.insights:
            if insight and insight not in insights:
                insights.append(insight)
        for chart in result.charts:
            key = (chart.title, chart.chart_type, json.dumps(chart.spec, sort_keys=True))
            if key not in chart_keys:
                chart_keys.add(key)
                charts.append(chart)
        text_analysis.extend(result.text_analysis)
    statistics["agent_loop_evidence"] = evidence_statistics
    contexts = [result.execution_context for _, result in evidence]
    execution_context = (
        contexts[0]
        if contexts
        and contexts[0] is not None
        and all(context == contexts[0] for context in contexts)
        else None
    )
    return PythonAnalysisResponse(
        statistics=statistics,
        insights=tuple(insights),
        charts=tuple(charts),
        text_analysis=tuple(text_analysis),
        execution_context=execution_context,
    )


def _loop_component_source(evidence: list[dict[str, Any]], component: str) -> str:
    tool_name = {
        "sql": "execute_safe_sql",
        "python": "execute_python_analysis",
    }[component]
    sources = {
        str(item.get("tool_name") or "")
        for item in evidence
        if item.get("status") == "succeeded" and item.get("tool_name")
    }
    if "legacy_fallback" in sources:
        return "legacy_fallback"
    if tool_name in sources:
        return "agent_loop"
    if component == "sql" and "execute_semantic_query" in sources:
        return "semantic"
    return "evidence_projection"


def _loop_adversarial_repair_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        issues = [
            item.model_dump(mode="json")
            for item in state.get("validation_issues", ())
            if item.severity.lower() in {"high", "critical", "error"}
        ]
        verification = state.get("statistical_verification")
        failed_checks = (
            [check for check in verification.checks if check.status == "failed"]
            if verification is not None
            else []
        )
        if any(check.code == "join_grain" for check in failed_checks):
            repair_guidance = _source_grain_repair_guidance(state)
        elif any(check.code == "request_coverage" for check in failed_checks):
            repair_guidance = _contract_gap_guidance(state)
        else:
            repair_guidance = {}
        _emit_loop_event(
            state,
            event_type="adversarial_repair",
            status="running",
            message="Adversarial validation requested one final evidence repair.",
            iteration=state.get("loop_iteration", 0),
            payload={"issues": issues[:10]},
        )
        return {
            "adversarial_repair_count": state.get("adversarial_repair_count", 0) + 1,
            "loop_repair_context": {
                "error_type": "validation_error",
                "message": "Repair the failed statistical checks with contract-completing evidence.",
                "issues": issues[:10],
                "failed_checks": [
                    {
                        "code": check.code,
                        "message": check.message,
                        "details": check.details,
                    }
                    for check in failed_checks
                ],
                **repair_guidance,
            },
            "loop_pending_call": None,
            "loop_terminal_reason": None,
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_ADVERSARIAL_REPAIR_NODE),
        }

    return run


def _public_loop_budget(budget: dict[str, Any], state: AnalysisWorkflowState) -> dict[str, Any]:
    return {
        "tool_calls_remaining": max(
            0, int(budget.get("max_tool_calls") or 12) - state.get("tool_call_count", 0)
        ),
        "decisions_remaining": max(
            0, int(budget.get("max_decisions") or 16) - state.get("loop_decision_count", 0)
        ),
        "tokens_remaining": max(
            0, int(budget.get("max_tokens") or 50_000) - int(budget.get("used_tokens") or 0)
        ),
    }


def _parse_loop_text_decision(content: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(str(content or ""))
    except json.JSONDecodeError:
        payload = {}
    action = str(payload.get("action") or "fallback")
    if action not in {"finish", "fallback", "tool_call"}:
        action = "fallback"
    result = {
        "action": action,
        "reason": str(payload.get("reason") or "No valid tool decision was returned.")[:500],
    }
    if action == "tool_call":
        result["tool_name"] = str(payload.get("tool_name") or "")
        result["arguments"] = (
            payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        )
    return result


def _is_tool_capability_error(exc: Exception) -> bool:
    if isinstance(exc, (TypeError, NotImplementedError)):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "tool calls unsupported",
            "tools unsupported",
            "tool_choice unsupported",
            "unknown field: tools",
            # This helper is only used around the native-tools attempt. Some
            # OpenAI-compatible Kimi endpoints report an undifferentiated 400
            # for an unsupported tool schema; retry once through JSON mode.
            "kimi api error 400",
        )
    )


def _planned_analysis_payload(plan: PlannedAnalysis) -> dict[str, Any]:
    return {
        "route": plan.route,
        "metric": plan.metric_column,
        "dimension": plan.category_column,
        "time": plan.time_column,
        "time_grain": plan.time_grain,
        "aggregations": [
            item.model_dump(mode="json") for item in plan.aggregations
        ],
        "filters": [item.model_dump(mode="json") for item in plan.filters],
        "requested_dimensions": list(plan.requested_dimensions),
        "derived_metrics": list(plan.derived_metrics),
        "steps": list(plan.steps),
    }


def _loop_result_summary(result: Any) -> str:
    if not isinstance(result, dict):
        return "No structured result was produced."
    if isinstance(result.get("rows"), list):
        return f"Produced {len(result['rows'])} bounded row(s)."
    if result.get("python_result"):
        return "Produced sandboxed Python statistics, insights and charts."
    if result.get("sql_result"):
        return "Produced deterministic SQL and Python fallback evidence."
    return "Produced structured evidence: " + ", ".join(
        str(key) for key in islice(result, 6)
    )


def _emit_loop_event(
    state: AnalysisWorkflowState,
    *,
    event_type: str,
    status: str,
    message: str,
    iteration: int,
    payload: dict[str, Any],
    tool_name: str | None = None,
) -> None:
    _notify_node_event(
        state,
        {
            "node": "agent_loop",
            "status": status,
            "message": message,
            "attempt": 0,
            "event_type": event_type,
            "iteration": iteration,
            "tool_name": tool_name,
            "payload": payload,
        },
    )


@dataclass(frozen=True)
class PythonCodeExecutionResult:
    result: PythonAnalysisResponse | None
    code: str | None
    error: str | None
    attempts: tuple[PythonCodeAttemptResponse, ...]
    provider: str | None = None
    model: str | None = None


def _execute_generated_python_with_repairs(
    *,
    model_router: AnalysisModelRouter,
    agent: str,
    question: str,
    dataframe: pd.DataFrame,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None,
    metadata: dict[str, object],
    python_executor: PythonAnalysisExecutor,
    hypothesis: str | None = None,
) -> PythonCodeExecutionResult:
    dataframe = _apply_plan_filters(dataframe, planned_analysis.filters)
    attempts: list[PythonCodeAttemptResponse] = []
    provider: str | None = None
    model: str | None = None
    last_code: str | None = None
    last_error: str | None = None

    for attempt_number in range(1, 4):
        try:
            response = model_router.complete(
                messages=(
                    _python_messages(
                        question=question,
                        profile=profile,
                        planned_analysis=planned_analysis,
                        sql_result=sql_result,
                        multi_dataset_context=multi_dataset_context,
                    )
                    if agent == "python" and attempt_number == 1
                    else _python_chart_messages(
                        question=question,
                        hypothesis=hypothesis,
                        profile=profile,
                        planned_analysis=planned_analysis,
                        sql_result=sql_result,
                        multi_dataset_context=multi_dataset_context,
                    )
                    if agent in {"python_charts", "round_python_charts"} and attempt_number == 1
                    else _round_python_messages(
                        question=question,
                        hypothesis=hypothesis or question,
                        profile=profile,
                        planned_analysis=planned_analysis,
                        sql_result=sql_result,
                        multi_dataset_context=multi_dataset_context,
                    )
                    if agent == "round_python" and attempt_number == 1
                    else _python_repair_messages(
                        question=question,
                        profile=profile,
                        planned_analysis=planned_analysis,
                        sql_result=sql_result,
                        attempts=tuple(attempts),
                        phase=agent,
                        multi_dataset_context=multi_dataset_context,
                        hypothesis=hypothesis,
                    )
                ),
                temperature=0.0,
                max_tokens=3200,
                metadata={"agent": agent, "attempt": attempt_number, **metadata},
            )
        except Exception as exc:
            return PythonCodeExecutionResult(
                result=None,
                code=last_code,
                error=str(exc),
                attempts=tuple(attempts),
                provider=provider,
                model=model,
            )
        provider = response.provider
        model = response.model
        code = _extract_python_code(response.content)
        last_code = code
        try:
            generated_result = python_executor(code, dataframe)
            attempts.append(
                PythonCodeAttemptResponse(
                    attempt=attempt_number,
                    phase=agent,
                    status="succeeded",
                    code=code,
                    error=None,
                    provider=response.provider,
                    model=response.model,
                )
            )
            return PythonCodeExecutionResult(
                result=generated_result,
                code=code,
                error=None,
                attempts=tuple(attempts),
                provider=response.provider,
                model=response.model,
            )
        except Exception as exc:
            last_error = str(exc)
            attempts.append(
                PythonCodeAttemptResponse(
                    attempt=attempt_number,
                    phase=agent,
                    status="failed",
                    code=code,
                    error=last_error,
                    provider=response.provider,
                    model=response.model,
                )
            )

    return PythonCodeExecutionResult(
        result=None,
        code=last_code,
        error=last_error or "LLM Python code failed after 3 attempts.",
        attempts=tuple(attempts),
        provider=provider,
        model=model,
    )


def _combine_python_generated_code(
    *,
    statistics_code: str | None,
    chart_code: str | None,
) -> str | None:
    if statistics_code and chart_code:
        return f"{statistics_code}\n\n# --- DataMind chart generation phase ---\n{chart_code}"
    return statistics_code or chart_code


class SQLValidationResult(TypedDict):
    ok: bool
    sql: str
    message: str | None


_FORBIDDEN_SQL_KEYWORDS = (
    "drop",
    "delete",
    "update",
    "insert",
    "attach",
    "copy",
    "create",
    "alter",
    "truncate",
    "merge",
    "replace",
    "vacuum",
    "pragma",
    "call",
)


def _extract_sql(content: str) -> str:
    text = content.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return text


def _extract_python_code(content: str) -> str:
    text = content.strip()
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        raise ValueError("Model did not return Python code.")
    return text


def _merge_python_results(
    *,
    generated_result: PythonAnalysisResponse,
    baseline_result: PythonAnalysisResponse,
) -> PythonAnalysisResponse:
    statistics = dict(baseline_result.statistics)
    statistics["model_generated"] = generated_result.statistics

    insights: list[str] = []
    for insight in (*generated_result.insights, *baseline_result.insights):
        normalized_insight = _localize_python_insight(insight)
        if normalized_insight and normalized_insight not in insights:
            insights.append(normalized_insight)

    charts = (
        (*generated_result.charts, *baseline_result.charts)
        if generated_result.charts
        else baseline_result.charts
    )
    return PythonAnalysisResponse(
        statistics=statistics,
        insights=tuple(insights),
        charts=tuple(charts),
        text_analysis=(*generated_result.text_analysis, *baseline_result.text_analysis),
    )


def _localize_python_insight(insight: str) -> str:
    text = str(insight).strip()
    replacements = {
        "The dataset contains": "数据集包含",
        "Average text length is": "平均文本长度为",
        "The most frequent keywords are": "最高频关键词为",
        "The data does not contain": "数据中不包含",
        "To perform the requested analysis": "要完成该分析",
        "To proceed with the requested analysis": "要继续该分析",
        "This workbook and related data is provided by": "该工作簿及相关数据由",
        "Dataset contains": "数据集包含",
        "Text length ranges from": "文本长度范围为",
        "Most frequent keyword is": "最高频关键词为",
        "Texts are primarily": "文本主要是",
        "No sales, profit, or customer/product/region data available": "当前没有可用于该问题的销售额、利润、客户、产品或地区数据",
        "Without numeric metrics": "缺少数值指标",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("rows", "行").replace("columns", "列")
    text = text.replace("characters", "字符").replace("records", "条记录")
    text = text.replace("appearing", "出现")
    text = text.replace("times", "次")
    return text.replace(" and ", " 和 ")


def _validate_dataset_select_sql(sql: str) -> SQLValidationResult:
    normalized = _strip_sql_comments(sql).strip()
    if not normalized:
        return {"ok": False, "sql": "", "message": "Model did not return SQL."}
    if normalized.count(";") > 1 or (";" in normalized and not normalized.endswith(";")):
        return {"ok": False, "sql": normalized, "message": "Only one SQL statement is allowed."}
    normalized = normalized.removesuffix(";").strip()
    lowered = normalized.lower()
    if not lowered.startswith("select "):
        return {"ok": False, "sql": normalized, "message": "Only SELECT statements are allowed."}
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            return {
                "ok": False,
                "sql": normalized,
                "message": f"Forbidden SQL keyword was used: {keyword}.",
            }

    from_segments = re.findall(
        r"\bfrom\b\s+(.*?)(?=\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|\bhaving\b|\bqualify\b|\bunion\b|$)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if any("," in segment for segment in from_segments):
        return {
            "ok": False,
            "sql": normalized,
            "message": "Comma joins are not allowed; SQL can only query dataset.",
        }

    table_refs = re.findall(
        r"\b(?:from|join)\s+([\"`]?[\w.]+[\"`]?)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not table_refs:
        return {
            "ok": False,
            "sql": normalized,
            "message": "SQL must read from the dataset table.",
        }
    invalid_tables = [ref for ref in table_refs if ref.strip('"`').lower() != "dataset"]
    if invalid_tables:
        return {
            "ok": False,
            "sql": normalized,
            "message": f"SQL can only query the dataset table: {', '.join(invalid_tables)}.",
        }
    return {"ok": True, "sql": normalized, "message": None}


def _strip_sql_comments(sql: str) -> str:
    without_line_comments = re.sub(r"--.*?(?=\r?\n|$)", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", without_line_comments, flags=re.DOTALL)


def _execute_dataset_sql(
    dataframe: pd.DataFrame,
    sql: str,
    *,
    explanation: str,
) -> SQLAnalysisResponse:
    import duckdb

    connection = duckdb.connect(":memory:")
    try:
        connection.register("dataset", dataframe)
        rows = _records(connection.execute(sql).fetchdf())
    finally:
        connection.close()
    return SQLAnalysisResponse(sql=sql, rows=tuple(rows), explanation=explanation)


def _parse_model_plan(content: str, *, fallback: PlannedAnalysis) -> PlannedAnalysis:
    payload = _extract_json_object(content)
    if payload is None:
        raise ValueError("Model Router planner did not return JSON.")
    route = str(payload.get("route") or fallback.route).lower()
    steps_payload = payload.get("steps")
    steps = (
        tuple(str(step) for step in steps_payload if str(step).strip())
        if isinstance(steps_payload, list)
        else fallback.steps
    )
    return PlannedAnalysis(
        route=route,
        category_column=_optional_str(payload.get("category_column")),
        metric_column=_optional_str(payload.get("metric_column")) or fallback.metric_column,
        time_column=_optional_str(payload.get("time_column")) or fallback.time_column,
        steps=steps or fallback.steps,
        aggregations=fallback.aggregations,
        filters=fallback.filters,
        requested_dimensions=fallback.requested_dimensions,
        derived_metrics=fallback.derived_metrics,
        time_grain=fallback.time_grain,
    )


def _validate_plan_harness(
    *,
    planned_analysis: PlannedAnalysis,
    profile: DatasetProfileResponse,
    finding_ref: str,
) -> tuple[ValidationIssueResponse, ...]:
    return validate_analysis_plan(
        finding_ref=finding_ref,
        route=planned_analysis.route,
        category_column=planned_analysis.category_column,
        metric_column=planned_analysis.metric_column,
        time_column=planned_analysis.time_column,
        steps=planned_analysis.steps,
        available_columns={column.name for column in profile.columns},
    )


def _plan_validation_summary(issues: tuple[ValidationIssueResponse, ...]) -> str:
    return "; ".join(issue.issue for issue in issues)


def _parse_model_framework(
    content: str,
    *,
    fallback: AnalysisFrameworkResponse,
    profile: DatasetProfileResponse,
    question: str,
) -> AnalysisFrameworkResponse:
    payload = _extract_json_object(content)
    if payload is None:
        raise ValueError("Model Router design_framework did not return JSON.")
    available_columns = {column.name for column in profile.columns}
    dimensions = tuple(
        item
        for item in _string_list(payload.get("candidate_dimensions"))
        if item in available_columns
    )
    metrics = tuple(
        item for item in _string_list(payload.get("candidate_metrics")) if item in available_columns
    )
    routes = tuple(
        route
        for route in (item.lower() for item in _string_list(payload.get("likely_routes")))
        if route in {"sql", "python", "hybrid"}
    )
    return AnalysisFrameworkResponse(
        business_question=str(payload.get("business_question") or question),
        candidate_dimensions=dimensions or fallback.candidate_dimensions,
        candidate_metrics=metrics or fallback.candidate_metrics,
        likely_routes=routes or fallback.likely_routes,
        initial_hypotheses=tuple(_string_list(payload.get("initial_hypotheses")))
        or fallback.initial_hypotheses,
        risk_notes=tuple(_string_list(payload.get("risk_notes"))) or fallback.risk_notes,
        dimensions=dimensions or fallback.dimensions,
        key_questions=tuple(_string_list(payload.get("key_questions"))) or fallback.key_questions,
        success_criteria=str(payload.get("success_criteria") or fallback.success_criteria),
    )


def _execute_analysis_round(
    *,
    dataset_id: UUID,
    question: str,
    dataframe: pd.DataFrame,
    profile: DatasetProfileResponse,
    round_number: int,
    hypothesis: str,
    previous_rounds: tuple[AnalysisRoundResponse, ...],
    total_rounds: int,
    model_router: AnalysisModelRouter | None,
    multi_dataset_context: MultiDatasetProfileResponse | None,
    fanout_mode: str,
    python_executor: PythonAnalysisExecutor,
) -> tuple[
    AnalysisRoundResponse,
    PythonAnalysisResponse,
    str | None,
    str | None,
    str | None,
    tuple[ValidationIssueResponse, ...],
]:
    round_plan = _plan(f"{question}\n本轮假设：{hypothesis}", profile)
    plan_validation_issues: tuple[ValidationIssueResponse, ...] = ()
    model_router_provider: str | None = None
    model_router_model: str | None = None
    model_router_error: str | None = None
    if model_router is not None:
        try:
            response = model_router.complete(
                messages=_round_plan_messages(
                    question=question,
                    hypothesis=hypothesis,
                    profile=profile,
                    previous_rounds=previous_rounds,
                    fallback_plan=round_plan,
                ),
                temperature=0.1,
                max_tokens=900,
                metadata={
                    "agent": "round_plan",
                    "round_number": round_number,
                    "fanout_mode": fanout_mode,
                    "dataset_id": str(dataset_id),
                },
            )
            model_plan = _parse_model_plan(response.content, fallback=round_plan)
            plan_validation_issues = _validate_plan_harness(
                planned_analysis=model_plan,
                profile=profile,
                finding_ref=f"round_plan_{round_number}",
            )
            if plan_validation_issues:
                raise ValueError(_plan_validation_summary(plan_validation_issues))
            round_plan = _sanitize_model_plan(
                model_plan,
                profile,
                allow_time=round_plan.time_column is not None,
            )
            model_router_provider = response.provider
            model_router_model = response.model
        except Exception as exc:
            model_router_error = str(exc)

    round_sql_result: SQLAnalysisResponse | None = None
    if round_plan.route in {"sql", "hybrid"}:
        round_sql_result = _run_sql(dataframe, round_plan)
    round_python_result = _run_python(
        dataframe,
        round_plan,
        round_sql_result,
        question=question,
    )
    round_python_source = "rules"
    round_python_generated_code: str | None = None
    round_python_execution_error: str | None = None
    round_python_attempts: tuple[PythonCodeAttemptResponse, ...] = ()
    if model_router is not None:
        execution = _execute_generated_python_with_repairs(
            model_router=model_router,
            agent="round_python",
            question=question,
            hypothesis=hypothesis,
            dataframe=dataframe,
            profile=profile,
            planned_analysis=round_plan,
            sql_result=round_sql_result,
            multi_dataset_context=multi_dataset_context,
            metadata={
                "round_number": round_number,
                "fanout_mode": fanout_mode,
                "dataset_id": str(dataset_id),
            },
            python_executor=python_executor,
        )
        chart_execution: PythonCodeExecutionResult | None = None
        if execution.result is not None:
            chart_execution = _execute_generated_python_with_repairs(
                model_router=model_router,
                agent="round_python_charts",
                question=question,
                hypothesis=hypothesis,
                dataframe=dataframe,
                profile=profile,
                planned_analysis=round_plan,
                sql_result=round_sql_result,
                multi_dataset_context=multi_dataset_context,
                metadata={
                    "round_number": round_number,
                    "fanout_mode": fanout_mode,
                    "dataset_id": str(dataset_id),
                },
                python_executor=python_executor,
            )
        round_python_attempts = (
            *execution.attempts,
            *(chart_execution.attempts if chart_execution else ()),
        )
        round_python_generated_code = _combine_python_generated_code(
            statistics_code=execution.code,
            chart_code=chart_execution.code if chart_execution else None,
        )
        round_python_execution_error = execution.error
        model_router_provider = (
            (chart_execution.provider if chart_execution else None)
            or execution.provider
            or model_router_provider
        )
        model_router_model = (
            (chart_execution.model if chart_execution else None)
            or execution.model
            or model_router_model
        )
        if execution.result is not None:
            generated_result = execution.result
            if chart_execution and chart_execution.result is not None:
                generated_result = PythonAnalysisResponse(
                    statistics=execution.result.statistics,
                    insights=execution.result.insights,
                    charts=chart_execution.result.charts,
                    text_analysis=execution.result.text_analysis,
                )
            elif chart_execution and chart_execution.error:
                model_router_error = chart_execution.error
            round_python_result = _merge_python_results(
                generated_result=generated_result,
                baseline_result=round_python_result,
            )
            round_python_source = "model_router"
            round_python_execution_error = None
        elif round_python_execution_error:
            model_router_error = round_python_execution_error

    round_reflection_text = _round_reflection_text(
        hypothesis=hypothesis,
        sql_result=round_sql_result,
        python_result=round_python_result,
    )
    round_response = AnalysisRoundResponse(
        round_number=round_number,
        hypothesis=AnalysisHypothesisResponse(
            statement=hypothesis,
            judgment_criteria="Use only uploaded dataset columns and computed round results.",
            expected_direction="unknown",
        ),
        plan=AnalysisRoundPlanResponse(
            round_number=round_number,
            analytical_step=_analytical_step_for_round_plan(round_number, round_plan),
            question=question,
            route=round_plan.route,
            metric_column=round_plan.metric_column,
            category_column=round_plan.category_column,
            time_column=round_plan.time_column,
        ),
        reflection=AnalysisReflectionResponse(
            insight_text=round_reflection_text,
            impact_pct=100 if round_number == 1 else 0,
            has_insight=bool(round_reflection_text),
            decision="CONTINUE" if round_number < total_rounds else "STOP",
            data_source="round.execution_result",
        ),
        execution_result=_round_execution_payload(
            sql_result=round_sql_result,
            python_result=round_python_result,
            python_source=round_python_source,
            python_generated_code=round_python_generated_code,
            python_execution_error=round_python_execution_error,
            python_attempts=round_python_attempts,
        )
        | {
            "fanout_mode": fanout_mode,
            "fanout_group": "rounds_2_3" if "fanout" in fanout_mode else "round_1",
            "plan_validation_issues": [
                issue.model_dump(mode="json") for issue in plan_validation_issues
            ],
        },
        charts=round_python_result.charts,
        validation_status="warning" if plan_validation_issues else "passed",
    )
    return (
        round_response,
        round_python_result,
        model_router_provider,
        model_router_model,
        model_router_error,
        plan_validation_issues,
    )


def _round_hypotheses(
    *,
    question: str,
    framework: AnalysisFrameworkResponse | None,
    python_result: PythonAnalysisResponse,
) -> tuple[str, ...]:
    candidates: list[str] = []
    if framework:
        candidates.extend(framework.initial_hypotheses)
        candidates.extend(framework.key_questions)
    candidates.extend(python_result.insights)
    candidates.append(question)
    unique: list[str] = []
    for candidate in candidates:
        text = str(candidate).strip()
        if text and text not in unique:
            unique.append(text)
    return tuple(unique[:3] or (question,))


def _round_output(
    *,
    round_item: AnalysisRoundResponse,
    python_result: PythonAnalysisResponse,
    provider: str | None,
    model: str | None,
    error: str | None,
    plan_issues: tuple[ValidationIssueResponse, ...],
) -> dict[str, Any]:
    return {
        "round": round_item,
        "python_result": python_result,
        "provider": provider,
        "model": model,
        "error": error,
        "plan_issues": plan_issues,
    }


def _prepare_multimodal_inputs(
    multimodal_inputs: tuple[MultimodalInputResponse, ...],
) -> tuple[MultimodalInputResponse, ...]:
    prepared: list[MultimodalInputResponse] = []
    for item in multimodal_inputs:
        media_type = item.media_type or ""
        is_pdf = item.kind == "pdf_page" or media_type == "application/pdf"
        if not is_pdf or not item.data_url:
            processing_status = item.processing_status
            if not processing_status:
                processing_status = (
                    "native_image_payload"
                    if item.data_url and media_type.startswith("image/")
                    else "text_context"
                )
            prepared.append(
                item.model_copy(
                    update={
                        "description": _truncate_text(item.description, 8000),
                        "processing_status": processing_status,
                    }
                )
            )
            continue

        extracted_text, extraction_note = _extract_pdf_text_from_data_url(item.data_url)
        description_parts = [_truncate_text(item.description, 2000)]
        processing_status = "pdf_text_unavailable"
        text_excerpt = None
        if extracted_text:
            processing_status = "pdf_text_extracted"
            text_excerpt = _truncate_text(extracted_text, 1200)
            description_parts.append(
                f"PDF text excerpt extracted for analysis:\n{_truncate_text(extracted_text, 6000)}"
            )
        elif extraction_note:
            description_parts.append(f"PDF extraction note: {extraction_note}")
        prepared.append(
            item.model_copy(
                update={
                    "description": "\n\n".join(part for part in description_parts if part),
                    "data_url": None,
                    "processing_status": processing_status,
                    "text_excerpt": text_excerpt,
                }
            )
        )
    return tuple(prepared)


def _extract_pdf_text_from_data_url(data_url: str) -> tuple[str | None, str | None]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return (
            None,
            "pypdf is not installed, so the PDF was kept as source metadata only. "
            "Install pypdf to extract PDF page text for Kimi report context.",
        )

    if "," not in data_url:
        return None, "Invalid PDF data URL."
    try:
        encoded_payload = data_url.split(",", 1)[1]
        pdf_bytes = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError):
        return None, "Invalid base64 PDF payload."

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_text: list[str] = []
        for index, page in enumerate(reader.pages[:3], start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_text.append(f"[Page {index}]\n{text}")
        if not page_text:
            return None, "No extractable text was found in the first 3 PDF pages."
        return "\n\n".join(page_text), None
    except Exception as exc:  # pragma: no cover - parser-specific failures are data dependent.
        return None, f"PDF text extraction failed: {exc}"


def _analytical_step_for_round_plan(round_number: int, plan: PlannedAnalysis) -> str:
    if round_number == 1:
        return f"建立基础证据：使用 {plan.route} 验证主要假设。"
    if round_number == 2:
        return f"探索补充维度：使用 {plan.route} 检查结构性差异。"
    return f"收敛验证：使用 {plan.route} 复核异常、分布或数据缺口。"


def _round_reflection_text(
    *,
    hypothesis: str,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse,
) -> str:
    if python_result.insights:
        return f"围绕“{hypothesis}”，本轮发现：{python_result.insights[0]}"
    if sql_result is not None:
        return f"围绕“{hypothesis}”，本轮 SQL 返回 {len(sql_result.rows)} 行结果。"
    return f"围绕“{hypothesis}”，本轮未发现足够强的量化证据。"


def _round_execution_payload(
    *,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse,
    python_source: str = "rules",
    python_generated_code: str | None = None,
    python_execution_error: str | None = None,
    python_attempts: tuple[PythonCodeAttemptResponse, ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_source": python_source,
        "python_generated_code": python_generated_code,
        "python_execution_error": python_execution_error,
        "python_attempts": [attempt.model_dump(mode="json") for attempt in python_attempts],
        "python_statistics": python_result.statistics,
        "python_insights": list(python_result.insights),
        "chart_count": len(python_result.charts),
        "text_analysis_count": len(python_result.text_analysis),
    }
    if sql_result is not None:
        payload["sql"] = sql_result.sql
        payload["sql_row_count"] = len(sql_result.rows)
        payload["sql_rows"] = [dict(row) for row in sql_result.rows[:20]]
        payload["sql_explanation"] = sql_result.explanation
    return payload


def _merge_round_python_results(
    *,
    baseline_result: PythonAnalysisResponse,
    round_results: tuple[PythonAnalysisResponse, ...],
) -> PythonAnalysisResponse:
    statistics = dict(baseline_result.statistics)
    statistics["rounds"] = [
        {
            "statistics": result.statistics,
            "insight_count": len(result.insights),
            "chart_count": len(result.charts),
        }
        for result in round_results
    ]
    insights: list[str] = []
    for result in (*round_results, baseline_result):
        for insight in result.insights:
            if insight and insight not in insights:
                insights.append(insight)
    charts: list[ChartResponse] = []
    seen_chart_keys: set[tuple[str, str]] = set()
    text_analysis = []
    for result in (*round_results, baseline_result):
        text_analysis.extend(result.text_analysis)
        for chart in result.charts:
            key = (chart.title, chart.chart_type)
            if key in seen_chart_keys:
                continue
            seen_chart_keys.add(key)
            charts.append(chart)
    return PythonAnalysisResponse(
        statistics=statistics,
        insights=tuple(insights),
        charts=tuple(charts),
        text_analysis=tuple(text_analysis),
    )


def _apply_model_reflections(
    content: str,
    rounds: tuple[AnalysisRoundResponse, ...],
) -> tuple[AnalysisRoundResponse, ...]:
    payload = _extract_json_object(content)
    if payload is None:
        return rounds
    reflections = _string_list(payload.get("reflections"))
    if not reflections:
        return rounds
    updated: list[AnalysisRoundResponse] = []
    for round_item in rounds:
        insight_text = (
            reflections[round_item.round_number - 1]
            if round_item.round_number <= len(reflections)
            else round_item.reflection.insight_text
        )
        updated.append(
            AnalysisRoundResponse(
                round_number=round_item.round_number,
                hypothesis=round_item.hypothesis,
                plan=round_item.plan,
                reflection=AnalysisReflectionResponse(
                    insight_text=insight_text,
                    impact_pct=round_item.reflection.impact_pct,
                    has_insight=round_item.reflection.has_insight,
                    decision=round_item.reflection.decision,
                    data_source=round_item.reflection.data_source,
                ),
                execution_result=round_item.execution_result,
                charts=round_item.charts,
                validation_status=round_item.validation_status,
            )
        )
    return tuple(updated)


def _parse_model_insights(
    content: str,
    *,
    fallback: tuple[InsightFindingResponse, ...],
) -> tuple[InsightFindingResponse, ...]:
    payload = _extract_json_object(content)
    if payload is None:
        raise ValueError("Model Router integrate_insights did not return JSON.")
    raw_items = payload.get("insights")
    if not isinstance(raw_items, list):
        raise ValueError("Model Router integrate_insights returned no insights array.")
    findings: list[InsightFindingResponse] = []
    for index, item in enumerate(raw_items[:6], 1):
        if not isinstance(item, dict):
            continue
        content_text = str(item.get("content") or "").strip()
        data_source = str(item.get("data_source") or item.get("evidence") or "").strip()
        if not content_text or not data_source:
            continue
        findings.append(
            InsightFindingResponse(
                title=str(item.get("title") or f"洞察 {index}"),
                content=content_text,
                data_source=data_source,
                impact_pct=_float_payload(item.get("impact_pct")),
                evidence=str(item.get("evidence") or data_source),
                confidence=str(item.get("confidence") or "medium"),
                business_impact=str(item.get("business_impact") or ""),
                recommended_action=str(item.get("recommended_action") or ""),
            )
        )
    return tuple(findings) or fallback


def _apply_model_chart_explanations(
    content: str,
    charts: tuple[ChartResponse, ...],
) -> tuple[ChartResponse, ...]:
    payload = _extract_json_object(content)
    if payload is None:
        return charts
    raw_items = payload.get("chart_explanations")
    if not isinstance(raw_items, list):
        return charts
    explanation_by_title = {
        str(item.get("title")): str(item.get("explanation") or "").strip()
        for item in raw_items
        if isinstance(item, dict) and str(item.get("explanation") or "").strip()
    }
    if not explanation_by_title:
        return charts
    return tuple(
        ChartResponse(
            title=chart.title,
            chart_type=chart.chart_type,
            spec=chart.spec,
            data=chart.data,
            explanation=_preserve_chart_denominator_scope(
                chart,
                explanation_by_title.get(chart.title, chart.explanation),
            ),
            related_finding_ids=chart.related_finding_ids,
        )
        for chart in charts
    )


def _sanitize_model_plan(
    planned_analysis: PlannedAnalysis,
    profile: DatasetProfileResponse,
    *,
    allow_time: bool = True,
) -> PlannedAnalysis:
    available_columns = {column.name for column in profile.columns}
    metric_columns = set(profile.numeric_columns)
    requested_dimensions = tuple(
        column
        for column in planned_analysis.requested_dimensions
        if column in available_columns
    )
    aggregations = tuple(
        item
        for item in planned_analysis.aggregations
        if item.column is None or item.column in available_columns
    )
    filters = tuple(
        item for item in planned_analysis.filters if item.column in available_columns
    )
    requested_metric = next(
        (
            item.column
            for item in aggregations
            if item.operation in {"sum", "avg", "min", "max"}
            and item.column in metric_columns
        ),
        None,
    )
    metric_column = requested_metric or _column_or_none(
        planned_analysis.metric_column, metric_columns
    )
    category_column = (
        requested_dimensions[0]
        if requested_dimensions
        else None
        if aggregations
        else _column_or_none(planned_analysis.category_column, available_columns)
    )
    time_column = (
        _column_or_none(planned_analysis.time_column, available_columns)
        if allow_time
        else None
    )
    if category_column == metric_column:
        category_column = next(
            (column for column in profile.categorical_columns if column != metric_column),
            None,
        )
    if time_column == metric_column:
        time_column = None
    route = (
        planned_analysis.route if planned_analysis.route in {"sql", "python", "hybrid"} else "sql"
    )
    return PlannedAnalysis(
        route=route,
        category_column=category_column,
        metric_column=metric_column,
        time_column=time_column,
        steps=planned_analysis.steps,
        aggregations=aggregations,
        filters=filters,
        requested_dimensions=requested_dimensions,
        derived_metrics=planned_analysis.derived_metrics,
        time_grain=planned_analysis.time_grain if time_column else None,
    )


def _analysis_fast_path_eligible(
    *,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    multi_dataset_context: MultiDatasetProfileResponse | None,
    multimodal_inputs: tuple[MultimodalInputResponse, ...],
) -> bool:
    settings = get_settings()
    return bool(
        settings.analysis_fast_path_enabled
        and profile.row_count <= settings.analysis_fast_path_max_rows
        and planned_analysis.route == "sql"
        and multi_dataset_context is None
        and not multimodal_inputs
    )


def _planner_metadata(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    source: str,
    error: str | None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> PlannerMetadataResponse:
    metric_priority = (
        *((planned_analysis.metric_column,) if planned_analysis.metric_column else ()),
        *profile.numeric_columns,
    )
    candidate_metrics = tuple(
        dict.fromkeys(
            column
            for column in metric_priority
            if not _looks_like_identifier_column(column)
        )
    )[:8]
    excluded_dimension_columns = {
        *(item.column for item in planned_analysis.filters),
        *(item.column for item in planned_analysis.aggregations if item.column),
        *((planned_analysis.metric_column,) if planned_analysis.metric_column else ()),
    }
    selected_dimensions = tuple(
        column
        for column in (
        *planned_analysis.requested_dimensions,
        *((planned_analysis.category_column,) if planned_analysis.category_column else ()),
        )
        if column not in excluded_dimension_columns
    )
    non_identifier_dimensions = tuple(
        column
        for column in profile.categorical_columns
        if not _looks_like_identifier_column(column)
    )
    identifier_dimensions = tuple(
        column
        for column in profile.categorical_columns
        if _looks_like_identifier_column(column)
    )
    fallback_dimensions = (
        (*non_identifier_dimensions, *identifier_dimensions)
        if not selected_dimensions
        else ()
    )
    candidate_dimensions = tuple(
        dict.fromkeys(
            column
            for column in (*selected_dimensions, *fallback_dimensions)
            if column not in excluded_dimension_columns
        )
    )[:8]
    candidate_time_fields = tuple(
        column.name
        for column in profile.columns
        if any(
            token in column.name.lower()
            for token in ("date", "time", "month", "日期", "时间", "月份")
        )
    )[:8]
    candidate_text_fields = tuple(
        column.name
        for column in profile.columns
        if not column.is_numeric and column.distinct_count > min(profile.row_count * 0.5, 30)
    )[:8]
    confidence = 0.58
    if planned_analysis.metric_column:
        confidence += 0.16
    if planned_analysis.category_column:
        confidence += 0.12
    if planned_analysis.time_column:
        confidence += 0.06
    if source == "model_router":
        confidence += 0.08
    if error:
        confidence -= 0.14
    clarifying_questions: list[str] = []
    lowered = question.lower()
    if not candidate_metrics and any(
        token in lowered for token in ("sales", "revenue", "profit", "amount")
    ):
        clarifying_questions.append(
            "当前没有可用数值指标，是否需要先把金额/销量字段标记为 metric？"
        )
    if len(candidate_metrics) > 3 and planned_analysis.metric_column is None:
        clarifying_questions.append("存在多个候选指标，建议明确本次优先分析哪个 metric。")
    if len(candidate_dimensions) > 5 and planned_analysis.category_column is None:
        clarifying_questions.append("存在多个候选维度，建议明确按哪个维度拆解。")
    route_reason = (
        f"选择 {planned_analysis.route}："
        f"指标={planned_analysis.metric_column or '无'}，"
        f"维度={planned_analysis.category_column or '无'}，"
        f"时间={planned_analysis.time_column or '无'}。"
    )
    multi_dataset_summary: dict[str, Any] = {}
    if multi_dataset_context is not None:
        multi_dataset_summary = {
            "primary_dataset": multi_dataset_context.primary_dataset.name,
            "additional_datasets": [
                dataset.name for dataset in multi_dataset_context.additional_datasets
            ],
            "join_summary": multi_dataset_context.join_summary,
            "column_source_map": multi_dataset_context.column_source_map,
        }
        join_count = len(multi_dataset_context.join_plan)
        route_reason += (
            f" 多数据集上下文：主表 {multi_dataset_context.primary_dataset.name}，"
            f"附加 {len(multi_dataset_context.additional_datasets)} 个数据集，"
            f"已确认 {join_count} 个 join。"
        )
    if error:
        route_reason += " 模型规划不可用，已使用确定性规则执行。"
    return PlannerMetadataResponse(
        confidence=max(0, min(confidence, 1)),
        route_reason=route_reason,
        candidate_metrics=candidate_metrics,
        candidate_dimensions=candidate_dimensions,
        candidate_time_fields=candidate_time_fields,
        candidate_text_fields=candidate_text_fields,
        clarifying_questions=tuple(clarifying_questions),
        multi_dataset_summary=multi_dataset_summary,
    )


def _merge_additional_dataset_ids(
    additional_dataset_ids: tuple[UUID, ...],
    join_plan: tuple[DatasetJoinConfig, ...],
    *,
    primary_dataset_id: UUID,
) -> tuple[UUID, ...]:
    seen: set[UUID] = {primary_dataset_id}
    result: list[UUID] = []
    for dataset_id in additional_dataset_ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        result.append(dataset_id)
    for config in join_plan:
        for dataset_id in (config.left_dataset_id, config.right_dataset_id):
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            result.append(dataset_id)
    return tuple(result)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _column_or_none(value: str | None, available_columns: set[str]) -> str | None:
    return value if value in available_columns else None
