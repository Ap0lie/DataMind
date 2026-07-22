from __future__ import annotations

import base64
import binascii
import io
import json
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

from app.analysis.agent_loop import (
    TOOL_DEFINITIONS,
    AgentToolRuntime,
    LoopErrorType,
    canonical_action_hash,
    error_fingerprint,
)
from app.analysis.checkpoints import get_analysis_checkpointer
from app.analysis.model_router import AnalysisModelRouter, MCPAnalysisModelRouter
from app.analysis.multidataset import prepare_multi_dataset_context
from app.analysis.prompt_utils import compact_prompt_columns, compact_prompt_records
from app.analysis.python_execution import PythonAnalysisExecutor
from app.analysis.python_sandbox import run_generated_python_analysis
from app.analysis.services import (
    PlannedAnalysis,
    _build_final_insights,
    _build_validation_issues,
    _dataframe,
    _format_report_charts,
    _framework_from_profile,
    _plan,
    _records,
    _run_python,
    _run_sql,
    _structured_report,
    render_structured_report_html,
)
from app.analysis.validators import validate_analysis_plan
from app.analysis.workflow_prompt_context import (
    compact_multi_dataset_context as _compact_multi_dataset_context,
)
from app.analysis.workflow_prompt_context import (
    experience_context as _experience_context,
)
from app.analysis.workflow_prompt_context import (
    prompt_system as _prompt_system,
)
from app.core.settings import get_settings
from app.harness.node import NodeExecutionHarness
from app.schemas.analysis import (
    AnalysisFrameworkResponse,
    AnalysisHypothesisResponse,
    AnalysisPlanResponse,
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
    StructuredReportResponse,
    ValidationIssueResponse,
    WorkflowTraceNodeResponse,
)
from app.storage.dataset_store import DatasetStoreRepository

PLANNER_NODE = "planner"
JOIN_PREPARE_NODE = "join_prepare"
DESIGN_FRAMEWORK_NODE = "design_framework"
SQL_NODE = "sql_agent"
PYTHON_NODE = "python_agent"
ROUND_PREPARE_NODE = "iterative_prepare_rounds"
ROUND_FOUNDATION_NODE = "iterative_round_1"
ROUND_FANOUT_NODE = "iterative_fanout_round"
ROUND_REFLECT_NODE = "iterative_reflect_and_merge"
INTEGRATE_INSIGHTS_NODE = "integrate_insights"
ADVERSARIAL_VALIDATE_NODE = "adversarial_validate"
FORMAT_CHARTS_NODE = "format_charts"
REPORT_NODE = "report_agent"
REPORT_DECIDE_NODE = "report_decide"
REPORT_EXECUTE_NODE = "report_execute"
REPORT_VERIFY_NODE = "report_verify"
REPORT_REPAIR_NODE = "report_repair"
REPORT_FALLBACK_NODE = "report_fallback"
REPORT_COMMIT_NODE = "report_commit"
LOOP_BOOTSTRAP_NODE = "loop_bootstrap"
LOOP_DECIDE_NODE = "loop_decide"
LOOP_EXECUTE_NODE = "loop_execute"
LOOP_OBSERVE_NODE = "loop_observe"
LOOP_VERIFY_NODE = "loop_verify"
LOOP_REPAIR_NODE = "loop_repair"
LOOP_FALLBACK_NODE = "loop_fallback"
LOOP_FINALIZE_NODE = "loop_finalize"
LOOP_ADVERSARIAL_REPAIR_NODE = "loop_adversarial_repair"

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
    loop_repair_context: NotRequired[dict[str, Any] | None]
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
    graph: Any = StateGraph(AnalysisWorkflowState)
    harness = NodeExecutionHarness(event_callback=_notify_node_event)
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
        ADVERSARIAL_VALIDATE_NODE: _adversarial_validate_node(model_router),
        FORMAT_CHARTS_NODE: _format_charts_node(model_router),
        REPORT_NODE: _report_node(repository, model_router),
        REPORT_DECIDE_NODE: _report_decide_node(model_router),
        REPORT_EXECUTE_NODE: _report_execute_node(model_router),
        REPORT_VERIFY_NODE: _report_verify_node(),
        REPORT_REPAIR_NODE: _report_repair_node(),
        REPORT_FALLBACK_NODE: _report_fallback_node(),
        REPORT_COMMIT_NODE: _report_commit_node(repository),
        LOOP_BOOTSTRAP_NODE: _loop_bootstrap_node(),
        LOOP_DECIDE_NODE: _loop_decide_node(model_router),
        LOOP_EXECUTE_NODE: _loop_execute_node(repository, resolved_python_executor),
        LOOP_OBSERVE_NODE: _loop_observe_node(repository),
        LOOP_VERIFY_NODE: _loop_verify_node(),
        LOOP_REPAIR_NODE: _loop_repair_node(),
        LOOP_FALLBACK_NODE: _loop_fallback_node(repository, resolved_python_executor),
        LOOP_FINALIZE_NODE: _loop_finalize_node(repository),
        LOOP_ADVERSARIAL_REPAIR_NODE: _loop_adversarial_repair_node(),
    }
    for node_name, handler in nodes.items():
        graph.add_node(node_name, harness.wrap(node_name, handler))

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
    graph.add_edge(LOOP_FINALIZE_NODE, INTEGRATE_INSIGHTS_NODE)
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
    graph.add_edge(FORMAT_CHARTS_NODE, ADVERSARIAL_VALIDATE_NODE)
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
                planned_analysis = _sanitize_model_plan(model_plan, profile)
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
            "plan_validation_issues": plan_validation_issues,
            "planner_source": planner_source,
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
        if model_router is not None:
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

        if model_router is not None:
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
        python_result = _require(
            state.get("python_result"),
            "Python Agent did not produce analysis results.",
        )
        final_insights = _build_final_insights(
            python_result=python_result,
            sql_result=state.get("sql_result"),
        )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if model_router is not None:
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


def _adversarial_validate_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=ADVERSARIAL_VALIDATE_NODE,
            progress=93,
            message="Reviewing analysis quality and gaps.",
        )
        mandatory_findings = _mandatory_evidence_findings(state)
        final_insights = _merge_report_findings(
            mandatory_findings,
            state.get("final_insights", ()),
        )
        python_result = _require(
            state.get("python_result"),
            "Python Agent did not produce analysis results.",
        )
        validation_issues = _build_validation_issues(
            findings=final_insights,
            charts=python_result.charts,
            extra_issues=state.get("plan_validation_issues", ()),
        )
        if state.get("python_execution_error") and len(state.get("python_attempts", ())) >= 3:
            validation_issues = (
                *validation_issues,
                ValidationIssueResponse(
                    severity="warning",
                    finding_ref="python_agent",
                    issue="LLM Python code failed after 3 attempts; rule fallback was used.",
                    suggestion=str(state.get("python_execution_error") or ""),
                ),
            )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if model_router is not None:
            try:
                response = model_router.complete(
                    messages=_review_messages(
                        question=state["question"],
                        final_insights=final_insights,
                        charts=python_result.charts,
                        sql_result=state.get("sql_result"),
                        multimodal_inputs=state.get("multimodal_inputs", ()),
                        multi_dataset_context=state.get("multi_dataset_context"),
                    ),
                    temperature=0.1,
                    max_tokens=1000,
                    metadata={
                        "agent": "review",
                        "dataset_id": str(state["dataset_id"]),
                    },
                )
                validation_issues = (
                    *validation_issues,
                    *_parse_model_validation_issues(response.content),
                )
                model_router_provider = response.provider
                model_router_model = response.model
            except Exception as exc:
                model_router_error = str(exc)
                validation_issues = (
                    *validation_issues,
                    ValidationIssueResponse(
                        severity="info",
                        finding_ref="adversarial_validate",
                        issue="LLM review unavailable; rule validation was used.",
                        suggestion=str(exc),
                    ),
                )
        return {
            "validation_issues": validation_issues,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
            "executed_nodes": (*state.get("executed_nodes", ()), ADVERSARIAL_VALIDATE_NODE),
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
        python_result = _require(
            state.get("python_result"),
            "Python Agent did not produce analysis results.",
        )
        report_charts = _format_report_charts(
            charts=python_result.charts,
            findings=state.get("final_insights", ()),
        )
        model_router_provider = state.get("model_router_provider")
        model_router_model = state.get("model_router_model")
        model_router_error = state.get("model_router_error")
        if model_router is not None and report_charts:
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


_REPORT_STRATEGY_TOOL = {
    "type": "function",
    "function": {
        "name": "select_report_strategy",
        "description": "Select a report generation strategy from the available verified evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["template", "llm", "request_evidence", "fallback"],
                },
                "reason": {"type": "string"},
            },
            "required": ["strategy", "reason"],
            "additionalProperties": False,
        },
    },
}


def _report_decide_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        settings = get_settings()
        count = state.get("report_decision_count", 0) + 1
        started_epoch = state.get("report_started_epoch") or time.time()
        _notify_progress(
            state,
            stage=REPORT_DECIDE_NODE,
            progress=95,
            message="AI is selecting the report strategy from verified evidence.",
        )
        findings = state.get("final_insights", ())
        evidence_gaps = [
            item.title for item in findings if not item.evidence and not item.data_source
        ]
        if not findings and state.get("report_evidence_return_count", 0) < 1:
            strategy, reason = "evidence_gap", "No final findings are available."
        elif count > settings.report_loop_max_decisions:
            strategy, reason = "rules_fallback", "Report decision budget exhausted."
        elif time.time() - started_epoch >= settings.report_loop_timeout_seconds:
            strategy, reason = "rules_fallback", "Report time budget exhausted."
        elif state.get("report_used_tokens", 0) >= settings.report_loop_max_tokens:
            strategy, reason = "rules_fallback", "Report token budget exhausted."
        else:
            strategy, reason = "llm", "Default structured LLM report strategy."
            if model_router is not None:
                try:
                    response = model_router.complete(
                        messages=[
                            {
                                "role": "system",
                                "content": "你是 DataMind 报告策略控制器。根据已验证证据选择 template、llm、request_evidence 或 fallback。不得生成事实，只选择策略。证据不足才 request_evidence，且只能请求一次。",
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "question": state["question"],
                                        "finding_count": len(findings),
                                        "evidence_gaps": evidence_gaps,
                                        "chart_count": len(state.get("report_charts", ())),
                                        "analysis_return_used": state.get(
                                            "report_evidence_return_count", 0
                                        )
                                        > 0,
                                        "validation_issues": [
                                            item.model_dump(mode="json")
                                            for item in state.get("validation_issues", ())[:8]
                                        ],
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        temperature=0.0,
                        max_tokens=500,
                        metadata={"agent": "report_decide", "job_id": str(state["run_id"])},
                        tools=[_REPORT_STRATEGY_TOOL],
                        tool_choice="auto",
                    )
                    arguments: dict[str, Any] = {}
                    if len(response.tool_calls) == 1:
                        function = response.tool_calls[0].get("function") or {}
                        raw = function.get("arguments") or {}
                        arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    elif response.content:
                        arguments = json.loads(response.content)
                    selected = str(arguments.get("strategy") or "llm")
                    reason = str(arguments.get("reason") or reason)[:400]
                    strategy = {
                        "template": "template",
                        "llm": "llm",
                        "fallback": "rules_fallback",
                        "request_evidence": "evidence_gap",
                    }.get(selected, "llm")
                    used_tokens = state.get("report_used_tokens", 0) + int(
                        response.token_usage.get("total_tokens") or 0
                    )
                except Exception as exc:
                    reason = f"Report strategy provider fallback: {type(exc).__name__}."
                    used_tokens = state.get("report_used_tokens", 0)
            else:
                used_tokens = state.get("report_used_tokens", 0)
        evidence_returns = state.get("report_evidence_return_count", 0)
        result: dict[str, Any] = {
            "report_decision_count": count,
            "report_strategy": strategy,
            "report_started_epoch": started_epoch,
            "report_used_tokens": locals().get("used_tokens", state.get("report_used_tokens", 0)),
        }
        if strategy == "evidence_gap":
            if evidence_returns >= 1:
                result.update(
                    {
                        "report_strategy": "rules_fallback",
                        "report_terminal_reason": "evidence_gap_after_reanalysis",
                    }
                )
            else:
                result.update(
                    {
                        "report_evidence_return_count": evidence_returns + 1,
                        "loop_repair_context": {
                            "error_type": "report_evidence_gap",
                            "message": reason,
                            "requested_tools": [
                                "execute_semantic_query",
                                "execute_safe_sql",
                                "execute_python_analysis",
                            ],
                        },
                        "loop_terminal_reason": None,
                    }
                )
                _emit_loop_event(
                    state,
                    event_type="evidence_request",
                    status="completed",
                    message=reason,
                    iteration=evidence_returns + 1,
                    payload={"return_count": evidence_returns + 1},
                )
        _emit_loop_event(
            state,
            event_type="report_decision",
            status="completed",
            message=f"Report strategy selected: {result.get('report_strategy', strategy)}.",
            iteration=count,
            payload={"strategy": result.get("report_strategy", strategy), "reason": reason},
        )
        return result

    return run


def _build_report_draft(
    state: AnalysisWorkflowState,
    model_router: AnalysisModelRouter | None,
    *,
    strategy: str,
    agent: str = "report_execute",
) -> dict[str, Any]:
    question = state["question"]
    profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
    python_result = _require(
        state.get("python_result"), "Python Agent did not produce analysis results."
    )
    sql_result = state.get("sql_result")
    rounds = state.get("rounds", ())
    findings = _attach_finding_evidence_ids(
        state.get("final_insights", ()), state.get("tool_evidence", ())
    )
    mandatory_findings = _mandatory_evidence_findings(state)
    findings = _merge_report_findings(mandatory_findings, findings)
    issues = state.get("validation_issues", ())
    charts = state.get("report_charts", python_result.charts)
    framework = state.get("analysis_framework")
    structured = _structured_report(
        question=question,
        profile=profile,
        sql_result=sql_result,
        python_result=python_result,
        rounds=rounds,
        final_insights=findings,
        validation_issues=issues,
        analysis_framework=framework,
        charts=charts,
    )
    source = "rules"
    provider = state.get("model_router_provider")
    model = state.get("model_router_model")
    error = state.get("model_router_error")
    augmented_model_content: str | None = None
    if strategy == "llm" and model_router is not None:
        try:
            report_messages = _report_messages(
                question=question,
                profile=profile,
                structured_report=structured,
                multimodal_inputs=state.get("multimodal_inputs", ()),
                multi_dataset_context=state.get("multi_dataset_context"),
            )
            if state.get("report_validation"):
                report_messages.append(
                    {
                        "role": "user",
                        "content": "上一版报告未通过确定性校验。只修复以下问题，不新增无证据数字或图表引用："
                        + json.dumps(state["report_validation"], ensure_ascii=False),
                    }
                )
            response = model_router.complete(
                messages=report_messages,
                temperature=0.2,
                max_tokens=2200,
                metadata={
                    "agent": agent,
                    "dataset_id": str(state["dataset_id"]),
                    "revision": state.get("report_revision_count", 0) + 1,
                },
            )
            token_usage = int(response.token_usage.get("total_tokens") or 0)
            try:
                structured = _parse_model_structured_report(
                    response.content,
                    fallback=structured,
                    provider=response.provider,
                    model=response.model,
                )
                source = "model_router_structured"
            except ValueError as parse_error:
                try:
                    repaired = model_router.complete(
                        messages=_json_repair_messages(
                            stage="report",
                            invalid_content=response.content,
                            error=str(parse_error),
                            contract=(
                                "Return one JSON object with executive_summary (at least 20 "
                                "characters), analysis_context, key_findings, chart_explanations, "
                                "data_gaps, validation_issues, and recommended_next_steps."
                            ),
                        ),
                        temperature=0.0,
                        max_tokens=2200,
                        metadata={
                            "agent": agent,
                            "dataset_id": str(state["dataset_id"]),
                            "revision": state.get("report_revision_count", 0) + 1,
                            "structured_repair": True,
                        },
                    )
                    token_usage += int(
                        repaired.token_usage.get("total_tokens") or 0
                    )
                    structured = _parse_model_structured_report(
                        repaired.content,
                        fallback=structured,
                        provider=repaired.provider,
                        model=repaired.model,
                    )
                    response = repaired
                    source = "model_router_structured_repair"
                except Exception as repair_error:
                    narrative = str(response.content or "").strip()
                    if len(narrative) >= 40 and not narrative.startswith("[mock:"):
                        augmented_model_content = narrative
                        source = "model_router_augmented"
                    else:
                        source = "rules"
                    error = (
                        f"Structured report repair failed: {type(repair_error).__name__}: "
                        f"{repair_error}"
                    )
            provider, model = response.provider, response.model
            if source != "rules":
                error = None
        except Exception as exc:
            error = str(exc)
            source = "rules"
    structured = _preserve_mandatory_report_findings(structured, mandatory_findings)
    markdown = _markdown_from_structured_report(structured)
    if augmented_model_content:
        markdown = _merge_model_report(
            base_report=markdown,
            model_content=augmented_model_content,
        )
    html = render_structured_report_html(structured, title="DataMind 分析报告")
    return {
        "structured_report": structured,
        "report_markdown": markdown,
        "html_report": html,
        "report_source": source,
        "model_router_provider": provider,
        "model_router_model": model,
        "model_router_error": error,
        "report_charts": charts,
        "final_insights": findings,
        "validation_issues": issues,
        "report_draft_tokens": locals().get("token_usage", 0),
    }


def _report_execute_node(model_router: AnalysisModelRouter | None) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=REPORT_EXECUTE_NODE,
            progress=96,
            message="Generating a traceable report draft.",
        )
        revision = state.get("report_revision_count", 0) + 1
        draft = _build_report_draft(
            state, model_router, strategy=state.get("report_strategy") or "llm"
        )
        _emit_loop_event(
            state,
            event_type="report_draft",
            status="completed",
            message=f"Report draft revision {revision} generated.",
            iteration=revision,
            payload={
                "strategy": state.get("report_strategy"),
                "source": draft.get("report_source"),
            },
        )
        return {
            **draft,
            "report_revision_count": revision,
            "report_used_tokens": state.get("report_used_tokens", 0)
            + int(draft.get("report_draft_tokens") or 0),
            "report_draft_ready": True,
            "executed_nodes": (*state.get("executed_nodes", ()), REPORT_EXECUTE_NODE),
        }

    return run


def _report_verify_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        settings = get_settings()
        _notify_progress(
            state,
            stage=REPORT_VERIFY_NODE,
            progress=97,
            message="Validating report claims, evidence and chart references.",
        )
        structured = _require(state.get("structured_report"), "Report draft is missing.")
        unsupported: list[str] = []
        evidence_gaps: list[str] = []
        known_evidence_ids = {
            str(item.get("evidence_id"))
            for item in state.get("tool_evidence", ())
            if item.get("evidence_id")
        }
        for finding in structured.key_findings:
            has_number = bool(re.search(r"(?<![\w])[-+]?\d+(?:\.\d+)?%?", finding.content))
            cited_ids = {
                evidence_id for evidence_id in known_evidence_ids if evidence_id in finding.evidence
            }
            if has_number and not cited_ids:
                unsupported.append(finding.title)
            if not finding.evidence and not finding.data_source:
                evidence_gaps.append(finding.title)
        chart_titles = {chart.title for chart in state.get("report_charts", ())}
        missing_chart_refs = [
            chart.title for chart in structured.charts if chart.title not in chart_titles
        ]
        revision = state.get("report_revision_count", 0)
        evidence_returns = state.get("report_evidence_return_count", 0)
        budget_exhausted = (
            time.time() - float(state.get("report_started_epoch") or time.time())
            >= settings.report_loop_timeout_seconds
            or state.get("report_used_tokens", 0) >= settings.report_loop_max_tokens
        )
        if budget_exhausted:
            outcome = "fallback"
        elif evidence_gaps and evidence_returns < 1:
            outcome = "evidence_gap"
        elif (unsupported or missing_chart_refs) and revision < settings.report_loop_max_revisions:
            outcome = "report_issue"
        elif unsupported or missing_chart_refs:
            outcome = "fallback"
        else:
            outcome = "sufficient"
        validation = {
            "outcome": outcome,
            "unsupported_numeric_findings": unsupported,
            "evidence_gaps": evidence_gaps,
            "missing_chart_references": missing_chart_refs,
        }
        update: dict[str, Any] = {"report_validation": validation}
        if outcome == "evidence_gap":
            update.update(
                {
                    "report_evidence_return_count": evidence_returns + 1,
                    "loop_repair_context": {
                        "error_type": "report_evidence_gap",
                        "message": f"Evidence missing for: {', '.join(evidence_gaps[:5])}",
                    },
                    "loop_terminal_reason": None,
                }
            )
        if outcome == "sufficient":
            update["report_terminal_reason"] = "validated"
        _emit_loop_event(
            state,
            event_type="report_validation",
            status="completed" if outcome == "sufficient" else "failed",
            message=f"Report validation outcome: {outcome}.",
            iteration=revision,
            payload=validation,
        )
        return update

    return run


def _report_repair_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        validation = state.get("report_validation") or {}
        _notify_progress(
            state,
            stage=REPORT_REPAIR_NODE,
            progress=97,
            message="Repairing unsupported report claims and references.",
        )
        _emit_loop_event(
            state,
            event_type="report_repair",
            status="completed",
            message="Report draft will be regenerated with validation feedback.",
            iteration=state.get("report_revision_count", 0),
            payload=validation,
        )
        return {
            "report_strategy": "llm",
            "report_draft_ready": False,
            "executed_nodes": (*state.get("executed_nodes", ()), REPORT_REPAIR_NODE),
        }

    return run


def _report_fallback_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=REPORT_FALLBACK_NODE,
            progress=98,
            message="Using deterministic report fallback with explicit data gaps.",
        )
        draft = _build_report_draft(state, None, strategy="template")
        structured = draft["structured_report"]
        gaps = tuple(
            dict.fromkeys(
                (
                    *structured.data_gaps,
                    *[
                        str(item)
                        for item in (state.get("report_validation") or {}).get("evidence_gaps", [])
                    ],
                )
            )
        )
        structured = structured.model_copy(update={"data_gaps": gaps})
        draft.update(
            {
                "structured_report": structured,
                "report_markdown": _markdown_from_structured_report(structured),
                "html_report": render_structured_report_html(structured, title="DataMind 分析报告"),
                "report_source": "rules",
            }
        )
        _emit_loop_event(
            state,
            event_type="report_fallback",
            status="completed",
            message="Rule report fallback prepared.",
            iteration=state.get("report_revision_count", 0),
            payload={"data_gap_count": len(gaps)},
        )
        return {
            **draft,
            "report_strategy": "rules_fallback",
            "report_terminal_reason": state.get("report_terminal_reason") or "rules_fallback",
            "report_draft_ready": True,
            "executed_nodes": (*state.get("executed_nodes", ()), REPORT_FALLBACK_NODE),
        }

    return run


def _report_commit_node(repository: DatasetStoreRepository) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=REPORT_COMMIT_NODE,
            progress=99,
            message="Committing the validated report idempotently.",
        )
        dataset_id = state["dataset_id"]
        planned = _require(
            state.get("planned_analysis"), "Planner did not produce an analysis plan."
        )
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        python_result = _require(
            state.get("python_result"), "Python Agent did not produce analysis results."
        )
        structured = _require(state.get("structured_report"), "Validated report draft is missing.")
        markdown = _require(state.get("report_markdown"), "Validated report markdown is missing.")
        html = _require(state.get("html_report"), "Validated report HTML is missing.")
        charts = state.get("report_charts", python_result.charts)
        executed = (*state.get("executed_nodes", ()), REPORT_COMMIT_NODE)
        report_source = state.get("report_source", "rules")
        trace = _workflow_trace(
            state=state,
            executed_nodes=executed,
            report_source=report_source,
            provider=state.get("model_router_provider"),
            model=state.get("model_router_model"),
        )
        analysis_summary = state.get("loop_summary", {})
        combined_summary = {
            **analysis_summary,
            "analysis": analysis_summary,
            "report": {
                "strategy": state.get("report_strategy"),
                "revision_count": state.get("report_revision_count", 0),
                "terminal_reason": state.get("report_terminal_reason"),
                "validation": state.get("report_validation") or {},
                "evidence_return_count": state.get("report_evidence_return_count", 0),
            },
        }
        analysis_framework = state.get("analysis_framework")
        planner_metadata = state.get("planner_metadata")
        multi_dataset_context = state.get("multi_dataset_context")
        report_id = repository.save_report(
            dataset_id=dataset_id,
            title="DataMind 分析报告",
            markdown=markdown,
            job_id=state["run_id"],
            metadata={
                "question": state["question"],
                "route": planned.route,
                "workflow": "langgraph_analysis",
                "prompt_overrides": dict(state.get("prompt_overrides") or {}),
                "nodes": list(executed),
                "analysis_framework": analysis_framework.model_dump(mode="json")
                if analysis_framework
                else None,
                "multi_dataset_context": multi_dataset_context.model_dump(mode="json")
                if multi_dataset_context
                else None,
                "primary_dataset_id": str(dataset_id),
                "dataset_group_id": str(state.get("dataset_group_id"))
                if state.get("dataset_group_id")
                else None,
                "additional_dataset_ids": [
                    str(item) for item in state.get("additional_dataset_ids", ())
                ],
                "join_plan": [item.model_dump(mode="json") for item in state.get("join_plan", ())],
                "relationship_plan": [
                    item.model_dump(mode="json") for item in state.get("relationship_plan", ())
                ],
                "join_summary": multi_dataset_context.join_summary
                if multi_dataset_context
                else {},
                "multimodal_inputs": [
                    item.model_dump(mode="json") for item in state.get("multimodal_inputs", ())
                ],
                "planner_metadata": planner_metadata.model_dump(mode="json")
                if planner_metadata
                else None,
                "planner_source": state.get("planner_source", "rules"),
                "python_source": state.get("python_source", "rules"),
                "python_generated_code": state.get("python_generated_code"),
                "python_execution_error": state.get("python_execution_error"),
                "python_attempts": [
                    attempt.model_dump(mode="json")
                    for attempt in state.get("python_attempts", ())
                ],
                "model_router_provider": state.get("model_router_provider"),
                "model_router_model": state.get("model_router_model"),
                "model_router_error": state.get("model_router_error"),
                "sql_source": state.get("sql_source", "none"),
                "sql_validation_error": state.get("sql_validation_error"),
                "structured_report": structured.model_dump(mode="json"),
                "html_report": html,
                "validation_issue_count": len(state.get("validation_issues", ())),
                "workflow_trace": [item.model_dump(mode="json") for item in trace],
                "report_source": report_source,
                "report_strategy": state.get("report_strategy"),
                "report_revision_count": state.get("report_revision_count", 0),
                "report_terminal_reason": state.get("report_terminal_reason"),
                "agent_mode": state.get("agent_mode", "legacy"),
                "loop_summary": combined_summary,
                "loop_terminal_reason": state.get("loop_terminal_reason"),
                "semantic_model_id": str(
                    (state.get("planner_decision") or {}).get("semantic_model_id") or ""
                )
                or None,
                "semantic_model_version": (state.get("planner_decision") or {}).get(
                    "semantic_model_version"
                ),
                "evidence_ids": [
                    item.get("evidence_id") or item.get("artifact_id")
                    for item in state.get("tool_evidence", ())
                    if item.get("evidence_id") or item.get("artifact_id")
                ],
                "finding_evidence": [
                    {
                        "title": finding.title,
                        "evidence": finding.evidence,
                        "data_source": finding.data_source,
                    }
                    for finding in structured.key_findings
                ],
            },
        )
        for chart in charts:
            repository.save_chart(
                dataset_id=dataset_id,
                title=chart.title,
                chart_type=chart.chart_type,
                chart_spec=chart.spec,
                chart_data=list(chart.data),
            )
        response = AnalysisRunResponse(
            dataset_id=dataset_id,
            dataset_group_id=state.get("dataset_group_id"),
            report_id=report_id,
            question=state["question"],
            multimodal_inputs=state.get("multimodal_inputs", ()),
            plan=AnalysisPlanResponse(route=planned.route, steps=planned.steps),
            planner_metadata=state.get("planner_metadata"),
            multi_dataset_context=state.get("multi_dataset_context"),
            profile=profile,
            analysis_framework=state.get("analysis_framework"),
            sql_result=state.get("sql_result"),
            python_result=python_result,
            rounds=state.get("rounds", ()),
            final_insights=state.get("final_insights", ()),
            validation_issues=state.get("validation_issues", ()),
            structured_report=structured,
            html_report=html,
            python_source=state.get("python_source", "rules"),
            python_generated_code=state.get("python_generated_code"),
            python_execution_error=state.get("python_execution_error"),
            python_attempts=state.get("python_attempts", ()),
            workflow_trace=trace,
            report_markdown=markdown,
            agent_mode=state.get("agent_mode", "legacy"),
            loop_summary=combined_summary,
            loop_terminal_reason=state.get("loop_terminal_reason"),
            report_strategy=state.get("report_strategy"),
            report_revision_count=state.get("report_revision_count", 0),
            report_terminal_reason=state.get("report_terminal_reason"),
        )
        _emit_loop_event(
            state,
            event_type="report_commit",
            status="completed",
            message="Validated report committed idempotently.",
            iteration=state.get("report_revision_count", 0),
            payload={"report_id": str(report_id)},
        )
        return {
            "final_response": response,
            "workflow_trace": trace,
            "executed_nodes": executed,
            "loop_summary": combined_summary,
        }

    return run


def _attach_finding_evidence_ids(
    findings: tuple[InsightFindingResponse, ...],
    evidence: tuple[dict[str, Any], ...],
) -> tuple[InsightFindingResponse, ...]:
    successful = [
        item
        for item in evidence
        if item.get("evidence_id") and item.get("status") in {"succeeded", "completed", None}
    ]
    if not successful:
        return findings
    output: list[InsightFindingResponse] = []
    for finding in findings:
        if any(str(item["evidence_id"]) in finding.evidence for item in successful):
            output.append(finding)
            continue
        source = f"{finding.data_source} {finding.evidence}".lower()
        preferred = next(
            (
                item
                for item in successful
                if (
                    (
                        "sql" in source
                        and (
                            "sql" in str(item.get("tool_name") or "")
                            or "semantic" in str(item.get("tool_name") or "")
                        )
                    )
                    or ("python" in source and "python" in str(item.get("tool_name") or ""))
                    or ("文本" in source and "text" in str(item.get("tool_name") or ""))
                )
            ),
            successful[0],
        )
        evidence_text = finding.evidence.strip()
        reference = f"evidence_id:{preferred['evidence_id']}"
        output.append(
            finding.model_copy(
                update={"evidence": f"{evidence_text}; {reference}" if evidence_text else reference}
            )
        )
    return tuple(output)


def _mandatory_evidence_findings(
    state: AnalysisWorkflowState,
) -> tuple[InsightFindingResponse, ...]:
    evidence = [
        item
        for item in state.get("tool_evidence", ())
        if item.get("evidence_id")
        and item.get("status") in {"succeeded", "completed", None}
        and isinstance(item.get("result"), dict)
    ]
    aggregates = [
        item for item in evidence if (item.get("result") or {}).get("native_grain") is True
    ]
    relationship_item = next(
        (
            item
            for item in evidence
            if item.get("relationship_guard") is True
            and isinstance((item.get("result") or {}).get("relationships"), list)
        ),
        None,
    )
    relationship_requested = _relationship_analysis_requested(
        state.get("question", "")
    )
    if not aggregates and not (relationship_item and relationship_requested):
        return ()

    fact_sources = {
        str((item.get("result") or {}).get("source_dataset") or "")
        for item in aggregates
        if (item.get("result") or {}).get("source_dataset")
    }
    relationships = (
        list((relationship_item.get("result") or {}).get("relationships") or [])
        if relationship_item
        else []
    )
    relationship_phrases: list[str] = []
    risk_phrases: list[str] = []
    seen_relationships: set[tuple[str, str, str]] = set()
    direct_fact_risk_keys = {
        str(item.get("left_column") or item.get("right_column") or "")
        for item in relationships
        if isinstance(item, dict)
        and item.get("relationship_type") == "many_to_many"
        and {
            str(item.get("left_dataset") or ""),
            str(item.get("right_dataset") or ""),
        }.issubset(fact_sources)
    }
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        left_name = str(relationship.get("left_dataset") or "")
        right_name = str(relationship.get("right_dataset") or "")
        key = str(relationship.get("left_column") or relationship.get("right_column") or "")
        relationship_type = str(relationship.get("relationship_type") or "")
        if fact_sources and left_name not in fact_sources and right_name not in fact_sources:
            continue
        parent_name = child_name = ""
        child_rows = child_distinct = child_duplicates = 0
        if relationship_type == "one_to_many":
            parent_name, child_name = left_name, right_name
            child_rows = int(relationship.get("right_non_null_count") or 0)
            child_distinct = int(relationship.get("right_distinct_count") or 0)
            child_duplicates = int(relationship.get("right_duplicate_count") or 0)
        elif relationship_type == "many_to_one":
            parent_name, child_name = right_name, left_name
            child_rows = int(relationship.get("left_non_null_count") or 0)
            child_distinct = int(relationship.get("left_distinct_count") or 0)
            child_duplicates = int(relationship.get("left_duplicate_count") or 0)
        if (
            parent_name
            and (not fact_sources or child_name in fact_sources)
            and (not direct_fact_risk_keys or key in direct_fact_risk_keys)
        ):
            signature = (parent_name, child_name, key)
            if signature not in seen_relationships and len(relationship_phrases) < 8:
                relationship_phrases.append(
                    f"{parent_name}.{key} → {child_name}.{key} 为 1:N"
                    f"（子表 {child_rows} 个非空键值、{child_distinct} 个唯一键，"
                    f"重复 {child_duplicates} 行）"
                )
                seen_relationships.add(signature)
        if (
            relationship_type == "many_to_many"
            and (not fact_sources or {left_name, right_name}.issubset(fact_sources))
            and len(risk_phrases) < 4
        ):
            risk_phrases.append(
                f"{left_name} 与 {right_name} 的 {key} 在两侧都重复，"
                "直接逐行连接会形成多对多乘积并重复累计指标"
            )

    aggregate_phrases: list[str] = []
    evidence_ids: list[str] = []
    for item in aggregates[:12]:
        result = item.get("result") or {}
        rows = result.get("rows") or []
        if not rows or not isinstance(rows[0], dict):
            continue
        metric = str(result.get("metric") or "metric")
        aggregation = str(result.get("aggregation") or "sum")
        value_key = f"{aggregation}_{metric}"
        value = rows[0].get(value_key)
        if value is None:
            continue
        aggregate_phrases.append(
            f"{result.get('source_dataset')}.{metric} 的 {aggregation.upper()}="
            f"{_format_evidence_number(value)}（源表 {int(result.get('source_row_count') or 0)} 行）"
        )
        evidence_ids.append(str(item["evidence_id"]))
    if relationship_item:
        evidence_ids.insert(0, str(relationship_item["evidence_id"]))

    sections: list[str] = []
    if relationship_phrases:
        sections.append("关系画像：" + "；".join(relationship_phrases))
    if risk_phrases:
        sections.append("基数风险：" + "；".join(risk_phrases))
    if aggregate_phrases:
        sections.append("原生粒度结果：" + "；".join(aggregate_phrases))
    if aggregates or risk_phrases:
        sections.append(
            "防重复方法：金额或数量先在各原始事实表按共同业务键分别预聚合到一行，"
            "再连接聚合结果；本次总额直接来自源表粒度，未对展开后的 join 行求和"
        )
    if not sections:
        return ()
    evidence_reference = "; ".join(
        f"evidence_id:{item}" for item in dict.fromkeys(evidence_ids)
    )
    return (
        InsightFindingResponse(
            title="多表关系、事实粒度与防重复口径",
            content="。".join(sections) + "。",
            data_source="tool_evidence.source_relationships_and_native_aggregates",
            evidence=evidence_reference,
            confidence="high",
            business_impact=(
                "避免把多个一对多事实表直接连接后重复放大金额、数量或其他可加指标。"
            ),
            recommended_action=(
                "跨事实表分析时保留源表聚合 SQL 和 evidence_id，并在共同业务键粒度预聚合后再连接。"
            ),
        ),
    )


def _relationship_analysis_requested(question: str) -> bool:
    folded = question.casefold()
    return any(
        token in folded
        for token in (
            "关系",
            "关联",
            "连接",
            "基数",
            "粒度",
            "重复",
            "放大",
            "一对多",
            "多对多",
            "join",
        )
    )


def _format_evidence_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.2f}"


def _merge_report_findings(
    preferred: tuple[InsightFindingResponse, ...],
    existing: tuple[InsightFindingResponse, ...],
) -> tuple[InsightFindingResponse, ...]:
    output: list[InsightFindingResponse] = []
    seen: set[str] = set()
    for finding in (*preferred, *existing):
        key = finding.title.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return tuple(output)


def _preserve_mandatory_report_findings(
    report: StructuredReportResponse,
    mandatory: tuple[InsightFindingResponse, ...],
) -> StructuredReportResponse:
    if not mandatory:
        return report
    existing_titles = {item.title.strip().casefold() for item in report.key_findings}
    missing = [
        item for item in mandatory if item.title.strip().casefold() not in existing_titles
    ]
    merged = _merge_report_findings(mandatory, report.key_findings)
    summary = report.executive_summary.strip()
    if missing and missing[0].content not in summary:
        summary = f"{summary} {missing[0].content}".strip()
    next_steps = tuple(
        dict.fromkeys(
            (
                *report.recommended_next_steps,
                *(
                    item.recommended_action
                    for item in mandatory
                    if item.recommended_action
                ),
            )
        )
    )
    return report.model_copy(
        update={
            "executive_summary": summary,
            "key_findings": merged,
            "recommended_next_steps": next_steps[:8],
        }
    )


def _report_node(
    repository: DatasetStoreRepository,
    model_router: AnalysisModelRouter | None,
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        _notify_progress(
            state,
            stage=REPORT_NODE,
            progress=97,
            message="Generating and saving report.",
        )
        dataset_id = state["dataset_id"]
        question = state["question"]
        profile = _require(state.get("profile"), "Planner did not produce a dataset profile.")
        planned_analysis = _require(
            state.get("planned_analysis"),
            "Planner did not produce an analysis plan.",
        )
        python_result = _require(
            state.get("python_result"),
            "Python Agent did not produce analysis results.",
        )
        sql_result = state.get("sql_result")
        rounds = state.get("rounds", ())
        analysis_framework = state.get("analysis_framework")
        draft = _build_report_draft(
            state,
            model_router,
            strategy="llm",
            agent="report",
        )
        final_insights = draft["final_insights"]
        validation_issues = draft["validation_issues"]
        report_charts = draft["report_charts"]
        structured_report = draft["structured_report"]
        report_markdown = draft["report_markdown"]
        html_report = draft["html_report"]
        report_source = draft["report_source"]
        model_router_provider = draft["model_router_provider"]
        model_router_model = draft["model_router_model"]
        model_router_error = draft["model_router_error"]
        if model_router is not None and report_source == "rules" and model_router_error:
            validation_issues = (
                *validation_issues,
                ValidationIssueResponse(
                    severity="info",
                    finding_ref="generate_structured_report",
                    issue="LLM structured report unavailable; rule report was used.",
                    suggestion=str(model_router_error),
                ),
            )
            structured_report = structured_report.model_copy(
                update={"validation_issues": validation_issues}
            )
            report_markdown = _markdown_from_structured_report(structured_report)
            html_report = render_structured_report_html(
                structured_report,
                title="DataMind 分析报告",
            )
        executed_nodes = (*state.get("executed_nodes", ()), REPORT_NODE)
        planner_metadata = state.get("planner_metadata")
        multi_dataset_context = state.get("multi_dataset_context")
        workflow_trace = _workflow_trace(
            state=state,
            executed_nodes=executed_nodes,
            report_source=report_source,
            provider=model_router_provider,
            model=model_router_model,
        )
        report_id = repository.save_report(
            dataset_id=dataset_id,
            title="DataMind 分析报告",
            markdown=report_markdown,
            job_id=state["run_id"],
            metadata={
                "question": question,
                "route": planned_analysis.route,
                "workflow": "langgraph_analysis",
                "nodes": list(executed_nodes),
                "analysis_framework": (
                    analysis_framework.model_dump(mode="json") if analysis_framework else None
                ),
                "planner_metadata": (
                    planner_metadata.model_dump(mode="json") if planner_metadata else None
                ),
                "multi_dataset_context": (
                    multi_dataset_context.model_dump(mode="json") if multi_dataset_context else None
                ),
                "primary_dataset_id": str(dataset_id),
                "dataset_group_id": (
                    str(state.get("dataset_group_id")) if state.get("dataset_group_id") else None
                ),
                "additional_dataset_ids": [
                    str(item) for item in state.get("additional_dataset_ids", ())
                ],
                "join_plan": [item.model_dump(mode="json") for item in state.get("join_plan", ())],
                "relationship_plan": [
                    item.model_dump(mode="json") for item in state.get("relationship_plan", ())
                ],
                "join_summary": (
                    multi_dataset_context.join_summary if multi_dataset_context else {}
                ),
                "workflow_trace": [node.model_dump(mode="json") for node in workflow_trace],
                "multimodal_inputs": [
                    item.model_dump(mode="json") for item in state.get("multimodal_inputs", ())
                ],
                "planner_source": state.get("planner_source", "rules"),
                "python_source": state.get("python_source", "rules"),
                "python_generated_code": state.get("python_generated_code"),
                "python_execution_error": state.get("python_execution_error"),
                "python_attempts": [
                    attempt.model_dump(mode="json") for attempt in state.get("python_attempts", ())
                ],
                "report_source": report_source,
                "model_router_provider": model_router_provider,
                "model_router_model": model_router_model,
                "model_router_error": model_router_error,
                "sql_source": state.get("sql_source", "none"),
                "sql_validation_error": state.get("sql_validation_error"),
                "structured_report": structured_report.model_dump(mode="json"),
                "html_report": html_report,
                "validation_issue_count": len(validation_issues),
                "agent_mode": state.get("agent_mode", "legacy"),
                "loop_summary": state.get("loop_summary", {}),
                "loop_terminal_reason": state.get("loop_terminal_reason"),
            },
        )
        for chart in report_charts:
            repository.save_chart(
                dataset_id=dataset_id,
                title=chart.title,
                chart_type=chart.chart_type,
                chart_spec=chart.spec,
                chart_data=list(chart.data),
            )

        final_response = AnalysisRunResponse(
            dataset_id=dataset_id,
            dataset_group_id=state.get("dataset_group_id"),
            report_id=report_id,
            question=question,
            multimodal_inputs=state.get("multimodal_inputs", ()),
            plan=AnalysisPlanResponse(route=planned_analysis.route, steps=planned_analysis.steps),
            planner_metadata=planner_metadata,
            multi_dataset_context=multi_dataset_context,
            profile=profile,
            analysis_framework=analysis_framework,
            sql_result=sql_result,
            python_result=python_result,
            rounds=rounds,
            final_insights=final_insights,
            validation_issues=validation_issues,
            structured_report=structured_report,
            html_report=html_report,
            python_source=state.get("python_source", "rules"),
            python_generated_code=state.get("python_generated_code"),
            python_execution_error=state.get("python_execution_error"),
            python_attempts=state.get("python_attempts", ()),
            workflow_trace=workflow_trace,
            report_markdown=report_markdown,
            agent_mode=state.get("agent_mode", "legacy"),
            loop_summary=state.get("loop_summary", {}),
            loop_terminal_reason=state.get("loop_terminal_reason"),
        )
        return {
            "rounds": rounds,
            "final_insights": final_insights,
            "validation_issues": validation_issues,
            "structured_report": structured_report,
            "html_report": html_report,
            "report_markdown": report_markdown,
            "report_charts": report_charts,
            "final_response": final_response,
            "executed_nodes": executed_nodes,
            "report_source": report_source,
            "model_router_provider": model_router_provider,
            "model_router_model": model_router_model,
            "model_router_error": model_router_error,
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
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_BOOTSTRAP_NODE),
        }

    return run


def _loop_decide_node(model_router: AnalysisModelRouter | None) -> Any:
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
        if model_router is None:
            return {"loop_pending_call": None, "loop_terminal_reason": "provider_unavailable"}
        settings = get_settings()
        decision_count = state.get("loop_decision_count", 0) + 1
        evidence = [
            {
                "evidence_id": item.get("evidence_id"),
                "tool_name": item.get("tool_name"),
                "status": item.get("status"),
                "summary": item.get("summary"),
                "error_type": item.get("error_type"),
            }
            for item in state.get("tool_evidence", ())[-8:]
        ]
        repair_context = state.get("loop_repair_context")
        decision_messages = [
            {
                "role": "system",
                "content": (
                    "You are DataMind's bounded analysis controller. Select at most one provided tool per turn. "
                    "Use only known columns and evidence IDs. Never request writes, files, network access, identity, or scope. "
                    "For multi-table questions that name original fact tables, first inspect source datasets and use "
                    "aggregate_source_dataset separately for every requested monetary metric. Never finish or fallback "
                    "while an explicitly requested source-table metric lacks native-grain evidence. "
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
                            for column in _require(
                                state.get("profile"), "Missing profile."
                            ).columns[:80]
                        ],
                        "plan": _planned_analysis_payload(
                            _require(state.get("planned_analysis"), "Missing plan.")
                        ),
                        "multi_dataset_context": _compact_multi_dataset_context(
                            state.get("multi_dataset_context")
                        ),
                        "evidence": evidence,
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
                    tools=list(TOOL_DEFINITIONS),
                    tool_choice="auto",
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
            if len(response.tool_calls) > 1:
                _emit_loop_event(
                    state,
                    event_type="invalid_decision",
                    status="failed",
                    message="Model returned multiple tool calls; one call per decision is required.",
                    iteration=state.get("loop_iteration", 0),
                    payload={"tool_call_count": len(response.tool_calls)},
                )
                return {
                    "loop_decision_count": decision_count,
                    "loop_budget": budget,
                    "loop_pending_call": {"action": "retry_decision"},
                    "loop_repair_context": {
                        "error_type": "invalid_decision",
                        "message": "Return exactly one tool call.",
                    },
                }
            if response.tool_calls:
                raw = response.tool_calls[0]
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
                    arguments = {}
                    return {
                        "loop_decision_count": decision_count,
                        "loop_budget": budget,
                        "loop_pending_call": {"action": "retry_decision"},
                        "loop_repair_context": {
                            "error_type": "invalid_arguments",
                            "message": "Tool arguments were not valid JSON.",
                        },
                    }
                pending = {
                    "action": "tool_call",
                    "tool_call_id": str(raw.get("id") or f"call_{decision_count}"),
                    "tool_name": name,
                    "arguments": arguments,
                    "reason": str(function.get("reason") or "")[:240],
                }
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
                    "loop_repair_context": None,
                    "model_router_provider": response.provider,
                    "model_router_model": response.model,
                }
            decision = _parse_loop_text_decision(response.content)
            if decision["action"] == "tool_call":
                name = str(decision.get("tool_name") or "")
                arguments = dict(decision.get("arguments") or {})
                pending = {
                    "action": "tool_call",
                    "tool_call_id": f"json_call_{decision_count}",
                    "tool_name": name,
                    "arguments": arguments,
                    "reason": decision.get("reason"),
                }
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
                    "loop_repair_context": None,
                    "model_router_provider": response.provider,
                    "model_router_model": response.model,
                }
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


def _loop_execute_node(
    repository: DatasetStoreRepository, python_executor: PythonAnalysisExecutor
) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        pending = state.get("loop_pending_call") or {}
        tool_name = str(pending.get("tool_name") or "")
        arguments = dict(pending.get("arguments") or {})
        action_hash = canonical_action_hash(tool_name, arguments)
        next_iteration = state.get("loop_iteration", 0) + 1
        idempotency_key = canonical_action_hash(
            str(state["run_id"]),
            {"iteration": next_iteration, "tool_name": tool_name, "arguments_hash": action_hash},
        )
        idempotency_artifact_id = uuid5(NAMESPACE_URL, f"datamind-agent-action:{idempotency_key}")
        for evidence in state.get("tool_evidence", ()):
            if evidence.get("action_hash") == action_hash and evidence.get("status") == "succeeded":
                return {
                    "loop_last_execution": {**evidence, "cached": True},
                    "loop_iteration": state.get("loop_iteration", 0) + 1,
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
                "cached": True,
            }
            return {
                "loop_iteration": next_iteration,
                "loop_last_execution": execution,
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
                repository.save_artifact(
                    dataset_id=state["dataset_id"],
                    artifact_type="agent_loop_action",
                    content={
                        "idempotency_key": idempotency_key,
                        "tool_name": tool_name,
                        "result": result.result,
                    },
                    file_name=f"{idempotency_key}.json",
                    artifact_id=idempotency_artifact_id,
                    if_absent=True,
                )
                execution["idempotency_artifact_id"] = str(idempotency_artifact_id)
        _emit_loop_event(
            state,
            event_type="tool_execution",
            status=execution["status"],
            message=f"{tool_name} {execution['status']}.",
            iteration=next_iteration,
            tool_name=tool_name,
            payload={
                "arguments_hash": action_hash,
                "idempotency_key": idempotency_key,
                "error_type": execution.get("error_type"),
                "result_summary": _loop_result_summary(execution.get("result")),
            },
        )
        return {
            "loop_iteration": next_iteration,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "tool_attempts": attempts,
            "loop_last_execution": execution,
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_EXECUTE_NODE),
        }

    return run


def _loop_observe_node(repository: DatasetStoreRepository) -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        execution = dict(state.get("loop_last_execution") or {})
        if execution.get("cached") and any(
            item.get("action_hash") == execution.get("action_hash")
            for item in state.get("tool_evidence", ())
        ):
            return {}
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
            "evidence_id": evidence_id,
            "artifact_id": artifact_id,
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
            outcome = (
                "sufficient"
                if valid
                and execution.get("tool_name")
                in {
                    "execute_semantic_query",
                    "execute_safe_sql",
                    "execute_python_analysis",
                    "validate_evidence",
                }
                else "need_more_evidence"
            )
            _emit_loop_event(
                state,
                event_type="verification",
                status="completed" if valid else "failed",
                message=f"Evidence verification: {outcome}.",
                iteration=state.get("loop_iteration", 0),
                tool_name=str(execution.get("tool_name") or "") or None,
                payload={"outcome": outcome, "valid": valid},
            )
            return {
                "loop_pending_call": {"action": "verified", "outcome": outcome},
                "loop_repair_context": None,
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
            },
        }

    return run


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
        evidence = (
            *state.get("tool_evidence", ()),
            {
                "evidence_id": f"ev_{len(state.get('tool_evidence', ())) + 1}",
                "tool_name": "legacy_fallback",
                "action_hash": "legacy_fallback",
                "status": "succeeded",
                "result": fallback,
                "summary": "Deterministic SQL/Python fallback completed.",
            },
        )
        _emit_loop_event(
            state,
            event_type="fallback",
            status="completed",
            message="Deterministic legacy fallback completed.",
            iteration=state.get("loop_iteration", 0),
            tool_name="legacy_fallback",
            payload={"reason": state.get("loop_terminal_reason") or "verification_fallback"},
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
        covered_source_aggregates = {
            (
                str((item.get("result") or {}).get("source_dataset_id") or ""),
                str((item.get("result") or {}).get("metric") or ""),
                str((item.get("result") or {}).get("aggregation") or ""),
            )
            for item in evidence_items
            if isinstance(item.get("result"), dict)
            and (item.get("result") or {}).get("native_grain") is True
        }
        source_guard_count = sum(
            1 for item in evidence_items if item.get("source_guard") is True
        )
        runtime = _loop_runtime(repository, state, run_generated_python_analysis)
        for source_result in runtime.required_source_aggregates():
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
        relationship_guard_count = sum(
            1 for item in evidence_items if item.get("relationship_guard") is True
        )
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
                    item.get("result")
                    for item in evidence_items
                    if item.get("relationship_guard") is True
                    and isinstance(item.get("result"), dict)
                ),
                {},
            )
            relationship_profile = (
                existing_relationship if isinstance(existing_relationship, dict) else {}
            )
        sql_evidence: list[tuple[str, SQLAnalysisResponse]] = []
        python_result: PythonAnalysisResponse | None = None
        charts: list[ChartResponse] = []
        for evidence in evidence_items:
            result = evidence.get("result") if isinstance(evidence.get("result"), dict) else {}
            if not result and evidence.get("artifact_id"):
                artifact = repository.get_artifact(
                    state["dataset_id"], UUID(str(evidence["artifact_id"]))
                )
                content = artifact.get("content")
                result = content if isinstance(content, dict) else {}
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
                python_result = PythonAnalysisResponse.model_validate(result["python_result"])
            if result.get("chart"):
                charts.append(ChartResponse.model_validate(result["chart"]))
        sql_result = _combined_loop_sql_result(sql_evidence)
        if sql_result is None:
            sql_result = _run_sql(dataframe, plan)
        if python_result is None:
            python_result = _run_python(dataframe, plan, sql_result, state["question"])
        if charts:
            python_result = python_result.model_copy(
                update={"charts": (*python_result.charts, *charts)}
            )
        terminal = state.get("loop_terminal_reason") or "evidence_sufficient"
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
        }
        _emit_loop_event(
            state,
            event_type="loop_finalize",
            status="completed",
            message=f"Autonomous loop finalized: {terminal}.",
            iteration=state.get("loop_iteration", 0),
            payload=summary,
        )
        return {
            "sql_result": sql_result,
            "python_result": python_result,
            "rounds": (),
            "tool_evidence": tuple(evidence_items),
            "loop_summary": summary,
            "loop_terminal_reason": terminal,
            "sql_source": "agent_loop",
            "python_source": "agent_loop",
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
        for row in result.rows[:200]:
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "query_index": query_index,
                    **row,
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


def _loop_adversarial_repair_node() -> Any:
    def run(state: AnalysisWorkflowState) -> dict[str, Any]:
        issues = [
            item.model_dump(mode="json")
            for item in state.get("validation_issues", ())
            if item.severity.lower() in {"high", "critical", "error"}
        ]
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
                "message": "Address adversarial validation issues.",
                "issues": issues[:10],
            },
            "loop_pending_call": None,
            "loop_terminal_reason": None,
            "executed_nodes": (*state.get("executed_nodes", ()), LOOP_ADVERSARIAL_REPAIR_NODE),
        }

    return run


def _route_after_framework(state: AnalysisWorkflowState) -> str:
    if state.get("agent_mode") == "loop":
        return LOOP_BOOTSTRAP_NODE
    return _route_after_planner(state)


def _route_after_loop_decide(state: AnalysisWorkflowState) -> str:
    pending = state.get("loop_pending_call") or {}
    action = pending.get("action")
    if action == "tool_call":
        return LOOP_EXECUTE_NODE
    if action == "retry_decision":
        return LOOP_DECIDE_NODE
    if action == "finish" and state.get("tool_evidence"):
        return LOOP_FINALIZE_NODE
    return LOOP_FALLBACK_NODE


def _route_after_loop_verify(state: AnalysisWorkflowState) -> str:
    action = (state.get("loop_pending_call") or {}).get("action")
    if action == "repairable":
        return LOOP_REPAIR_NODE
    if action in {"fallback", "budget_exhausted"}:
        return LOOP_FALLBACK_NODE
    if _loop_budget_exhaustion(state):
        return LOOP_FINALIZE_NODE if state.get("tool_evidence") else LOOP_FALLBACK_NODE
    return LOOP_DECIDE_NODE


def _route_after_adversarial_validate(state: AnalysisWorkflowState) -> str:
    if state.get("agent_mode") != "loop" or state.get("adversarial_repair_count", 0) >= 1:
        return (
            REPORT_DECIDE_NODE
            if state.get("agent_mode") == "loop" and get_settings().report_loop_enabled
            else REPORT_NODE
        )
    high = any(
        item.severity.lower() in {"high", "critical", "error"}
        for item in state.get("validation_issues", ())
    )
    if high:
        return LOOP_ADVERSARIAL_REPAIR_NODE
    return REPORT_DECIDE_NODE if get_settings().report_loop_enabled else REPORT_NODE


def _route_after_report_decide(state: AnalysisWorkflowState) -> str:
    strategy = state.get("report_strategy")
    if strategy == "evidence_gap":
        return LOOP_BOOTSTRAP_NODE
    if strategy == "rules_fallback":
        return REPORT_FALLBACK_NODE
    return REPORT_EXECUTE_NODE


def _route_after_report_verify(state: AnalysisWorkflowState) -> str:
    outcome = str((state.get("report_validation") or {}).get("outcome") or "fallback")
    if outcome == "sufficient":
        return REPORT_COMMIT_NODE
    if outcome == "evidence_gap":
        return LOOP_BOOTSTRAP_NODE
    if outcome == "report_issue":
        return REPORT_REPAIR_NODE
    return REPORT_FALLBACK_NODE


def _loop_budget_exhaustion(state: AnalysisWorkflowState) -> str | None:
    budget = state.get("loop_budget", {})
    if state.get("tool_call_count", 0) >= int(budget.get("max_tool_calls") or 12):
        return "tool_budget_exhausted"
    if state.get("loop_decision_count", 0) >= int(budget.get("max_decisions") or 16):
        return "decision_budget_exhausted"
    if int(budget.get("used_tokens") or 0) >= int(budget.get("max_tokens") or 50_000):
        return "token_budget_exhausted"
    if budget.get("deadline_epoch") and time.time() >= float(budget["deadline_epoch"]):
        return "time_budget_exhausted"
    return None


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


def _route_after_planner(state: AnalysisWorkflowState) -> str:
    planned_analysis = _require(
        state.get("planned_analysis"),
        "Planner did not produce an analysis plan.",
    )
    if planned_analysis.route in {"sql", "hybrid"}:
        return SQL_NODE
    return PYTHON_NODE


def _framework_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
                "distinct_count": column.distinct_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=5),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "framework", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind design_framework. Return only JSON with keys: "
                "business_question, candidate_dimensions, candidate_metrics, likely_routes, "
                "initial_hypotheses, risk_notes, key_questions, success_criteria. "
                "Return at most 3 initial_hypotheses and at most 6 items in every other list. "
                "Use only provided column names. Use experience_context only for business "
                "priority, risk framing, and analysis style; never invent evidence from it."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _round_plan_messages(
    *,
    question: str,
    hypothesis: str,
    profile: DatasetProfileResponse,
    previous_rounds: tuple[AnalysisRoundResponse, ...],
    fallback_plan: PlannedAnalysis,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "hypothesis": _truncate_text(hypothesis, 1200),
        "fallback_plan": {
            "route": fallback_plan.route,
            "metric_column": fallback_plan.metric_column,
            "category_column": fallback_plan.category_column,
            "time_column": fallback_plan.time_column,
            "steps": list(fallback_plan.steps),
        },
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "previous_rounds": [
            {
                "round_number": round_item.round_number,
                "hypothesis": round_item.hypothesis.statement,
                "route": round_item.plan.route,
                "reflection": round_item.reflection.insight_text,
            }
            for round_item in previous_rounds
        ],
        "experience_context": _experience_context(
            "round_plan", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind generate_plan inside an iterative analysis loop. "
                "Return only compact JSON with keys: route, category_column, metric_column, "
                "time_column, steps. route must be sql, python, or hybrid. Use only provided "
                "column names and avoid repeating previous rounds. Use experience_context to "
                "prioritize useful checks, but keep every step executable against the dataset. "
                "For joined data, respect column_source_map and row-expansion warnings before choosing metrics."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _round_python_messages(
    *,
    question: str,
    hypothesis: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "hypothesis": _truncate_text(hypothesis, 1200),
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=10),
        "sql_rows": list(sql_result.rows[:20]) if sql_result else [],
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "python", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind round_python Agent. Generate Python code only, no Markdown. "
                "Define exactly one function analyze(df). Allowed imports are only pandas, numpy, "
                "math, statistics, re, json, datetime, collections, and itertools. This call is only for "
                "statistics and insights, not charts. Analyze both numeric "
                "and text columns. For review/comment/text data, compute text length, keyword "
                "frequency, and group comparisons such as sentiment/label. "
                "Prefer pandas/numpy analysis code that directly answers the user's question. "
                "All insight strings and human-readable labels must be Chinese. "
                "Return a dict with keys statistics, insights, charts. charts must be an empty list []. "
                "Keep code compact: no long comments, no chart builders, at most 5 insights. "
                "Use experience_context only to choose practical statistics. Keep statistics compact: "
                "do not return full describe() tables for many grouped columns or row-level records. "
                "For joined data, verify metric source and grain before aggregation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _reflection_messages(
    *,
    question: str,
    rounds: tuple[AnalysisRoundResponse, ...],
    python_result: PythonAnalysisResponse,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "rounds": [round_item.model_dump(mode="json") for round_item in rounds],
        "python_insights": list(python_result.insights),
        "experience_context": _experience_context("reflection"),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind reflect_on_result. Return only JSON with key reflections, "
                "an array of concise Chinese reflection strings aligned to the rounds. Use "
                "experience_context for review criteria, not as evidence."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _integrate_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    rounds: tuple[AnalysisRoundResponse, ...],
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse,
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "profile": _compact_profile(profile),
        "rounds": [_compact_round(round_item) for round_item in rounds],
        "sql_result": _compact_sql_result(sql_result),
        "python_statistics": _compact_python_statistics(python_result.statistics),
        "python_insights": list(python_result.insights),
        "charts": [_compact_chart(chart) for chart in python_result.charts],
        "multimodal_context": _multimodal_payload(multimodal_inputs),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "integrate", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind integrate_insights. Return only JSON with key insights. "
                "Each insight must include title, content, data_source, evidence, confidence, "
                "business_impact, recommended_action, and impact_pct. Do not invent evidence. "
                "Return at most 8 insights. "
                "Multimodal context may guide interpretation but dataset-backed claims must cite "
                "SQL, Python, chart, or round evidence. Use experience_context to rank insight "
                "importance and choose report language."
            ),
        },
        {
            "role": "user",
            "content": _multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def _review_messages(
    *,
    question: str,
    final_insights: tuple[InsightFindingResponse, ...],
    charts: tuple[ChartResponse, ...],
    sql_result: SQLAnalysisResponse | None,
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "final_insights": [finding.model_dump(mode="json") for finding in final_insights],
        "charts": [_compact_chart(chart) for chart in charts],
        "sql_result": _compact_sql_result(sql_result),
        "multimodal_context": _multimodal_payload(multimodal_inputs),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context("review"),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind adversarial_validate. Return only JSON with key issues. "
                "Each issue has severity, finding_ref, issue, suggestion. Return at most 10 issues. "
                "Flag unsupported claims, chart/text mismatch, over-attribution, data gaps, hallucinated fields, "
                "and any misuse of multimodal context as if it were tabular evidence. Apply "
                "experience_context as additional review criteria."
            ),
        },
        {
            "role": "user",
            "content": _multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def _chart_refine_messages(
    *,
    question: str,
    charts: tuple[ChartResponse, ...],
    final_insights: tuple[InsightFindingResponse, ...],
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "charts": [_compact_chart(chart) for chart in charts],
        "final_insights": [finding.model_dump(mode="json") for finding in final_insights],
        "multimodal_context": _multimodal_payload(multimodal_inputs),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context("chart_refine"),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind chart_refine. Return only JSON with key chart_explanations. "
                "Each item has title and explanation. Explain what the chart proves and the "
                "business reading. Do not invent numbers. Use experience_context for chart "
                "communication style and priority."
            ),
        },
        {
            "role": "user",
            "content": _multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def _planner_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    schema = {
        "columns": [
            {
                "name": column.name,
                "is_numeric": column.is_numeric,
                "dtype": column.dtype,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "planner", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind Planner Agent. Return only compact JSON with keys: "
                "route, category_column, metric_column, time_column, steps. "
                "route must be one of sql, python, hybrid. Use only provided column names. "
                "Use experience_context as planning guidance, not as data evidence. For joined data, "
                "use multi_dataset_context to respect field provenance, skipped joins, and row expansion."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"question": _truncate_text(question, 2000), "dataset_schema": schema},
                ensure_ascii=False,
            ),
        },
    ]


def _report_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    structured_report: StructuredReportResponse,
    multimodal_inputs: tuple[MultimodalInputResponse, ...] = (),
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "fallback_structured_report": _compact_structured_report(structured_report),
        "multimodal_context": _multimodal_payload(multimodal_inputs),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "report", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind generate_structured_report. Return only JSON, no Markdown. "
                "Generate a high-quality Chinese structured data analysis report. Required keys: "
                "executive_summary, analysis_context, key_findings, chart_explanations, "
                "data_gaps, validation_issues, recommended_next_steps. key_findings must be an "
                "array with title, content, data_source, evidence, confidence, business_impact, "
                "recommended_action, impact_pct. Do not invent columns, rows, or metrics beyond "
                "the payload. Keep all dataset claims traceable to SQL, Python, chart, or round "
                "evidence. Findings whose data_source starts with tool_evidence are mandatory: "
                "preserve their source-table totals, relationship direction, fact grain, cardinality "
                "risk, prevention method, and evidence_id in both the executive summary and key findings. "
                "Multimodal context can enrich explanation and data-gap notes but must "
                "not be treated as verified tabular data unless the payload provides matching evidence. "
                "Use experience_context to shape narrative quality, prioritization, and review discipline."
            ),
        },
        {
            "role": "user",
            "content": _multimodal_content(
                json.dumps(payload, ensure_ascii=False), multimodal_inputs
            ),
        },
    ]


def _python_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=10),
        "sql_rows": list(sql_result.rows[:20]) if sql_result else [],
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "python", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind Python Agent. Generate Python code only, no Markdown. "
                "Define exactly one function analyze(df). Allowed imports are only pandas, numpy, "
                "math, statistics, re, json, datetime, collections, and itertools. This call is only for "
                "statistics and insights, not charts. Analyze both numeric "
                "and text columns. For review/comment/text data, compute text length, keyword "
                "frequency, and group comparisons such as sentiment/label. "
                "Prefer pandas/numpy analysis code that directly answers the user's question. "
                "All insight strings and human-readable labels must be Chinese. Return a dict "
                "with keys statistics, insights, charts. charts must be an empty list []. "
                "Keep code compact: no long comments, no chart builders, at most 7 insights. "
                "Use experience_context to choose useful statistics, but all outputs must come from df. "
                "Keep statistics compact: do not return full describe() tables for many grouped columns "
                "or row-level records. For joined data, verify metric source and grain before aggregation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _python_chart_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    hypothesis: str | None = None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "hypothesis": _truncate_text(hypothesis, 1200) if hypothesis else None,
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=12),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=12),
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=8),
        "sql_rows": list(sql_result.rows[:20]) if sql_result else [],
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind Python chart Agent. Generate Python code only, no Markdown. "
                "Define exactly one function analyze(df). This call is only for chart construction. "
                "Allowed imports are only pandas, numpy, math, statistics, re, json, datetime, collections, and itertools. "
                "Return {'statistics': {}, 'insights': [], 'charts': charts}. Generate the charts needed "
                "to answer the question, but keep code short and avoid repetitive chart builders. "
                "Each chart must be a serializable dict with title, chart_type, spec, data. "
                "Supported chart_type values: bar, line, pie, histogram, box_plot, correlation_heatmap. "
                "Do not return raw row-level data for large charts. Hard rules: each chart data list must "
                "have at most 500 rows and at most 8 fields per row; histogram must use pre-binned rows "
                "such as bin_start/bin_end/count with at most 30 bins; box_plot must use five-number "
                "summary rows such as group/min/q1/median/q3/max/count, not raw observations; any "
                "to_dict('records') output must be aggregated or sampled/head-limited before returning. "
                "Keep code compact and complete. Avoid comments and long strings. All chart titles and labels must be Chinese."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


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


def _python_repair_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    sql_result: SQLAnalysisResponse | None,
    attempts: tuple[PythonCodeAttemptResponse, ...],
    phase: str,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
    hypothesis: str | None = None,
) -> list[dict[str, str]]:
    truncation_like = _has_truncation_like_python_failure(attempts)
    payload = {
        "question": _truncate_text(question, 2000),
        "hypothesis": _truncate_text(hypothesis, 1200) if hypothesis else None,
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=10),
        "sql_rows": list(sql_result.rows[:20]) if sql_result else [],
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "phase": phase,
        "failed_attempts": [
            {
                "attempt": item.attempt,
                "code": _truncate_text(item.code or "", 6000),
                "error": _truncate_text(item.error or "", 1200),
            }
            for item in attempts
        ],
        "repair_mode": "concise_truncation_repair" if truncation_like else "normal_repair",
        "repair_instructions": (
            "The previous code appears truncated by an output/token limit. Generate a much shorter "
            "complete function. Avoid long comments, long strings, repeated chart builders, and "
            "large inline chart specs. Return fewer insights and fewer charts only if needed to keep the code complete."
            if truncation_like
            else (
                "Fix the concrete runtime or validation errors without changing the output contract. "
                "If an error says generated output exceeded the size limit, reduce returned payload size: "
                "aggregate or sample to_dict('records'), pre-bin histograms to at most 30 bins, summarize "
                "box plots with min/q1/median/q3/max/count, and keep each chart data list under 500 rows."
            )
        ),
        "output_contract": _python_phase_contract(phase),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind Python code repair Agent. Generate corrected Python "
                "code only, no Markdown. The previous generated code failed in the "
                "sandbox. Read every failed_attempts item and fix the code without "
                "repeating the same error. Keep the code executable against the provided "
                "df and obey the phase-specific output_contract exactly. All human-readable insights and chart "
                "titles must be Chinese. If repair_mode is concise_truncation_repair, "
                "the likely cause is token truncation: write compact code, no comments, "
                "reduce insights/charts only as needed, and ensure every string/bracket is closed."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _python_phase_contract(phase: str) -> str:
    safety = (
        "Allowed imports: pandas, numpy, math, statistics, re, json, datetime, collections, itertools. "
        "Do not use filesystem, network, eval, exec, open, os, sys, subprocess, requests, or blocked calls."
    )
    if phase in {"python_charts", "round_python_charts"}:
        return (
            "Define exactly one function analyze(df). Return {'statistics': {}, 'insights': [], 'charts': charts}. "
            "Charts must contain title, chart_type, spec, data. Keep at most 500 rows per chart, pre-bin "
            "histograms to at most 30 bins, and summarize box plots. Do not return statistics or insights. "
            + safety
        )
    return (
        "Define exactly one function analyze(df). Return keys statistics, insights, charts. "
        "charts must be exactly []. Keep statistics compact and return at most 7 Chinese insights. "
        "Do not generate chart payloads in this phase. " + safety
    )


def _has_truncation_like_python_failure(attempts: tuple[PythonCodeAttemptResponse, ...]) -> bool:
    truncation_markers = (
        "unterminated string literal",
        "unexpected eof",
        "was never closed",
        "eof while scanning",
        "eol while scanning",
        "invalid syntax",
    )
    for attempt in attempts:
        error = (attempt.error or "").lower()
        code = attempt.code or ""
        if any(marker in error for marker in truncation_markers):
            if _looks_incomplete_python_code(code):
                return True
            if "unterminated string literal" in error or "was never closed" in error:
                return True
    return False


def _looks_incomplete_python_code(code: str) -> bool:
    stripped = code.rstrip()
    if not stripped:
        return False
    if stripped.endswith((",", "(", "[", "{", ":", "\\", "'x", '"x')):
        return True
    return (
        stripped.count("(") > stripped.count(")")
        or stripped.count("[") > stripped.count("]")
        or stripped.count("{") > stripped.count("}")
    )


def _combine_python_generated_code(
    *,
    statistics_code: str | None,
    chart_code: str | None,
) -> str | None:
    if statistics_code and chart_code:
        return f"{statistics_code}\n\n# --- DataMind chart generation phase ---\n{chart_code}"
    return statistics_code or chart_code


def _sql_messages(
    *,
    question: str,
    profile: DatasetProfileResponse,
    planned_analysis: PlannedAnalysis,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": _truncate_text(question, 2000),
        "route": planned_analysis.route,
        "category_column": planned_analysis.category_column,
        "metric_column": planned_analysis.metric_column,
        "time_column": planned_analysis.time_column,
        "columns": [
            {
                "name": column.name,
                "is_numeric": column.is_numeric,
                "dtype": column.dtype,
            }
            for column in profile.columns[:60]
        ],
        "numeric_columns": compact_prompt_columns(profile.numeric_columns, max_items=20),
        "categorical_columns": compact_prompt_columns(profile.categorical_columns, max_items=20),
        "multi_dataset_context": _compact_multi_dataset_context(multi_dataset_context),
        "experience_context": _experience_context(
            "sql", tuple(column.name for column in profile.columns)
        ),
    }
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You are DataMind SQL Agent. Generate exactly one DuckDB SQL query. "
                "Return only SQL, no Markdown. The query must start with SELECT, "
                "must read only from the temporary table named dataset, and must not "
                "use DROP, DELETE, UPDATE, INSERT, ATTACH, or COPY. Use experience_context "
                "only to choose useful aggregations or filters; never reference columns outside "
                "the provided schema. For joined data, inspect multi_dataset_context: do not SUM or AVG "
                "a metric across expanded duplicate rows unless the query explicitly restores its source grain."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


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
        metric_column=_optional_str(payload.get("metric_column")),
        time_column=_optional_str(payload.get("time_column")),
        steps=steps or fallback.steps,
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
            round_plan = _sanitize_model_plan(model_plan, profile)
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


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated]"


def _compact_profile(profile: DatasetProfileResponse) -> dict[str, Any]:
    return {
        "dataset_id": str(profile.dataset_id),
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "missing_value_count": profile.missing_value_count,
        "missing_value_ratio": profile.missing_value_ratio,
        "duplicate_row_count": profile.duplicate_row_count,
        "numeric_columns": list(profile.numeric_columns[:20]),
        "categorical_columns": list(profile.categorical_columns[:20]),
        "columns": [
            {
                "name": column.name,
                "dtype": column.dtype,
                "is_numeric": column.is_numeric,
                "missing_count": column.missing_count,
                "distinct_count": column.distinct_count,
                "min_value": column.min_value,
                "max_value": column.max_value,
                "mean": column.mean,
            }
            for column in profile.columns[:60]
        ],
        "sample_records": compact_prompt_records(profile.sample_records, max_rows=5),
    }


def _compact_sql_result(sql_result: SQLAnalysisResponse | None) -> dict[str, Any] | None:
    if sql_result is None:
        return None
    return {
        "sql": sql_result.sql,
        "explanation": sql_result.explanation,
        "row_count": len(sql_result.rows),
        "rows_sample": list(sql_result.rows[:30]),
    }


def _compact_chart(chart: ChartResponse) -> dict[str, Any]:
    return {
        "title": chart.title,
        "chart_type": chart.chart_type,
        "spec": chart.spec,
        "data_row_count": len(chart.data),
        "data_sample": list(chart.data[:30]),
        "explanation": chart.explanation,
        "related_finding_ids": list(chart.related_finding_ids),
    }


def _compact_python_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in statistics.items():
        if key == "numeric_summary" and isinstance(value, dict):
            compact[key] = {
                str(column): summary
                for column, summary in islice(value.items(), 12)
                if isinstance(summary, dict)
            }
        elif key == "text_analysis" and isinstance(value, list):
            compact[key] = [
                {
                    "task": item.get("task"),
                    "text_column": item.get("text_column"),
                    "group_column": item.get("group_column"),
                    "summary": item.get("summary"),
                    "insights": item.get("insights"),
                }
                for item in value[:5]
                if isinstance(item, dict)
            ]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = value[:20]
        elif isinstance(value, dict):
            compact[key] = dict(islice(value.items(), 20))
    return compact


def _compact_round(round_item: AnalysisRoundResponse) -> dict[str, Any]:
    return {
        "round_number": round_item.round_number,
        "hypothesis": round_item.hypothesis.model_dump(mode="json"),
        "plan": round_item.plan.model_dump(mode="json"),
        "reflection": round_item.reflection.model_dump(mode="json"),
        "execution_result": _compact_execution_result(round_item.execution_result),
        "charts": [_compact_chart(chart) for chart in round_item.charts[:5]],
        "validation_status": round_item.validation_status,
    }


def _compact_execution_result(execution_result: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "sql_row_count",
        "chart_count",
        "text_analysis_count",
        "python_source",
        "python_execution_error",
        "fanout_mode",
        "fanout_group",
        "plan_validation_issues",
    }
    compact = {key: value for key, value in execution_result.items() if key in allowed_keys}
    if "insight" in execution_result:
        compact["insight"] = _truncate_text(str(execution_result["insight"]), 1200)
    if "statistics_keys" in execution_result:
        compact["statistics_keys"] = execution_result["statistics_keys"]
    return compact


def _compact_structured_report(report: StructuredReportResponse) -> dict[str, Any]:
    payload = report.model_dump(mode="json")
    payload["charts"] = [_compact_chart(chart) for chart in report.charts[:10]]
    payload["sql_results"] = list(report.sql_results[:30])
    payload["python_results"] = _compact_python_statistics(report.python_results)
    payload["analysis_trace"] = [
        _compact_round(round_item) for round_item in report.analysis_trace[:6]
    ]
    return payload


def _multimodal_payload(
    multimodal_inputs: tuple[MultimodalInputResponse, ...],
) -> list[dict[str, str | None]]:
    return [
        {
            "kind": item.kind,
            "title": _truncate_text(item.title or "", 240) or None,
            "description": _truncate_text(item.description or "", 1200) or None,
            "source_ref": _truncate_text(item.source_ref or "", 500) or None,
            "media_type": item.media_type,
            "has_native_payload": "true" if item.data_url else "false",
            "processing_status": item.processing_status,
            "text_excerpt": _truncate_text(item.text_excerpt or "", 1200) or None,
        }
        for item in multimodal_inputs[:8]
    ]


def _multimodal_content(
    text: str,
    multimodal_inputs: tuple[MultimodalInputResponse, ...],
) -> str | list[dict[str, Any]]:
    image_parts = [
        {
            "type": "image_url",
            "image_url": {
                "url": item.data_url,
                "detail": "auto",
            },
        }
        for item in multimodal_inputs[:4]
        if item.data_url
        and item.kind in {"image", "chart", "screenshot"}
        and (item.media_type or "").startswith("image/")
    ]
    if not image_parts:
        return text
    return [{"type": "text", "text": text}, *image_parts]


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


def _parse_model_validation_issues(content: str) -> tuple[ValidationIssueResponse, ...]:
    payload = _extract_json_object(content)
    if payload is None:
        return ()
    raw_items = payload.get("issues")
    if not isinstance(raw_items, list):
        return ()
    issues: list[ValidationIssueResponse] = []
    for item in raw_items[:10]:
        if not isinstance(item, dict):
            continue
        issue = str(item.get("issue") or "").strip()
        if not issue:
            continue
        issues.append(
            ValidationIssueResponse(
                severity=str(item.get("severity") or "warning"),
                finding_ref=str(item.get("finding_ref") or "report"),
                issue=issue,
                suggestion=str(item.get("suggestion") or ""),
            )
        )
    return tuple(issues)


def _parse_model_structured_report(
    content: str,
    *,
    fallback: StructuredReportResponse,
    provider: str,
    model: str,
) -> StructuredReportResponse:
    payload = _extract_json_object(content)
    if payload is None:
        raise ValueError("Model Router report did not return JSON.")
    executive_summary = str(payload.get("executive_summary") or "").strip()
    if len(executive_summary) < 20:
        raise ValueError("Model Router report executive summary was too short.")

    key_findings = _parse_report_findings(payload.get("key_findings"))
    validation_issues = _parse_report_validation_issues(payload.get("validation_issues"))
    chart_explanations = _parse_chart_explanation_strings(payload.get("chart_explanations"))
    report_charts = _apply_report_chart_explanations(
        charts=fallback.charts,
        chart_explanations=chart_explanations,
    )
    return StructuredReportResponse(
        executive_summary=executive_summary,
        analysis_context=str(payload.get("analysis_context") or fallback.analysis_context),
        key_findings=key_findings or fallback.key_findings,
        charts=report_charts,
        chart_explanations=chart_explanations or fallback.chart_explanations,
        sql_results=fallback.sql_results,
        python_results=fallback.python_results,
        data_gaps=tuple(_string_list(payload.get("data_gaps"))) or fallback.data_gaps,
        validation_issues=validation_issues or fallback.validation_issues,
        recommended_next_steps=(
            tuple(_string_list(payload.get("recommended_next_steps")))
            or fallback.recommended_next_steps
        ),
        analysis_trace=fallback.analysis_trace,
        provider=provider,
        model=model,
    )


def _parse_report_findings(value: object) -> tuple[InsightFindingResponse, ...]:
    if not isinstance(value, list):
        return ()
    findings: list[InsightFindingResponse] = []
    for index, item in enumerate(value[:8], 1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        data_source = str(item.get("data_source") or item.get("evidence") or "").strip()
        if not content or not data_source:
            continue
        findings.append(
            InsightFindingResponse(
                title=str(item.get("title") or f"洞察 {index}"),
                content=content,
                data_source=data_source,
                impact_pct=_float_payload(item.get("impact_pct")),
                evidence=str(item.get("evidence") or data_source),
                confidence=str(item.get("confidence") or "medium"),
                business_impact=str(item.get("business_impact") or ""),
                recommended_action=str(item.get("recommended_action") or ""),
            )
        )
    return tuple(findings)


def _parse_report_validation_issues(value: object) -> tuple[ValidationIssueResponse, ...]:
    if not isinstance(value, list):
        return ()
    issues: list[ValidationIssueResponse] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        issue = str(item.get("issue") or "").strip()
        if not issue:
            continue
        issues.append(
            ValidationIssueResponse(
                severity=str(item.get("severity") or "info"),
                finding_ref=str(item.get("finding_ref") or "report"),
                issue=issue,
                suggestion=str(item.get("suggestion") or ""),
            )
        )
    return tuple(issues)


def _parse_chart_explanation_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    explanations: list[str] = []
    for item in value[:12]:
        if isinstance(item, dict):
            explanation = str(item.get("explanation") or "").strip()
            title = str(item.get("title") or "").strip()
            if explanation:
                explanations.append(f"{title}: {explanation}" if title else explanation)
        else:
            explanation = str(item).strip()
            if explanation:
                explanations.append(explanation)
    return tuple(explanations)


def _apply_report_chart_explanations(
    *,
    charts: tuple[ChartResponse, ...],
    chart_explanations: tuple[str, ...],
) -> tuple[ChartResponse, ...]:
    if not chart_explanations:
        return charts
    updated: list[ChartResponse] = []
    for chart in charts:
        explanation = chart.explanation
        for item in chart_explanations:
            prefix = f"{chart.title}:"
            if item.startswith(prefix):
                explanation = item.removeprefix(prefix).strip()
                break
        updated.append(
            ChartResponse(
                title=chart.title,
                chart_type=chart.chart_type,
                spec=chart.spec,
                data=chart.data,
                explanation=explanation,
                related_finding_ids=chart.related_finding_ids,
            )
        )
    return tuple(updated)


def _markdown_from_structured_report(report: StructuredReportResponse) -> str:
    lines = [
        "# DataMind 分析报告",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
    ]
    if report.analysis_context:
        lines.extend(["## Analysis Context", report.analysis_context, ""])
    if report.key_findings:
        lines.append("## Key Findings")
        for finding in report.key_findings:
            lines.append(f"- **{finding.title}**: {finding.content}")
            if finding.evidence:
                lines.append(f"  - 证据: {finding.evidence}")
            if finding.recommended_action:
                lines.append(f"  - 建议: {finding.recommended_action}")
        lines.append("")
    if report.charts:
        lines.extend(
            [
                "## Visualizations",
                *[
                    f"- {chart.title} ({chart.chart_type}): {chart.explanation or '见图表数据。'}"
                    for chart in report.charts
                ],
                "",
            ]
        )
    if report.data_gaps:
        lines.extend(["## Data Gaps", *[f"- {gap}" for gap in report.data_gaps], ""])
    if report.validation_issues:
        lines.extend(
            [
                "## Validation Issues",
                *[
                    f"- {issue.severity} · {issue.finding_ref}: {issue.issue}"
                    for issue in report.validation_issues
                ],
                "",
            ]
        )
    if report.recommended_next_steps:
        lines.extend(
            [
                "## Recommended Next Steps",
                *[f"- {step}" for step in report.recommended_next_steps],
                "",
            ]
        )
    return "\n".join(lines)


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
            explanation=explanation_by_title.get(chart.title, chart.explanation),
            related_finding_ids=chart.related_finding_ids,
        )
        for chart in charts
    )


def _sanitize_model_plan(
    planned_analysis: PlannedAnalysis,
    profile: DatasetProfileResponse,
) -> PlannedAnalysis:
    available_columns = {column.name for column in profile.columns}
    metric_columns = set(profile.numeric_columns)
    metric_column = _column_or_none(planned_analysis.metric_column, metric_columns)
    category_column = _column_or_none(planned_analysis.category_column, available_columns)
    time_column = _column_or_none(planned_analysis.time_column, available_columns)
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
    candidate_metrics = tuple(profile.numeric_columns[:8])
    candidate_dimensions = tuple(profile.categorical_columns[:8])
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
        route_reason += f" 模型规划不可用，已使用规则 fallback：{error}"
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


def _workflow_trace(
    *,
    state: AnalysisWorkflowState,
    executed_nodes: tuple[str, ...],
    report_source: str,
    provider: str | None,
    model: str | None,
) -> tuple[WorkflowTraceNodeResponse, ...]:
    trace: list[WorkflowTraceNodeResponse] = []
    for node in executed_nodes:
        error = _node_error(state, node)
        trace.append(
            WorkflowTraceNodeResponse(
                node=node,
                status="fallback" if error else "completed",
                provider=provider
                if node
                in {
                    REPORT_NODE,
                    INTEGRATE_INSIGHTS_NODE,
                    ADVERSARIAL_VALIDATE_NODE,
                    FORMAT_CHARTS_NODE,
                }
                else state.get("model_router_provider"),
                model=model
                if node
                in {
                    REPORT_NODE,
                    INTEGRATE_INSIGHTS_NODE,
                    ADVERSARIAL_VALIDATE_NODE,
                    FORMAT_CHARTS_NODE,
                }
                else state.get("model_router_model"),
                input_summary=_node_input_summary(state, node),
                output_summary=_node_output_summary(state, node, report_source=report_source),
                fallback=_node_fallback(state, node, report_source=report_source),
                error=error,
            )
        )
    return tuple(trace)


def _node_error(state: AnalysisWorkflowState, node: str) -> str | None:
    if node == JOIN_PREPARE_NODE:
        context = state.get("multi_dataset_context")
        if context and context.validation_issues:
            return "; ".join(issue.issue for issue in context.validation_issues[:2])
        return None
    if node == PLANNER_NODE:
        return state.get("model_router_error") if state.get("planner_source") == "rules" else None
    if node == SQL_NODE:
        return state.get("sql_validation_error")
    if node == PYTHON_NODE:
        error = state.get("python_execution_error")
        attempts = state.get("python_attempts", ())
        if error and len(attempts) >= 3:
            return f"LLM Python code failed after 3 attempts: {error}"
        return error
    if node == REPORT_NODE:
        return state.get("model_router_error") if state.get("report_source") == "rules" else None
    return None


def _node_fallback(
    state: AnalysisWorkflowState,
    node: str,
    *,
    report_source: str,
) -> str | None:
    if node == PLANNER_NODE and state.get("planner_source") == "rules":
        return "rule_planner"
    if node == SQL_NODE and state.get("sql_source") == "rules":
        return "rule_sql"
    if node == PYTHON_NODE and state.get("python_source") == "rules":
        return "rule_python"
    if node == REPORT_NODE and report_source == "rules":
        return "rule_report"
    return None


def _node_input_summary(state: AnalysisWorkflowState, node: str) -> str:
    profile = state.get("profile")
    if node == JOIN_PREPARE_NODE:
        additional_count = len(state.get("additional_dataset_ids", ()))
        join_count = len(state.get("join_plan", ()))
        return f"主数据集 + {additional_count} 个附加数据集，{join_count} 个 join 配置。"
    if node == PLANNER_NODE and profile is not None:
        return f"{profile.row_count} 行，{profile.column_count} 列，问题：{state['question']}"
    if node == SQL_NODE:
        return "使用 planner 输出和 dataframe 执行安全 SELECT。"
    if node == PYTHON_NODE:
        return "使用 dataframe、planner 输出和 SQL 结果生成 Python 分析。"
    if node == REPORT_NODE:
        return "整合 SQL、Python、图表、验证问题和多轮分析 trace。"
    return "使用上游节点输出。"


def _node_output_summary(
    state: AnalysisWorkflowState,
    node: str,
    *,
    report_source: str,
) -> str:
    if node == PLANNER_NODE:
        plan = state.get("planned_analysis")
        return f"route={plan.route if plan else '-'}"
    if node == JOIN_PREPARE_NODE:
        context = state.get("multi_dataset_context")
        if not context:
            return "单数据集分析"
        summary = context.join_summary
        return (
            f"mode={summary.get('mode', '-')}, "
            f"joined_datasets={summary.get('joined_dataset_count', 1)}/{summary.get('dataset_count', 1)}, "
            f"joined_rows={summary.get('joined_row_count', '-')}, "
            f"joined_columns={summary.get('joined_column_count', '-')}, "
            f"row_expansion={summary.get('row_expansion_ratio', 1)}x, "
            f"skipped_joins={summary.get('skipped_join_count', 0)}"
        )
    if node == SQL_NODE:
        sql_result = state.get("sql_result")
        return f"{len(sql_result.rows) if sql_result else 0} 行 SQL 结果"
    if node == PYTHON_NODE:
        python_result = state.get("python_result")
        attempts = state.get("python_attempts", ())
        attempt_summary = f"，Python 代码尝试 {len(attempts)} 次" if attempts else ""
        return f"{len(python_result.insights) if python_result else 0} 条洞察，{len(python_result.charts) if python_result else 0} 个图表{attempt_summary}"
    if node == REPORT_NODE:
        return f"report_source={report_source}"
    if node == ROUND_REFLECT_NODE:
        return f"{len(state.get('rounds', ()))} 个分析轮次"
    return "节点完成"


def _merge_model_report(*, base_report: str, model_content: str) -> str:
    narrative = model_content.strip()
    if len(narrative) < 40:
        raise ValueError("Model Router report was too short.")
    if narrative.startswith("[mock:"):
        raise ValueError("Mock model output is not a usable report narrative.")
    return f"{base_report}\n\n## Model Router Narrative\n\n{narrative}\n"


def _json_repair_messages(
    *,
    stage: str,
    invalid_content: str | None,
    error: str,
    contract: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _prompt_system(
                "You repair structured model output. Return exactly one valid JSON object "
                "and no Markdown, commentary, or reasoning. Preserve only claims already "
                "present in the supplied output; do not invent facts or numbers."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "stage": stage,
                    "validation_error": _truncate_text(error, 1000),
                    "output_contract": contract,
                    "invalid_output": _truncate_text(invalid_content or "", 8000),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _extract_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            nested = _extract_json_object(payload)
            if nested is not None:
                return nested
    return None


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))


def _float_payload(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _column_or_none(value: str | None, available_columns: set[str]) -> str | None:
    return value if value in available_columns else None


def _require[T](value: T | None, message: str) -> T:
    if value is None:
        raise RuntimeError(message)
    return value
