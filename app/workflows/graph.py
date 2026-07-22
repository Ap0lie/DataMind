from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.workflows.models import (
    AgentNodeName,
    WorkflowCheckpoint,
    WorkflowError,
    WorkflowState,
    WorkflowStatus,
    WorkflowTraceEntry,
)
from app.workflows.nodes import WorkflowAgents

NodeOutput = dict[str, Any]
NodeCallable = Callable[[WorkflowState], Awaitable[NodeOutput]]


def build_datamind_workflow(
    agents: WorkflowAgents,
    *,
    checkpointer: Any | None = None,
    debug: bool = False,
) -> Any:
    graph: Any = StateGraph(WorkflowState)
    _add_node(graph, AgentNodeName.PLANNER, _wrap(AgentNodeName.PLANNER, _planner_node(agents)))
    _add_node(graph, AgentNodeName.SEARCH, _wrap(AgentNodeName.SEARCH, _search_node(agents)))
    _add_node(graph, AgentNodeName.CRAWL, _wrap(AgentNodeName.CRAWL, _crawl_node(agents)))
    _add_node(graph, AgentNodeName.PARSER, _wrap(AgentNodeName.PARSER, _parser_node(agents)))
    _add_node(graph, AgentNodeName.NLP, _wrap(AgentNodeName.NLP, _nlp_node(agents)))
    _add_node(
        graph,
        AgentNodeName.KNOWLEDGE,
        _wrap(AgentNodeName.KNOWLEDGE, _knowledge_node(agents)),
    )
    _add_node(
        graph,
        AgentNodeName.REVIEWER,
        _wrap(AgentNodeName.REVIEWER, _reviewer_node(agents)),
    )
    _add_node(graph, AgentNodeName.REPORT, _wrap(AgentNodeName.REPORT, _report_node(agents)))

    graph.add_edge(START, AgentNodeName.PLANNER.value)
    graph.add_conditional_edges(
        AgentNodeName.PLANNER.value,
        _route_after_planner,
        {
            AgentNodeName.SEARCH.value: AgentNodeName.SEARCH.value,
            END: END,
        },
    )
    graph.add_edge(AgentNodeName.SEARCH.value, AgentNodeName.CRAWL.value)
    graph.add_edge(AgentNodeName.CRAWL.value, AgentNodeName.PARSER.value)
    graph.add_edge(AgentNodeName.PARSER.value, AgentNodeName.NLP.value)
    graph.add_edge(AgentNodeName.NLP.value, AgentNodeName.KNOWLEDGE.value)
    graph.add_edge(AgentNodeName.KNOWLEDGE.value, AgentNodeName.REVIEWER.value)
    graph.add_conditional_edges(
        AgentNodeName.REVIEWER.value,
        _route_after_reviewer,
        {
            AgentNodeName.REPORT.value: AgentNodeName.REPORT.value,
            AgentNodeName.SEARCH.value: AgentNodeName.SEARCH.value,
            END: END,
        },
    )
    graph.add_edge(AgentNodeName.REPORT.value, END)
    return graph.compile(checkpointer=checkpointer, debug=debug, name="datamind_multi_agent")


def initial_workflow_state(
    task: Any,
    *,
    max_node_retries: int = 1,
    max_review_retries: int = 1,
) -> WorkflowState:
    return WorkflowState(
        task=task,
        trace_id=uuid4(),
        status=WorkflowStatus.PENDING,
        retry_counts={},
        checkpoints=(),
        trace=(),
        errors=(),
        plan=None,
        search_results=(),
        crawl_results=(),
        parsed_documents=(),
        nlp_extractions=(),
        knowledge_documents=(),
        knowledge_result=None,
        review=None,
        report=None,
        max_node_retries=max_node_retries,
        max_review_retries=max_review_retries,
        review_retries=0,
    )


def _add_node(graph: Any, node: AgentNodeName, fn: NodeCallable) -> None:
    graph.add_node(node.value, cast(Any, fn))


def _planner_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        plan = await agents.planner.plan(state["task"])
        return {"plan": plan, "status": WorkflowStatus.RUNNING}

    return run


def _search_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        plan = _require(state.get("plan"), "Planner did not produce an ExecutionPlan.")
        return {"search_results": await agents.search.search(plan)}

    return run


def _crawl_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        return {"crawl_results": await agents.crawl.crawl(state.get("search_results", ()))}

    return run


def _parser_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        return {"parsed_documents": await agents.parser.parse(state.get("crawl_results", ()))}

    return run


def _nlp_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        return {"nlp_extractions": await agents.nlp.analyze(state.get("parsed_documents", ()))}

    return run


def _knowledge_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        documents, result = await agents.knowledge.update(
            state.get("parsed_documents", ()),
            state.get("nlp_extractions", ()),
        )
        return {"knowledge_documents": documents, "knowledge_result": result}

    return run


def _reviewer_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        review = await agents.reviewer.review(_state_summary(state))
        review_retries = state.get("review_retries", 0)
        status = WorkflowStatus.RUNNING if review.passed else WorkflowStatus.REVIEW_RETRY
        if not review.passed:
            review_retries += 1
        return {"review": review, "review_retries": review_retries, "status": status}

    return run


def _report_node(agents: WorkflowAgents) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        report = await agents.report.report(_state_summary(state))
        return {"report": report, "status": WorkflowStatus.COMPLETED}

    return run


def _wrap(node: AgentNodeName, fn: NodeCallable) -> NodeCallable:
    async def run(state: WorkflowState) -> NodeOutput:
        started_at = datetime.now(UTC)
        retry_counts = dict(state.get("retry_counts", {}))
        max_retries = state.get("max_node_retries", 1)
        attempts = retry_counts.get(node.value, 0)
        while attempts <= max_retries:
            try:
                output = await fn(state)
                return _with_observability(state, node, output, attempts + 1, started_at)
            except Exception as exc:
                attempts += 1
                retry_counts[node.value] = attempts
                if attempts > max_retries:
                    return _failure(state, node, exc, attempts, retry_counts, started_at)
        return _failure(
            state,
            node,
            RuntimeError("Node failed after retry loop."),
            max(attempts, 1),
            retry_counts,
            started_at,
        )

    return run


def _with_observability(
    state: WorkflowState,
    node: AgentNodeName,
    output: NodeOutput,
    attempts: int,
    started_at: datetime,
) -> NodeOutput:
    latency_ms = (datetime.now(UTC) - started_at).total_seconds() * 1000
    checkpoint = WorkflowCheckpoint(
        node=node,
        status=cast(WorkflowStatus, output.get("status", state["status"])),
        state_keys=tuple(sorted(output.keys())),
    )
    trace_entry = WorkflowTraceEntry(
        trace_id=state["trace_id"],
        node=node,
        message=f"{node.value} completed",
        latency_ms=latency_ms,
    )
    retry_counts = dict(state.get("retry_counts", {}))
    retry_counts[node.value] = attempts
    return {
        **output,
        "retry_counts": retry_counts,
        "checkpoints": (*state.get("checkpoints", ()), checkpoint),
        "trace": (*state.get("trace", ()), trace_entry),
    }


def _failure(
    state: WorkflowState,
    node: AgentNodeName,
    exc: Exception,
    attempts: int,
    retry_counts: dict[str, int],
    started_at: datetime,
) -> NodeOutput:
    latency_ms = (datetime.now(UTC) - started_at).total_seconds() * 1000
    error = WorkflowError(node=node, message=str(exc), attempt=attempts)
    checkpoint = WorkflowCheckpoint(
        node=node,
        status=WorkflowStatus.FAILED,
        state_keys=("errors", "status"),
    )
    trace_entry = WorkflowTraceEntry(
        trace_id=state["trace_id"],
        node=node,
        message=f"{node.value} failed: {exc}",
        latency_ms=latency_ms,
    )
    return {
        "status": WorkflowStatus.FAILED,
        "retry_counts": retry_counts,
        "errors": (*state.get("errors", ()), error),
        "checkpoints": (*state.get("checkpoints", ()), checkpoint),
        "trace": (*state.get("trace", ()), trace_entry),
    }


def _route_after_planner(state: WorkflowState) -> str:
    if state["status"] == WorkflowStatus.FAILED:
        return END
    if state.get("plan") is None:
        return END
    return AgentNodeName.SEARCH.value


def _route_after_reviewer(state: WorkflowState) -> str:
    if state["status"] == WorkflowStatus.FAILED:
        return END
    review = state.get("review")
    if review is not None and review.passed:
        return AgentNodeName.REPORT.value
    if state.get("review_retries", 0) <= state.get("max_review_retries", 1):
        return AgentNodeName.SEARCH.value
    return END


def _state_summary(state: WorkflowState) -> dict[str, object]:
    plan = state.get("plan")
    knowledge_result = state.get("knowledge_result")
    review = state.get("review")
    return {
        "task": state["task"].model_dump(mode="json"),
        "plan": plan.model_dump(mode="json") if plan else None,
        "search_results": [
            item.model_dump(mode="json") for item in state.get("search_results", ())
        ],
        "crawl_results": [item.model_dump(mode="json") for item in state.get("crawl_results", ())],
        "parsed_documents": [
            item.model_dump(mode="json") for item in state.get("parsed_documents", ())
        ],
        "nlp_extractions": [
            item.model_dump(mode="json") for item in state.get("nlp_extractions", ())
        ],
        "knowledge_result": knowledge_result.model_dump(mode="json") if knowledge_result else None,
        "review": review.model_dump(mode="json") if review else None,
    }


def _require[T](value: T | None, message: str) -> T:
    if value is None:
        raise RuntimeError(message)
    return value
