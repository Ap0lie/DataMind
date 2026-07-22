from __future__ import annotations

from typing import cast

import pytest

from app.core.entities import TaskIntent
from app.workflows.examples import DataAnalysisAgents, run_data_analysis_example
from app.workflows.graph import build_datamind_workflow, initial_workflow_state
from app.workflows.models import WorkflowStatus
from app.workflows.nodes import WorkflowAgents


@pytest.mark.asyncio
async def test_data_analysis_workflow_completes() -> None:
    state = await run_data_analysis_example()

    assert state["status"] == WorkflowStatus.COMPLETED
    assert state["plan"] is not None
    assert state["plan"].objective == (
        "Analyze regional sales performance and unusual revenue patterns."
    )
    assert state["report"] is not None
    assert "销售表现分析" in state["report"].markdown
    assert len(state["checkpoints"]) >= 8
    assert not state["errors"]


@pytest.mark.asyncio
async def test_workflow_routes_back_to_search_when_reviewer_fails_once() -> None:
    task = TaskIntent(
        tenant_id="demo",
        user_id="system",
        prompt="Analyze regional sales performance.",
    )
    agents = DataAnalysisAgents(reviewer_failures_before_pass=1)
    workflow = build_datamind_workflow(cast(WorkflowAgents, agents))

    state = await workflow.ainvoke(initial_workflow_state(task, max_review_retries=1))

    assert state["status"] == WorkflowStatus.COMPLETED
    assert state["review_retries"] == 1
    assert agents.reviewer.calls == 2


@pytest.mark.asyncio
async def test_workflow_stops_when_reviewer_keeps_failing() -> None:
    task = TaskIntent(tenant_id="demo", user_id="system", prompt="Analyze sales")
    agents = DataAnalysisAgents(reviewer_failures_before_pass=99)
    workflow = build_datamind_workflow(cast(WorkflowAgents, agents))

    state = await workflow.ainvoke(initial_workflow_state(task, max_review_retries=0))

    assert state["status"] == WorkflowStatus.REVIEW_RETRY
    assert state["report"] is None
    assert agents.reviewer.calls == 1
