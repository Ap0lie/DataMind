from __future__ import annotations

import pytest

from app.workflows.examples import run_data_analysis_example
from app.workflows.models import WorkflowStatus


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_analysis_parse_nlp_knowledge_report_pipeline() -> None:
    state = await run_data_analysis_example()

    assert state["status"] == WorkflowStatus.COMPLETED
    assert state["search_results"]
    assert state["crawl_results"]
    assert state["parsed_documents"]
    assert state["nlp_extractions"]
    assert state["knowledge_result"] is not None
    assert state["report"] is not None
