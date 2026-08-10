from __future__ import annotations

import pytest

from app.assistant.routing import compact_message_text, should_skip_tool_router

pytestmark = pytest.mark.unit


REPORT = {
    "title": "区域销售分析报告",
    "question": "销售趋势如何？",
    "executive_summary": "华东销售额增长 20%，利润率仍需核查。",
    "key_findings": [{"title": "销售增长", "content": "销售额同比增长 20%。"}],
}


@pytest.mark.parametrize(
    "question",
    (
        "概括最近一份分析报告",
        "销售趋势如何？",
        "报告中有哪些数据质量风险？",
    ),
)
def test_report_questions_skip_tool_router(question: str) -> None:
    assert should_skip_tool_router(
        question=question,
        execution_mode="ask",
        scope_type="auto",
        retrieved_reports=(REPORT,),
    )


@pytest.mark.parametrize(
    "question",
    (
        "重新分析销售数据",
        "美化并精简报告",
        "这个数据集有哪些字段？",
        "任务状态怎么样？",
    ),
)
def test_action_and_status_questions_keep_tool_router(question: str) -> None:
    assert not should_skip_tool_router(
        question=question,
        execution_mode="ask",
        scope_type="auto",
        retrieved_reports=(REPORT,),
    )


def test_execute_mode_never_skips_permissioned_tool_router() -> None:
    assert not should_skip_tool_router(
        question="概括报告",
        execution_mode="execute",
        scope_type="report",
        retrieved_reports=(REPORT,),
    )


def test_unrelated_fallback_report_does_not_skip_router() -> None:
    assert not should_skip_tool_router(
        question="库存周转情况如何？",
        execution_mode="ask",
        scope_type="auto",
        retrieved_reports=(REPORT,),
    )


def test_history_compaction_preserves_head_and_tail() -> None:
    content = "A" * 5000 + "TAIL"
    compacted = compact_message_text(content, max_chars=1000)
    assert len(compacted) <= 1005
    assert compacted.startswith("A")
    assert compacted.endswith("TAIL")
    assert "[truncated]" in compacted
