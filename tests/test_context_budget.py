from __future__ import annotations

import json

import pytest

from app.analysis.model_router import MCPAnalysisModelRouter
from app.core.settings import Settings
from app.harness.context import (
    ContextBudgetExceeded,
    ContextBudgetManager,
    PromptEnvelope,
    estimate_text_tokens,
)
from app.harness.node import NodeExecutionHarness, NodeHarnessPolicy, current_node_name


def _manager(*, mode: str, max_chars: int) -> ContextBudgetManager:
    return ContextBudgetManager(
        enabled=True,
        mode=mode,
        context_window_tokens=65_536,
        max_chars=max_chars,
        safety_ratio=0.15,
    )


def test_token_estimator_is_conservative_for_chinese_and_json() -> None:
    assert estimate_text_tokens("数据分析工作流") == 7
    assert estimate_text_tokens("a" * 40) == 10
    assert estimate_text_tokens('{"rows":[1,2,3]}') > 4


def test_shadow_mode_reports_reduction_without_changing_transmitted_messages() -> None:
    messages = [
        {"role": "system", "content": "immutable system contract"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": "按地区汇总销售额",
                    "sample_records": [
                        {"region": f"region-{index}", "note": "x" * 120}
                        for index in range(200)
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    prepared = _manager(mode="shadow", max_chars=800).prepare(
        PromptEnvelope.from_messages(messages),
        profile="planner",
        output_tokens=1024,
    )

    assert prepared.messages == messages
    assert prepared.report.compressed is True
    assert prepared.report.proposed_chars <= 800
    assert prepared.report.transmitted_chars == prepared.report.original_chars


def test_enforce_mode_preserves_question_and_sends_compacted_context() -> None:
    payload = {
        "question": "按地区汇总销售额",
        "analysis_contract": {"metric": "sales", "dimension": "region"},
        "sample_records": [{"region": "east", "note": "x" * 200} for _ in range(100)],
    }
    prepared = _manager(mode="enforce", max_chars=900).prepare(
        PromptEnvelope.from_messages(
            [
                {"role": "system", "content": "immutable system contract"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        ),
        profile="planner",
        output_tokens=1024,
    )

    compacted = json.loads(prepared.messages[-1]["content"])
    assert compacted["question"] == payload["question"]
    assert compacted["analysis_contract"] == payload["analysis_contract"]
    assert len(compacted["sample_records"]) < len(payload["sample_records"])
    assert prepared.report.transmitted_chars <= 900


def test_python_repair_keeps_every_error_and_function_signature() -> None:
    code = "def analyze(df):\n" + "\n".join(
        f"    value_{index} = {index}" for index in range(800)
    )
    payload = {
        "question": "分析销售额",
        "python_attempts": [
            {"attempt": 1, "code": code, "error": "ValueError at line 401: invalid value"},
            {"attempt": 2, "code": code, "error": "TypeError at line 520: invalid type"},
        ],
    }
    prepared = _manager(mode="enforce", max_chars=7_000).prepare(
        PromptEnvelope.from_messages(
            [
                {"role": "system", "content": "Return repaired Python."},
                {"role": "user", "content": json.dumps(payload)},
            ]
        ),
        profile="python",
        output_tokens=2048,
    )
    compacted = json.loads(prepared.messages[-1]["content"])

    assert [item["error"] for item in compacted["python_attempts"]] == [
        item["error"] for item in payload["python_attempts"]
    ]
    assert all("def analyze(df):" in item["code"] for item in compacted["python_attempts"])
    assert all("context compressed" in item["code"] for item in compacted["python_attempts"])


def test_required_plain_text_is_rejected_in_enforce_mode() -> None:
    envelope = PromptEnvelope.from_messages(
        [
            {"role": "system", "content": "s" * 500},
            {"role": "user", "content": "q" * 500},
        ]
    )
    with pytest.raises(ContextBudgetExceeded, match="Required LLM context"):
        _manager(mode="enforce", max_chars=200).prepare(
            envelope,
            profile="planner",
            output_tokens=1024,
        )


def test_node_harness_exposes_context_identity_to_router() -> None:
    observed: list[str | None] = []

    def handler(_state: object) -> dict[str, bool]:
        observed.append(current_node_name())
        return {"ok": True}

    harness = NodeExecutionHarness(NodeHarnessPolicy())
    assert harness.wrap("analysis.report_loop", handler)({}) == {"ok": True}
    assert observed == ["analysis.report_loop"]
    assert current_node_name() is None


def test_router_uses_agent_profile_without_invoking_provider() -> None:
    router = MCPAnalysisModelRouter(
        Settings(
            llm_provider="mock",
            context_budget_mode="shadow",
            llm_prompt_max_chars=10_000,
        )
    )
    messages, report = router._prepare_messages(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        context=None,
        metadata={"agent": "report_execute"},
        max_tokens=2048,
        streaming=False,
    )

    assert messages[-1]["content"] == "question"
    assert report["profile"] == "report"
    assert report["input_token_budget"] == 32_768


def test_router_emits_budget_event_through_node_harness() -> None:
    events: list[dict[str, object]] = []
    router = MCPAnalysisModelRouter(
        Settings(
            llm_provider="mock",
            context_budget_mode="shadow",
            llm_prompt_max_chars=10_000,
        )
    )
    harness = NodeExecutionHarness(
        event_callback=lambda _state, payload: events.append(dict(payload))
    )

    def prepare(_state: object) -> dict[str, bool]:
        router._prepare_messages(
            messages=[
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Summarize the evidence."},
            ],
            context=None,
            metadata={"agent": "report_execute"},
            max_tokens=512,
            streaming=False,
        )
        return {"ok": True}

    harness.wrap("report_node", prepare)({})

    event = next(
        item for item in events if item.get("event_type") == "context.budget_evaluated"
    )
    assert event["node"] == "report_node"
    assert event["payload"]["profile"] == "report"
    assert "messages" not in event["payload"]
