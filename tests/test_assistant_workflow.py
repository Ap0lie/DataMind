from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from app.assistant.control import (
    AssistantRunCanceled,
    AssistantRunPaused,
    ensure_run_continuable,
)
from app.assistant.evidence import canonical_reliability, safe_excerpt
from app.assistant.tools import AssistantToolRuntime, _display_timestamp
from app.assistant.workflow import (
    AssistantWorkflowRunner,
    _assistant_output_budget,
    _assistant_temperature,
    _deduplicate_continuation,
    _emit_latency_warning,
    _ensure_requested_evidence_details,
    _evidence_instruction,
    _public_citations,
    _repair_evidence_conflict,
)
from app.core.settings import Settings
from app.mcp.tool_schemas import ModelRouterResponse
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository

pytestmark = pytest.mark.workflow


def test_canonical_reliability_uses_worst_explicit_lineage_status() -> None:
    reliability = canonical_reliability(
        {"status": "passed", "summary": "报告统计审查已通过。"},
        {"status": "rejected", "summary": "任务统计审查未通过。"},
    )

    assert reliability == {
        "status": "rejected",
        "summary": "任务统计审查未通过。",
    }


def test_safe_excerpt_never_cuts_a_numeric_token() -> None:
    summary = f"{'背景说明' * 75}支付总额为 798,503.42，后续内容仍需查看。"

    excerpt = safe_excerpt(summary)

    assert len(excerpt) <= 320
    assert excerpt.endswith("…（摘要已截断）")
    assert "798,50" not in excerpt


def test_evidence_conflict_repair_does_not_turn_excerpt_into_answer() -> None:
    citations = [
        {
            "source_type": "report",
            "source_id": "report-1",
            "label": "销售报告",
            "excerpt": "支付总额为 798,503.42。",
            "reliability": {
                "status": "verified",
                "summary": "DataMind 统计审查已通过。",
            },
        }
    ]

    repaired, changed = _repair_evidence_conflict(
        "目前没有足够的 DataMind 分析证据回答这个问题。",
        citations,
    )

    assert changed is True
    assert "支付总额为 798,503.42" not in repaired
    assert "DataMind 统计审查状态为通过" in repaired
    assert "核验" not in repaired
    assert "independently verified" in _evidence_instruction(citations)


def test_evidence_conflict_repair_uses_structured_report_facts() -> None:
    citations = [
        {
            "source_type": "report",
            "source_id": "report-1",
            "label": "Olist 报告",
            "excerpt": "截断预览不应成为答案。",
            "reliability": {
                "status": "verified",
                "summary": "DataMind 统计审查已通过。",
            },
            "facts": {
                "datasets_used": [
                    "olist_order_payments_dataset.csv",
                    "olist_orders_dataset.csv",
                    "olist_customers_dataset.csv",
                ],
                "executive_summary": (
                    "delivered 总支付额为 643,547.75，SP 州支付总额为 248,007.22。"
                ),
                "key_findings": [],
            },
        }
    ]

    repaired, changed = _repair_evidence_conflict(
        "目前没有足够的 DataMind 分析证据回答这个问题。",
        citations,
    )

    assert changed is True
    assert "olist_order_payments_dataset.csv" in repaired
    assert "olist_orders_dataset.csv" in repaired
    assert "olist_customers_dataset.csv" in repaired
    assert "643,547.75" in repaired
    assert "248,007.22" in repaired
    assert "截断预览不应成为答案" not in repaired
    assert "核验" not in repaired
    public = _public_citations(citations)
    assert "facts" not in public[0]
    assert public[0]["source_id"] == "report-1"


def test_kimi_k26_uses_provider_supported_temperature() -> None:
    assert _assistant_temperature("kimi-k2.6") == 1.0
    assert _assistant_temperature("moonshot-v1-32k") == 0.1


def test_latency_warning_only_emits_after_budget_is_reached() -> None:
    events: list[dict[str, object]] = []

    def emit(**event: object) -> None:
        events.append(event)

    _emit_latency_warning(emit, stage="retrieval", value_ms=2_999, threshold_ms=3_000)
    _emit_latency_warning(emit, stage="retrieval", value_ms=3_000, threshold_ms=3_000)

    assert events == [
        {
            "event_type": "performance.warning",
            "status": "completed",
            "message": "Assistant retrieval latency exceeded its budget.",
            "payload": {
                "stage": "retrieval",
                "value_ms": 3_000,
                "threshold_ms": 3_000,
            },
        }
    ]


class AnswerRouter:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def complete(self, **kwargs: object) -> ModelRouterResponse:
        self.calls.append(list(kwargs.get("messages") or []))
        return ModelRouterResponse(
            provider="mock",
            model="kimi-test",
            content="已有报告显示销售额增长，建议继续核查利润率。",
            token_usage={"total_tokens": 12},
        )


class StreamingAnswerRouter(AnswerRouter):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls = 0
        self.stream_max_tokens: int | None = None

    def stream_complete(self, **kwargs: object) -> ModelRouterResponse:
        self.stream_calls += 1
        self.stream_max_tokens = int(kwargs["max_tokens"])
        callback = kwargs["on_delta"]
        callback("基于现有报告，")
        callback("销售额保持增长。")
        return ModelRouterResponse(
            provider="mock",
            model="kimi-stream-test",
            content="基于现有报告，销售额保持增长。",
            finish_reason="stop",
            token_usage={"total_tokens": 9},
        )


class TruncatedStreamingAnswerRouter(StreamingAnswerRouter):
    def stream_complete(self, **kwargs: object) -> ModelRouterResponse:
        self.stream_calls += 1
        self.stream_max_tokens = int(kwargs["max_tokens"])
        callback = kwargs["on_delta"]
        if self.stream_calls == 1:
            callback("第一部分尚未完成，")
            return ModelRouterResponse(
                provider="mock",
                model="kimi-stream-test",
                content="第一部分尚未完成，",
                finish_reason="length",
                token_usage={"total_tokens": 9},
            )
        if self.stream_calls == 2:
            callback("尚未完成，第二部分仍未完成，")
            return ModelRouterResponse(
                provider="mock",
                model="kimi-stream-test",
                content="尚未完成，第二部分仍未完成，",
                finish_reason="length",
                token_usage={"total_tokens": 7},
            )
        messages = list(kwargs.get("messages") or [])
        assert messages[-1]["role"] == "user"
        assert "被截断" in str(messages[-1]["content"])
        callback("仍未完成，第三部分已补全。")
        return ModelRouterResponse(
            provider="mock",
            model="kimi-stream-test",
            content="仍未完成，第三部分已补全。",
            finish_reason="stop",
            token_usage={"total_tokens": 6},
        )


class AlwaysTruncatedStreamingAnswerRouter(StreamingAnswerRouter):
    def stream_complete(self, **kwargs: object) -> ModelRouterResponse:
        self.stream_calls += 1
        self.stream_max_tokens = int(kwargs["max_tokens"])
        callback = kwargs["on_delta"]
        callback(f"未完成片段{self.stream_calls}，")
        return ModelRouterResponse(
            provider="mock",
            model="kimi-stream-test",
            content=f"未完成片段{self.stream_calls}，",
            finish_reason="length",
            token_usage={"completion_tokens": self.stream_max_tokens},
        )


class RepeatingTruncatedStreamingAnswerRouter(StreamingAnswerRouter):
    def stream_complete(self, **kwargs: object) -> ModelRouterResponse:
        self.stream_calls += 1
        self.stream_max_tokens = int(kwargs["max_tokens"])
        kwargs["on_delta"]("完全相同的未完成片段。")
        return ModelRouterResponse(
            provider="mock",
            model="kimi-stream-test",
            content="完全相同的未完成片段。",
            finish_reason="length",
            token_usage={"completion_tokens": self.stream_max_tokens},
        )


class OmittedReportDetailsRouter(StreamingAnswerRouter):
    def stream_complete(self, **kwargs: object) -> ModelRouterResponse:
        self.stream_calls += 1
        self.stream_max_tokens = int(kwargs["max_tokens"])
        kwargs["on_delta"]("报告结论已生成。")
        return ModelRouterResponse(
            provider="mock",
            model="kimi-stream-test",
            content="报告结论已生成。",
            finish_reason="stop",
            token_usage={"completion_tokens": 8, "total_tokens": 20},
        )


class MissingFinishReasonRouter(StreamingAnswerRouter):
    def stream_complete(self, **kwargs: object) -> ModelRouterResponse:
        self.stream_calls += 1
        self.stream_max_tokens = int(kwargs["max_tokens"])
        kwargs["on_delta"]("看似完整但终止原因缺失。")
        return ModelRouterResponse(
            provider="mock",
            model="kimi-stream-test",
            content="看似完整但终止原因缺失。",
            finish_reason=None,
            token_usage={"completion_tokens": 20},
        )


class InsufficientEvidenceRouter(AnswerRouter):
    def complete(self, **kwargs: object) -> ModelRouterResponse:
        self.calls.append(list(kwargs.get("messages") or []))
        return ModelRouterResponse(
            provider="mock",
            model="kimi-test",
            content="目前没有足够的 DataMind 分析证据回答这个问题。",
            token_usage={"total_tokens": 8},
        )


class PauseAfterRoutingRouter(AnswerRouter):
    def __init__(self, repository: AssistantRepository, run_id: UUID) -> None:
        super().__init__()
        self.repository = repository
        self.run_id = run_id

    def complete(self, **kwargs: object) -> ModelRouterResponse:
        response = super().complete(**kwargs)
        self.repository.request_pause(self.run_id)
        return response


def test_assistant_run_pauses_and_resumes_with_persisted_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="可恢复对话", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="分析销售表现",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    with pytest.raises(AssistantRunPaused):
        AssistantWorkflowRunner(
            store=store,
            assistant_store=assistant_store,
            model_router=PauseAfterRoutingRouter(assistant_store, run.id),
            settings=settings,
        ).run(run.id)

    assert assistant_store.get_run(run.id).status == "pause_requested"
    paused = assistant_store.mark_paused(run.id)
    assert paused.status == "paused"
    assert (
        assistant_store.get_conversation(conversation["conversation_id"])["active_run_status"]
        == "paused"
    )
    resumed = assistant_store.resume_run(run.id)
    assert resumed.status == "queued"
    assert resumed.current_stage == "resuming"

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=AnswerRouter(),
        settings=settings,
    ).run(run.id)

    assert assistant_store.get_run(run.id).status == "completed"
    event_types = [item["event_type"] for item in assistant_store.list_events(run.id)]
    assert "run.pause_requested" in event_types
    assert "run.paused" in event_types
    assert "run.resumed" in event_types


def test_canceling_queued_assistant_run_is_immediately_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="待取消", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="分析"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )

    canceled = assistant_store.request_cancel(run.id)

    assert canceled.status == "canceled"
    assert canceled.completed_at is not None
    assert assistant_store.get_message(assistant["message_id"])["status"] == "canceled"
    assert (
        assistant_store.get_conversation(conversation["conversation_id"])["active_run_id"] is None
    )


def test_canceling_running_assistant_run_is_immediate_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="运行中取消", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="分析"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="正在准备回答...",
        status="streaming",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    assistant_store.update_run(run.id, status="running", current_stage="compose")

    canceled = assistant_store.request_cancel(run.id)
    duplicate = assistant_store.request_cancel(run.id)

    assert canceled.status == duplicate.status == "canceled"
    assert canceled.cancel_requested is True
    assert canceled.completed_at is not None
    message = assistant_store.get_message(assistant["message_id"])
    assert message["status"] == "canceled"
    assert message["content"].startswith("已结束本次 Kimi 任务")
    assert [item["event_type"] for item in assistant_store.list_events(run.id)].count(
        "run.canceled"
    ) == 1
    assert (
        assistant_store.complete_run_answer(
            run.id,
            content="迟到回答",
            provider="mock",
            model="mock",
            citations=(),
            token_usage={},
            metadata={},
            event_payload={},
        )
        is False
    )
    assert assistant_store.get_message(assistant["message_id"])["content"].startswith(
        "已结束本次 Kimi 任务"
    )
    with pytest.raises(AssistantRunCanceled):
        ensure_run_continuable(assistant_store, run.id)


def test_deleting_conversation_cancels_its_queued_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="删除中的对话", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="分析"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )

    assistant_store.delete_conversation(conversation["conversation_id"])

    assert assistant_store.get_run(run.id).status == "canceled"
    assert assistant_store.get_message(assistant["message_id"])["status"] == "canceled"
    with pytest.raises(RuntimeError, match="conversation"):
        assistant_store.get_conversation(conversation["conversation_id"])


def test_assistant_emits_real_streaming_router_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    store.save_report(
        dataset_id=dataset.id,
        title="销售报告",
        markdown="销售额增长。",
        metadata={"structured_report": {"executive_summary": "销售额增长。"}},
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="流式回答", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="销售如何？"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    router = StreamingAnswerRouter()
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=router,
        settings=settings,
    ).run(run.id)

    deltas = [
        item["payload"]["delta"]
        for item in assistant_store.list_events(run.id)
        if item["event_type"] == "message.delta"
    ]
    first_delta = next(
        item
        for item in assistant_store.list_events(run.id)
        if item["event_type"] == "message.delta"
    )
    completed = assistant_store.get_message(assistant["message_id"])
    assert router.calls == []
    assert router.stream_calls == 1
    assert router.stream_max_tokens >= settings.assistant_completion_min_tokens
    assert router.stream_max_tokens <= settings.assistant_ask_max_tokens
    assert deltas == ["基于现有报告，", "销售额保持增长。"]
    assert completed["content"] == "".join(deltas)
    assert completed["metadata"]["fast_path"] is True
    assert completed["metadata"]["latency"]["tool_routing_ms"] == 0
    assert first_delta["payload"]["latency"]["fast_path"] is True
    assert completed["token_usage"]["total_tokens"] == 9


def test_assistant_continues_until_third_response_finishes_and_deduplicates_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    store.save_report(
        dataset_id=dataset.id,
        title="销售报告",
        markdown="销售额为 120。",
        metadata={"structured_report": {"executive_summary": "销售额为 120。"}},
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="截断续写", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="给出完整结论"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    router = TruncatedStreamingAnswerRouter()
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=router,
        settings=settings,
    ).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert router.stream_calls == 3
    assert completed["content"] == "第一部分尚未完成，第二部分仍未完成，第三部分已补全。"
    assert completed["token_usage"]["total_tokens"] == 22
    assert completed["metadata"]["output_budget"]["continuation_count"] == 2
    assert completed["metadata"]["output_budget"]["finish_reason"] == "stop"
    assert any(
        item["event_type"] == "message.continuing" for item in assistant_store.list_events(run.id)
    )


def test_continuation_overlap_is_removed_without_dropping_new_content() -> None:
    assert (
        _deduplicate_continuation(
            "第一部分尚未完成，",
            "尚未完成，第二部分已补全。",
        )
        == "第二部分已补全。"
    )


def test_assistant_output_budget_expands_for_multi_table_detailed_evidence() -> None:
    settings = Settings(environment="test")
    simple = _assistant_output_budget(
        question="结论是什么？",
        citations=[],
        execution_mode="ask",
        settings=settings,
    )
    complex_budget = _assistant_output_budget(
        question="请完整列出多表报告实际使用表、所有关键金额和统计审查状态。",
        citations=[
            {
                "source_type": "report",
                "source_id": "report-1",
                "facts": {
                    "datasets_used": ["customers.csv", "orders.csv", "payments.csv"],
                    "executive_summary": "delivered 总额 643,547.75，SP 为 248,007.22。",
                    "row_count": 1_000_000,
                },
            }
        ],
        execution_mode="ask",
        settings=settings,
    )

    assert simple["per_call_tokens"] >= 1_536
    assert simple["per_call_tokens"] != 700
    assert complex_budget["per_call_tokens"] > simple["per_call_tokens"]
    assert complex_budget["table_count"] == 3
    assert complex_budget["row_count"] == 1_000_000
    assert complex_budget["total_tokens"] == settings.assistant_completion_total_max_tokens


def test_assistant_output_budget_never_exceeds_hard_total_cap() -> None:
    settings = Settings(
        environment="test",
        assistant_completion_min_tokens=512,
        assistant_ask_max_tokens=4_096,
        assistant_completion_total_max_tokens=1_024,
    )

    budget = _assistant_output_budget(
        question="请完整详细分析全部多表、大表、数字和统计审查状态。",
        citations=[
            {
                "source_type": "dataset",
                "source_id": "dataset-1",
                "excerpt": "2,000,000 rows; 80 columns",
                "facts": {
                    "datasets_used": ["large.csv"],
                    "row_count": 2_000_000,
                },
            }
        ],
        execution_mode="ask",
        settings=settings,
    )

    assert budget["per_call_tokens"] == 1_024
    assert budget["total_tokens"] == 1_024


def test_assistant_fails_instead_of_committing_partial_answer_when_budget_exhausts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="输出预算耗尽", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="请给出完整结论",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    router = AlwaysTruncatedStreamingAnswerRouter()
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        assistant_completion_min_tokens=512,
        assistant_ask_max_tokens=512,
        assistant_completion_total_max_tokens=1_024,
        assistant_max_continuations=5,
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=router,
        settings=settings,
    ).run(run.id)

    failed = assistant_store.get_message(assistant["message_id"])
    assert router.stream_calls == 2
    assert assistant_store.get_run(run.id).status == "failed"
    assert failed["status"] == "failed"
    assert "未保存半截答案" in failed["content"]
    assert "未完成片段" not in failed["content"]
    assert failed["metadata"]["reason"] == "total_token_budget"
    events = assistant_store.list_events(run.id)
    assert not any(item["event_type"] == "message.completed" for item in events)
    assert any(
        item["event_type"] == "message.reset" and item["status"] == "failed" for item in events
    )


def test_assistant_stops_continuations_immediately_when_they_make_no_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="续写无进展", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="请完整回答",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    router = RepeatingTruncatedStreamingAnswerRouter()
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=router,
        settings=settings,
    ).run(run.id)

    failed = assistant_store.get_message(assistant["message_id"])
    assert router.stream_calls == 2
    assert failed["status"] == "failed"
    assert failed["metadata"]["reason"] == "no_progress"
    assert "完全相同的未完成片段" not in failed["content"]


def test_multi_table_report_answer_restores_exact_tables_numbers_and_review_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(
        name="olist_customers_dataset.csv", source_type="csv", source_metadata={}
    )
    store.append_raw_records(dataset_id=dataset.id, records=[{"customer_id": "c1"}])
    store.save_report(
        dataset_id=dataset.id,
        title="Olist 多表支付报告",
        markdown="delivered 总支付额 643,547.75；SP 248,007.22。",
        metadata={
            "question": "Olist delivered 支付总额和 SP 州总额",
            "structured_report": {
                "executive_summary": (
                    "delivered 总支付额为 643,547.75，SP 州支付总额为 248,007.22。"
                )
            },
            "multi_dataset_context": {
                "primary_dataset": {"name": "olist_customers_dataset.csv"},
                "additional_datasets": [
                    {"name": "olist_orders_dataset.csv"},
                    {"name": "olist_order_payments_dataset.csv"},
                ],
            },
            "statistical_verification": {
                "status": "passed",
                "summary": "统计审查通过。",
            },
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="Olist 报告复核", scope_type="auto", scope_id=None
    )
    question = "请列出该 Olist 报告实际使用表、关键金额和统计审查状态。"
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content=question,
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    router = OmittedReportDetailsRouter()
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=router,
        settings=settings,
    ).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert completed["status"] == "completed"
    assert "olist_customers_dataset.csv" in completed["content"]
    assert "olist_orders_dataset.csv" in completed["content"]
    assert "olist_order_payments_dataset.csv" in completed["content"]
    assert "643,547.75" in completed["content"]
    assert "248,007.22" in completed["content"]
    assert "DataMind 统计审查状态：通过" in completed["content"]
    assert completed["metadata"]["requested_details_repaired"] is True
    assert router.stream_max_tokens > settings.assistant_completion_min_tokens


def test_report_evidence_exposes_verification_join_and_chart_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(
        name="olist_order_payments_dataset.csv", source_type="csv", source_metadata={}
    )
    store.append_raw_records(dataset_id=dataset.id, records=[{"order_id": "o1"}])
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="Olist 多表支付报告",
        markdown="delivered 总支付额 643,547.75；SP 248,007.22。",
        metadata={
            "question": "仅使用 customers、orders、order_payments 分析 delivered 支付总额",
            "multi_dataset_context": {
                "primary_dataset": {"name": "olist_order_payments_dataset.csv"},
                "additional_datasets": [
                    {"name": "olist_orders_dataset.csv"},
                    {"name": "olist_customers_dataset.csv"},
                ],
            },
            "structured_report": {
                "executive_summary": (
                    "delivered 总支付额为 643,547.75，SP 州支付总额为 248,007.22。"
                ),
                "analysis_context": ("payments 通过 order_id 连接 orders；连接后无行扩展。"),
                "charts": [
                    {
                        "title": "payment_value 总额分组对比",
                        "chart_type": "bar",
                        "spec": {"x": "customer_state", "y": "total_payment_value"},
                        "data": [
                            {
                                "customer_state": f"S{index:02d}",
                                "total_payment_value": 27 - index,
                            }
                            for index in range(27)
                        ],
                        "explanation": "SP 占全部 27 州支付总额的 38.5%。",
                    }
                ],
            },
            "join_summary": {
                "mode": "joined",
                "dataset_count": 3,
                "joined_dataset_count": 3,
                "joined_row_count": 5783,
                "row_expansion_ratio": 1.0,
                "skipped_join_count": 0,
                "joins": [
                    {
                        "left_column": "order_id",
                        "right_column": "order_id",
                        "join_type": "left",
                        "before_rows": 5783,
                        "after_rows": 5783,
                        "row_expansion_ratio": 1.0,
                    }
                ],
            },
            "analysis_lineage": {
                "relationship_graph": {
                    "nodes": [
                        {
                            "entity_id": "payments",
                            "name": "olist_order_payments_dataset.csv",
                        },
                        {"entity_id": "orders", "name": "olist_orders_dataset.csv"},
                    ]
                },
                "grain_plan": {
                    "safe": True,
                    "metric_grain": ["one row per payment record"],
                    "join_path": [
                        {
                            "left_entity_id": "payments",
                            "right_entity_id": "orders",
                            "cardinality": "many_to_one",
                            "join_type": "left",
                        }
                    ],
                },
            },
            "statistical_verification": {
                "status": "passed",
                "summary": "统计审查通过：8 项通过。",
                "checks": [
                    {
                        "code": "request_coverage",
                        "status": "passed",
                        "message": "请求覆盖完整。",
                        "details": {
                            "required_dimensions": ["customer_state"],
                            "required_filters": ["order_status=delivered"],
                            "required_aggregations": ["sum(payment_value)"],
                            "covered_by": "sql_statement_1",
                        },
                    },
                    {
                        "code": "join_grain",
                        "status": "passed",
                        "message": "Join 粒度审查通过。",
                        "details": {"row_expansion_ratio": 1.0},
                    },
                ],
            },
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="报告证据", scope_type="report", scope_id=report_id
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="说明 request_coverage、Join 粒度和柱状图 24/27 的分母口径。",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    runtime = AssistantToolRuntime(
        store=store,
        assistant_store=assistant_store,
        settings=Settings(dataset_store_path=str(root), environment="test"),
        run_id=run.id,
        conversation=conversation,
        event=lambda **_: None,
    )

    result = runtime.execute("get_report", {"report_id": str(report_id)})

    coverage = next(
        check
        for check in result["statistical_verification"]["checks"]
        if check["code"] == "request_coverage"
    )
    assert coverage["details"]["covered_by"] == "sql_statement_1"
    assert result["analysis_context"].startswith("payments 通过 order_id")
    assert result["join_context"]["paths"][0]["cardinality"] == "many_to_one"
    assert result["join_context"]["row_expansion_ratio"] == 1.0
    assert result["chart_context"][0]["displayed_category_count"] == 24
    assert result["chart_context"][0]["data_point_count"] == 27
    assert result["chart_context"][0]["denominator_scope"] == ("not_applicable_for_bar_chart")

    repaired, changed = _ensure_requested_evidence_details(
        "当前证据未提供这些信息。",
        question="说明 request_coverage、Join 粒度和柱状图 24/27 的分母口径。",
        citations=list(runtime.evidence.values()),
    )

    assert changed is True
    assert "covered_by=sql_statement_1" in repaired
    assert "N:1" in repaired
    assert "row_expansion_ratio=1.0" in repaired
    assert "前 24 / 27 个类别" in repaired
    assert "柱状图本身不使用百分比分母" in repaired


def test_assistant_uses_existing_report_as_validated_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"region": "East", "sales": 120}])
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售分析",
        markdown="销售额增长 20%。",
        metadata={
            "question": "销售趋势",
            "structured_report": {"executive_summary": "销售额增长 20%。"},
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="新对话", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="销售趋势如何？"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )

    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )
    AssistantWorkflowRunner(
        store=store, assistant_store=assistant_store, model_router=AnswerRouter(), settings=settings
    ).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert completed["status"] == "completed"
    assert completed["citations"][0]["source_id"] == str(report_id)
    assert assistant_store.get_run(run.id).status == "completed"
    assert any(
        item["event_type"] == "message.delta" for item in assistant_store.list_events(run.id)
    )


def test_assistant_repairs_evidence_denial_when_citations_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"region": "East", "sales": 120}])
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售分析",
        markdown="销售额为 120。",
        metadata={
            "question": "销售表现",
            "structured_report": {"executive_summary": "销售额为 120。"},
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="证据一致性",
        scope_type="auto",
        scope_id=None,
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="销售表现如何？",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=InsufficientEvidenceRouter(),
        settings=settings,
    ).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert completed["citations"][0]["source_id"] == str(report_id)
    assert "没有足够的 DataMind 分析证据" not in completed["content"]
    assert "销售分析" in completed["content"]
    assert completed["metadata"]["evidence_consistency_repaired"] is True
    assert any(
        item["event_type"] == "message.reset"
        and item["payload"].get("reason") == "evidence_consistency"
        for item in assistant_store.list_events(run.id)
    )


def test_assistant_discloses_rejected_report_reliability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    store.save_report(
        dataset_id=dataset.id,
        title="未通过审查的销售报告",
        markdown="销售额为 120。",
        metadata={
            "structured_report": {"executive_summary": "销售额为 120。"},
            "statistical_verification": {
                "status": "failed",
                "summary": "统计审查失败：请求维度未覆盖。",
                "requires_replan": True,
            },
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="可靠性披露", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="销售表现如何？"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=InsufficientEvidenceRouter(),
        settings=settings,
    ).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert completed["citations"][0]["reliability"]["status"] == "rejected"
    assert "统计审查未通过" in completed["content"]
    assert "已读取并核验" not in completed["content"]


def test_auto_retrieve_keeps_only_latest_report_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    legacy_dataset = store.create_dataset(
        name="legacy_sales.csv", source_type="csv", source_metadata={}
    )
    store.append_raw_records(dataset_id=legacy_dataset.id, records=[{"sales": 100}])
    old_report_id = store.save_report(
        dataset_id=legacy_dataset.id,
        title="销售分析",
        markdown="旧口径销售额为 100。",
        metadata={
            "question": "销售表现如何？",
            "structured_report": {"executive_summary": "旧口径销售额为 100。"},
        },
    )
    latest_report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售分析",
        markdown="新口径销售额为 120。",
        metadata={
            "question": "销售表现如何？",
            "structured_report": {
                "executive_summary": "新口径销售额为 120。",
                "key_findings": [{"content": "华东销售额为 120。"}],
            },
            "multi_dataset_context": {
                "primary_dataset": {"name": "sales.csv"},
                "additional_datasets": [{"name": "customers.csv"}],
            },
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="最新报告", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="销售表现如何？",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    runtime = AssistantToolRuntime(
        store=store,
        assistant_store=assistant_store,
        settings=Settings(dataset_store_path=str(root), environment="test"),
        run_id=run.id,
        conversation=conversation,
        event=lambda **_: None,
    )

    reports = runtime.auto_retrieve("销售表现")

    assert [item["report_id"] for item in reports] == [str(latest_report_id)]
    assert reports[0]["datasets_used"] == ["sales.csv", "customers.csv"]
    assert reports[0]["key_findings"][0]["content"] == "华东销售额为 120。"
    assert f"report:{latest_report_id}" in runtime.evidence
    assert f"report:{old_report_id}" not in runtime.evidence


def test_auto_retrieve_does_not_fallback_to_unrelated_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="DataMind 分析报告",
        markdown="支付总额为 180,096.80。DataMind 统计审查通过，并回答用户问题。",
        metadata={
            "question": ("仅使用 customers、orders、order_payments，统计 delivered 支付总额和 SP。")
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="发送验证", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="发送按钮测试",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    runtime = AssistantToolRuntime(
        store=store,
        assistant_store=assistant_store,
        settings=Settings(dataset_store_path=str(root), environment="test"),
        run_id=run.id,
        conversation=conversation,
        event=lambda **_: None,
    )

    reports = runtime.auto_retrieve("请只回答：发送按钮测试通过。")

    assert reports == ()
    assert runtime.evidence == {}
    assert f"report:{report_id}" not in runtime.evidence

    assert runtime.auto_retrieve("解释统计结论的含义") == ()
    assert runtime.evidence == {}

    latest_follow_up = runtime.auto_retrieve("请读取刚完成的报告")
    assert [item["report_id"] for item in latest_follow_up] == [str(report_id)]

    relevant = runtime.auto_retrieve("delivered 支付总额和 SP")
    assert [item["report_id"] for item in relevant] == [str(report_id)]


def test_unrelated_question_persists_no_report_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    store.save_report(
        dataset_id=dataset.id,
        title="销售分析",
        markdown="支付总额为 180,096.80。",
        metadata={"question": "销售表现如何？"},
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="发送验证", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="发送按钮测试",
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )

    AssistantWorkflowRunner(
        store=store,
        assistant_store=assistant_store,
        model_router=AnswerRouter(),
        settings=Settings(
            dataset_store_path=str(root),
            checkpoint_backend="none",
            assistant_llm_provider="mock",
            assistant_llm_model="kimi-test",
            environment="test",
        ),
    ).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert not completed["citations"]
    assert "180,096.80" not in completed["content"]


def test_assistant_report_timestamp_uses_configured_display_timezone() -> None:
    assert (
        _display_timestamp("2026-07-30T09:15:00+00:00", "Asia/Singapore")
        == "2026-07-30T17:15:00+08:00"
    )


def test_assistant_omits_empty_messages_left_by_failed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="历史对话", scope_type="auto", scope_id=None
    )
    assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="第一次分析"
    )
    assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="failed",
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="请重新分析"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    router = AnswerRouter()
    settings = Settings(
        dataset_store_path=str(root),
        checkpoint_backend="none",
        assistant_llm_provider="mock",
        assistant_llm_model="kimi-test",
        environment="test",
    )

    AssistantWorkflowRunner(
        store=store, assistant_store=assistant_store, model_router=router, settings=settings
    ).run(run.id)

    sent = router.calls[0]
    assert {"role": "user", "content": "请重新分析"} in sent
    assert not any(
        item.get("role") == "assistant" and not str(item.get("content") or "").strip()
        for item in sent
    )


def test_completed_analysis_registers_rendered_report_as_primary_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"region": "East", "sales": 120}])
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="重新生成的销售分析报告",
        markdown="# 销售分析\n完整报告内容。",
        metadata={
            "question": "重新分析销售趋势",
            "structured_report": {"executive_summary": "销售趋势完整分析已经生成。"},
        },
    )
    job = store.create_analysis_job(dataset_id=dataset.id, question="重新分析销售趋势")
    store.update_analysis_job(
        job.id,
        status="completed",
        progress=100,
        current_stage="complete",
        result={"structured_report": {"executive_summary": "销售趋势完整分析已经生成。"}},
        report_id=report_id,
        completed=True,
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="新对话", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="生成完整报告"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    runtime = AssistantToolRuntime(
        store=store,
        assistant_store=assistant_store,
        settings=Settings(dataset_store_path=str(root), environment="test"),
        run_id=run.id,
        conversation=conversation,
        event=lambda **_: None,
    )

    result = runtime.execute("get_analysis_result", {"job_id": str(job.id)})

    assert result["report_id"] == str(report_id)
    assert result["report"]["title"] == "重新生成的销售分析报告"
    assert runtime.evidence[f"report:{report_id}"]["label"] == "重新生成的销售分析报告"
    assert runtime.evidence[f"report:{report_id}"]["artifact_role"] == "deliverable"


def test_analysis_and_report_evidence_share_worst_lineage_reliability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"sales": 120}])
    job = store.create_analysis_job(dataset_id=dataset.id, question="分析销售")
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售报告",
        markdown="销售额为 120。",
        job_id=job.id,
        metadata={
            "structured_report": {"executive_summary": "销售额为 120。"},
            "statistical_verification": {"status": "passed", "summary": "报告审查通过。"},
        },
    )
    store.update_analysis_job(
        job.id,
        status="completed",
        result={
            "structured_report": {"executive_summary": "销售额为 120。"},
            "statistical_verification": {
                "status": "failed",
                "summary": "任务审查未通过。",
                "requires_replan": True,
            },
        },
        report_id=report_id,
        completed=True,
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="可靠性血缘", scope_type="auto", scope_id=None
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="分析销售"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
    )
    runtime = AssistantToolRuntime(
        store=store,
        assistant_store=assistant_store,
        settings=Settings(dataset_store_path=str(root), environment="test"),
        run_id=run.id,
        conversation=conversation,
        event=lambda **_: None,
    )

    result = runtime.execute("get_analysis_result", {"job_id": str(job.id)})

    assert result["reliability"]["status"] == "rejected"
    assert result["report"]["reliability"]["status"] == "rejected"
    assert runtime.evidence[f"analysis_job:{job.id}"]["reliability"]["status"] == "rejected"
    assert runtime.evidence[f"report:{report_id}"]["reliability"]["status"] == "rejected"


def test_report_revision_reuses_frozen_metrics_without_new_analysis_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    source_report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售分析",
        markdown="# 销售分析",
        metadata={
            "question": "销售表现如何？",
            "structured_report": {
                "executive_summary": "总销售额为 100 元。平均订单金额为 25 元。还有更多背景说明。",
                "key_findings": [
                    {
                        "title": f"发现 {index}",
                        "content": f"指标 {index} 的原始结论保持不变。",
                        "data_source": "sql_result.rows",
                    }
                    for index in range(1, 7)
                ],
                "charts": [
                    {
                        "title": "销售额",
                        "chart_type": "bar",
                        "spec": {"x": "region", "y": "sales"},
                        "data": [{"region": "East", "sales": 100}],
                    }
                ],
                "sql_results": [{"total_sales": 100, "average_order_value": 25}],
                "python_results": {"total_sales": 100, "average_order_value": 25},
                "recommended_next_steps": ["核查利润", "观察趋势", "按区域拆分", "检查异常"],
            },
        },
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(
        title="报告修订", scope_type="report", scope_id=source_report_id
    )
    assistant_store.save_permission_grant(
        asset_type="report", asset_id=source_report_id, capabilities=("report_manage",)
    )
    user = assistant_store.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="美化图表并精简报告"
    )
    assistant = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="assistant",
        content="",
        status="pending",
    )
    run = assistant_store.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
        execution_mode="execute",
    )
    runtime = AssistantToolRuntime(
        store=store,
        assistant_store=assistant_store,
        settings=Settings(dataset_store_path=str(root), environment="test"),
        run_id=run.id,
        conversation=conversation,
        event=lambda **_: None,
    )
    jobs_before = len(store.list_analysis_jobs(limit=100))

    result = runtime.execute(
        "revise_report",
        {"report_id": str(source_report_id), "instruction": "图表美化并精简报告"},
    )

    assert result["analysis_rerun"] is False
    assert len(store.list_analysis_jobs(limit=100)) == jobs_before
    revised = store.get_report(UUID(result["report_id"]))
    source_structured = store.get_report(source_report_id)["metadata"]["structured_report"]
    revised_structured = revised["metadata"]["structured_report"]
    assert revised_structured["sql_results"] == source_structured["sql_results"]
    assert revised_structured["python_results"] == source_structured["python_results"]
    assert revised_structured["charts"][0]["data"] == source_structured["charts"][0]["data"]
    assert len(revised_structured["key_findings"]) == 4
    assert revised["metadata"]["report_revision"]["evidence_frozen"] is True
    assert runtime.evidence[f"report:{result['report_id']}"]["artifact_role"] == "deliverable"
