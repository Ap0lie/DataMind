from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from app.assistant.tools import AssistantToolRuntime
from app.assistant.workflow import AssistantWorkflowRunner, _assistant_temperature
from app.core.settings import Settings
from app.mcp.tool_schemas import ModelRouterResponse
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository

pytestmark = pytest.mark.workflow


def test_kimi_k26_uses_provider_supported_temperature() -> None:
    assert _assistant_temperature("kimi-k2.6") == 1.0
    assert _assistant_temperature("moonshot-v1-32k") == 0.1


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


def test_assistant_uses_existing_report_as_validated_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    store.append_raw_records(dataset_id=dataset.id, records=[{"region": "East", "sales": 120}])
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售分析",
        markdown="销售额增长 20%。",
        metadata={"question": "销售趋势", "structured_report": {"executive_summary": "销售额增长 20%。"}},
    )
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(title="新对话", scope_type="auto", scope_id=None)
    user = assistant_store.create_message(conversation_id=conversation["conversation_id"], role="user", content="销售趋势如何？")
    assistant = assistant_store.create_message(conversation_id=conversation["conversation_id"], role="assistant", content="", status="pending")
    run = assistant_store.create_run(conversation_id=conversation["conversation_id"], user_message_id=user["message_id"], assistant_message_id=assistant["message_id"])

    settings = Settings(dataset_store_path=str(root), checkpoint_backend="none", assistant_llm_provider="mock", assistant_llm_model="kimi-test", environment="test")
    AssistantWorkflowRunner(store=store, assistant_store=assistant_store, model_router=AnswerRouter(), settings=settings).run(run.id)

    completed = assistant_store.get_message(assistant["message_id"])
    assert completed["status"] == "completed"
    assert completed["citations"][0]["source_id"] == str(report_id)
    assert assistant_store.get_run(run.id).status == "completed"
    assert any(item["event_type"] == "message.delta" for item in assistant_store.list_events(run.id))


def test_assistant_omits_empty_messages_left_by_failed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAMIND_DATABASE_URL", "")
    root = tmp_path / "datasets"
    store = DatasetStoreRepository(str(root), user_id="alice")
    assistant_store = AssistantRepository(str(root), user_id="alice")
    conversation = assistant_store.create_conversation(title="历史对话", scope_type="auto", scope_id=None)
    assistant_store.create_message(conversation_id=conversation["conversation_id"], role="user", content="第一次分析")
    assistant_store.create_message(conversation_id=conversation["conversation_id"], role="assistant", content="", status="failed")
    user = assistant_store.create_message(conversation_id=conversation["conversation_id"], role="user", content="请重新分析")
    assistant = assistant_store.create_message(conversation_id=conversation["conversation_id"], role="assistant", content="", status="pending")
    run = assistant_store.create_run(conversation_id=conversation["conversation_id"], user_message_id=user["message_id"], assistant_message_id=assistant["message_id"])
    router = AnswerRouter()
    settings = Settings(dataset_store_path=str(root), checkpoint_backend="none", assistant_llm_provider="mock", assistant_llm_model="kimi-test", environment="test")

    AssistantWorkflowRunner(store=store, assistant_store=assistant_store, model_router=router, settings=settings).run(run.id)

    sent = router.calls[0]
    assert {"role": "user", "content": "请重新分析"} in sent
    assert not any(item.get("role") == "assistant" and not str(item.get("content") or "").strip() for item in sent)


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
        conversation_id=conversation["conversation_id"], role="assistant", content="", status="pending"
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
