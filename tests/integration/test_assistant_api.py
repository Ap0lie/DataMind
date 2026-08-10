from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from PIL import Image

from app.core.settings import get_settings
from app.main import create_app
from app.storage.assistant_repository import AssistantRepository


@pytest.fixture
def assistant_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "datasets"))
    monkeypatch.setenv("DATAMIND_AUTH_MODE", "legacy")
    monkeypatch.setenv("DATAMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAMIND_CHECKPOINT_BACKEND", "none")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_assistant_conversation_message_and_user_isolation(assistant_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.v1.assistant.start_assistant_run", lambda **_: None)
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/v1/assistant/conversations", json={"scope_type": "auto"}, headers={"X-DataMind-User": "alice"})
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        run = await client.post(f"/api/v1/assistant/conversations/{conversation_id}/messages", json={"content": "总结我的报告", "attachment_ids": []}, headers={"X-DataMind-User": "alice"})
        assert run.status_code == 202
        messages = await client.get(f"/api/v1/assistant/conversations/{conversation_id}/messages", headers={"X-DataMind-User": "alice"})
        assert [item["role"] for item in messages.json()["messages"]] == ["user", "assistant"]
        hidden = await client.get(f"/api/v1/assistant/conversations/{conversation_id}", headers={"X-DataMind-User": "bob"})
        assert hidden.status_code == 404


@pytest.mark.asyncio
async def test_assistant_run_can_pause_resume_and_remain_attached_to_conversation(
    assistant_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[str] = []
    monkeypatch.setattr(
        "app.api.v1.assistant.start_assistant_run",
        lambda **kwargs: dispatched.append(str(kwargs["run_id"])),
    )
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    headers = {"X-DataMind-User": "alice"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/assistant/conversations",
            json={"scope_type": "auto"},
            headers=headers,
        )
        conversation_id = created.json()["conversation_id"]
        submitted = await client.post(
            f"/api/v1/assistant/conversations/{conversation_id}/messages",
            json={"content": "分析销售表现", "attachment_ids": []},
            headers=headers,
        )
        run_id = submitted.json()["run_id"]

        paused = await client.post(
            f"/api/v1/assistant/runs/{run_id}/pause",
            json={},
            headers=headers,
        )
        assert paused.status_code == 202
        assert paused.json()["status"] == "paused"

        conversations = await client.get(
            "/api/v1/assistant/conversations",
            headers=headers,
        )
        selected = next(
            item
            for item in conversations.json()["conversations"]
            if item["conversation_id"] == conversation_id
        )
        assert selected["active_run_id"] == run_id
        assert selected["active_run_status"] == "paused"

        resumed = await client.post(
            f"/api/v1/assistant/runs/{run_id}/resume",
            json={},
            headers=headers,
        )
        assert resumed.status_code == 202
        assert resumed.json()["status"] == "queued"
        assert dispatched == [run_id, run_id]

        AssistantRepository(
            assistant_settings.dataset_store_path, user_id="alice"
        ).update_run(
            UUID(run_id),
            status="running",
            current_stage="compose",
        )
        canceled = await client.post(
            f"/api/v1/assistant/runs/{run_id}/cancel",
            json={},
            headers=headers,
        )
        assert canceled.status_code == 200
        assert canceled.json()["status"] == "canceled"
        messages = await client.get(
            f"/api/v1/assistant/conversations/{conversation_id}/messages",
            headers=headers,
        )
        assert messages.json()["messages"][-1]["status"] == "canceled"
        assert messages.json()["messages"][-1]["content"].startswith(
            "已结束本次 Kimi 任务"
        )

        conversations = await client.get(
            "/api/v1/assistant/conversations",
            headers=headers,
        )
        selected = next(
            item
            for item in conversations.json()["conversations"]
            if item["conversation_id"] == conversation_id
        )
        assert selected["active_run_id"] is None
        assert selected["active_run_status"] is None


@pytest.mark.asyncio
async def test_assistant_image_upload_is_validated_and_protected(assistant_settings) -> None:
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    image = Image.new("RGB", (48, 32), color=(20, 120, 90))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/v1/assistant/conversations", json={"scope_type": "auto"}, headers={"X-DataMind-User": "alice"})
        conversation_id = created.json()["conversation_id"]
        uploaded = await client.post(f"/api/v1/assistant/conversations/{conversation_id}/attachments", files={"file": ("chart.png", buffer.getvalue(), "image/png")}, headers={"X-DataMind-User": "alice"})
        assert uploaded.status_code == 201
        payload = uploaded.json()
        assert payload["width"] == 48
        content = await client.get(f"/api/v1/assistant/attachments/{payload['attachment_id']}/content", headers={"X-DataMind-User": "alice"})
        assert content.status_code == 200
        forbidden = await client.get(f"/api/v1/assistant/attachments/{payload['attachment_id']}/content", headers={"X-DataMind-User": "bob"})
        assert forbidden.status_code == 404


@pytest.mark.asyncio
async def test_assistant_multi_file_import_creates_group_and_full_grant(assistant_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.v1.assistant.start_cleaning_job", lambda **_: None)
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    headers = {"X-DataMind-User": "alice"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/v1/assistant/conversations", json={"scope_type": "auto"}, headers=headers)
        conversation_id = created.json()["conversation_id"]
        attachment_ids = []
        for name, content in (
            ("orders.csv", b"order_id,customer_id,amount\n1,c1,10\n2,c2,20\n"),
            ("customers.csv", b"customer_id,segment\nc1,A\nc2,B\n"),
        ):
            uploaded = await client.post(
                f"/api/v1/assistant/conversations/{conversation_id}/attachments",
                files={"file": (name, content, "text/csv")},
                headers=headers,
            )
            assert uploaded.status_code == 201
            assert uploaded.json()["attachment_kind"] == "data_file"
            attachment_ids.append(uploaded.json()["attachment_id"])

        preview = await client.post(
            "/api/v1/assistant/import-batches/preview",
            json={"conversation_id": conversation_id, "attachment_ids": attachment_ids},
            headers=headers,
        )
        assert preview.status_code == 201
        assert preview.json()["preview"]["valid_count"] == 2

        committed = await client.post(
            f"/api/v1/assistant/import-batches/{preview.json()['batch_id']}/commit",
            json={"name": "Commerce package", "allow_partial": False},
            headers=headers,
        )
        assert committed.status_code == 202
        payload = committed.json()
        assert len(payload["dataset_ids"]) == 2
        assert payload["dataset_group_id"]

        grants = await client.get("/api/v1/assistant/permission-grants", headers=headers)
        assert grants.status_code == 200
        assert grants.json()["grants"][0]["asset_type"] == "dataset_group"
        assert set(grants.json()["grants"][0]["capabilities"]) == {
            "data_prepare",
            "relationship_manage",
            "analysis_manage",
            "report_manage",
            "semantic_manage",
            "asset_recycle",
        }


@pytest.mark.asyncio
async def test_assistant_recycle_bin_restores_original_dataset_id(assistant_settings) -> None:
    from app.storage.dataset_store import DatasetStoreRepository

    store = DatasetStoreRepository(assistant_settings.dataset_store_path, user_id="alice")
    dataset = store.create_dataset(name="temporary.csv", source_type="csv", source_metadata={})
    store.soft_delete_asset(asset_type="dataset", asset_id=dataset.id)
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    headers = {"X-DataMind-User": "alice"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        hidden = await client.get(f"/api/v1/store/datasets/{dataset.id}", headers=headers)
        assert hidden.status_code == 404
        recycled = await client.get("/api/v1/assistant/recycle-bin", headers=headers)
        assert recycled.status_code == 200
        assert recycled.json()["assets"][0]["asset_id"] == str(dataset.id)
        restored = await client.post(f"/api/v1/assistant/recycle-bin/dataset/{dataset.id}/restore", json={}, headers=headers)
        assert restored.status_code == 200
        visible = await client.get(f"/api/v1/store/datasets/{dataset.id}", headers=headers)
        assert visible.status_code == 200


@pytest.mark.asyncio
async def test_assistant_actions_hide_internal_audit_fields(assistant_settings) -> None:
    repository = AssistantRepository(assistant_settings.dataset_store_path, user_id="alice")
    created = repository.create_action(
        run_id=None,
        conversation_id=None,
        grant_id=None,
        tool_name="start_analysis",
        arguments_hash="arguments-hash",
        idempotency_key="assistant-action-list-contract",
        asset_type=None,
        asset_id=None,
    )
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/assistant/actions?limit=100",
            headers={"X-DataMind-User": "alice"},
        )

    assert response.status_code == 200
    action = response.json()["actions"][0]
    assert action["action_id"] == str(created["action_id"])
    assert action["tool_name"] == "start_analysis"
    assert "arguments_hash" not in action
    assert "idempotency_key" not in action
    assert "before_state" not in action


@pytest.mark.asyncio
async def test_historical_analysis_citation_is_enriched_with_report_artifact(
    assistant_settings,
) -> None:
    from app.storage.dataset_store import DatasetStoreRepository

    store = DatasetStoreRepository(assistant_settings.dataset_store_path, user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="销售分析完整报告",
        markdown="# 销售分析\n完整内容",
        metadata={
            "structured_report": {"executive_summary": "完整销售分析已经生成。"},
            "statistical_verification": {
                "status": "passed",
                "summary": "统计审查已通过。",
            },
        },
    )
    job = store.create_analysis_job(dataset_id=dataset.id, question="分析销售")
    store.update_analysis_job(
        job.id,
        status="completed",
        result={"structured_report": {"executive_summary": "完整销售分析已经生成。"}},
        report_id=report_id,
        completed=True,
    )
    repository = AssistantRepository(assistant_settings.dataset_store_path, user_id="alice")
    conversation = repository.create_conversation(title="报告对话", scope_type="auto", scope_id=None)
    message = repository.create_message(
        conversation_id=conversation["conversation_id"], role="assistant", content="分析已完成。"
    )
    repository.update_message(
        message["message_id"],
        content="分析已完成。",
        status="completed",
        citations=(
            {
                "source_type": "analysis_job",
                "source_id": str(job.id),
                "label": "分析销售",
                "excerpt": "分析任务已完成。",
                "dataset_id": str(dataset.id),
                "reliability": {
                    "status": "rejected",
                    "summary": "旧任务统计审查未通过。",
                },
            },
        ),
    )
    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/assistant/conversations/{conversation['conversation_id']}/messages",
            headers={"X-DataMind-User": "alice"},
        )

    assert response.status_code == 200
    citations = response.json()["messages"][0]["citations"]
    assert [item["source_type"] for item in citations] == ["analysis_job", "report"]
    assert citations[1]["source_id"] == str(report_id)
    assert citations[1]["label"] == "销售分析完整报告"
    assert citations[1]["artifact_role"] == "evidence"
    assert citations[0]["reliability"]["status"] == "rejected"
    assert citations[1]["reliability"]["status"] == "rejected"
    assert citations[1]["reliability"] == citations[0]["reliability"]


@pytest.mark.asyncio
async def test_completed_report_action_marks_only_latest_report_as_deliverable(
    assistant_settings,
) -> None:
    from app.storage.dataset_store import DatasetStoreRepository

    store = DatasetStoreRepository(assistant_settings.dataset_store_path, user_id="alice")
    dataset = store.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    source_id = store.save_report(
        dataset_id=dataset.id,
        title="原报告",
        markdown="原内容",
        metadata={"structured_report": {"executive_summary": "原结论"}},
    )
    revised_id = store.save_report(
        dataset_id=dataset.id,
        title="精简报告",
        markdown="精简内容",
        metadata={"structured_report": {"executive_summary": "精简结论"}},
    )
    repository = AssistantRepository(assistant_settings.dataset_store_path, user_id="alice")
    conversation = repository.create_conversation(title="报告修订", scope_type="auto", scope_id=None)
    user = repository.create_message(
        conversation_id=conversation["conversation_id"], role="user", content="精简报告"
    )
    assistant = repository.create_message(
        conversation_id=conversation["conversation_id"], role="assistant", content="已生成。"
    )
    run = repository.create_run(
        conversation_id=conversation["conversation_id"],
        user_message_id=user["message_id"],
        assistant_message_id=assistant["message_id"],
        execution_mode="execute",
    )
    action = repository.create_action(
        run_id=run.id,
        conversation_id=conversation["conversation_id"],
        grant_id=None,
        tool_name="revise_report",
        arguments_hash="hash",
        idempotency_key="latest-report-deliverable",
        asset_type="report",
        asset_id=source_id,
    )
    repository.complete_action(
        action["action_id"],
        result={"report_id": str(revised_id)},
        before_state={},
        after_state={"created_report_id": str(revised_id)},
        reversible=True,
    )
    repository.update_message(
        assistant["message_id"],
        content="已生成。",
        status="completed",
        citations=(
            {
                "source_type": "report",
                "source_id": str(source_id),
                "label": "原报告",
                "excerpt": "原结论",
                "dataset_id": str(dataset.id),
            },
            {
                "source_type": "report",
                "source_id": str(revised_id),
                "label": "精简报告",
                "excerpt": "精简结论",
                "dataset_id": str(dataset.id),
            },
        ),
    )

    app = create_app(assistant_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/assistant/conversations/{conversation['conversation_id']}/messages",
            headers={"X-DataMind-User": "alice"},
        )

    assert response.status_code == 200
    citations = response.json()["messages"][1]["citations"]
    assert [item["artifact_role"] for item in citations] == ["evidence", "deliverable"]
