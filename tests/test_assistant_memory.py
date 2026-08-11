from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.assistant.memory import (
    AssistantMemoryService,
    extract_memory_candidates,
)
from app.core.settings import get_settings
from app.semantic.embedding import DisabledEmbeddingProvider
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository


@pytest.fixture
def memory_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "datasets"
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(root))
    monkeypatch.setenv("DATAMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAMIND_ASSISTANT_MEMORY_SUMMARY_MESSAGES", "12")
    monkeypatch.setenv("DATAMIND_ASSISTANT_MEMORY_SUMMARY_CHARS", "24000")
    get_settings.cache_clear()
    settings = get_settings()
    store = DatasetStoreRepository(str(root), user_id="alice")
    repository = AssistantMemoryRepository(str(root), user_id="alice")
    service = AssistantMemoryService(
        repository=repository,
        store=store,
        settings=settings,
        embedding_provider=DisabledEmbeddingProvider(),
    )
    yield settings, store, repository, service
    get_settings.cache_clear()


def test_explicit_and_inferred_memory_are_separated_and_sensitive_values_are_rejected() -> None:
    explicit = extract_memory_candidates("请记住，以后的分析报告默认使用中文并保持简洁。")
    assert len(explicit) == 1
    assert explicit[0].explicit is True
    assert explicit[0].memory_type == "workflow_preference"

    inferred = extract_memory_candidates("我喜欢绿色的图表风格。")
    assert len(inferred) == 1
    assert inferred[0].explicit is False

    assert extract_memory_candidates("这次请把报告改短一点。") == ()
    assert extract_memory_candidates("请记住 API_KEY=not-a-real-secret-value") == ()


def test_explicit_conflict_creates_auditable_version_chain(memory_context) -> None:
    _settings, store, repository, service = memory_context
    first = service.create_manual(
        memory_type="workflow_preference",
        scope_type="user",
        scope_id=None,
        content="默认用中文生成简洁报告",
    )
    second = service.create_manual(
        memory_type="workflow_preference",
        scope_type="user",
        scope_id=None,
        content="默认用中文生成详细报告",
    )
    assert first["memory_id"] != second["memory_id"]
    assert repository.get(first["memory_id"])["status"] == "superseded"
    assert repository.get(first["memory_id"])["superseded_by_id"] == second["memory_id"]
    assert second["version"] == 2
    assert second["supersedes_id"] == first["memory_id"]
    assert second["content"] == "默认用中文生成详细报告"
    assert [item["version"] for item in repository.history(second["memory_id"])] == [2, 1]

    reactivated = repository.reactivate(first["memory_id"])
    assert reactivated["version"] == 3
    assert reactivated["content"] == first["content"]
    assert repository.get(second["memory_id"])["status"] == "superseded"

    bob = AssistantMemoryRepository(store.root_path, user_id="bob")
    assert bob.list() == ()


def test_inferred_conflict_remains_pending_until_confirmed(memory_context) -> None:
    _settings, store, repository, service = memory_context
    assistant_store = AssistantRepository(store.root_path, user_id="alice")
    conversation = assistant_store.create_conversation(
        title="偏好冲突",
        scope_type="auto",
        scope_id=None,
    )
    first_message = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="请记住，以后的报告默认保持简洁。",
    )
    service.capture_user_memories(conversation=conversation, user_message=first_message)
    active = repository.list(status="active")[0]

    candidate_message = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="我喜欢详细的报告。",
    )
    events = service.capture_user_memories(
        conversation=conversation,
        user_message=candidate_message,
    )
    assert events[0]["event_type"] == "memory.conflict"
    pending = repository.list(status="pending")[0]
    assert repository.get(active["memory_id"])["status"] == "active"
    assert pending["supersedes_id"] is None

    confirmed = repository.confirm(pending["memory_id"])
    assert confirmed["status"] == "active"
    assert confirmed["supersedes_id"] == active["memory_id"]
    assert repository.get(active["memory_id"])["status"] == "superseded"


def test_memory_maintenance_job_is_idempotent_and_lease_claimed_once(memory_context) -> None:
    _settings, _store, repository, _service = memory_context
    run_id = uuid4()
    conversation_id = uuid4()
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    first = repository.create_maintenance_job(
        run_id=run_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    repeated = repository.create_maintenance_job(
        run_id=run_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )

    assert repeated["job_id"] == first["job_id"]
    claimed = repository.claim_maintenance_job(
        first["job_id"], worker_id="worker-a", lease_seconds=60
    )
    assert claimed is not None
    assert claimed["attempt_count"] == 1
    assert repository.claim_maintenance_job(
        first["job_id"], worker_id="worker-b", lease_seconds=60
    ) is None

    with repository._connect() as connection:
        connection.execute(
            "UPDATE assistant_memory_maintenance_jobs SET lease_expires_at=? WHERE id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), str(first["job_id"])),
        )
    recoverable = repository.list_all_recoverable_maintenance_jobs()
    assert [item["job_id"] for item in recoverable] == [first["job_id"]]
    recovered = repository.claim_maintenance_job(
        first["job_id"], worker_id="worker-b", lease_seconds=60
    )
    assert recovered is not None
    assert recovered["attempt_count"] == 2

    repository.finish_maintenance_job(first["job_id"])
    completed = repository.get_maintenance_job(first["job_id"])
    assert completed["status"] == "completed"
    assert repository.create_maintenance_job(
        run_id=run_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )["status"] == "completed"


def test_asset_memory_inherits_to_group_member_but_not_unrelated_dataset(memory_context) -> None:
    _settings, store, repository, service = memory_context
    member = store.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    unrelated = store.create_dataset(name="inventory.csv", source_type="csv", source_metadata={})
    group = store.create_dataset_group(name="电商数据包", dataset_ids=(member.id,))
    service.create_manual(
        memory_type="metric_definition",
        scope_type="dataset_group",
        scope_id=group.id,
        content="销售额口径是 payment_value 的总和",
    )

    inherited = service.retrieve(
        question="销售额口径是什么？",
        conversation={"scope_type": "dataset", "scope_id": member.id},
    )
    assert [item["content"] for item in inherited] == ["销售额口径是 payment_value 的总和"]

    isolated = service.retrieve(
        question="销售额口径是什么？",
        conversation={"scope_type": "dataset", "scope_id": unrelated.id},
    )
    assert isolated == ()
    assert repository.get(inherited[0]["memory_id"])["last_used_at"] is not None


def test_conversation_summary_keeps_latest_eight_messages_and_advances_cursor(memory_context) -> None:
    _settings, store, _repository, service = memory_context
    assistant_store = AssistantRepository(store.root_path, user_id="alice")
    conversation = assistant_store.create_conversation(
        title="摘要测试",
        scope_type="auto",
        scope_id=None,
    )
    for index in range(14):
        assistant_store.create_message(
            conversation_id=conversation["conversation_id"],
            role="user" if index % 2 == 0 else "assistant",
            content=f"第 {index} 条消息，关键事实 {index}",
        )

    updated = service.update_conversation_summary(
        assistant_store=assistant_store,
        conversation_id=conversation["conversation_id"],
        summarizer=lambda source: "模型摘要：" + source,
    )
    assert updated is not None
    assert updated["summary_version"] == 1
    assert updated["summary_through_message_id"] is not None
    remaining = assistant_store.list_messages_after(
        conversation["conversation_id"],
        after_message_id=updated["summary_through_message_id"],
    )
    assert len(remaining) == 8
    assert "关键事实 0" in updated["summary"]
    assert updated["summary_payload"]["facts"]
    summarized_ids = {
        str(source_id)
        for values in updated["summary_payload"].values()
        for item in values
        for source_id in item["source_message_ids"]
    }
    all_message_ids = {
        str(item["message_id"])
        for item in assistant_store.list_messages(conversation["conversation_id"])
    }
    assert summarized_ids <= all_message_ids
    assert service.update_conversation_summary(
        assistant_store=assistant_store,
        conversation_id=conversation["conversation_id"],
    ) is None


def test_pending_confirm_recycle_restore_and_stale_lifecycle(memory_context) -> None:
    settings, store, repository, service = memory_context
    conversation = AssistantRepository(store.root_path, user_id="alice").create_conversation(
        title="记忆测试",
        scope_type="auto",
        scope_id=None,
    )
    assistant_store = AssistantRepository(store.root_path, user_id="alice")
    message = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="我喜欢图表使用绿色。",
    )
    events = service.capture_user_memories(
        conversation=conversation,
        user_message=message,
    )
    assert events[0]["event_type"] == "memory.candidate"
    pending = repository.list(status="pending")[0]
    confirmed = repository.update(pending["memory_id"], status="active")
    assert confirmed["status"] == "active"

    recycled = repository.recycle(
        confirmed["memory_id"],
        retention_days=settings.assistant_memory_recycle_days,
    )
    assert recycled["status"] == "recycled"
    assert repository.restore(confirmed["memory_id"])["status"] == "active"

    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    with repository._connect() as connection:
        connection.execute(
            "UPDATE assistant_memories SET updated_at=?,last_used_at=NULL WHERE id=?",
            (old, str(confirmed["memory_id"])),
        )
    assert repository.recycle_stale(
        active_days=180,
        pending_days=30,
        retention_days=30,
    ) == 1
    assert repository.get(confirmed["memory_id"])["status"] == "recycled"


def test_memory_switch_disables_long_term_reads_and_writes_but_not_summary(
    memory_context,
) -> None:
    _settings, store, repository, service = memory_context
    service.create_manual(
        memory_type="business_context",
        scope_type="user",
        scope_id=None,
        content="华东区域是当前重点市场",
    )
    repository.update_settings(enabled=False)
    assert service.retrieve(
        question="重点市场在哪里？",
        conversation={"scope_type": "auto", "scope_id": None},
    ) == ()

    assistant_store = AssistantRepository(store.root_path, user_id="alice")
    conversation = assistant_store.create_conversation(
        title="关闭长期记忆",
        scope_type="auto",
        scope_id=None,
    )
    for index in range(12):
        assistant_store.create_message(
            conversation_id=conversation["conversation_id"],
            role="user" if index % 2 == 0 else "assistant",
            content=f"第 {index} 条上下文",
        )
    new_message = assistant_store.create_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content="请记住，以后默认用中文。",
    )
    assert service.capture_user_memories(
        conversation=conversation,
        user_message=new_message,
    ) == ()
    assert service.update_conversation_summary(
        assistant_store=assistant_store,
        conversation_id=conversation["conversation_id"],
    ) is not None


def test_retrieval_applies_relevance_gate_and_records_actual_usage(memory_context) -> None:
    _settings, _store, repository, service = memory_context
    relevant = service.create_manual(
        memory_type="business_context",
        scope_type="user",
        scope_id=None,
        content="华东区域是重点销售市场",
    )
    service.create_manual(
        memory_type="business_context",
        scope_type="user",
        scope_id=None,
        content="客服团队每周一召开例会",
    )
    run_id = uuid4()
    recalled = service.retrieve(
        question="华东销售市场表现如何？",
        conversation={"scope_type": "auto", "scope_id": None},
        run_id=run_id,
    )
    assert [item["memory_id"] for item in recalled] == [relevant["memory_id"]]
    usages = repository.list_usage(run_id=run_id)
    assert [item["memory_id"] for item in usages] == [relevant["memory_id"]]
    assert usages[0]["reason"]


def test_only_validated_analysis_becomes_experience_and_drift_marks_it_stale(
    memory_context,
) -> None:
    _settings, store, repository, service = memory_context
    dataset = store.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    cleaning_run_id = store.save_cleaning_result(
        dataset_id=dataset.id,
        provider="rules",
        model="rules-v1",
        prompt="",
        result_markdown="",
        cleaned_dataset={"records": []},
    )
    report_id = store.save_report(
        dataset_id=dataset.id,
        title="订单分析",
        markdown="# 订单分析",
        metadata={
            "structured_report": {"executive_summary": "订单金额保持增长。"},
            "analysis_contract": {"metrics": ["order_value"]},
        },
    )
    job = store.create_analysis_job(dataset_id=dataset.id, question="分析订单金额")
    store.update_analysis_job(
        job.id,
        status="completed",
        result={
            "analysis_contract": {"metrics": ["order_value"]},
            "statistical_verification": {"status": "passed", "checks": []},
            "validation_issues": [],
        },
        report_id=report_id,
        report_terminal_reason="validated",
        completed=True,
    )
    experience = service.save_analysis_experience(job.id)
    assert experience is not None
    assert experience["memory_kind"] == "episodic"
    assert experience["source_job_id"] == job.id
    assert experience["structured_value"]["asset_fingerprint"]["cleaning_versions"] == {
        str(dataset.id): str(cleaning_run_id)
    }

    recalled = service.retrieve_analysis_experiences(
        question="分析订单金额",
        dataset_id=dataset.id,
        run_id=uuid4(),
    )
    assert [item["memory_id"] for item in recalled] == [experience["memory_id"]]

    with store._connect() as connection:
        connection.execute(
            "UPDATE datasets SET updated_at=? WHERE id=? AND user_id=?",
            (
                (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                str(dataset.id),
                "alice",
            ),
        )
    assert service.retrieve_analysis_experiences(
        question="分析订单金额",
        dataset_id=dataset.id,
    ) == ()
    stale = repository.get(experience["memory_id"])
    assert stale["status"] == "stale"
    assert "变化" in stale["structured_value"]["stale_reason"]
