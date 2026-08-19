from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pytest

from app.assistant.memory import AssistantMemoryService
from app.core.settings import get_settings
from app.memory.namespaces import build_memory_namespace
from app.memory.projections import memory_store_key, project_agent_memories
from app.memory.store import DataMindMemoryStore
from app.semantic.embedding import DisabledEmbeddingProvider
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.dataset_store import DatasetStoreRepository


@pytest.fixture
def store_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "datasets"
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(root))
    monkeypatch.setenv("DATAMIND_ENVIRONMENT", "test")
    get_settings.cache_clear()
    repository = AssistantMemoryRepository(str(root), user_id="alice")
    memory_store = DataMindMemoryStore(repository, recycle_retention_days=30)
    yield root, repository, memory_store
    get_settings.cache_clear()


def test_langmem_dependency_and_store_contract_are_pinned() -> None:
    from langgraph.store.base import BaseStore

    assert version("langmem") == "0.0.30"
    assert issubclass(DataMindMemoryStore, BaseStore)


def test_memory_namespace_is_strict_and_user_scoped() -> None:
    dataset_id = uuid4()
    namespace = build_memory_namespace(
        user_id="alice",
        scope_type="dataset",
        scope_id=dataset_id,
        memory_kind="semantic",
    )
    assert namespace == ("alice", "dataset", str(dataset_id), "semantic")

    with pytest.raises(ValueError, match="cannot have a scope id"):
        build_memory_namespace(
            user_id="alice",
            scope_type="user",
            scope_id=dataset_id,
            memory_kind="semantic",
        )

    with pytest.raises(ValueError, match="not a valid LangGraph namespace"):
        build_memory_namespace(
            user_id="alice.example",
            scope_type="user",
            scope_id=None,
            memory_kind="semantic",
        )


def test_store_round_trip_search_and_soft_delete(store_context) -> None:
    _root, repository, store = store_context
    namespace = build_memory_namespace(
        user_id="alice",
        scope_type="user",
        scope_id=None,
        memory_kind="semantic",
    )
    key = "business_context:business_context:focus_market"
    store.put(
        namespace,
        key,
        {
            "memory_type": "business_context",
            "subject_key": "business_context:focus_market",
            "content": "华东区域是当前重点市场",
            "status": "active",
            "explicit": True,
            "confidence": 1.0,
        },
    )

    item = store.get(namespace, key)
    assert item is not None
    assert item.value["content"] == "华东区域是当前重点市场"
    assert item.key == key
    results = store.search(
        namespace,
        query="重点市场",
        filter={"status": "active"},
        limit=8,
    )
    assert [result.key for result in results] == [key]
    assert store.list_namespaces(prefix=("alice",)) == [namespace]

    store.delete(namespace, key)
    assert store.get(namespace, key) is None
    assert repository.list(status="recycled")[0]["content"] == "华东区域是当前重点市场"


def test_store_projects_existing_versions_without_changing_ids(store_context) -> None:
    _root, repository, store = store_context
    saved = repository.save(
        memory_type="workflow_preference",
        scope_type="user",
        scope_id=None,
        normalized_key="workflow_preference:language",
        subject_key="workflow_preference:language",
        content="报告默认使用中文",
        explicit=True,
        confidence=1.0,
        status="active",
    )
    namespace = build_memory_namespace(
        user_id="alice",
        scope_type="user",
        scope_id=None,
        memory_kind="semantic",
    )
    item = store.get(namespace, memory_store_key(saved))
    assert item is not None
    assert item.value["memory_id"] == str(saved["memory_id"])
    assert item.value["version"] == 1

    store.put(
        namespace,
        item.key,
        item.value | {"content": "报告默认使用中英文双语"},
    )
    updated = store.get(namespace, item.key)
    assert updated is not None
    assert updated.value["version"] == 2
    assert updated.value["memory_id"] != item.value["memory_id"]
    assert repository.get(saved["memory_id"])["status"] == "superseded"

    with pytest.raises(PermissionError):
        store.search(("bob",), limit=8)


def test_service_recall_uses_store_as_its_only_source(store_context, monkeypatch) -> None:
    root, repository, _store = store_context
    dataset_store = DatasetStoreRepository(str(root), user_id="alice")
    service = AssistantMemoryService(
        repository=repository,
        store=dataset_store,
        settings=get_settings(),
        embedding_provider=DisabledEmbeddingProvider(),
    )
    service.create_manual(
        memory_type="business_context",
        scope_type="user",
        scope_id=None,
        content="华东区域是当前重点市场",
    )

    calls = 0
    original = service.langmem_store.search

    def observed_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service.langmem_store, "search", observed_search)
    recalled = service.retrieve(
        question="重点市场在哪里？",
        conversation={"scope_type": "auto", "scope_id": None},
    )

    assert calls == 1
    assert len(recalled) == 1


@pytest.mark.asyncio
async def test_store_async_api_uses_the_same_repository_boundary(store_context) -> None:
    _root, _repository, store = store_context
    namespace = build_memory_namespace(
        user_id="alice",
        scope_type="user",
        scope_id=None,
        memory_kind="semantic",
    )
    await store.aput(
        namespace,
        "preference:preference:answer_style",
        {
            "memory_type": "preference",
            "subject_key": "preference:answer_style",
            "content": "回答保持简洁",
            "status": "active",
        },
    )
    item = await store.aget(namespace, "preference:preference:answer_style")
    assert item is not None
    assert item.value["content"] == "回答保持简洁"


def test_agent_memory_projection_enforces_role_boundaries() -> None:
    semantic = {
        "memory_id": uuid4(),
        "memory_kind": "semantic",
        "memory_type": "metric_definition",
        "content": "GMV 指支付金额总和",
        "scope_type": "dataset",
        "scope_id": uuid4(),
    }
    preference = semantic | {
        "memory_id": uuid4(),
        "memory_type": "workflow_preference",
        "content": "报告默认简洁",
    }
    experience = semantic | {
        "memory_id": uuid4(),
        "memory_kind": "episodic",
        "memory_type": "analysis_experience",
        "content": "已验证分析路线",
        "structured_value": {"analysis_contract": {"metric": "gmv"}},
    }

    assert [item["memory_type"] for item in project_agent_memories("planner", (semantic, preference, experience))] == [
        "metric_definition",
        "analysis_experience",
    ]
    assert [item["memory_type"] for item in project_agent_memories("report", (semantic, preference, experience))] == [
        "workflow_preference",
        "analysis_experience",
    ]
    assert project_agent_memories("sql", (semantic, preference, experience)) == ()
    assert project_agent_memories("python", (semantic, preference, experience)) == ()


def test_store_failure_skips_memory_without_repository_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(root))
    monkeypatch.setenv("DATAMIND_ENVIRONMENT", "test")
    get_settings.cache_clear()
    try:
        repository = AssistantMemoryRepository(str(root), user_id="alice")
        service = AssistantMemoryService(
            repository=repository,
            store=DatasetStoreRepository(str(root), user_id="alice"),
            settings=get_settings(),
            embedding_provider=DisabledEmbeddingProvider(),
        )
        saved = service.create_manual(
            memory_type="business_context",
            scope_type="user",
            scope_id=None,
            content="华东区域是重点市场",
        )
        monkeypatch.setattr(
            service.langmem_store,
            "search",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("store down")),
        )
        recalled = service.retrieve(
            question="重点市场",
            conversation={"scope_type": "auto", "scope_id": None},
        )

        assert saved["memory_id"]
        assert recalled == ()
    finally:
        get_settings.cache_clear()


def test_analysis_memory_contexts_are_projected_per_agent(store_context, monkeypatch) -> None:
    root, repository, _memory_store = store_context
    service = AssistantMemoryService(
        repository=repository,
        store=DatasetStoreRepository(str(root), user_id="alice"),
        settings=get_settings(),
        embedding_provider=DisabledEmbeddingProvider(),
    )
    metric = {
        "memory_id": uuid4(),
        "memory_kind": "semantic",
        "memory_type": "metric_definition",
        "content": "GMV 指支付金额总和",
    }
    preference = {
        "memory_id": uuid4(),
        "memory_kind": "semantic",
        "memory_type": "workflow_preference",
        "content": "报告保持简洁",
    }
    experience = {
        "memory_id": uuid4(),
        "memory_kind": "episodic",
        "memory_type": "analysis_experience",
        "content": "已验证的分析路线",
        "structured_value": {"analysis_contract": {"metric": "gmv"}},
    }
    monkeypatch.setattr(
        service,
        "_recall_kind",
        lambda **kwargs: (experience,)
        if kwargs["memory_kind"] == "episodic"
        else (metric, preference),
    )

    contexts = service.retrieve_analysis_memory_contexts(
        question="分析 GMV",
        dataset_id=uuid4(),
    )

    assert [item["memory_type"] for item in contexts["planner"]] == [
        "metric_definition",
        "analysis_experience",
    ]
    assert contexts["sql"] == ()
    assert contexts["python"] == ()
    assert [item["memory_type"] for item in contexts["reviewer"]] == [
        "metric_definition",
        "analysis_experience",
    ]
    assert [item["memory_type"] for item in contexts["report"]] == [
        "workflow_preference",
        "analysis_experience",
    ]
