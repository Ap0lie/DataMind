from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from app.core.settings import get_settings
from app.mcp.tool_schemas import ModelRouterResponse
from app.memory.guards import DataMindMemoryGuard
from app.memory.manager import LangMemFormationManager
from app.memory.store import DataMindMemoryStore
from app.storage.assistant_memory_repository import AssistantMemoryRepository


def test_langmem_manager_accepts_datamind_base_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langmem import create_memory_store_manager

    root = tmp_path / "datasets"
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(root))
    monkeypatch.setenv("DATAMIND_ENVIRONMENT", "test")
    get_settings.cache_clear()
    try:
        repository = AssistantMemoryRepository(str(root), user_id="alice")
        store = DataMindMemoryStore(repository)
        manager = create_memory_store_manager(
            FakeListChatModel(responses=["ok"]),
            namespace=("alice", "user", "user", "semantic"),
            store=store,
        )
        assert type(manager).__name__ == "MemoryStoreManager"
    finally:
        get_settings.cache_clear()


class _FormationRouter:
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        tools = tuple(kwargs.get("tools") or ())
        schema = next(
            item
            for item in tools
            if item.get("function", {}).get("name") == "LangMemSemanticMemory"
        )
        assert schema["function"]["parameters"]["properties"]["source_message_ids"]
        source_id = str(kwargs["messages"][-1]["content"]).split("Source message id: ", 1)[1].splitlines()[0]
        return ModelRouterResponse(
            provider="mock",
            model="memory-model",
            content=None,
            tool_calls=(
                {
                    "id": "memory-call-1",
                    "type": "function",
                    "function": {
                        "name": "LangMemSemanticMemory",
                        "arguments": (
                            '{"memory_type":"metric_definition","entity_key":"gmv",'
                            '"predicate":"definition","typed_value":{"type":"text",'
                            '"value":"支付金额总和"},"unit":"CNY",'
                            '"content":"GMV 定义为支付金额总和",'
                            '"evidence":"请记住，GMV 定义为支付金额总和",'
                            f'"source_message_ids":["{source_id}"],'
                            '"confidence":0.94,"correction":false,'
                            '"scope":"conversation"}'
                        ),
                    },
                },
            ),
            finish_reason="tool_calls",
        )


def test_langmem_manager_output_is_guarded_before_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAMIND_ENVIRONMENT", "test")
    get_settings.cache_clear()
    try:
        message_id = uuid4()
        message = {
            "message_id": message_id,
            "role": "user",
            "content": "请记住，GMV 定义为支付金额总和",
        }
        formed = LangMemFormationManager(
            get_settings(),
            router=_FormationRouter(),
        ).extract(
            text=message["content"],
            source_message_id=str(message_id),
        )
        result = DataMindMemoryGuard().validate(
            formed,
            source_message=message,
            source_message_id=message_id,
        )

        assert result.rejected_codes == ()
        assert len(result.accepted) == 1
        assert result.accepted[0].memory_type == "metric_definition"
        assert result.accepted[0].explicit is True

        rejected = DataMindMemoryGuard().validate(
            (formed[0] | {"source_message_ids": [str(uuid4())]},),
            source_message=message,
            source_message_id=message_id,
        )
        assert rejected.accepted == ()
        assert rejected.rejected_codes == ("source_mismatch",)
    finally:
        get_settings.cache_clear()
