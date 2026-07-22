from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.analysis.prompt_override_router import PromptOverrideModelRouter
from app.assistant.tools import assistant_tools_for_mode
from app.mcp.tool_schemas import ModelRouterResponse
from app.schemas.prompt_overrides import AgentPromptOverrides
from app.storage.dataset_store import DatasetStoreRepository


class CapturingRouter:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.metadata: dict[str, object] = {}

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ModelRouterResponse:
        self.messages = messages
        self.metadata = dict(metadata or {})
        return ModelRouterResponse(provider="mock", model="mock", content="ok")


def test_prompt_override_is_scoped_and_does_not_replace_system_message() -> None:
    delegate = CapturingRouter()
    router = PromptOverrideModelRouter(
        delegate,
        {
            "all": "回答保持简洁。",
            "report": "报告只保留有证据的关键结论。",
            "python": "生成短代码。",
        },
    )

    router.complete(
        messages=[{"role": "system", "content": "immutable safety"}],
        metadata={"agent": "report_execute"},
    )

    assert delegate.messages[0] == {"role": "system", "content": "immutable safety"}
    injected = str(delegate.messages[-1]["content"])
    assert "回答保持简洁" in injected
    assert "报告只保留" in injected
    assert "生成短代码" not in injected
    assert delegate.metadata["prompt_override_stage"] == "report"
    assert delegate.metadata["prompt_override_hash"]


def test_prompt_override_total_budget_is_enforced() -> None:
    with pytest.raises(ValidationError, match="12000"):
        AgentPromptOverrides(all="a" * 4000, report="b" * 4000, python="c" * 4000, sql="d")


def test_job_prompt_overrides_are_persisted(tmp_path: Path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    overrides = {"visualization": "使用清晰配色。", "report": "生成精简管理摘要。"}

    analysis = repository.create_analysis_job(
        dataset_id=dataset.id,
        question="总结销售情况",
        prompt_overrides=overrides,
    )
    cleaning = repository.create_cleaning_job(
        dataset_id=dataset.id,
        requirement="保守清洗",
        prompt_overrides={"cleaning": "不要删除有效业务行。"},
    )

    assert analysis.prompt_overrides == overrides
    assert cleaning.prompt_overrides == {"cleaning": "不要删除有效业务行。"}


def test_execute_mode_exposes_report_revision_and_stage_overrides() -> None:
    tools = {
        item["function"]["name"]: item["function"] for item in assistant_tools_for_mode("execute")
    }

    assert "revise_report" in tools
    analysis_properties = tools["start_analysis"]["parameters"]["properties"]
    cleaning_properties = tools["start_cleaning"]["parameters"]["properties"]
    assert set(analysis_properties["prompt_overrides"]["properties"]) == {
        "all",
        "cleaning",
        "planner",
        "sql",
        "python",
        "visualization",
        "review",
        "report",
    }
    assert "prompt_overrides" in cleaning_properties
