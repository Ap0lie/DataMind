from __future__ import annotations

from typing import Any

import pytest

from app.analysis.cleaning_workflow import CleaningWorkflowRunner
from app.mcp.tool_schemas import ModelRouterResponse
from app.storage.dataset_store import DatasetStoreRepository

pytestmark = pytest.mark.workflow


class UnsafeCleaningRouter:
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
        if str((metadata or {}).get("agent")) == "cleaning_decide":
            return ModelRouterResponse(
                provider="mock", model="cleaning", content=None,
                tool_calls=({"id": "choose", "type": "function", "function": {"name": "select_cleaning_strategy", "arguments": '{"strategy":"llm","reason":"需要语义标准化"}'}},),
                finish_reason="tool_calls", token_usage={"total_tokens": 7},
            )
        return ModelRouterResponse(
            provider="mock", model="cleaning",
            content="```python\ndef clean_dataset(df):\n    open('forbidden.txt', 'w')\n    return df\n```",
            finish_reason="stop", token_usage={"total_tokens": 11},
        )


def test_rules_cleaning_loop_commits_one_active_version(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="中文订单.txt", source_type="txt", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"客户": " 张三 ", "金额": "10"}, {"客户": " 张三 ", "金额": "10"}],
    )
    job = repository.create_cleaning_job(
        dataset_id=dataset.id, requirement="去空格并去重", cleaning_strategy="rules"
    )

    result = CleaningWorkflowRunner(repository).run(job_id=job.id)

    assert result["selected_strategy"] == "rules"
    assert repository.read_cleaned_records(dataset.id) == [{"客户": "张三", "金额": 10}]
    runs = repository.list_cleaning_runs(dataset.id)
    assert len(runs) == 1
    assert runs[0]["is_active"] is True
    assert runs[0]["job_id"] == str(job.id)


def test_cleaning_job_lease_claim_is_exclusive(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="lease.csv", source_type="csv", source_metadata={})
    job = repository.create_cleaning_job(dataset_id=dataset.id, cleaning_strategy="rules")

    first = repository.claim_cleaning_job(job.id, worker_id="worker-a", lease_seconds=120)
    second = repository.claim_cleaning_job(job.id, worker_id="worker-b", lease_seconds=120)

    assert first is not None
    assert first.lease_owner == "worker-a"
    assert second is None


def test_unsafe_llm_cleaning_repairs_then_falls_back_without_raw_event_rows(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="客户.txt", source_type="txt", source_metadata={})
    repository.append_raw_records(dataset_id=dataset.id, records=[{"客户": " Alice ", "等级": "重要"}])
    job = repository.create_cleaning_job(
        dataset_id=dataset.id, requirement="统一客户等级", cleaning_strategy="auto"
    )
    events: list[dict[str, Any]] = []

    result = CleaningWorkflowRunner(repository, model_router=UnsafeCleaningRouter()).run(
        job_id=job.id, event_callback=events.append
    )

    assert result["selected_strategy"] == "rules"
    assert result["failures"]
    assert any(event.get("event_type") == "cleaning_repair" for event in events)
    assert any(event.get("event_type") == "cleaning_commit" for event in events)
    serialized = str(events)
    assert "Alice" not in serialized
    assert "重要" not in serialized
