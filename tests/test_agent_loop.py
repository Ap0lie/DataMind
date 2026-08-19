from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pandas as pd
import pytest

from app.analysis.agent_loop import (
    AgentToolRuntime,
    LoopErrorType,
    _validate_safe_dataset_sql,
    canonical_action_hash,
    classify_tool_error,
)
from app.analysis.services import PlannedAnalysis
from app.api.v1.analysis import _resolve_agent_mode
from app.core.settings import Settings
from app.evaluation.agent_loop import AgentLoopBenchmarkOutcome, evaluate_agent_loop_benchmark
from app.storage.dataset_store import DatasetStoreRepository


def test_loop_engineering_is_the_default_analysis_policy() -> None:
    settings = Settings()

    assert settings.agent_loop_enabled is True
    assert settings.agent_loop_default_mode == "loop"


def test_safe_loop_sql_allows_scoped_select_and_rejects_external_or_write_sql() -> None:
    _validate_safe_dataset_sql(
        'WITH totals AS (SELECT "区域", SUM("销售额") AS total FROM dataset GROUP BY "区域") SELECT * FROM totals'
    )

    for sql in (
        "DELETE FROM dataset",
        "SELECT * FROM read_csv_auto('secret.csv')",
        "SELECT * FROM system.information_schema.tables",
        "SELECT * FROM dataset CROSS JOIN dataset AS other",
    ):
        with pytest.raises(ValueError):
            _validate_safe_dataset_sql(sql)


def test_safe_loop_sql_rejects_prepared_dataset_self_join() -> None:
    sql = (
        "SELECT c.customer_state, SUM(p.total_payment) FROM "
        "(SELECT order_id, SUM(payment_value) AS total_payment FROM dataset GROUP BY order_id) p "
        "JOIN dataset o ON o.order_id = p.order_id "
        "JOIN dataset c ON c.customer_id = o.customer_id "
        "GROUP BY c.customer_state"
    )

    with pytest.raises(ValueError, match="scan the prepared dataset exactly once"):
        _validate_safe_dataset_sql(sql)


def test_loop_action_hash_is_canonical_and_error_classification_is_specific() -> None:
    assert canonical_action_hash("aggregate_dataset", {"metric": "销售额", "group_by": "区域"}) == canonical_action_hash(
        "aggregate_dataset", {"group_by": "区域", "metric": "销售额"}
    )
    assert classify_tool_error("execute_safe_sql", ValueError("Binder error")) == LoopErrorType.SQL_ERROR
    assert classify_tool_error("execute_python_analysis", RuntimeError("NameError")) == LoopErrorType.PYTHON_ERROR
    assert classify_tool_error("execute_safe_sql", ValueError("forbidden statement")) == LoopErrorType.POLICY_ERROR


def test_agent_tool_runtime_cannot_open_another_users_dataset(tmp_path) -> None:
    owner = DatasetStoreRepository(str(tmp_path), user_id="owner")
    dataset = owner.create_dataset(name="private.csv", source_type="csv", source_metadata={})
    attacker = DatasetStoreRepository(str(tmp_path), user_id="attacker")

    with pytest.raises(RuntimeError, match="not found"):
        AgentToolRuntime(
            repository=attacker,
            job_id=uuid4(),
            dataset_id=dataset.id,
            allowed_dataset_ids=(dataset.id,),
            dataframe=cast(Any, None),
            question="steal data",
            profile=cast(Any, None),
            plan=cast(Any, None),
            planner_decision=None,
            python_executor=cast(Any, None),
        )


def test_agent_tool_runtime_hydrates_archived_evidence_only_on_demand(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="owner")
    dataset = repository.create_dataset(
        name="sales.csv", source_type="csv", source_metadata={}
    )
    calls = 0

    def load_result(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"rows": [{"total": 42}]}

    monkeypatch.setattr("app.analysis.agent_loop.load_evidence_result", load_result)
    runtime = AgentToolRuntime(
        repository=repository,
        job_id=uuid4(),
        dataset_id=dataset.id,
        allowed_dataset_ids=(dataset.id,),
        dataframe=pd.DataFrame([{"total": 1}]),
        question="total sales",
        profile=cast(Any, None),
        plan=PlannedAnalysis(
            route="sql",
            category_column=None,
            metric_column=None,
            time_column=None,
            steps=("sum",),
        ),
        planner_decision=None,
        python_executor=cast(Any, None),
        evidence=(
            {
                "evidence_id": "ev_1",
                "tool_result_artifact_id": str(uuid4()),
                "result": {"headline": "bounded summary"},
            },
        ),
    )

    assert calls == 0
    assert runtime._required_evidence("ev_1")["result"]["rows"][0]["total"] == 42
    assert runtime._required_evidence("ev_1")["result"]["rows"][0]["total"] == 42
    assert calls == 1


def test_loop_request_mode_obeys_deployment_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.v1.analysis.get_settings", lambda: Settings(agent_loop_enabled=False))
    with pytest.raises(ValueError, match="disabled"):
        _resolve_agent_mode("loop")
    assert _resolve_agent_mode("auto") == "legacy"

    monkeypatch.setattr(
        "app.api.v1.analysis.get_settings",
        lambda: Settings(agent_loop_enabled=True, agent_loop_default_mode="loop"),
    )
    assert _resolve_agent_mode("auto") == "loop"


def test_idempotent_action_artifact_keeps_first_successful_result(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="owner")
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    artifact_id = uuid4()

    repository.save_artifact(
        dataset_id=dataset.id,
        artifact_type="agent_loop_action",
        content={"result": {"rows": [{"total": 10}]}},
        artifact_id=artifact_id,
        if_absent=True,
    )
    repository.save_artifact(
        dataset_id=dataset.id,
        artifact_type="agent_loop_action",
        content={"result": {"rows": [{"total": 999}]}},
        artifact_id=artifact_id,
        if_absent=True,
    )

    assert repository.get_artifact(dataset.id, artifact_id)["content"]["result"]["rows"] == [
        {"total": 10}
    ]


def test_offline_loop_benchmark_enforces_release_thresholds() -> None:
    outcomes = tuple(
        AgentLoopBenchmarkOutcome(
            case_id=f"case-{index}",
            selected_tool="execute_safe_sql",
            expected_tools=frozenset({"execute_safe_sql", "execute_semantic_query"}),
            legal_call=index < 96,
            recoverable_error=index < 20,
            recovered=index < 18,
            tool_calls=2 if index < 20 else 1,
        )
        for index in range(100)
    )

    report = evaluate_agent_loop_benchmark(outcomes)

    assert report.legal_call_rate == 0.96
    assert report.repair_success_rate == 0.9
    assert report.duplicate_successful_actions == 0
    assert report.passed
