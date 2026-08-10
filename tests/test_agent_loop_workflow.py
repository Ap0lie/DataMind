from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from app.analysis.agent_loop import AgentToolRuntime, canonical_action_hash
from app.analysis.services import DatasetProfiler, PlannedAnalysis
from app.analysis.statistical_verifier import analysis_contract_gaps
from app.analysis.workflow import (
    AnalysisWorkflowRunner,
    _combined_loop_python_result,
    _mandatory_evidence_findings,
)
from app.core.settings import get_settings
from app.mcp.tool_schemas import ModelRouterResponse
from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisContractResponse,
    AnalysisFilterResponse,
    DatasetJoinConfig,
    PythonAnalysisResponse,
    SQLAnalysisResponse,
    StatisticalCheckResponse,
    StatisticalVerificationResponse,
)
from app.storage.dataset_store import DatasetStoreRepository
from tests.fakes import ScriptedPythonExecutor

pytestmark = pytest.mark.workflow


class RepairingLoopRouter:
    def __init__(self) -> None:
        self.loop_calls = 0
        self.loop_messages: list[list[dict[str, Any]]] = []

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
        agent = str((metadata or {}).get("agent") or "")
        if agent == "planner":
            content = '{"route":"sql","category_column":"区域","metric_column":"销售额","time_column":null,"steps":["按区域汇总销售额"]}'
        elif agent == "design_framework":
            content = '{"business_question":"区域销售额","candidate_dimensions":["区域"],"candidate_metrics":["销售额"],"likely_routes":["sql"],"initial_hypotheses":[],"risk_notes":[],"key_questions":[],"success_criteria":"可追溯"}'
        elif agent == "agent_loop":
            self.loop_calls += 1
            self.loop_messages.append(messages)
            if self.loop_calls == 1:
                return _tool_response("execute_safe_sql", '{"sql":"SELECT \\\"不存在\\\" FROM dataset"}')
            if self.loop_calls == 2:
                return _tool_response("execute_safe_sql", '{"sql":"SELECT \\\"区域\\\" AS category, SUM(\\\"销售额\\\") AS total_sales FROM dataset GROUP BY \\\"区域\\\" ORDER BY total_sales DESC"}')
            return ModelRouterResponse(provider="mock", model="loop", content='{"action":"finish","reason":"聚合证据充分"}', finish_reason="stop", token_usage={"total_tokens": 5})
        elif agent == "review":
            content = '{"issues":[]}'
        else:
            content = "{}"
        return ModelRouterResponse(provider="mock", model="loop", content=content, finish_reason="stop", token_usage={"total_tokens": 5})


def _tool_response(name: str, arguments: str) -> ModelRouterResponse:
    return ModelRouterResponse(
        provider="mock",
        model="loop",
        content=None,
        tool_calls=(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            },
        ),
        finish_reason="tool_calls",
        token_usage={"total_tokens": 5},
    )


def test_monthly_sql_kpis_are_mandatory_without_python_or_evidence_ids() -> None:
    findings = _mandatory_evidence_findings(
        {
            "question": "计算 GMV、订单数、客单价、准时率并分析月度趋势。",
            "sql_result": SQLAnalysisResponse(
                sql="SELECT period, total_price, order_count, average_order_value FROM dataset",
                rows=(
                    {
                        "period": "2026-01",
                        "total_price": 100.0,
                        "order_count": 10,
                        "average_order_value": 10.0,
                    },
                    {
                        "period": "2026-02",
                        "total_price": 200.0,
                        "order_count": 20,
                        "average_order_value": 10.0,
                    },
                ),
                explanation="月度指标。",
            ),
        }
    )

    report_text = " ".join(finding.content for finding in findings)
    assert "price 总额=300.00" in report_text
    assert "订单数=30" in report_text
    assert "客单价=10.00" in report_text
    assert any(finding.title == "月度趋势与异常期间" for finding in findings)


class MultiToolDecisionRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        self.loop_messages.append(kwargs["messages"])
        if self.loop_calls == 1:
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=None,
                tool_calls=(
                    {
                        "id": "inspect",
                        "type": "function",
                        "function": {
                            "name": "inspect_analysis_context",
                            "arguments": "{}",
                        },
                    },
                    {
                        "id": "sql",
                        "type": "function",
                        "function": {
                            "name": "execute_safe_sql",
                            "arguments": (
                                '{"sql":"SELECT \\"区域\\" AS category, '
                                'SUM(\\"销售额\\") AS total_sales FROM dataset '
                                'GROUP BY \\"区域\\""}'
                            ),
                        },
                    },
                ),
                finish_reason="tool_calls",
                token_usage={"total_tokens": 5},
            )
        if self.loop_calls == 2:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":"SELECT \\"区域\\" AS category, SUM(\\"销售额\\") '
                'AS total_sales FROM dataset GROUP BY \\"区域\\" '
                'ORDER BY total_sales DESC"}',
            )
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content='{"action":"finish","reason":"证据充分"}',
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_loop_normalizes_parallel_tool_suggestions_to_one_safe_call(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"区域": "华东", "销售额": 120}, {"区域": "华南", "销售额": 80}],
    )
    router = MultiToolDecisionRouter()
    events: list[dict[str, Any]] = []

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="比较各区域销售额",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    normalized = next(
        event for event in events if event.get("event_type") == "normalized_decision"
    )
    assert normalized["payload"]["selected_tool"] == "inspect_analysis_context"
    assert normalized["payload"]["deferred_tools"] == ["execute_safe_sql"]
    assert not any(event.get("event_type") == "invalid_decision" for event in events)
    assert result.sql_result is not None
    assert {
        row["category"]: row["total_sales"] for row in result.sql_result.rows
    } == {"华东": 120, "华南": 80}
    assert result.python_result is None
    assert result.sql_source == "agent_loop"
    assert result.python_source == "not_run"
    assert result.loop_summary["analysis_components"] == ("sql",)
    assert result.loop_terminal_reason in {"evidence_sufficient", "model_finished"}


class SqlThenPythonRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content=None,
            tool_calls=(
                {
                    "id": "sql",
                    "type": "function",
                    "function": {
                        "name": "execute_safe_sql",
                        "arguments": (
                            '{"sql":"SELECT \\"区域\\" AS category, '
                            'SUM(\\"销售额\\") AS total_sales FROM dataset '
                            'GROUP BY \\"区域\\""}'
                        ),
                    },
                },
                {
                    "id": "python",
                    "type": "function",
                    "function": {
                        "name": "execute_python_analysis",
                        "arguments": json.dumps(
                            {
                                "code": "def analyze(df):\n    return {}",
                                "evidence_id": "ev_1",
                            }
                        ),
                    },
                },
            ),
            finish_reason="tool_calls",
            token_usage={"total_tokens": 5},
        )


def test_loop_executes_deferred_sql_then_python_with_sql_rows_as_input(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"区域": "华东", "销售额": 120}, {"区域": "华南", "销售额": 80}],
    )
    python_executor = ScriptedPythonExecutor(
        outcomes=(
            PythonAnalysisResponse(
                statistics={"spread": 40},
                insights=("区域汇总结果的极差为 40。",),
                charts=(),
            ),
        )
    )
    router = SqlThenPythonRouter()

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=python_executor,
    ).run(
        dataset_id=dataset.id,
        question="比较各区域销售额",
        agent_mode="loop",
    )

    assert router.loop_calls == 1
    assert result.sql_result is not None
    assert result.python_result is not None
    assert result.loop_summary["analysis_components"] == ("sql", "python")
    assert list(python_executor.calls[0][1].columns) == ["category", "total_sales"]
    assert len(python_executor.calls[0][1]) == 2
    assert any(
        "极差为 40" in finding.content for finding in result.final_insights
    ), [finding.content for finding in result.final_insights]


class PythonOnlyLoopRouter:
    def __init__(self) -> None:
        self.loop_calls = 0

    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        agent = str((kwargs.get("metadata") or {}).get("agent") or "")
        if agent == "planner":
            content = '{"route":"python","category_column":null,"metric_column":null,"time_column":null,"steps":["描述分布与异常"]}'
        elif agent == "design_framework":
            content = '{"business_question":"描述数据特征","candidate_dimensions":[],"candidate_metrics":[],"likely_routes":["python"],"initial_hypotheses":[],"risk_notes":[],"key_questions":[],"success_criteria":"给出可追溯描述"}'
        elif agent == "agent_loop":
            self.loop_calls += 1
            if self.loop_calls == 1:
                return _tool_response(
                    "execute_python_analysis",
                    '{"code":"def analyze(df):\\n    return {}"}',
                )
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content='{"action":"finish","reason":"分布证据充分"}',
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        elif agent == "review":
            content = '{"issues":[]}'
        else:
            content = "{}"
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content=content,
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_loop_keeps_python_only_execution_python_only(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="observations.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"score": 10, "comment": "稳定"}, {"score": 30, "comment": "异常"}],
    )
    python_executor = ScriptedPythonExecutor(
        outcomes=(
            PythonAnalysisResponse(
                statistics={"rows": 2, "score_summary": {"mean": 20}},
                insights=("score 均值为 20。",),
                charts=(),
            ),
        )
    )

    result = AnalysisWorkflowRunner(
        repository,
        model_router=PythonOnlyLoopRouter(),
        python_executor=python_executor,
    ).run(
        dataset_id=dataset.id,
        question="描述数值分布和文本特征，不做聚合排名",
        agent_mode="loop",
    )

    assert result.sql_result is None
    assert result.python_result is not None
    assert result.sql_source == "not_run"
    assert result.python_source == "agent_loop"
    assert result.loop_summary["analysis_components"] == ("python",)
    assert len(python_executor.calls) == 1


class PythonFallbackRouter(PythonOnlyLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        if str((kwargs.get("metadata") or {}).get("agent") or "") == "agent_loop":
            raise RuntimeError("loop provider unavailable")
        return super().complete(**kwargs)


def test_python_route_fallback_does_not_execute_sql(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="observations.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"score": 10}, {"score": 30}],
    )

    result = AnalysisWorkflowRunner(
        repository,
        model_router=PythonFallbackRouter(),
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="描述数值分布和异常，不做聚合排名",
        agent_mode="loop",
    )

    assert result.sql_result is None
    assert result.python_result is not None
    assert result.sql_source == "not_run"
    assert result.python_source == "legacy_fallback"
    assert result.loop_summary["analysis_components"] == ("python",)


def test_python_evidence_carries_server_verified_filter_and_dimension_scope(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="scoped-sales.csv",
        source_type="csv",
        source_metadata={},
    )
    records = [
        {"region": "East", "status": "active", "sales": 10},
        {"region": "West", "status": "inactive", "sales": 90},
    ]
    repository.append_raw_records(dataset_id=dataset.id, records=records)
    dataframe = pd.DataFrame.from_records(records)
    profile = DatasetProfiler().profile(dataset_id=dataset.id, records=records)
    applied_filter = AnalysisFilterResponse(
        column="status",
        operator="=",
        value="active",
    )
    aggregation = AnalysisAggregationResponse(
        operation="avg",
        column="sales",
        alias="avg_sales",
    )
    plan = PlannedAnalysis(
        route="python",
        category_column="region",
        metric_column="sales",
        time_column=None,
        steps=("compare active sales by region",),
        aggregations=(aggregation,),
        filters=(applied_filter,),
        requested_dimensions=("region",),
    )
    executor = ScriptedPythonExecutor(
        outcomes=(
            PythonAnalysisResponse(
                statistics={"avg_sales": 10},
                insights=("East active average is 10.",),
                charts=(),
            ),
        )
    )
    runtime = AgentToolRuntime(
        repository=repository,
        job_id=uuid4(),
        dataset_id=dataset.id,
        allowed_dataset_ids=(dataset.id,),
        dataframe=dataframe,
        question="Compare average sales by region where status=active",
        profile=profile,
        plan=plan,
        planner_decision=None,
        python_executor=executor,
    )

    execution = runtime.execute(
        "execute_python_analysis",
        {
            "code": (
                "def analyze(df):\n"
                "    grouped = df.groupby('region')['sales'].mean()\n"
                "    return {}"
            )
        },
    )

    assert execution.succeeded
    python_result = PythonAnalysisResponse.model_validate(execution.result["python_result"])
    assert python_result.execution_context is not None
    assert python_result.execution_context.source_row_count == 2
    assert python_result.execution_context.input_row_count == 1
    assert python_result.execution_context.applied_filters == (applied_filter,)
    assert python_result.execution_context.referenced_columns == ("region", "sales")
    assert len(executor.calls[0][1]) == 1
    contract = AnalysisContractResponse(
        objective="Compare average sales by region where status=active",
        population="active rows",
        analysis_type="comparison",
        metric="sales",
        dimensions=("region",),
        aggregations=(aggregation,),
        filters=(applied_filter,),
        grain=("region",),
        method="grouped average",
    )
    assert not any(
        analysis_contract_gaps(
            contract=contract,
            sql_result=None,
            python_result=python_result,
        ).values()
    )


def test_safe_sql_action_hash_uses_ast_semantics() -> None:
    first = canonical_action_hash(
        "execute_safe_sql",
        {"sql": 'SELECT SUM("销售额") AS total_sales FROM dataset'},
    )
    equivalent = canonical_action_hash(
        "execute_safe_sql",
        {"sql": ' select sum(d."销售额") result FROM dataset AS d '},
    )

    assert first == equivalent


def test_loop_finalize_combines_all_python_evidence() -> None:
    first = PythonAnalysisResponse(
        statistics={"rows": 2, "mean_score": 20},
        insights=("均值为 20。",),
        charts=(),
    )
    second = PythonAnalysisResponse(
        statistics={"outlier_count": 1},
        insights=("发现 1 个异常值。",),
        charts=(),
    )

    combined = _combined_loop_python_result([("ev_1", first), ("ev_2", second)])

    assert combined is not None
    assert combined.statistics["mean_score"] == 20
    assert combined.statistics["outlier_count"] == 1
    assert [item["evidence_id"] for item in combined.statistics["agent_loop_evidence"]] == [
        "ev_1",
        "ev_2",
    ]
    assert combined.insights == ("均值为 20。", "发现 1 个异常值。")


class TwoPythonCallsRouter(PythonOnlyLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        if str((kwargs.get("metadata") or {}).get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content=None,
            tool_calls=(
                {
                    "id": "python-1",
                    "type": "function",
                    "function": {
                        "name": "execute_python_analysis",
                        "arguments": json.dumps(
                            {"code": "def analyze(df):\n    return {}"}
                        ),
                    },
                },
                {
                    "id": "python-2",
                    "type": "function",
                    "function": {
                        "name": "execute_python_analysis",
                        "arguments": json.dumps(
                            {"code": "def analyze(df):\n    return {'second': True}"}
                        ),
                    },
                },
            ),
            finish_reason="tool_calls",
            token_usage={"total_tokens": 5},
        )


def test_loop_combines_two_real_python_tool_results(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="observations.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"score": 10}, {"score": 30}],
    )
    executor = ScriptedPythonExecutor(
        outcomes=(
            PythonAnalysisResponse(
                statistics={"mean_score": 20},
                insights=("均值为 20。",),
                charts=(),
            ),
            PythonAnalysisResponse(
                statistics={"outlier_count": 1},
                insights=("发现 1 个异常值。",),
                charts=(),
            ),
        )
    )

    result = AnalysisWorkflowRunner(
        repository,
        model_router=TwoPythonCallsRouter(),
        python_executor=executor,
    ).run(
        dataset_id=dataset.id,
        question="描述数值分布和异常，不做聚合排名",
        agent_mode="loop",
    )

    assert len(executor.calls) == 2
    assert result.python_result is not None
    assert result.python_result.statistics["mean_score"] == 20
    assert result.python_result.statistics["outlier_count"] == 1
    assert [
        item["evidence_id"]
        for item in result.python_result.statistics["agent_loop_evidence"]
    ] == ["ev_1", "ev_2"]
    assert result.loop_summary["analysis_components"] == ("python",)


class DuplicateDecisionLoopRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        self.loop_messages.append(kwargs["messages"])
        if self.loop_calls == 1:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":"SELECT SUM(\\"销售额\\") AS total FROM dataset"}',
            )
        if self.loop_calls == 2:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":" select sum(d.\\"销售额\\") amount FROM dataset AS d "}',
            )
        payload = json.loads(kwargs["messages"][1]["content"])
        assert payload["repair"]["error_type"] == "duplicate_action"
        assert payload["repair"]["required_tool"] == "execute_safe_sql"
        assert payload["repair"]["contract_gaps"]["dimensions"] == ["区域"]
        assert [item["function"]["name"] for item in kwargs["tools"]] == [
            "execute_safe_sql"
        ]
        assert kwargs["tool_choice"] == "required"
        return _tool_response(
            "execute_safe_sql",
            '{"sql":"SELECT \\"区域\\", SUM(\\"销售额\\") AS total '
            'FROM dataset GROUP BY \\"区域\\""}',
        )


def test_loop_reports_duplicate_decision_and_finishes_when_contract_is_covered(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"区域": "华东", "销售额": 120}, {"区域": "华南", "销售额": 80}],
    )
    events: list[dict[str, Any]] = []
    router = DuplicateDecisionLoopRouter()

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="比较各区域销售额",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert router.loop_calls == 3
    assert result.loop_summary["tool_calls"] == 2
    assert result.loop_terminal_reason == "evidence_sufficient"
    duplicate = next(event for event in events if event.get("event_type") == "duplicate_action")
    assert duplicate["payload"]["result_hash"]


class LargeSqlResultRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        return _tool_response(
            "execute_safe_sql",
            '{"sql":"SELECT \\"区域\\", SUM(\\"销售额\\") AS total_sales '
            'FROM dataset GROUP BY \\"区域\\""}',
        )


class LargeSqlThenChartRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content=None,
            tool_calls=(
                {
                    "id": "sql",
                    "type": "function",
                    "function": {
                        "name": "execute_safe_sql",
                        "arguments": (
                            '{"sql":"SELECT \\"区域\\", SUM(\\"销售额\\") '
                            'AS total_sales FROM dataset GROUP BY \\"区域\\""}'
                        ),
                    },
                },
                {
                    "id": "chart",
                    "type": "function",
                    "function": {
                        "name": "generate_chart",
                        "arguments": json.dumps(
                            {
                                "evidence_id": "ev_1",
                                "chart_type": "bar",
                                "x": "区域",
                                "y": "total_sales",
                                "title": "区域销售额",
                            },
                            ensure_ascii=False,
                        ),
                    },
                },
            ),
            finish_reason="tool_calls",
            token_usage={"total_tokens": 5},
        )


def test_large_sql_artifact_keeps_contract_metadata_and_converges(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="large-sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"区域": f"区域-{index:04d}-" + "长名称" * 20, "销售额": index + 1}
            for index in range(600)
        ],
    )
    events: list[dict[str, Any]] = []
    router = LargeSqlResultRouter()

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="按区域汇总销售额",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert router.loop_calls == 1
    assert result.loop_terminal_reason == "evidence_sufficient"
    observation = next(
        event for event in events if event.get("event_type") == "observation"
    )
    assert observation["payload"]["artifact_id"]
    assert result.sql_source == "agent_loop"
    assert result.python_result is None
    assert result.loop_summary["analysis_components"] == ("sql",)
    verification = next(
        event for event in events if event.get("event_type") == "verification"
    )
    assert verification["payload"]["contract_covered"] is True


def test_artifact_backed_sql_can_feed_deferred_chart_without_repeating_sql(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="large-sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"区域": f"区域-{index:04d}-" + "长名称" * 20, "销售额": index + 1}
            for index in range(600)
        ],
    )
    events: list[dict[str, Any]] = []
    router = LargeSqlThenChartRouter()

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="按区域汇总销售额并生成图表",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert router.loop_calls == 1
    assert result.loop_summary["tool_calls"] == 2
    assert result.loop_summary["analysis_components"] == ("sql",)
    assert [
        event.get("tool_name")
        for event in events
        if event.get("event_type") == "observation"
    ] == ["execute_safe_sql", "generate_chart"]
    assert any(
        chart.chart_type == "bar"
        and chart.spec.get("x") == "区域"
        and chart.spec.get("y") == "total_sales"
        for chart in (result.structured_report.charts if result.structured_report else ())
    )


def test_loop_repairs_safe_sql_error_and_finishes_without_iterative_rounds(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="中文销售.txt", source_type="txt", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"区域": "华东", "销售额": 120}, {"区域": "华南", "销售额": 80}],
    )
    events: list[dict[str, Any]] = []

    router = RepairingLoopRouter()
    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="哪个区域销售额最高？",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert result.agent_mode == "loop"
    assert result.rounds == ()
    assert result.sql_result is not None
    assert result.sql_result.rows[0]["category"] == "华东"
    assert result.loop_summary["tool_calls"] == 2, (result.loop_summary, events)
    assert result.loop_summary["failed_tools"] == 1
    assert result.loop_terminal_reason == "evidence_sufficient"
    assert any(event.get("event_type") == "repair" for event in events)
    assert any(event.get("event_type") == "loop_finalize" for event in events)
    assert any(event.get("event_type") == "report_decision" for event in events)
    assert any(event.get("event_type") == "report_validation" for event in events)
    assert any(event.get("event_type") == "report_commit" for event in events)
    usage_event = next(event for event in events if event.get("event_type") == "model_usage")
    assert usage_event["token_usage"]["total_tokens"] > 0
    assert result.report_revision_count <= 2
    assert result.loop_summary["report"]["terminal_reason"] == "validated"
    loop_payload = json.loads(router.loop_messages[0][1]["content"])
    assert set(loop_payload["columns"][0]) == {
        "name",
        "dtype",
        "is_numeric",
        "missing_count",
        "distinct_count",
    }


class PolicyRepairingLoopRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        if self.loop_calls == 1:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":"SELECT * FROM read_csv_auto(\\"secret.csv\\")"}',
            )
        if self.loop_calls == 2:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":"SELECT SUM(\\"销售额\\") AS total_sales FROM dataset"}',
            )
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content='{"action":"finish","reason":"合法聚合证据充分"}',
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_loop_repairs_one_blocked_policy_call_without_relaxing_policy(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"销售额": 120}, {"销售额": 80}],
    )
    events: list[dict[str, Any]] = []

    result = AnalysisWorkflowRunner(
        repository,
        model_router=PolicyRepairingLoopRouter(),
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="总销售额是多少？",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert result.loop_terminal_reason == "model_finished"
    assert result.loop_summary["failed_tools"] == 1
    assert result.sql_result is not None
    assert result.sql_result.rows[0]["total_sales"] == 200
    assert any(event.get("event_type") == "repair" for event in events)


class StatisticalReplanningRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        agent = str(metadata.get("agent") or "")
        if agent == "planner":
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"route":"sql","category_column":null,'
                    '"metric_column":"order_total","time_column":null,'
                    '"steps":["汇总订单金额"]}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent == "design_framework":
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"business_question":"订单总额","candidate_dimensions":[],'
                    '"candidate_metrics":["order_total"],"likely_routes":["sql"],'
                    '"initial_hypotheses":[],"risk_notes":["检查 Join 粒度"],'
                    '"key_questions":[],"success_criteria":"原生粒度证据"}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent == "agent_loop":
            self.loop_calls += 1
            self.loop_messages.append(kwargs["messages"])
            if self.loop_calls == 1:
                return _tool_response(
                    "execute_safe_sql",
                    '{"sql":"SELECT SUM(order_total) AS total FROM dataset"}',
                )
            if self.loop_calls == 2:
                return _tool_response(
                    "aggregate_source_dataset",
                    '{"dataset":"orders.csv","metric":"order_total","aggregation":"sum"}',
                )
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content='{"action":"finish","reason":"已补充源表粒度证据"}',
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent == "review":
            content = '{"issues":[]}'
        elif agent == "report_decide":
            content = '{"strategy":"template","reason":"使用已验证证据"}'
        else:
            content = "{}"
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content=content,
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_statistical_failure_replans_and_collects_native_grain_evidence(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    orders = repository.create_dataset(
        name="orders.csv", source_type="csv", source_metadata={}
    )
    items = repository.create_dataset(
        name="items.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[
            {"order_id": "o1", "order_total": 100.0},
            {"order_id": "o2", "order_total": 200.0},
        ],
    )
    repository.append_raw_records(
        dataset_id=items.id,
        records=[
            {"order_id": "o1", "sku": "a"},
            {"order_id": "o1", "sku": "b"},
            {"order_id": "o2", "sku": "c"},
        ],
    )
    router = StatisticalReplanningRouter()
    events: list[dict[str, Any]] = []

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=orders.id,
        additional_dataset_ids=(items.id,),
        join_plan=(
            DatasetJoinConfig(
                left_dataset_id=orders.id,
                right_dataset_id=items.id,
                left_column="order_id",
                right_column="order_id",
                join_type="left",
            ),
        ),
        question="计算订单总额，并避免 Join 后重复累计。",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert router.loop_calls == 2
    assert result.statistical_verification is not None
    assert result.statistical_verification.status != "failed"
    join_check = next(
        check
        for check in result.statistical_verification.checks
        if check.code == "join_grain"
    )
    assert join_check.details["native_grain_evidence"] is True
    preflight_events = [
        event
        for event in events
        if event.get("event_type") == "statistical_preflight"
    ]
    assert [event["status"] for event in preflight_events] == [
        "failed",
        "completed",
    ]
    assert any(
        event.get("event_type") == "adversarial_repair" for event in events
    )
    repair_messages = router.loop_messages[1]
    repair_payload = next(
        message for message in repair_messages if message["role"] == "user"
    )["content"]
    assert '"required_tool": "aggregate_source_dataset"' in repair_payload
    assert '"dataset": "orders.csv"' in repair_payload


def test_source_grain_guard_does_not_emit_unfiltered_cross_table_total(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    payments = repository.create_dataset(
        name="order_payments", source_type="csv", source_metadata={}
    )
    orders = repository.create_dataset(
        name="orders", source_type="csv", source_metadata={}
    )
    customers = repository.create_dataset(
        name="customers", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=payments.id,
        records=[{"order_id": "o1", "payment_value": 100.0}],
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[{"order_id": "o1", "order_status": "delivered"}],
    )
    repository.append_raw_records(
        dataset_id=customers.id,
        records=[{"customer_id": "c1", "customer_state": "SP"}],
    )
    frame = pd.DataFrame(
        {
            "order_payments__payment_value": [100.0],
            "orders__order_status": ["delivered"],
            "customers__customer_state": ["SP"],
        }
    )
    runtime = AgentToolRuntime(
        repository=repository,
        job_id=uuid4(),
        dataset_id=payments.id,
        allowed_dataset_ids=(payments.id, orders.id, customers.id),
        dataframe=frame,
        question=(
            "仅使用 customers、orders、order_payments，按 customer_state "
            "统计 order_status=delivered 的 payment_value 支付总额"
        ),
        profile=DatasetProfiler().profile(
            dataset_id=payments.id,
            records=frame.to_dict(orient="records"),
        ),
        plan=PlannedAnalysis(
            route="sql",
            category_column="customers__customer_state",
            metric_column="order_payments__payment_value",
            time_column=None,
            steps=("aggregate",),
            filters=(
                AnalysisFilterResponse(
                    column="orders__order_status", value="delivered"
                ),
            ),
            requested_dimensions=("customers__customer_state",),
        ),
        planner_decision=None,
        python_executor=ScriptedPythonExecutor(),
    )

    assert runtime.required_source_aggregates() == ()
    execution = runtime.execute(
        "aggregate_source_dataset",
        {
            "dataset": "order_payments",
            "metric": "payment_value",
            "aggregation": "sum",
        },
    )
    assert execution.succeeded is False
    assert "cannot satisfy the contract population/grain" in str(execution.error)


def test_persistent_statistical_failure_blocks_report_commit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    orders = repository.create_dataset(
        name="orders.csv", source_type="csv", source_metadata={}
    )
    items = repository.create_dataset(
        name="items.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[{"order_id": "o1", "order_total": 100.0}],
    )
    repository.append_raw_records(
        dataset_id=items.id,
        records=[{"order_id": "o1", "sku": "a"}, {"order_id": "o1", "sku": "b"}],
    )
    failed = StatisticalVerificationResponse(
        status="failed",
        summary="统计审查失败：客户州维度缺失。",
        requires_replan=True,
        checks=(
            StatisticalCheckResponse(
                code="request_coverage",
                status="failed",
                severity="error",
                message="缺少客户州维度。",
            ),
        ),
    )
    monkeypatch.setattr(
        "app.analysis.workflow.verify_statistical_analysis",
        lambda **_kwargs: failed,
    )

    with pytest.raises(RuntimeError, match="report commit was blocked"):
        AnalysisWorkflowRunner(
            repository,
            model_router=StatisticalReplanningRouter(),
            python_executor=ScriptedPythonExecutor(),
        ).run(
            dataset_id=orders.id,
            additional_dataset_ids=(items.id,),
            join_plan=(
                DatasetJoinConfig(
                    left_dataset_id=orders.id,
                    right_dataset_id=items.id,
                    left_column="order_id",
                    right_column="order_id",
                    join_type="left",
                ),
            ),
            question="按客户州分析订单总额。",
            agent_mode="loop",
        )

    assert repository.list_reports(orders.id) == ()


class MultiSqlEvidenceLoopRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        if self.loop_calls == 1:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":"SELECT SUM(\\"销售额\\") AS total_sales FROM dataset"}',
            )
        if self.loop_calls == 2:
            return _tool_response(
                "execute_safe_sql",
                '{"sql":"SELECT \\"区域\\", SUM(\\"销售额\\") AS regional_sales FROM dataset GROUP BY \\"区域\\" ORDER BY regional_sales DESC"}',
            )
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content='{"action":"finish","reason":"多项证据充分"}',
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_loop_finalize_preserves_all_successful_sql_evidence(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"区域": "华东", "销售额": 120},
            {"区域": "华南", "销售额": 80},
        ],
    )

    result = AnalysisWorkflowRunner(
        repository,
        model_router=MultiSqlEvidenceLoopRouter(),
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="总销售额及区域销售额分别是多少？",
        agent_mode="loop",
    )

    assert result.sql_result is not None
    assert "-- ev_1" in result.sql_result.sql
    assert "-- ev_2" in result.sql_result.sql
    assert {row["query_index"] for row in result.sql_result.rows} == {1, 2}
    assert any(row.get("total_sales") == 200 for row in result.sql_result.rows)
    assert any(row.get("regional_sales") == 120 for row in result.sql_result.rows)


class StructuredJsonFallbackLoopRouter(RepairingLoopRouter):
    def __init__(self) -> None:
        super().__init__()
        self.structured_calls = 0

    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        assert metadata.get("allow_provider_fallback") is False
        if kwargs.get("tools"):
            raise RuntimeError("Kimi API error 400: unsupported tool schema")
        self.structured_calls += 1
        if self.structured_calls == 1:
            return ModelRouterResponse(
                provider="kimi",
                model="kimi-k2.6",
                content=(
                    '{"action":"tool_call","tool_name":"execute_safe_sql",'
                    '"arguments":{"sql":"SELECT SUM(\\"销售额\\") AS total_sales '
                    'FROM dataset"},"reason":"计算总额"}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        return ModelRouterResponse(
            provider="kimi",
            model="kimi-k2.6",
            content='{"action":"finish","reason":"证据充分"}',
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_loop_uses_structured_json_when_kimi_native_tools_return_400(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"销售额": 120}, {"销售额": 80}],
    )

    result = AnalysisWorkflowRunner(
        repository,
        model_router=StructuredJsonFallbackLoopRouter(),
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="总销售额是多少？",
        agent_mode="loop",
    )

    assert result.loop_terminal_reason == "model_finished"
    assert result.loop_summary["tool_calls"] == 1
    assert result.sql_result is not None
    assert result.sql_result.rows[0]["total_sales"] == 200


class MissingArgumentStructuredLoopRouter(RepairingLoopRouter):
    def __init__(self) -> None:
        super().__init__()
        self.structured_calls = 0

    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        if str(metadata.get("agent") or "") != "agent_loop":
            return super().complete(**kwargs)
        if kwargs.get("tools"):
            raise RuntimeError("Kimi API error 400: unsupported tool schema")
        self.structured_calls += 1
        if self.structured_calls > 1:
            return ModelRouterResponse(
                provider="kimi",
                model="kimi-k2.6",
                content='{"action":"finish","reason":"证据充分"}',
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        return ModelRouterResponse(
            provider="kimi",
            model="kimi-k2.6",
            content=(
                '{"action":"tool_call","tool_name":"execute_safe_sql",'
                '"arguments":{},"reason":"计算总额"}'
            ),
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_structured_adapter_rejects_missing_required_arguments_before_execution(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="sales.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"销售额": 120}, {"销售额": 80}],
    )
    router = MissingArgumentStructuredLoopRouter()
    events: list[dict[str, Any]] = []

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="总销售额是多少？",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert router.structured_calls >= 1
    assert result.loop_terminal_reason in {"evidence_sufficient", "model_finished"}
    assert result.loop_summary["tool_calls"] == 1
    assert result.loop_summary["failed_tools"] == 0
    assert result.sql_result is not None
    assert 200 in result.sql_result.rows[0].values()
    assert any(
        event.get("event_type") == "decision"
        and "deterministic SQL" in event.get("message", "")
        for event in events
    )
    snapshot = next(
        event for event in events if event.get("event_type") == "loop_finalize"
    )["payload"]["analysis_snapshot"]
    assert {"plan", "contract", "sql", "verification", "evidence"} <= set(snapshot)
    assert "rows" not in snapshot["sql"]


class AggregateContractOnPythonRouteRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        agent = str(metadata.get("agent") or "")
        if agent == "planner":
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"route":"python","category_column":null,'
                    '"metric_column":"price","time_column":null,'
                    '"steps":["汇总指标"]}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent != "agent_loop":
            return super().complete(**kwargs)
        self.loop_calls += 1
        if self.loop_calls > 1:
            raise AssertionError("aggregate contract repair must not depend on another model call")
        return _tool_response("profile_dataset", "{}")


def test_aggregate_contract_uses_deterministic_sql_even_on_python_route(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="orders.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"order_id": "o1", "price": 120.0},
            {"order_id": "o1", "price": 80.0},
            {"order_id": "o2", "price": 50.0},
        ],
    )
    router = AggregateContractOnPythonRouteRouter()
    events: list[dict[str, Any]] = []

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="以 price 之和计算总额，并以唯一 order_id 计算订单数。",
        agent_mode="loop",
        node_event_callback=events.append,
    )

    assert router.loop_calls == 1
    assert result.sql_result is not None
    assert "SUM" in result.sql_result.sql
    assert "COUNT(DISTINCT" in result.sql_result.sql
    assert any(row.get("total_price") == 250.0 for row in result.sql_result.rows)
    assert any(row.get("order_count") == 2 for row in result.sql_result.rows)
    assert any(
        event.get("event_type") == "decision"
        and "deterministic SQL" in event.get("message", "")
        for event in events
    )


class SourceAggregateFallbackRouter(RepairingLoopRouter):
    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        agent = str(metadata.get("agent") or "")
        if agent == "report_execute":
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"executive_summary":"模型生成了格式正确但遗漏多表证据的报告。",'
                    '"analysis_context":"仅提供通用描述。","key_findings":[],'
                    '"chart_explanations":[],"data_gaps":[],"validation_issues":[],'
                    '"recommended_next_steps":[]}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent != "agent_loop":
            return super().complete(**kwargs)
        return ModelRouterResponse(
            provider="mock",
            model="loop",
            content='{"action":"fallback","reason":"交由确定性恢复路径完成"}',
            finish_reason="stop",
            token_usage={"total_tokens": 5},
        )


def test_loop_preserves_requested_fact_table_totals_at_native_grain(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    items = repository.create_dataset(
        name="order_items", source_type="csv", source_metadata={}
    )
    payments = repository.create_dataset(
        name="order_payments", source_type="csv", source_metadata={}
    )
    orders = repository.create_dataset(
        name="orders", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=items.id,
        records=[
            {"order_id": "o1", "price": 100.0, "freight_value": 10.0},
            {"order_id": "o1", "price": 50.0, "freight_value": 5.0},
            {"order_id": "o2", "price": 200.0, "freight_value": 20.0},
        ],
    )
    repository.append_raw_records(
        dataset_id=payments.id,
        records=[
            {"order_id": "o1", "payment_value": 100.0},
            {"order_id": "o1", "payment_value": 50.0},
            {"order_id": "o2", "payment_value": 180.0},
        ],
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[
            {"order_id": "o1", "status": "delivered"},
            {"order_id": "o2", "status": "delivered"},
        ],
    )

    result = AnalysisWorkflowRunner(
        repository,
        model_router=SourceAggregateFallbackRouter(),
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=items.id,
        additional_dataset_ids=(payments.id, orders.id),
        question=(
            "说明 orders、order_items、order_payments 的关系和事实表粒度；"
            "分别计算 order_items 的商品收入和运费、order_payments 的支付总额，"
            "不得把两个一对多表直接连接后重复累计，并说明防重复方法。"
        ),
        agent_mode="loop",
    )

    assert result.sql_result is not None
    rows = result.sql_result.rows
    assert any(row.get("sum_price") == 350.0 for row in rows)
    assert any(row.get("sum_freight_value") == 35.0 for row in rows)
    assert any(row.get("sum_payment_value") == 330.0 for row in rows)
    assert 'FROM "order_items"' in result.sql_result.sql
    assert 'FROM "order_payments"' in result.sql_result.sql
    assert result.loop_summary["source_aggregate_guards"] == 3
    assert result.loop_summary["source_relationship_guards"] == 1
    assert result.loop_summary["source_relationship_risk_count"] >= 3
    # Relationship evidence is rendered in the inspected left-to-right direction:
    # each fact row belongs to one order, so both fact-table edges are N:1.
    assert "order_items.order_id → orders.order_id 为 N:1" in result.report_markdown
    assert "order_payments.order_id → orders.order_id 为 N:1" in result.report_markdown
    assert "直接逐行连接会形成多对多乘积" in result.report_markdown
    assert "预聚合" in result.report_markdown
    assert "order_items.price 的 SUM=350.00" in result.report_markdown
    assert "order_items.freight_value 的 SUM=35.00" in result.report_markdown
    assert "order_payments.payment_value 的 SUM=330.00" in result.report_markdown
    assert "evidence_id:relationship_ev_1" in result.report_markdown
    assert "核心指标概览" not in result.report_markdown
    assert result.loop_terminal_reason == "model_requested_fallback"


class StructuredStageRepairRouter(RepairingLoopRouter):
    def __init__(self) -> None:
        super().__init__()
        self.repaired: set[str] = set()

    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        metadata = kwargs.get("metadata") or {}
        agent = str(metadata.get("agent") or "")
        repair = metadata.get("structured_repair") is True
        if agent == "planner":
            if not repair:
                return ModelRouterResponse(
                    provider="mock",
                    model="loop",
                    content="先分析字段，再输出计划。",
                    finish_reason="stop",
                    token_usage={"total_tokens": 5},
                )
            self.repaired.add(agent)
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"route":"sql","category_column":"区域",'
                    '"metric_column":"销售额","time_column":null,'
                    '"steps":["按区域汇总销售额"]}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent == "integrate":
            if not repair:
                return ModelRouterResponse(
                    provider="mock",
                    model="loop",
                    content="分析完成，但省略了 JSON。",
                    finish_reason="stop",
                    token_usage={"total_tokens": 5},
                )
            self.repaired.add(agent)
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"insights":[{"title":"区域收入",'
                    '"content":"华东销售额为120。","data_source":"ev_1",'
                    '"evidence":"ev_1","confidence":"high",'
                    '"business_impact":"识别重点区域",'
                    '"recommended_action":"复核华东增长"}]}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        if agent == "report_execute":
            if not repair:
                return ModelRouterResponse(
                    provider="mock",
                    model="loop",
                    content="简短报告",
                    finish_reason="stop",
                    token_usage={"total_tokens": 5},
                )
            self.repaired.add(agent)
            return ModelRouterResponse(
                provider="mock",
                model="loop",
                content=(
                    '{"executive_summary":"结构化修复后生成了完整且可追溯的区域销售额报告。",'
                    '"analysis_context":"仅使用已验证证据。","key_findings":[],'
                    '"chart_explanations":[],"data_gaps":[],"validation_issues":[],'
                    '"recommended_next_steps":["继续复核区域趋势"]}'
                ),
                finish_reason="stop",
                token_usage={"total_tokens": 5},
            )
        return super().complete(**kwargs)


def test_structured_stages_retry_once_and_keep_model_outputs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATAMIND_ANALYSIS_FAST_PATH_ENABLED", "false")
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(
        name="sales.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"区域": "华东", "销售额": 120}, {"区域": "华南", "销售额": 80}],
    )
    router = StructuredStageRepairRouter()

    result = AnalysisWorkflowRunner(
        repository,
        model_router=router,
        python_executor=ScriptedPythonExecutor(),
    ).run(
        dataset_id=dataset.id,
        question="哪个区域销售额最高？",
        agent_mode="loop",
    )

    assert router.repaired == {"planner", "integrate", "report_execute"}
    assert result.plan.route == "sql"
    assert any(item.title == "区域收入" for item in result.final_insights)
    assert "结构化修复后生成了完整且可追溯" in result.report_markdown
    planner_trace = next(item for item in result.workflow_trace if item.node == "planner")
    assert planner_trace.fallback is None
    assert planner_trace.error is None


def test_loop_event_metadata_is_persisted_without_raw_rows(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    job = repository.create_analysis_job(dataset_id=dataset.id, question="Analyze", agent_mode="loop")

    stored = repository.append_analysis_job_event(
        job.id,
        node="agent_loop",
        status="completed",
        message="Tool completed",
        event_type="tool_execution",
        iteration=2,
        tool_name="profile_dataset",
        payload={"arguments_hash": "abc", "result_summary": "2 rows"},
    )

    assert stored["event_type"] == "tool_execution"
    assert stored["iteration"] == 2
    assert stored["tool_name"] == "profile_dataset"
    assert stored["payload"] == {"arguments_hash": "abc", "result_summary": "2 rows"}
    assert repository.get_analysis_job(job.id).agent_mode == "loop"
