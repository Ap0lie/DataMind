from __future__ import annotations

import json
from typing import Any

import pytest

from app.analysis.workflow import AnalysisWorkflowRunner
from app.mcp.tool_schemas import ModelRouterResponse
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
    assert result.loop_terminal_reason == "model_finished"
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
    assert "orders.order_id → order_items.order_id 为 1:N" in result.report_markdown
    assert "orders.order_id → order_payments.order_id 为 1:N" in result.report_markdown
    assert "直接逐行连接会形成多对多乘积" in result.report_markdown
    assert "预聚合" in result.report_markdown
    assert "order_items.price 的 SUM=350.00" in result.report_markdown
    assert "order_items.freight_value 的 SUM=35.00" in result.report_markdown
    assert "order_payments.payment_value 的 SUM=330.00" in result.report_markdown
    assert "evidence_id:relationship_ev_1" in result.report_markdown
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


def test_structured_stages_retry_once_and_keep_model_outputs(tmp_path) -> None:
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
