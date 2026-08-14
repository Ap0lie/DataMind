import json
from types import SimpleNamespace

import pytest

from app.analysis.analysis_contract import _analysis_type
from app.analysis.services import PlannedAnalysis
from app.analysis.workflow import (
    AnalysisWorkflowRunner,
    _claim_result,
    _mandatory_evidence_findings,
    _prepare_multimodal_inputs,
    _preserve_verified_report_findings,
    _sanitize_report_cardinality_claims,
    _unsupported_summary_numbers,
    _validate_dataset_select_sql,
)
from app.core.settings import Settings, get_settings
from app.mcp.tool_schemas import ModelRouterResponse
from app.schemas.analysis import (
    InsightFindingResponse,
    MultimodalInputResponse,
    StatisticalFindingVerdictResponse,
    StatisticalVerificationResponse,
    StructuredReportResponse,
)
from app.storage.dataset_store import DatasetStoreRepository


def test_negated_reviews_dataset_does_not_select_text_analysis() -> None:
    question = (
        "仅使用 customers、orders、order_payments 三张表，过滤 "
        "order_status=delivered，按 customer_state 统计 payment_value 总额，"
        "并给出总体支付总额和 SP 州支付总额。不要使用 order_items、reviews、"
        "products、sellers 或 geolocation，也不要按 order_status 或 "
        "payment_type 分组。"
    )
    plan = PlannedAnalysis(
        route="sql",
        category_column="customer_state",
        metric_column="payment_value",
        time_column=None,
        steps=("aggregate",),
    )

    assert _analysis_type(question, plan) == "descriptive"
    assert _analysis_type("分析 reviews 评论中的关键词", plan) == "text"


class FakeAnalysisModelRouter:
    def __init__(
        self,
        *,
        planner_content: str | None = None,
        sql_content: str | None = None,
        python_content: str | list[str] | None = None,
        python_chart_content: str | list[str] | None = None,
        round_plan_content: str | None = None,
        round_python_content: str | list[str] | None = None,
        round_python_chart_content: str | list[str] | None = None,
        report_content: str | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._planner_content = planner_content
        self._sql_content = sql_content or (
            'SELECT "region" AS category, '
            'SUM(CAST("sales" AS DOUBLE)) AS total_sales '
            'FROM dataset GROUP BY "region" ORDER BY total_sales DESC LIMIT 20'
        )
        self._python_content = python_content
        self._python_chart_content = python_chart_content
        self._round_plan_content = round_plan_content
        self._round_python_content = round_python_content
        self._round_python_chart_content = round_python_chart_content
        self._report_content = report_content

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ModelRouterResponse:
        self.calls.append({"messages": messages, "metadata": metadata or {}, "max_tokens": max_tokens})
        agent = (metadata or {}).get("agent")
        if agent == "planner":
            content = self._planner_content or (
                '{"route":"sql","category_column":"region","metric_column":"sales",'
                '"time_column":null,"steps":["Model planner selected SQL aggregation."]}'
            )
        elif agent == "design_framework":
            content = (
                '{"business_question":"Which region has the highest sales?",'
                '"candidate_dimensions":["region"],"candidate_metrics":["sales"],'
                '"likely_routes":["sql","python","hybrid"],'
                '"initial_hypotheses":["region explains sales variance"],'
                '"risk_notes":[],"key_questions":["Which region ranks first?"],'
                '"success_criteria":"Trace every claim to data."}'
            )
        elif agent == "sql":
            content = self._sql_content
        elif agent == "python":
            content = self._next_content("python", self._python_content) or (
                "def analyze(df):\n"
                "    total_sales = float(df['sales'].sum())\n"
                "    return {\n"
                "        'statistics': {'kimi_total_sales': total_sales},\n"
                "        'insights': ['DeepSeek Python Agent completed question-aware analysis.'],\n"
                "        'charts': []\n"
                "    }\n"
            )
        elif agent == "python_charts":
            content = self._next_content("python_charts", self._python_chart_content) or (
                "def analyze(df):\n"
                "    grouped = df.groupby('region', dropna=True)['sales'].sum().reset_index()\n"
                "    return {\n"
                "        'statistics': {},\n"
                "        'insights': [],\n"
                "        'charts': [{\n"
                "            'title': 'Python 区域销售图',\n"
                "            'chart_type': 'bar',\n"
                "            'spec': {'x': 'region', 'y': 'sales'},\n"
                "            'data': grouped.to_dict(orient='records')\n"
                "        }]\n"
                "    }\n"
            )
        elif agent == "round_plan":
            content = self._round_plan_content or (
                '{"route":"sql","category_column":"region","metric_column":"sales",'
                '"time_column":null,"steps":["Round-specific SQL aggregation."]}'
            )
        elif agent == "round_python":
            content = self._next_content("round_python", self._round_python_content) or (
                "def analyze(df):\n"
                "    total_sales = float(df['sales'].sum())\n"
                "    return {\n"
                "        'statistics': {'round_total_sales': total_sales},\n"
                "        'insights': ['Round Python Agent verified regional sales.'],\n"
                "        'charts': []\n"
                "    }\n"
            )
        elif agent == "round_python_charts":
            content = self._next_content("round_python_charts", self._round_python_chart_content) or (
                "def analyze(df):\n"
                "    grouped = df.groupby('region', dropna=True)['sales'].sum().reset_index()\n"
                "    return {\n"
                "        'statistics': {},\n"
                "        'insights': [],\n"
                "        'charts': [{\n"
                "            'title': 'Round Python 区域销售图',\n"
                "            'chart_type': 'bar',\n"
                "            'spec': {'x': 'region', 'y': 'sales'},\n"
                "            'data': grouped.to_dict(orient='records')\n"
                "        }]\n"
                "    }\n"
            )
        elif agent == "reflection":
            content = '{"reflections":["模型反思：South 区域需要进一步解释。"]}'
        elif agent == "integrate":
            content = (
                '{"insights":[{"title":"区域销售差异","content":"South 区域销售额最高。",'
                '"data_source":"sql_result.rows","evidence":"SQL 聚合结果显示 South 排名第一。",'
                '"confidence":"high","business_impact":"可优先检查 South 的订单结构。",'
                '"recommended_action":"继续检查利润率和订单数量。","impact_pct":100}]}'
            )
        elif agent == "review":
            content = '{"issues":[]}'
        elif agent == "chart_refine":
            content = (
                '{"chart_explanations":[{"title":"SQL 结果图",'
                '"explanation":"该图显示不同区域销售额对比，South 最高。"}]}'
            )
        elif agent == "report":
            content = self._report_content or (
                '{"executive_summary":"Kimi 结构化总结：South 区域销售额最高，建议继续检查利润率。",'
                '"analysis_context":"基于多轮 SQL 与 Python sandbox 分析生成。",'
                '"key_findings":[{"title":"Kimi 报告洞察","content":"South 区域在销售额聚合结果中领先。",'
                '"data_source":"sql_result.rows","evidence":"SQL rows show South ranked first.",'
                '"confidence":"high","business_impact":"可优先检查 South 的收入结构。",'
                '"recommended_action":"继续检查 South 的利润率、订单量和客单价。","impact_pct":100}],'
                '"chart_explanations":[{"title":"SQL 结果图","explanation":"柱状图显示 South 销售额最高。"}],'
                '"data_gaps":["缺少订单量字段，无法拆解销量与客单价。"],'
                '"validation_issues":[],"recommended_next_steps":["补充订单量后继续分析利润率。"]}'
            )
        else:
            content = (
                "### 模型补充解读\n\n"
                "West 区域销售额最高, 建议继续检查利润率和订单结构。"
            )
        return ModelRouterResponse(
            provider="mock",
            model="fake-router",
            content=content,
            token_usage={"total_tokens": 12},
        )

    def _next_content(self, agent: str, value: str | list[str] | None) -> str | None:
        if isinstance(value, list):
            previous_count = sum(
                1
                for call in self.calls[:-1]
                if (call.get("metadata") or {}).get("agent") == agent
            )
            return value[min(previous_count, len(value) - 1)]
        return value


class ContractRepairRouter:
    def __init__(self, *, question: str, left_id, right_id) -> None:
        self.question = question
        self.left_id = left_id
        self.right_id = right_id
        self.agents: list[str] = []

    def complete(self, **kwargs) -> ModelRouterResponse:
        agent = str((kwargs.get("metadata") or {}).get("agent") or "")
        self.agents.append(agent)
        if agent == "intent_compiler":
            content = json.dumps(
                {
                    "question": self.question,
                    "source": "llm",
                    "clauses": [
                        {
                            "clause_id": "relationship-1",
                            "kind": "relationship",
                            "polarity": "required",
                            "concept": "row_level_join",
                            "source_span": {
                                "text": self.question,
                                "start": 0,
                                "end": len(self.question),
                            },
                        }
                    ],
                    "relationship_constraints": [
                        {
                            "left_dataset_id": str(self.left_id),
                            "right_dataset_id": str(self.right_id),
                            "operation": "row_level_join",
                            "polarity": "required",
                            "source_span": {
                                "text": self.question,
                                "start": 0,
                                "end": len(self.question),
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            )
        elif agent == "planner":
            content = (
                '{"route":"sql","category_column":null,"metric_column":null,'
                '"time_column":null,"steps":["Inspect joined coverage."]}'
            )
        else:
            raise AssertionError(f"Unexpected model call after contract rejection: {agent}")
        return ModelRouterResponse(provider="mock", model="guard-test", content=content)


def test_contract_guard_replans_twice_then_stops_before_tools(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        environment="test",
        intent_compiler_mode="enforce",
        intent_compiler_max_repairs=2,
    )
    monkeypatch.setattr("app.analysis.workflow.get_settings", lambda: settings)
    monkeypatch.setattr("app.analysis.intent_compiler.get_settings", lambda: settings)
    repository = DatasetStoreRepository(str(tmp_path))
    orders = repository.create_dataset(
        name="orders.csv", source_type="csv", source_metadata={}
    )
    payments = repository.create_dataset(
        name="payments.csv", source_type="csv", source_metadata={}
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[{"order_id": "O1", "status": "complete"}],
    )
    repository.append_raw_records(
        dataset_id=payments.id,
        records=[{"order_id": "O1", "payment_value": 10.0}],
    )
    question = "将 orders.csv 与 payments.csv 关联后检查订单覆盖率。"
    router = ContractRepairRouter(
        question=question,
        left_id=orders.id,
        right_id=payments.id,
    )

    with pytest.raises(RuntimeError) as captured:
        AnalysisWorkflowRunner(repository, model_router=router).run(
            dataset_id=orders.id,
            additional_dataset_ids=(payments.id,),
            question=question,
        )

    assert router.agents == ["intent_compiler", "planner", "planner", "planner"]
    assert "did not preserve the approved user intent" in str(captured.value)
    assert repository.list_reports(orders.id) == ()


def test_analysis_workflow_runner_executes_planner_sql_python_report(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter()
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
        multimodal_inputs=(
            MultimodalInputResponse(
                kind="screenshot",
                title="Dashboard screenshot",
                description="截图备注 South 区域促销活动更集中。",
                source_ref="manual-test",
                media_type="image/png",
                data_url="data:image/png;base64,iVBORw0KGgo=",
            ),
        ),
    )
    reports = repository.list_reports(dataset.id)

    assert result.plan.route == "sql"
    assert result.multimodal_inputs[0].title == "Dashboard screenshot"
    assert result.plan.steps == ("Model planner selected SQL aggregation.",)
    assert result.sql_result is not None
    assert result.sql_result.rows[0]["category"] == "South"
    assert result.python_result is not None
    assert result.python_source == "model_router"
    assert result.python_generated_code is not None
    assert "def analyze" in result.python_generated_code
    assert "chart generation phase" in result.python_generated_code
    assert result.python_execution_error is None
    assert "DeepSeek Python Agent completed question-aware analysis." in result.python_result.insights
    assert result.analysis_framework is not None
    assert result.analysis_contract is not None
    assert result.analysis_contract.metric == "sales"
    assert result.statistical_verification is not None
    assert result.statistical_verification.numeric_evidence_coverage == 1
    assert result.analysis_lineage is not None
    assert any(node.node_type == "report" for node in result.analysis_lineage.nodes)
    assert result.structured_report is not None
    assert result.structured_report.analysis_lineage == result.analysis_lineage
    assert result.analysis_framework.candidate_dimensions == ("region",)
    assert len(result.rounds) == 3
    assert result.rounds[0].hypothesis.statement == "region explains sales variance"
    assert result.rounds[0].plan.route == "sql"
    assert result.rounds[0].execution_result["sql_row_count"] == 2
    assert result.rounds[0].execution_result["python_source"] == "model_router"
    assert "def analyze" in result.rounds[0].execution_result["python_generated_code"]
    assert result.rounds[0].execution_result["python_execution_error"] is None
    assert "chart generation phase" in result.rounds[0].execution_result["python_generated_code"]
    assert result.rounds[0].execution_result["chart_count"] >= 1
    assert result.rounds[0].execution_result["fanout_mode"] == "serial_foundation"
    assert result.rounds[1].execution_result["fanout_mode"] == "langgraph_send_fanout"
    assert result.rounds[2].execution_result["fanout_group"] == "rounds_2_3"
    assert "Round Python Agent verified regional sales." in result.python_result.insights
    assert result.final_insights[0].title == "区域销售差异"
    assert result.structured_report.executive_summary.startswith("Kimi 结构化总结")
    assert result.structured_report.key_findings[0].title == "Kimi 报告洞察"
    assert result.structured_report.chart_explanations
    assert "Kimi 结构化总结" in result.report_markdown
    assert reports[0]["metadata"]["workflow"] == "langgraph_analysis"
    assert reports[0]["metadata"]["multimodal_inputs"][0]["kind"] == "screenshot"
    assert reports[0]["metadata"]["multimodal_inputs"][0]["data_url"].startswith("data:image/png")
    assert reports[0]["metadata"]["planner_source"] == "model_router"
    assert reports[0]["metadata"]["sql_source"] == "model_router"
    assert reports[0]["metadata"]["python_source"] == "model_router"
    assert reports[0]["metadata"]["python_generated_code"]
    assert reports[0]["metadata"]["analysis_lineage"]["nodes"]
    assert reports[0]["metadata"]["python_execution_error"] is None
    assert reports[0]["metadata"]["sql_validation_error"] is None
    assert reports[0]["metadata"]["report_source"] == "model_router_structured"
    assert reports[0]["metadata"]["model_router_provider"] == "mock"
    assert reports[0]["metadata"]["model_router_model"] == "fake-router"
    assert reports[0]["metadata"]["nodes"] == [
        "intent_compile",
        "scope_resolve",
        "planner",
        "contract_validate",
        "design_framework",
        "sql_agent",
        "python_agent",
        "iterative_prepare_rounds",
        "iterative_round_1",
        "iterative_fanout_round",
        "iterative_reflect_and_merge",
        "integrate_insights",
        "format_charts",
        "statistical_verify",
        "adversarial_validate",
        "report_agent",
    ]
    assert result.intent_validation is not None
    assert result.contract_validation is not None
    assert result.contract_validation.status == "passed"
    agents = [call["metadata"]["agent"] for call in model_router.calls]
    assert agents[:4] == [
        "planner",
        "design_framework",
        "sql",
        "python",
    ]
    assert agents.count("python_charts") == 1
    assert agents.count("round_plan") == 3
    assert agents.count("round_python") == 3
    assert agents.count("round_python_charts") == 3
    assert agents[-5:] == ["reflection", "integrate", "chart_refine", "review", "report"]
    report_call = next(call for call in model_router.calls if call["metadata"]["agent"] == "report")
    report_content = report_call["messages"][1]["content"]
    assert isinstance(report_content, list)
    assert report_content[0]["type"] == "text"
    assert "multimodal_context" in report_content[0]["text"]
    assert "experience_context" in report_content[0]["text"]
    assert "Dashboard screenshot" in report_content[0]["text"]
    report_payload = json.loads(report_content[0]["text"])
    assert set(report_payload) == {
        "question",
        "fallback_structured_report",
        "multimodal_context",
        "multi_dataset_context",
        "experience_context",
    }
    assert report_content[1]["type"] == "image_url"
    planner_call = next(call for call in model_router.calls if call["metadata"]["agent"] == "planner")
    assert "experience_context" in planner_call["messages"][1]["content"]


def test_prepare_multimodal_inputs_marks_image_and_pdf_processing() -> None:
    prepared = _prepare_multimodal_inputs(
        (
            MultimodalInputResponse(
                kind="screenshot",
                title="Chart screenshot",
                description="视觉截图。",
                media_type="image/png",
                data_url="data:image/png;base64,iVBORw0KGgo=",
            ),
            MultimodalInputResponse(
                kind="pdf_page",
                title="Brief.pdf",
                description="PDF 辅助材料。",
                media_type="application/pdf",
                data_url="data:application/pdf;base64,not-valid-base64",
            ),
        )
    )

    assert prepared[0].processing_status == "native_image_payload"
    assert prepared[0].data_url is not None
    assert prepared[1].processing_status == "pdf_text_unavailable"
    assert prepared[1].data_url is None
    assert "PDF extraction note" in prepared[1].description


def test_analysis_workflow_round_python_uses_fallback_for_dangerous_imports(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        round_python_content=(
            "import os\n"
            "def analyze(df):\n"
            "    return {\n"
            "        'statistics': {'cwd_length': len(os.getcwd())},\n"
            "        'insights': ['Regular Python imports executed.'],\n"
            "        'charts': []\n"
            "    }\n"
        )
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    assert result.rounds[0].execution_result["python_source"] == "rules"
    assert "cannot import os" in str(result.rounds[0].execution_result["python_execution_error"])
    assert "Regular Python imports executed." not in result.python_result.insights
    assert result.structured_report is not None


def test_analysis_workflow_python_repairs_failed_generated_code(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    bad_code = (
        "def analyze(df):\n"
        "    return {'statistics': {'broken': float(df['missing'].sum())}, 'insights': [], 'charts': []}\n"
    )
    fixed_code = (
        "def analyze(df):\n"
        "    total_sales = float(df['sales'].sum())\n"
        "    return {'statistics': {'fixed_total_sales': total_sales}, 'insights': ['第二次修复成功。'], 'charts': []}\n"
    )
    model_router = FakeAnalysisModelRouter(python_content=[bad_code, fixed_code])
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    python_calls = [
        call for call in model_router.calls if (call.get("metadata") or {}).get("agent") == "python"
    ]
    assert len(python_calls) == 2
    assert result.python_source == "model_router"
    assert result.python_execution_error is None
    assert len(result.python_attempts) == 3
    assert result.python_attempts[0].status == "failed"
    assert result.python_attempts[1].status == "succeeded"
    assert result.python_attempts[2].phase == "python_charts"
    repair_payload = python_calls[1]["messages"][1]["content"]
    assert "missing" in repair_payload
    assert "broken" in repair_payload
    assert "def analyze" in repair_payload
    assert result.python_result.statistics["model_generated"]["fixed_total_sales"] == 280.0


def test_analysis_workflow_python_repair_detects_truncated_code(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    truncated_code = (
        "def analyze(df):\n"
        "    charts = []\n"
        "    charts.append({'title': '英文产品类别分布（前10）', 'chart_type': 'bar', 'spec': {'x"
    )
    fixed_code = (
        "def analyze(df):\n"
        "    total_sales = float(df['sales'].sum())\n"
        "    return {'statistics': {'fixed_total_sales': total_sales}, 'insights': ['截断后短代码修复成功。'], 'charts': []}\n"
    )
    model_router = FakeAnalysisModelRouter(python_content=[truncated_code, fixed_code])
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    python_calls = [
        call for call in model_router.calls if (call.get("metadata") or {}).get("agent") == "python"
    ]
    repair_payload = python_calls[1]["messages"][1]["content"]
    assert result.python_source == "model_router"
    assert result.python_execution_error is None
    assert "concise_truncation_repair" in repair_payload
    assert "output/token limit" in repair_payload
    assert "fewer charts" in repair_payload
    assert python_calls[0]["max_tokens"] == 3200


def test_analysis_workflow_python_returns_three_failures_to_user_with_fallback(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        python_content=[
            "def analyze(df):\n    return {'statistics': {'a': df['missing_a'].sum()}, 'insights': [], 'charts': []}\n",
            "def analyze(df):\n    return {'statistics': {'b': df['missing_b'].sum()}, 'insights': [], 'charts': []}\n",
            "def analyze(df):\n    return {'statistics': {'c': df['missing_c'].sum()}, 'insights': [], 'charts': []}\n",
        ]
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    python_calls = [
        call for call in model_router.calls if (call.get("metadata") or {}).get("agent") == "python"
    ]
    assert len(python_calls) == 3
    assert result.python_source == "rules"
    assert "missing_c" in str(result.python_execution_error)
    assert len(result.python_attempts) == 3
    assert all(attempt.status == "failed" for attempt in result.python_attempts)
    third_prompt = python_calls[2]["messages"][1]["content"]
    assert "missing_a" in third_prompt
    assert "missing_b" in third_prompt
    assert any("failed after 3 attempts" in issue.issue for issue in result.validation_issues)


def test_analysis_workflow_round_python_repairs_failed_code(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        round_python_content=[
            "def analyze(df):\n    return {'statistics': {'bad': df['bad_column'].sum()}, 'insights': [], 'charts': []}\n",
            "def analyze(df):\n    return {'statistics': {'still_bad': df['still_bad'].sum()}, 'insights': [], 'charts': []}\n",
            (
                "def analyze(df):\n"
                "    total_sales = float(df['sales'].sum())\n"
                "    return {'statistics': {'round_fixed_total': total_sales}, 'insights': ['第三次 round 修复成功。'], 'charts': []}\n"
            ),
        ]
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    round_attempts = result.rounds[0].execution_result["python_attempts"]
    assert len(round_attempts) == 4
    assert round_attempts[0]["status"] == "failed"
    assert round_attempts[1]["status"] == "failed"
    assert round_attempts[2]["status"] == "succeeded"
    assert round_attempts[3]["phase"] == "round_python_charts"
    assert result.rounds[0].execution_result["python_source"] == "model_router"
    round_python_calls = [
        call for call in model_router.calls if (call.get("metadata") or {}).get("agent") == "round_python"
    ]
    third_prompt = round_python_calls[2]["messages"][1]["content"]
    assert "bad_column" in third_prompt
    assert "still_bad" in third_prompt


def test_analysis_workflow_round_python_allows_imports_and_for_loops(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        round_python_content=(
            "import pandas as pd\n"
            "from collections import Counter\n"
            "def analyze(df):\n"
            "    counter = Counter()\n"
            "    for value in df['region'].dropna().astype(str).tolist():\n"
            "        counter[value] += 1\n"
            "    rows = [{'region': key, 'count': int(count)} for key, count in counter.items()]\n"
            "    total_sales = float(pd.to_numeric(df['sales'], errors='coerce').fillna(0).sum())\n"
            "    return {\n"
            "        'statistics': {'round_total_sales': total_sales, 'region_count': dict(counter)},\n"
            "        'insights': ['Safe import and for-loop analysis completed.'],\n"
            "        'charts': [{\n"
            "            'title': 'Region counts',\n"
            "            'chart_type': 'bar',\n"
            "            'spec': {'x': 'region', 'y': 'count'},\n"
            "            'data': rows\n"
            "        }]\n"
            "    }\n"
        )
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    assert result.rounds[0].execution_result["python_source"] == "model_router"
    assert result.rounds[0].execution_result["python_execution_error"] is None
    assert "Safe import 和 for-loop analysis completed." in result.python_result.insights


def test_analysis_workflow_localizes_common_english_python_insights(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        python_content=(
            "def analyze(df):\n"
            "    return {\n"
            "        'statistics': {},\n"
            "        'insights': ['The dataset contains 3 rows and 2 columns.'],\n"
            "        'charts': []\n"
            "    }\n"
        ),
        round_python_content=(
            "def analyze(df):\n"
            "    return {\n"
            "        'statistics': {},\n"
            "        'insights': ['Average text length is 95 characters.'],\n"
            "        'charts': []\n"
            "    }\n"
        ),
    )
    dataset = repository.create_dataset(name="reviews.json", source_type="json", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"review": "great movie", "sentiment": "positive"},
            {"review": "bad movie", "sentiment": "negative"},
            {"review": "excellent acting", "sentiment": "positive"},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="比较正面和负面评论的文本长度和关键词",
    )

    assert any("数据集包含 3 行 和 2 列" in insight for insight in result.python_result.insights)
    assert any("平均文本长度为 95 字符" in insight for insight in result.python_result.insights)


def test_analysis_workflow_records_plan_validation_issues_and_falls_back(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        planner_content=(
            '{"route":"sql","category_column":"missing_region","metric_column":"sales",'
            '"time_column":null,"steps":["Invalid planner step."]}'
        )
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    assert result.sql_result is not None
    assert '"region"' in result.sql_result.sql
    assert any(
        issue.finding_ref == "planner"
        and "unknown category_column" in issue.issue
        for issue in result.validation_issues
    )


def test_analysis_workflow_resolves_source_qualified_plan_columns(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        planner_content=(
            '{"route":"sql","category_column":"sales_dataset.csv__region",'
            '"metric_column":"sales_dataset.csv__sales","time_column":null,'
            '"steps":["Aggregate sales by region."]}'
        )
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100},
            {"region": "South", "sales": 180},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    assert result.planner_metadata.candidate_metrics[0] == "sales"
    assert not any(
        issue.finding_ref == "planner" and "unknown" in issue.issue
        for issue in result.validation_issues
    )


def test_analysis_workflow_records_round_plan_validation_issues(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        round_plan_content=(
            '{"route":"hybrid","category_column":"region","metric_column":"missing_sales",'
            '"time_column":null,"steps":["Invalid round step."]}'
        )
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )

    assert result.rounds[0].validation_status == "warning"
    assert result.rounds[0].execution_result["plan_validation_issues"]
    assert any(
        issue.finding_ref == "round_plan_1"
        and "unknown metric_column" in issue.issue
        for issue in result.validation_issues
    )


def test_analysis_workflow_sanitizes_round_field_role_conflicts(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        round_plan_content=(
            '{"route":"hybrid","category_column":"sales","metric_column":"sales",'
            '"time_column":"sales","steps":["Compare seller potential."]}'
        )
    )
    dataset = repository.create_dataset(name="sellers.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"seller_id": "S1", "seller_state": "SP", "sales": 100},
            {"seller_id": "S2", "seller_state": "MG", "sales": 180},
            {"seller_id": "S3", "seller_state": "SP", "sales": 150},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="哪些卖家的潜力最大？",
    )

    assert result.rounds
    assert all(round_item.plan.metric_column == "sales" for round_item in result.rounds)
    assert all(round_item.plan.category_column != "sales" for round_item in result.rounds)
    assert all(round_item.plan.time_column is None for round_item in result.rounds)


def test_analysis_workflow_report_falls_back_when_structured_json_is_invalid(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(report_content="short")
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )
    reports = repository.list_reports(dataset.id)

    assert result.structured_report is not None
    assert result.structured_report.executive_summary.startswith("围绕")
    assert reports[0]["metadata"]["report_source"] == "rules"
    assert any(
        issue.finding_ref == "generate_structured_report"
        for issue in result.validation_issues
    )


def test_analysis_workflow_rejects_unsupported_numeric_summary(tmp_path) -> None:
    report_content = json.dumps(
        {
            "executive_summary": (
                "Kimi 结构化总结：South 区域销售额达到 999，建议继续检查利润率。"
            ),
            "analysis_context": "基于 SQL 与 Python 分析生成。",
            "key_findings": [
                {
                    "title": "Kimi 报告洞察",
                    "content": "South 区域在销售额聚合结果中领先。",
                    "data_source": "sql_result.rows",
                    "evidence": "SQL rows show South ranked first.",
                    "confidence": "high",
                    "business_impact": "可优先检查 South 的收入结构。",
                    "recommended_action": "继续检查 South 的利润率。",
                    "impact_pct": 100,
                }
            ],
            "chart_explanations": [],
            "data_gaps": [],
            "validation_issues": [],
            "recommended_next_steps": ["继续检查利润率。"],
        },
        ensure_ascii=False,
    )
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(report_content=report_content)
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )
    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )
    reports = repository.list_reports(dataset.id)

    assert result.structured_report is not None
    assert "999" not in result.structured_report.executive_summary
    assert (
        reports[0]["metadata"]["report_source"]
        == "model_router_structured_numeric_sanitized"
    )
    assert any(
        issue.finding_ref == "executive_summary" and "999" in issue.issue
        for issue in result.structured_report.validation_issues
    )


def test_rules_fallback_does_not_restore_rejected_finding() -> None:
    rejected = InsightFindingResponse(
        title="已拒绝结论",
        content="观察结果导致业务增长。",
        data_source="tool_evidence",
        evidence="evidence_id:ev_1",
    )
    report = StructuredReportResponse(
        executive_summary="确定性回退报告。",
        key_findings=(rejected,),
    )
    verification = StatisticalVerificationResponse(
        status="failed",
        summary="统计审查失败。",
        finding_verdicts=(
            StatisticalFindingVerdictResponse(
                finding_ref="finding_1",
                title=rejected.title,
                status="failed",
            ),
        ),
        requires_replan=True,
        numeric_evidence_coverage=1,
    )

    preserved = _preserve_verified_report_findings(
        report,
        verified_findings=(),
        verification=verification,
    )

    assert preserved.key_findings == ()


def test_join_cardinality_uses_declared_direction_and_sanitizes_model_claim() -> None:
    findings = _mandatory_evidence_findings(
        {
            "question": "按客户州统计已交付支付总额",
            "tool_evidence": (
                {
                    "evidence_id": "ev_relationships",
                    "status": "succeeded",
                    "relationship_guard": True,
                    "result": {
                        "relationships": [
                            {
                                "left_dataset": "order_payments",
                                "right_dataset": "orders",
                                "left_column": "order_id",
                                "right_column": "order_id",
                                "relationship_type": "many_to_one",
                                "left_non_null_count": 5,
                                "left_distinct_count": 3,
                                "left_duplicate_count": 2,
                            }
                        ]
                    },
                },
                {
                    "evidence_id": "ev_native",
                    "status": "succeeded",
                    "result": {
                        "native_grain": True,
                        "source_dataset": "order_payments",
                        "metric": "payment_value",
                        "aggregation": "sum",
                        "source_row_count": 5,
                        "rows": [{"sum_payment_value": 100.0}],
                    },
                },
            ),
        }
    )
    relationship_finding = next(
        finding for finding in findings if "多表关系" in finding.title
    )
    assert "order_payments.order_id → orders.order_id 为 N:1" in relationship_finding.content

    bad_report = StructuredReportResponse(
        executive_summary="两次 Join 均为一对一，因此不存在重复风险。",
        key_findings=(
            InsightFindingResponse(
                title="Join 基数",
                content="order_payments 与 orders 是一对一关系。",
                data_source="model",
            ),
        ),
    )
    fallback = StructuredReportResponse(
        executive_summary=relationship_finding.content,
        key_findings=(relationship_finding,),
    )

    sanitized, changed = _sanitize_report_cardinality_claims(
        report=bad_report,
        fallback=fallback,
        verified_findings=(relationship_finding,),
    )

    assert changed is True
    assert "一对一" not in sanitized.executive_summary
    assert "N:1" in sanitized.executive_summary
    assert sanitized.key_findings == ()
    assert any(
        issue.finding_ref == "join_cardinality"
        for issue in sanitized.validation_issues
    )


def test_executive_summary_cannot_borrow_a_number_from_global_evidence() -> None:
    verified = InsightFindingResponse(
        title="SP 支付总额",
        content="SP 支付总额为100。",
        data_source="tool_evidence",
        evidence="evidence_id:ev_states",
    )
    verification = StatisticalVerificationResponse(
        status="passed",
        summary="统计审查通过。",
        finding_verdicts=(
            StatisticalFindingVerdictResponse(
                finding_ref="finding_1",
                title=verified.title,
                status="passed",
            ),
        ),
        requires_replan=False,
        numeric_evidence_coverage=1,
    )
    report = StructuredReportResponse(
        executive_summary="SP 支付总额为50。",
        key_findings=(verified,),
    )
    state = {
        "statistical_verification": verification,
        "final_insights": (verified,),
        "tool_evidence": (
            {
                "evidence_id": "ev_states",
                "status": "succeeded",
                "result": {
                    "rows": [
                        {"customer_state": "SP", "total_payment": 100.0},
                        {"customer_state": "MS", "total_payment": 50.0},
                    ]
                },
            },
        ),
    }

    assert _unsupported_summary_numbers(
        state,
        report,
        SimpleNamespace(row_count=2, column_count=2),
    ) == ["50"]


def test_analysis_workflow_falls_back_when_model_sql_is_unsafe(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATAMIND_ANALYSIS_FAST_PATH_ENABLED", "false")
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path))
    model_router = FakeAnalysisModelRouter(
        sql_content='DROP TABLE dataset; SELECT * FROM dataset'
    )
    dataset = repository.create_dataset(name="sales.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[
            {"region": "North", "sales": 100, "profit": 20},
            {"region": "South", "sales": 180, "profit": 45},
        ],
    )

    result = AnalysisWorkflowRunner(repository, model_router=model_router).run(
        dataset_id=dataset.id,
        question="Which region has the highest sales?",
    )
    reports = repository.list_reports(dataset.id)

    assert result.sql_result is not None
    assert 'SUM(CAST("sales" AS DOUBLE))' in result.sql_result.sql
    assert result.sql_result.rows[0]["category"] == "South"
    assert reports[0]["metadata"]["sql_source"] == "rules"
    assert "Only one SQL statement is allowed" in reports[0]["metadata"]["sql_validation_error"]


def test_sql_validator_allows_only_selects_against_dataset() -> None:
    safe = _validate_dataset_select_sql("SELECT region, SUM(sales) FROM dataset GROUP BY region")
    non_select = _validate_dataset_select_sql("UPDATE dataset SET sales = 0")
    forbidden = _validate_dataset_select_sql("SELECT * FROM dataset; COPY dataset TO 'x.csv'")
    external_table = _validate_dataset_select_sql("SELECT * FROM customers")
    comma_join = _validate_dataset_select_sql("SELECT * FROM dataset, customers")
    schema_table = _validate_dataset_select_sql("SELECT * FROM main.dataset")
    self_join = _validate_dataset_select_sql(
        "SELECT d.customer_state, SUM(d.payment_value) FROM dataset d "
        "JOIN dataset o ON o.order_id = d.order_id GROUP BY d.customer_state"
    )

    assert safe["ok"]
    assert not non_select["ok"]
    assert not forbidden["ok"]
    assert not external_table["ok"]
    assert not comma_join["ok"]
    assert not schema_table["ok"]
    assert not self_join["ok"]
    assert "scan the prepared dataset exactly once" in str(self_join["message"])
def test_claim_result_keeps_derived_total_for_artifact_backed_rows() -> None:
    result = _claim_result(
        {
            "sql": "SELECT region, SUM(sales) AS total_sales FROM dataset GROUP BY region",
            "rows": [
                {"region": "North", "total_sales": 100},
                {"region": "South", "total_sales": 180},
            ],
            "explanation": "grouped result",
        }
    )

    assert result is not None
    assert 280.0 in result["claim_values"]
