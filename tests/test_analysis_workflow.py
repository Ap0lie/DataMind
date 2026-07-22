import json

from app.analysis.workflow import (
    AnalysisWorkflowRunner,
    _prepare_multimodal_inputs,
    _validate_dataset_select_sql,
)
from app.mcp.tool_schemas import ModelRouterResponse
from app.schemas.analysis import MultimodalInputResponse
from app.storage.dataset_store import DatasetStoreRepository


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
    assert result.structured_report is not None
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
    assert reports[0]["metadata"]["python_execution_error"] is None
    assert reports[0]["metadata"]["sql_validation_error"] is None
    assert reports[0]["metadata"]["report_source"] == "model_router_structured"
    assert reports[0]["metadata"]["model_router_provider"] == "mock"
    assert reports[0]["metadata"]["model_router_model"] == "fake-router"
    assert reports[0]["metadata"]["nodes"] == [
        "planner",
        "design_framework",
        "sql_agent",
        "python_agent",
            "iterative_prepare_rounds",
            "iterative_round_1",
            "iterative_fanout_round",
            "iterative_reflect_and_merge",
            "integrate_insights",
            "format_charts",
            "adversarial_validate",
            "report_agent",
    ]
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


def test_analysis_workflow_falls_back_when_model_sql_is_unsafe(tmp_path) -> None:
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

    assert safe["ok"]
    assert not non_select["ok"]
    assert not forbidden["ok"]
    assert not external_table["ok"]
    assert not comma_join["ok"]
    assert not schema_table["ok"]
