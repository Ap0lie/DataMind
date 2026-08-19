from __future__ import annotations

import json
from uuid import UUID

from app.analysis.services import PlannedAnalysis
from app.analysis.workflow import (
    _extract_json_object,
    _planner_messages,
    _python_repair_messages,
    _sql_messages,
)
from app.schemas.analysis import (
    DatasetColumnProfile,
    DatasetJoinConfig,
    DatasetProfileResponse,
    DatasetReferenceResponse,
    MultiDatasetProfileResponse,
    PythonCodeAttemptResponse,
)

PRIMARY_ID = UUID("11111111-1111-1111-1111-111111111111")
RIGHT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _profile() -> DatasetProfileResponse:
    return DatasetProfileResponse(
        dataset_id=PRIMARY_ID,
        row_count=100,
        column_count=2,
        missing_value_count=0,
        missing_value_ratio=0,
        duplicate_row_count=0,
        numeric_columns=("orders__amount",),
        categorical_columns=("customer_id",),
        columns=(
            DatasetColumnProfile(
                name="customer_id",
                dtype="object",
                missing_count=0,
                distinct_count=80,
                is_numeric=False,
            ),
            DatasetColumnProfile(
                name="orders__amount",
                dtype="float64",
                missing_count=0,
                distinct_count=90,
                is_numeric=True,
            ),
        ),
        sample_records=({"customer_id": "c1", "orders__amount": 10.0},),
    )


def _multi_context() -> MultiDatasetProfileResponse:
    primary = DatasetReferenceResponse(
        dataset_id=PRIMARY_ID,
        name="customers.csv",
        status="cleaned",
        row_count=80,
        column_count=1,
        columns=("customer_id",),
    )
    right = DatasetReferenceResponse(
        dataset_id=RIGHT_ID,
        name="orders.csv",
        status="cleaned",
        row_count=100,
        column_count=2,
        columns=("customer_id", "amount"),
    )
    return MultiDatasetProfileResponse(
        primary_dataset=primary,
        additional_datasets=(right,),
        join_plan=(
            DatasetJoinConfig(
                left_dataset_id=PRIMARY_ID,
                right_dataset_id=RIGHT_ID,
                left_column="customer_id",
                right_column="customer_id",
                join_type="left",
            ),
        ),
        join_summary={
            "dataset_count": 2,
            "joined_dataset_count": 2,
            "joined_row_count": 120,
            "row_expansion_ratio": 1.5,
            "joins": [
                {
                    "status": "joined",
                    "right_dataset_name": "orders.csv",
                    "left_column": "customer_id",
                    "right_column": "customer_id",
                    "row_expansion_ratio": 1.5,
                    "right_key_unique": False,
                }
            ],
        },
        column_source_map={"orders__amount": "orders.csv.amount"},
    )


def test_planner_and_sql_prompts_include_join_grain_and_provenance() -> None:
    profile = _profile()
    context = _multi_context()
    planner_payload = json.loads(
        _planner_messages(question="total amount", profile=profile, multi_dataset_context=context)[1]["content"]
    )
    sql_messages = _sql_messages(
        question="total amount",
        profile=profile,
        planned_analysis=PlannedAnalysis("sql", "customer_id", "orders__amount", None, ("sum",)),
        multi_dataset_context=context,
    )
    sql_payload = json.loads(sql_messages[1]["content"])

    planner_context = planner_payload["dataset_schema"]["multi_dataset_context"]
    assert planner_context["row_expansion_ratio"] == 1.5
    assert planner_context["column_source_map"]["orders__amount"] == "orders.csv.amount"
    assert "Before SUM/AVG" in planner_context["grain_rule"]
    assert sql_payload["multi_dataset_context"] == planner_context
    assert "untrusted data" in sql_messages[0]["content"]


def test_planner_analysis_experience_is_compact_read_only_evidence() -> None:
    experience_id = UUID("33333333-3333-3333-3333-333333333333")
    messages = _planner_messages(
        question="total amount by customer",
        profile=_profile(),
        multi_dataset_context=_multi_context(),
        analysis_experiences=(
            {
                "memory_id": experience_id,
                "content": "A previous validated route used safe SQL.",
                "structured_value": {
                    "analysis_contract": {"required_metrics": ["amount"]},
                    "semantic_model_id": str(PRIMARY_ID),
                    "semantic_model_version": 2,
                    "join_plan": [{"left_column": "customer_id"}],
                    "tool_sequence": ["execute_safe_sql"],
                    "result_summary": {"row_count": 10},
                    "sql": "SELECT secret FROM host_file",
                    "python_code": "open('/etc/passwd').read()",
                    "raw_records": [{"secret": "not allowed"}],
                },
            },
        ),
    )
    payload = json.loads(messages[1]["content"])
    recalled = payload["dataset_schema"]["validated_analysis_experiences"]

    assert recalled == [
        {
            "experience_id": str(experience_id),
            "summary": "A previous validated route used safe SQL.",
            "analysis_contract": {"required_metrics": ["amount"]},
            "semantic_model_id": str(PRIMARY_ID),
            "semantic_model_version": 2,
            "join_plan": [{"left_column": "customer_id"}],
            "tool_sequence": ["execute_safe_sql"],
            "result_summary": {"row_count": 10},
        }
    ]
    assert "read-only route evidence" in messages[0]["content"]
    assert "host_file" not in messages[1]["content"]
    assert "/etc/passwd" not in messages[1]["content"]


def test_planner_memory_is_context_not_an_executable_requirement() -> None:
    memory_id = UUID("44444444-4444-4444-4444-444444444444")
    messages = _planner_messages(
        question="分析订单金额",
        profile=_profile(),
        memory_context=(
            {
                "memory_id": memory_id,
                "memory_type": "metric_definition",
                "content": "GMV 指订单金额总和",
                "scope_type": "dataset",
                "scope_id": PRIMARY_ID,
                "structured_value": {"value": "orders__amount"},
                "raw_records": [{"secret": "not allowed"}],
            },
        ),
    )
    payload = json.loads(messages[1]["content"])
    recalled = payload["dataset_schema"]["approved_memory_context"]

    assert recalled == [
        {
            "memory_id": str(memory_id),
            "memory_type": "metric_definition",
            "content": "GMV 指订单金额总和",
            "scope_type": "dataset",
            "scope_id": str(PRIMARY_ID),
            "structured_value": {"value": "orders__amount"},
        }
    ]
    assert "cannot add requirements" in messages[0]["content"]
    assert "secret" not in messages[1]["content"]


def test_python_repair_prompt_has_phase_specific_contract() -> None:
    attempt = PythonCodeAttemptResponse(
        attempt=1,
        phase="python_charts",
        status="failed",
        code="def analyze(df): return {'charts': df.to_dict('records')}",
        error="Generated Python output exceeded the size limit.",
    )
    plan = PlannedAnalysis("python", "customer_id", "orders__amount", None, ("analyze",))

    chart_payload = json.loads(
        _python_repair_messages(
            question="chart amount",
            profile=_profile(),
            planned_analysis=plan,
            sql_result=None,
            attempts=(attempt,),
            phase="python_charts",
            multi_dataset_context=_multi_context(),
        )[1]["content"]
    )
    stats_payload = json.loads(
        _python_repair_messages(
            question="analyze amount",
            profile=_profile(),
            planned_analysis=plan,
            sql_result=None,
            attempts=(attempt,),
            phase="python",
            multi_dataset_context=_multi_context(),
        )[1]["content"]
    )

    assert "Do not return statistics or insights" in chart_payload["output_contract"]
    assert "charts must be exactly []" in stats_payload["output_contract"]
    assert "pre-bin histograms" in chart_payload["output_contract"]


def test_extract_json_object_skips_reasoning_braces_and_trailing_objects() -> None:
    payload = _extract_json_object(
        '先检查 {not-json}，最终答案如下：\n'
        '```json\n{"route":"sql","steps":["sum"]}\n```\n'
        '{"diagnostic":"ignored"}'
    )

    assert payload == {"route": "sql", "steps": ["sum"]}
