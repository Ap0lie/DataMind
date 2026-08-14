from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest

from app.analysis.intent_compiler import (
    IntentCompilationContext,
    IntentCompilationHarness,
)
from app.analysis.intent_guard import validate_analysis_contract, validate_intent
from app.analysis.services import DatasetProfiler
from app.core.settings import Settings
from app.mcp.tool_schemas import ModelRouterResponse
from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisContractResponse,
    DatasetJoinConfig,
)
from app.schemas.analysis_intent import (
    FieldBinding,
    IntentSourceSpan,
    RelationshipConstraint,
)

pytestmark = pytest.mark.unit

ORDERS_ID = UUID("11111111-1111-4111-8111-111111111111")
PAYMENTS_ID = UUID("22222222-2222-4222-8222-222222222222")
SELLERS_ID = UUID("33333333-3333-4333-8333-333333333333")


class ScriptedRouter:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, **kwargs: Any) -> ModelRouterResponse:
        self.calls.append(kwargs["messages"])
        return ModelRouterResponse(
            provider="mock",
            model="intent-compiler",
            content=self.outputs.pop(0),
            finish_reason="stop",
        )


def _context() -> IntentCompilationContext:
    assets = (
        {
            "dataset_id": ORDERS_ID,
            "name": "orders.csv",
            "columns": (
                {
                    "name": "customer_state",
                    "reference": "orders__customer_state",
                    "dtype": "string",
                    "role": "dimension",
                },
                {
                    "name": "seller_state",
                    "reference": "orders__seller_state",
                    "dtype": "string",
                    "role": "dimension",
                },
                {
                    "name": "order_id",
                    "reference": "orders__order_id",
                    "dtype": "string",
                    "role": "id",
                },
            ),
        },
        {
            "dataset_id": PAYMENTS_ID,
            "name": "payments.csv",
            "columns": (
                {
                    "name": "order_id",
                    "reference": "payments__order_id",
                    "dtype": "string",
                    "role": "id",
                },
                {
                    "name": "payment_value",
                    "reference": "payments__payment_value",
                    "dtype": "float64",
                    "role": "metric",
                },
            ),
        },
        {
            "dataset_id": SELLERS_ID,
            "name": "sellers.csv",
            "columns": (
                {
                    "name": "seller_state",
                    "reference": "sellers__seller_state",
                    "dtype": "string",
                    "role": "dimension",
                },
            ),
        },
    )
    records = [
        {
            "orders__customer_state": "SP",
            "orders__seller_state": "RJ",
            "orders__order_id": "O1",
            "payments__order_id": "O1",
            "payments__payment_value": 120.0,
            "sellers__seller_state": "RJ",
        },
        {
            "orders__customer_state": "MG",
            "orders__seller_state": "SP",
            "orders__order_id": "O2",
            "payments__order_id": "O2",
            "payments__payment_value": 80.0,
            "sellers__seller_state": "SP",
        },
    ]
    bindings = {
        str(item["reference"]): FieldBinding(
            column=str(item["reference"]),
            dataset_id=UUID(str(asset["dataset_id"])),
            dataset_name=str(asset["name"]),
            dtype=str(item["dtype"]),
            role=str(item["role"]),
        )
        for asset in assets
        for item in asset["columns"]
    }
    return IntentCompilationContext(
        assets=assets,
        profile=DatasetProfiler().profile(dataset_id=ORDERS_ID, records=records),
        bindings=bindings,
    )


def _compile_rules(question: str):
    return IntentCompilationHarness(
        model_router=None,
        settings=Settings(
            environment="test",
            intent_compiler_mode="shadow",
        ),
    ).compile(question=question, context=_context())


def test_forbidden_relationship_is_not_promoted_to_dataset_scope() -> None:
    result = _compile_rules(
        "严禁将 orders.csv 与 payments.csv 逐行连接，按 customer_state 分析 payment_value。"
    )

    assert result.validation.status == "passed"
    assert result.intent.dataset_allowlist == ()
    assert len(result.intent.relationship_constraints) == 1
    relationship = result.intent.relationship_constraints[0]
    assert relationship.polarity == "forbidden"
    assert {relationship.left_dataset_id, relationship.right_dataset_id} == {
        ORDERS_ID,
        PAYMENTS_ID,
    }


def test_strict_allowlist_and_denylist_keep_independent_polarity() -> None:
    result = _compile_rules(
        "Only use orders.csv and payments.csv; do not use sellers.csv."
    )

    assert result.validation.status == "passed"
    assert result.intent.strict_dataset_scope is True
    assert set(result.intent.dataset_allowlist) == {ORDERS_ID, PAYMENTS_ID}
    assert result.intent.dataset_denylist == (SELLERS_ID,)


def test_required_relationship_keeps_textual_direction_without_widening_allowlist() -> None:
    result = _compile_rules(
        "将 payments.csv 与 orders.csv 关联后检查订单覆盖率。"
    )

    assert result.validation.status == "passed"
    assert result.intent.dataset_allowlist == ()
    relationship = result.intent.relationship_constraints[0]
    assert relationship.polarity == "required"
    assert relationship.left_dataset_id == PAYMENTS_ID
    assert relationship.right_dataset_id == ORDERS_ID


def test_llm_required_relationship_passes_guard() -> None:
    question = "将 orders.csv 与 payments.csv 关联后检查订单覆盖率。"
    expected = _compile_rules(question).intent.model_copy(update={"source": "llm"})
    router = ScriptedRouter([expected.model_dump_json()])

    result = IntentCompilationHarness(
        model_router=router,
        settings=Settings(environment="test", intent_compiler_mode="enforce"),
    ).compile(question=question, context=_context())

    assert result.attempts[0].status == "succeeded", result.attempts[0]
    assert result.validation.status == "passed", result.validation.issues


def test_llm_intent_is_repaired_with_prior_guard_errors() -> None:
    question = "不要按 seller_state 分组，按 customer_state 分析 payment_value。"
    baseline = _compile_rules(question).intent
    valid = baseline.model_copy(update={"source": "llm"})
    ghost = FieldBinding(
        column="ghost_amount",
        dataset_id=PAYMENTS_ID,
        dtype="float64",
        role="metric",
    )
    invalid = valid.model_copy(
        update={
            "required_metric": ghost,
            "clauses": tuple(
                clause.model_copy(update={"field": ghost, "concept": "ghost_amount"})
                if clause.kind == "metric" and clause.polarity == "required"
                else clause
                for clause in valid.clauses
            ),
        }
    )
    router = ScriptedRouter(
        [invalid.model_dump_json(), invalid.model_dump_json(), valid.model_dump_json()]
    )

    result = IntentCompilationHarness(
        model_router=router,
        settings=Settings(
            environment="test",
            intent_compiler_mode="enforce",
            intent_compiler_max_repairs=2,
        ),
    ).compile(question=question, context=_context())

    assert len(router.calls) == 3
    assert tuple(item.status for item in result.attempts) == (
        "failed",
        "failed",
        "succeeded",
    )
    assert result.intent.source == "llm"
    assert result.intent.required_metric == valid.required_metric
    assert "unknown_field" in router.calls[1][1]["content"]
    assert '"attempt": 2' in router.calls[2][1]["content"]


def test_two_failed_repairs_require_confirmation_without_execution() -> None:
    question = "不要按 seller_state 分组，按 customer_state 分析 payment_value。"
    baseline = _compile_rules(question).intent.model_copy(update={"source": "llm"})
    payload = baseline.model_dump(mode="json")
    payload["required_metric"] = {
        "column": "missing_metric",
        "dataset_id": str(PAYMENTS_ID),
    }
    for clause in payload["clauses"]:
        if clause["kind"] == "metric" and clause["polarity"] == "required":
            clause["field"] = payload["required_metric"]
            clause["concept"] = "missing_metric"
    router = ScriptedRouter([json.dumps(payload)] * 3)

    result = IntentCompilationHarness(
        model_router=router,
        settings=Settings(
            environment="test",
            intent_compiler_mode="enforce",
            intent_compiler_max_repairs=2,
        ),
    ).compile(question=question, context=_context())

    assert len(result.attempts) == 3
    assert result.validation.status == "confirmation_required"
    assert result.intent.requires_confirmation is True
    assert result.intent.confirmation_reasons


def test_guard_rejects_required_field_without_source_backed_clause() -> None:
    question = "按 customer_state 分析 payment_value。"
    intent = _compile_rules(question).intent.model_copy(update={"source": "llm"})
    extra = _context().bindings["orders__seller_state"]
    intent = intent.model_copy(
        update={"required_dimensions": (*intent.required_dimensions, extra)}
    )

    result = validate_intent(intent, question=question, assets=_context().assets)

    assert result.status == "repairable"
    assert "required_clause_missing_source" in {item.code for item in result.issues}


def test_contract_guard_rejects_forbidden_fields_and_relationships() -> None:
    question = "不要按 seller_state 分组，按 customer_state 汇总 payment_value。"
    intent = _compile_rules(question).intent
    contract = AnalysisContractResponse(
        objective=question,
        population="authorized rows",
        dataset_ids=(ORDERS_ID, PAYMENTS_ID),
        analysis_type="descriptive",
        metric="payments__payment_value",
        dimensions=("orders__seller_state",),
        aggregations=(
            AnalysisAggregationResponse(
                operation="sum",
                column="payments__payment_value",
                alias="total_payment_value",
            ),
        ),
        method="safe aggregate",
    )
    join = DatasetJoinConfig(
        left_dataset_id=ORDERS_ID,
        right_dataset_id=PAYMENTS_ID,
        left_column="order_id",
        right_column="order_id",
    )
    relationship_span = question.index("不要按")
    intent = intent.model_copy(
        update={
            "relationship_constraints": (
                RelationshipConstraint(
                    left_dataset_id=ORDERS_ID,
                    right_dataset_id=PAYMENTS_ID,
                    polarity="forbidden",
                    source_span=IntentSourceSpan(
                        text=question[relationship_span:relationship_span + 16],
                        start=relationship_span,
                        end=relationship_span + 16,
                    ),
                ),
            )
        }
    )

    result = validate_analysis_contract(contract, intent=intent, join_plan=(join,))

    assert result.status == "failed"
    assert {
        "contract_uses_forbidden_field",
        "forbidden_relationship_used",
        "contract_requirement_missing",
    }.issubset({item.code for item in result.issues})
