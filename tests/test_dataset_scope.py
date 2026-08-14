from __future__ import annotations

import pytest

from app.analysis.dataset_scope import resolve_analysis_dataset_scope
from app.analysis.intent_compiler import (
    IntentCompilationHarness,
    build_intent_compilation_context,
)
from app.core.settings import Settings
from app.schemas.analysis import DatasetJoinConfig
from app.storage.dataset_store import DatasetStoreRepository


def _commerce_package(repository: DatasetStoreRepository):
    datasets = {
        name: repository.create_dataset(
            name=f"olist_{name}_dataset.csv",
            source_type="csv",
            source_metadata={},
        )
        for name in (
            "order_items",
            "orders",
            "customers",
            "order_payments",
            "products",
            "product_translation",
            "sellers",
            "order_reviews",
            "geolocation",
        )
    }
    records = {
        "order_items": [{"order_id": "O1", "product_id": "P1", "seller_id": "S1"}],
        "orders": [
            {"order_id": "O1", "customer_id": "C1", "order_status": "delivered"}
        ],
        "customers": [
            {
                "customer_id": "C1",
                "customer_state": "SP",
                "customer_zip_code_prefix": "01000",
            }
        ],
        "order_payments": [
            {"order_id": "O1", "payment_type": "credit_card", "payment_value": 10.0}
        ],
        "products": [{"product_id": "P1", "product_category_name": "books"}],
        "product_translation": [
            {"product_category_name": "books", "product_category_name_english": "books"}
        ],
        "sellers": [{"seller_id": "S1", "seller_state": "SP"}],
        "order_reviews": [{"order_id": "O1", "review_comment_message": "ok"}],
        "geolocation": [
            {
                "geolocation_zip_code_prefix": "01000",
                "geolocation_state": "SP",
            }
        ],
    }
    for name, rows in records.items():
        repository.append_raw_records(dataset_id=datasets[name].id, records=rows)
    plan = (
        DatasetJoinConfig(
            left_dataset_id=datasets["order_items"].id,
            right_dataset_id=datasets["orders"].id,
            left_column="order_id",
            right_column="order_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["orders"].id,
            right_dataset_id=datasets["customers"].id,
            left_column="customer_id",
            right_column="customer_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["orders"].id,
            right_dataset_id=datasets["order_payments"].id,
            left_column="order_id",
            right_column="order_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["order_items"].id,
            right_dataset_id=datasets["products"].id,
            left_column="product_id",
            right_column="product_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["products"].id,
            right_dataset_id=datasets["product_translation"].id,
            left_column="product_category_name",
            right_column="product_category_name",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["order_items"].id,
            right_dataset_id=datasets["sellers"].id,
            left_column="seller_id",
            right_column="seller_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["orders"].id,
            right_dataset_id=datasets["order_reviews"].id,
            left_column="order_id",
            right_column="order_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["customers"].id,
            right_dataset_id=datasets["geolocation"].id,
            left_column="customer_zip_code_prefix",
            right_column="geolocation_zip_code_prefix",
        ),
    )
    return datasets, plan


def test_explicit_three_table_question_selects_minimal_metric_rooted_subtree(
    tmp_path, monkeypatch
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, plan = _commerce_package(repository)
    sampled_ids = []
    original_sample = repository.sample_analysis_records

    def track_sample(dataset_id, *, limit=1000):
        sampled_ids.append(dataset_id)
        return original_sample(dataset_id, limit=limit)

    monkeypatch.setattr(repository, "sample_analysis_records", track_sample)
    scope = resolve_analysis_dataset_scope(
        repository,
        question=(
            "仅使用 customers、orders、order_payments 三张表，过滤 "
            "order_status=delivered，按 customer_state 统计 payment_value 总额，"
            "并给出总体支付总额和 SP 州支付总额。不要使用 order_items、reviews、"
            "products、sellers 或 geolocation，也不要按 order_status 或 "
            "payment_type 分组。"
        ),
        dataset_id=datasets["order_items"].id,
        additional_dataset_ids=tuple(
            dataset.id
            for name, dataset in datasets.items()
            if name != "order_items"
        ),
        join_plan=plan,
    )

    assert set(sampled_ids) == {
        datasets["customers"].id,
        datasets["orders"].id,
        datasets["order_payments"].id,
    }
    assert set(scope.allowlist_dataset_ids) == {
        datasets["customers"].id,
        datasets["orders"].id,
        datasets["order_payments"].id,
    }
    assert set(scope.denylist_dataset_ids) == {
        datasets["order_items"].id,
        datasets["order_reviews"].id,
        datasets["products"].id,
        datasets["product_translation"].id,
        datasets["sellers"].id,
        datasets["geolocation"].id,
    }
    assert scope.dataset_id == datasets["order_payments"].id
    assert set(scope.additional_dataset_ids) == {
        datasets["orders"].id,
        datasets["customers"].id,
    }
    assert tuple(
        (item.left_dataset_id, item.right_dataset_id) for item in scope.join_plan
    ) == (
        (datasets["order_payments"].id, datasets["orders"].id),
        (datasets["orders"].id, datasets["customers"].id),
    )


def test_prohibited_direct_join_does_not_add_an_unneeded_fact_table(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, plan = _commerce_package(repository)

    scope = resolve_analysis_dataset_scope(
        repository,
        question=(
            "仅按 customers.customer_state 分析 order_status=delivered 的订单，"
            "以 order_payments.payment_value 为总支付金额。先将 order_payments "
            "按 order_id 预聚合，再经 orders.customer_id 连接 customers.customer_id；"
            "严禁将 order_items 与 order_payments 逐行连接。"
        ),
        dataset_id=datasets["order_payments"].id,
        additional_dataset_ids=tuple(
            dataset.id
            for name, dataset in datasets.items()
            if name != "order_payments"
        ),
        join_plan=plan,
    )

    assert set(scope.referenced_dataset_ids) == {
        datasets["order_payments"].id,
        datasets["orders"].id,
        datasets["customers"].id,
    }
    assert datasets["order_items"].id not in scope.additional_dataset_ids
    assert scope.dataset_id == datasets["order_payments"].id
    assert tuple(
        (item.left_dataset_id, item.right_dataset_id) for item in scope.join_plan
    ) == (
        (datasets["order_payments"].id, datasets["orders"].id),
        (datasets["orders"].id, datasets["customers"].id),
    )


def test_approved_forbidden_relationship_prunes_join_without_field_requirements(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, plan = _commerce_package(repository)
    prohibited = DatasetJoinConfig(
        left_dataset_id=datasets["order_items"].id,
        right_dataset_id=datasets["order_payments"].id,
        left_column="order_id",
        right_column="order_id",
    )
    submitted_plan = (*plan, prohibited)
    question = "严禁将 order_items 与 order_payments 逐行连接。"
    intent = IntentCompilationHarness(
        model_router=None,
        settings=Settings(environment="test", intent_compiler_mode="shadow"),
    ).compile(
        question=question,
        context=build_intent_compilation_context(
            repository,
            dataset_ids=tuple(dataset.id for dataset in datasets.values()),
        ),
    ).intent

    scope = resolve_analysis_dataset_scope(
        repository,
        question=question,
        dataset_id=datasets["order_items"].id,
        additional_dataset_ids=tuple(
            dataset.id
            for name, dataset in datasets.items()
            if name != "order_items"
        ),
        join_plan=submitted_plan,
        intent_spec=intent,
    )

    assert prohibited not in scope.join_plan
    assert set(scope.additional_dataset_ids) == {
        dataset.id
        for name, dataset in datasets.items()
        if name != "order_items"
    }


def test_required_relationship_rejects_an_unsubmitted_authorized_dataset(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, _ = _commerce_package(repository)
    question = "将 orders 与 customers 关联后分析。"
    intent = IntentCompilationHarness(
        model_router=None,
        settings=Settings(environment="test", intent_compiler_mode="shadow"),
    ).compile(
        question=question,
        context=build_intent_compilation_context(
            repository,
            dataset_ids=(datasets["orders"].id,),
            authorized_dataset_ids=tuple(
                dataset.id for dataset in datasets.values()
            ),
        ),
    ).intent

    with pytest.raises(ValueError, match=r"Required relationship.*not submitted"):
        resolve_analysis_dataset_scope(
            repository,
            question=question,
            dataset_id=datasets["orders"].id,
            additional_dataset_ids=(),
            join_plan=(),
            intent_spec=intent,
        )


def test_single_casual_table_mention_preserves_relationship_context(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, plan = _commerce_package(repository)

    scope = resolve_analysis_dataset_scope(
        repository,
        question="Compare order_payments by customer state.",
        dataset_id=datasets["order_items"].id,
        additional_dataset_ids=tuple(
            dataset.id
            for name, dataset in datasets.items()
            if name != "order_items"
        ),
        join_plan=plan,
    )

    assert scope.dataset_id == datasets["order_items"].id
    assert scope.join_plan == plan
    assert len(scope.additional_dataset_ids) == 8


def test_strict_single_table_scope_can_drop_the_rest_of_the_package(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, plan = _commerce_package(repository)

    scope = resolve_analysis_dataset_scope(
        repository,
        question="Only use order_payments to calculate total payment value.",
        dataset_id=datasets["order_items"].id,
        additional_dataset_ids=tuple(
            dataset.id
            for name, dataset in datasets.items()
            if name != "order_items"
        ),
        join_plan=plan,
    )

    assert scope.dataset_id == datasets["order_payments"].id
    assert scope.additional_dataset_ids == ()
    assert scope.join_plan == ()
    assert scope.allowlist_dataset_ids == (datasets["order_payments"].id,)
    assert set(scope.denylist_dataset_ids) == {
        dataset.id
        for name, dataset in datasets.items()
        if name != "order_payments"
    }


def test_unresolved_strict_allowlist_fails_closed(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, plan = _commerce_package(repository)

    with pytest.raises(ValueError, match="allowlist"):
        resolve_analysis_dataset_scope(
            repository,
            question="Only use missing_table to calculate total payment value.",
            dataset_id=datasets["order_items"].id,
            additional_dataset_ids=tuple(
                dataset.id
                for name, dataset in datasets.items()
                if name != "order_items"
            ),
            join_plan=plan,
        )


def test_strict_allowlist_rejects_owned_tables_missing_from_submitted_scope(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, _ = _commerce_package(repository)

    with pytest.raises(ValueError, match="not submitted") as exc_info:
        resolve_analysis_dataset_scope(
            repository,
            question=(
                "仅使用 customers、orders、order_payments 三张表，过滤 "
                "order_status=delivered，按 customer_state 统计 payment_value 总额。"
            ),
            dataset_id=datasets["order_payments"].id,
            additional_dataset_ids=(),
            join_plan=(),
        )

    message = str(exc_info.value)
    assert "olist_customers_dataset.csv" in message
    assert "olist_orders_dataset.csv" in message


def test_multiple_referenced_tables_without_join_plan_keep_only_referenced_scope(
    tmp_path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets, _ = _commerce_package(repository)
    additional_ids = tuple(
        dataset.id
        for name, dataset in datasets.items()
        if name != "order_items"
    )

    scope = resolve_analysis_dataset_scope(
        repository,
        question="Only use customers and order_payments to total payment value.",
        dataset_id=datasets["order_items"].id,
        additional_dataset_ids=additional_ids,
        join_plan=(),
    )

    assert scope.dataset_id == datasets["order_payments"].id
    assert scope.additional_dataset_ids == (datasets["customers"].id,)
    assert scope.join_plan == ()
