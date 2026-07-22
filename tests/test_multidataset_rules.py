from __future__ import annotations

from uuid import uuid4

import pytest

from app.analysis.dataset_groups import (
    select_automatic_dataset_relationships,
    suggest_dataset_group_relationships,
)
from app.analysis.multidataset import prepare_multi_dataset_context, suggest_dataset_joins
from app.schemas.analysis import DatasetJoinConfig
from app.schemas.dataset_store import DatasetRelationshipCandidate
from app.storage.dataset_store import DatasetStoreRepository, StoredDatasetGroup


def _candidate(
    left_id,
    right_id,
    left_column: str,
    right_column: str,
    *,
    confidence: float,
    relationship_type: str,
) -> DatasetRelationshipCandidate:
    return DatasetRelationshipCandidate(
        left_dataset_id=left_id,
        right_dataset_id=right_id,
        left_column=left_column,
        right_column=right_column,
        confidence=confidence,
        source="rules",
        reason="validated test relationship",
        estimated_match_rate=1.0,
        relationship_type=relationship_type,
    )


def test_automatic_relationships_build_safe_acyclic_tree() -> None:
    items_id, orders_id, customers_id, products_id, translation_id = (uuid4() for _ in range(5))
    group = StoredDatasetGroup(
        id=uuid4(),
        user_id="default",
        name="commerce package",
        description="",
        dataset_ids=(orders_id, customers_id, items_id, products_id, translation_id),
        relationships=(),
        metadata={},
    )
    candidates = (
        _candidate(items_id, orders_id, "order_id", "order_id", confidence=0.96, relationship_type="many_to_one"),
        _candidate(orders_id, customers_id, "customer_id", "customer_id", confidence=0.94, relationship_type="many_to_one"),
        _candidate(items_id, products_id, "product_id", "product_id", confidence=0.95, relationship_type="many_to_one"),
        _candidate(products_id, translation_id, "category", "category", confidence=0.92, relationship_type="many_to_one"),
    )

    selection = select_automatic_dataset_relationships(group, candidates)

    assert selection.primary_dataset_id == items_id
    assert len(selection.relationships) == 4
    assert selection.unresolved_dataset_ids == ()
    connected = {selection.primary_dataset_id}
    for relationship in selection.relationships:
        assert relationship.left_dataset_id in connected
        assert relationship.right_dataset_id not in connected
        assert relationship.relationship_type in {"many_to_one", "one_to_one"}
        connected.add(relationship.right_dataset_id)


def test_prepare_multi_dataset_context_executes_chain_and_tracks_sources(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    items = repository.create_dataset(name="order_items.csv", source_type="csv", source_metadata={})
    orders = repository.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    customers = repository.create_dataset(name="customers.csv", source_type="csv", source_metadata={})
    products = repository.create_dataset(name="products.csv", source_type="csv", source_metadata={})
    translation = repository.create_dataset(name="translation.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=items.id,
        records=[
            {"order_id": "O1", "product_id": "P1", "price": 10},
            {"order_id": "O2", "product_id": "P2", "price": 20},
        ],
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[{"order_id": "O1", "customer_id": "C1"}, {"order_id": "O2", "customer_id": "C2"}],
    )
    repository.append_raw_records(
        dataset_id=customers.id,
        records=[{"customer_id": "C1", "segment": "A"}, {"customer_id": "C2", "segment": "B"}],
    )
    repository.append_raw_records(
        dataset_id=products.id,
        records=[{"product_id": "P1", "category": "cat_a"}, {"product_id": "P2", "category": "cat_b"}],
    )
    repository.append_raw_records(
        dataset_id=translation.id,
        records=[{"category": "cat_a", "category_en": "A"}, {"category": "cat_b", "category_en": "B"}],
    )
    plan = (
        DatasetJoinConfig(left_dataset_id=items.id, right_dataset_id=orders.id, left_column="order_id", right_column="order_id"),
        DatasetJoinConfig(left_dataset_id=items.id, right_dataset_id=products.id, left_column="product_id", right_column="product_id"),
        DatasetJoinConfig(left_dataset_id=orders.id, right_dataset_id=customers.id, left_column="customer_id", right_column="customer_id"),
        DatasetJoinConfig(left_dataset_id=products.id, right_dataset_id=translation.id, left_column="category", right_column="category"),
    )

    prepared = prepare_multi_dataset_context(
        repository,
        dataset_id=items.id,
        additional_dataset_ids=(orders.id, customers.id, products.id, translation.id),
        join_plan=plan,
    )

    assert len(prepared.records) == 2
    assert prepared.response is not None
    assert prepared.response.join_summary["joined_dataset_count"] == 5
    assert prepared.response.join_summary["skipped_join_count"] == 0
    assert len(prepared.response.join_plan) == 4
    assert prepared.records[0]["orders_csv__customer_id"] == "C1"
    assert prepared.records[0]["customers_csv__segment"] == "A"
    assert prepared.records[0]["products_csv__category"] == "cat_a"
    assert prepared.records[0]["translation_csv__category_en"] == "A"
    assert prepared.response.column_source_map["translation_csv__category_en"] == "translation.csv"


def test_prepare_multi_dataset_context_skips_explosive_join(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    primary = repository.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    repeated = repository.create_dataset(name="events.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=primary.id,
        records=[{"order_id": "O1", "amount": 10}, {"order_id": "O2", "amount": 20}],
    )
    repository.append_raw_records(
        dataset_id=repeated.id,
        records=[{"order_id": "O1", "event": index} for index in range(25)],
    )
    plan = (
        DatasetJoinConfig(
            left_dataset_id=primary.id,
            right_dataset_id=repeated.id,
            left_column="order_id",
            right_column="order_id",
        ),
    )

    prepared = prepare_multi_dataset_context(
        repository,
        dataset_id=primary.id,
        additional_dataset_ids=(repeated.id,),
        join_plan=plan,
    )

    assert len(prepared.records) == 2
    assert prepared.response is not None
    assert prepared.response.join_summary["skipped_join_count"] == 1
    assert prepared.response.join_summary["joins"][0]["status"] == "skipped_row_expansion"
    assert any("超过安全上限" in issue.issue for issue in prepared.validation_issues)


def test_join_match_rate_is_directional_not_smaller_set_overlap(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    left = repository.create_dataset(name="events.csv", source_type="csv", source_metadata={})
    right = repository.create_dataset(name="lookup.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=left.id,
        records=[{"entity_id": f"E{index}"} for index in range(100)],
    )
    repository.append_raw_records(
        dataset_id=right.id,
        records=[{"entity_id": f"E{index}"} for index in range(10)],
    )

    response = suggest_dataset_joins(
        repository,
        dataset_id=left.id,
        additional_dataset_ids=(right.id,),
    )
    matching = next(
        candidate
        for candidate in response.suggestions
        if candidate.left_column == "entity_id" and candidate.right_column == "entity_id"
    )

    assert matching.estimated_match_rate == 0.1


def test_join_match_rate_is_stable_when_related_files_have_different_row_orders(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    orders = repository.create_dataset(name="orders.csv", source_type="csv", source_metadata={})
    payments = repository.create_dataset(name="payments.csv", source_type="csv", source_metadata={})
    order_ids = [f"O{index:04d}" for index in range(1200)]
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[{"order_id": order_id} for order_id in order_ids],
    )
    repository.append_raw_records(
        dataset_id=payments.id,
        records=[{"order_id": order_id, "payment_value": index} for index, order_id in enumerate(reversed(order_ids))],
    )

    response = suggest_dataset_joins(
        repository,
        dataset_id=payments.id,
        additional_dataset_ids=(orders.id,),
        records_by_dataset={
            payments.id: repository.sample_analysis_records(payments.id, limit=1000),
            orders.id: repository.sample_analysis_records(orders.id, limit=1000),
        },
    )
    matching = next(
        candidate
        for candidate in response.suggestions
        if candidate.left_column == "order_id" and candidate.right_column == "order_id"
    )

    assert matching.estimated_match_rate == 1.0


def test_delimited_list_foreign_key_is_detected_selected_and_joined(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    behavior = repository.create_dataset(
        name="user_behavior_data.txt",
        source_type="txt",
        source_metadata={},
    )
    products = repository.create_dataset(
        name="product_meta_data.txt",
        source_type="txt",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=behavior.id,
        records=[
            {"query": "q1", "candidate_wid_list": "P1_P2"},
            {"query": "q2", "candidate_wid_list": "P2_P3"},
        ],
    )
    repository.append_raw_records(
        dataset_id=products.id,
        records=[
            {"wid": "P1", "category": "A"},
            {"wid": "P2", "category": "B"},
            {"wid": "P3", "category": "C"},
        ],
    )
    group = repository.create_dataset_group(
        name="JDsearch",
        dataset_ids=(behavior.id, products.id),
    )

    class UnavailableRouter:
        def complete(self, **_kwargs):
            raise RuntimeError("offline")

    suggestions = suggest_dataset_group_relationships(
        repository,
        group_id=group.id,
        router=UnavailableRouter(),  # type: ignore[arg-type]
    )
    candidate = next(
        item
        for item in suggestions.candidates
        if item.left_column == "candidate_wid_list" and item.right_column == "wid"
    )
    assert candidate.estimated_match_rate == 1.0
    assert candidate.left_value_mode == "delimited"
    assert candidate.left_delimiter == "_"
    assert candidate.relationship_type == "many_to_one"

    selection = select_automatic_dataset_relationships(group, suggestions.candidates)
    assert selection.unresolved_dataset_ids == ()
    assert len(selection.relationships) == 1
    relationship = selection.relationships[0]
    assert relationship.left_value_mode == "delimited"

    prepared = prepare_multi_dataset_context(
        repository,
        dataset_id=selection.primary_dataset_id,
        additional_dataset_ids=(products.id,),
        join_plan=(
            DatasetJoinConfig.model_validate(
                relationship.model_dump(
                    include={
                        "left_dataset_id",
                        "right_dataset_id",
                        "left_column",
                        "right_column",
                        "join_type",
                        "left_value_mode",
                        "right_value_mode",
                        "left_delimiter",
                        "right_delimiter",
                    }
                )
            ),
        ),
    )
    assert len(prepared.records) == 4
    assert prepared.response is not None
    join = prepared.response.join_summary["joins"][0]
    assert join["status"] == "joined"
    assert join["left_value_mode"] == "delimited"
    assert join["unmatched_rows"] == 0


def test_relationship_plan_validation_rejects_bad_columns_and_non_root_jobs(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    first = repository.create_dataset(name="first.csv", source_type="csv", source_metadata={})
    second = repository.create_dataset(name="second.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(dataset_id=first.id, records=[{"first_id": "A"}])
    repository.append_raw_records(dataset_id=second.id, records=[{"second_id": "A"}])
    group = repository.create_dataset_group(name="validation", dataset_ids=(first.id, second.id))

    with pytest.raises(ValueError, match="column was not found"):
        repository.update_dataset_group_relationships(
            group_id=group.id,
            relationships=(
                {
                    "left_dataset_id": str(first.id),
                    "right_dataset_id": str(second.id),
                    "left_column": "missing_id",
                    "right_column": "second_id",
                    "join_type": "left",
                },
            ),
        )

    with pytest.raises(ValueError, match="root must match"):
        repository.create_analysis_job(
            dataset_id=second.id,
            additional_dataset_ids=(first.id,),
            join_plan=(
                {
                    "left_dataset_id": str(first.id),
                    "right_dataset_id": str(second.id),
                    "left_column": "first_id",
                    "right_column": "second_id",
                    "join_type": "left",
                },
            ),
            question="invalid root",
        )
