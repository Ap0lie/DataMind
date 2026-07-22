from __future__ import annotations

from app.analysis.dataset_groups import (
    RELATIONSHIP_SAMPLE_LIMIT,
    suggest_dataset_group_relationships,
)
from app.storage.dataset_store import DatasetStoreRepository


def test_relationship_suggestions_use_bounded_samples_instead_of_full_records(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    customers = repository.create_dataset(
        name="customers.csv",
        source_type="csv",
        source_metadata={},
    )
    orders = repository.create_dataset(
        name="orders.csv",
        source_type="csv",
        source_metadata={},
    )
    records = [
        {"customer_id": f"C{index}", "segment": "A"}
        for index in range(RELATIONSHIP_SAMPLE_LIMIT + 25)
    ]
    repository.append_raw_records(dataset_id=customers.id, records=records)
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[
            {"order_id": f"O{index}", "customer_id": f"C{index}"}
            for index in range(len(records))
        ],
    )
    group = repository.create_dataset_group(
        name="orders and customers",
        dataset_ids=(orders.id, customers.id),
    )

    def fail_on_full_read(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("relationship inference must not materialise every record")

    repository.read_analysis_records = fail_on_full_read  # type: ignore[method-assign]
    suggestions = suggest_dataset_group_relationships(repository, group_id=group.id)

    assert repository.count_analysis_records(customers.id) == len(records)
    assert (
        len(repository.sample_analysis_records(customers.id, limit=RELATIONSHIP_SAMPLE_LIMIT))
        == RELATIONSHIP_SAMPLE_LIMIT
    )
    assert suggestions.candidates
    assert suggestions.compact_context["tables"][0]["row_count"] == len(records)
    assert len(suggestions.compact_context["tables"][0]["preview_records"]) <= 5
