from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.analysis.data_cleaning import DataCleaningService
from app.data_reliability.drift import DataDriftService, compare_snapshots
from app.storage.dataset_store import DatasetStoreRepository


def test_compare_snapshots_detects_schema_and_distribution_drift() -> None:
    previous = {
        "row_count": 100,
        "columns": [
            {
                "name": "amount",
                "dtype": "number",
                "missing_rate": 0.0,
                "unique_rate": 0.8,
                "mean": 10.0,
                "std": 2.0,
                "value_signature": ["a", "b"],
            },
            {
                "name": "region",
                "dtype": "text",
                "missing_rate": 0.0,
                "unique_rate": 0.1,
                "value_signature": ["c"],
            },
        ],
    }
    current = {
        "row_count": 150,
        "columns": [
            {
                "name": "sales_amount",
                "dtype": "number",
                "missing_rate": 0.0,
                "unique_rate": 0.8,
                "mean": 10.0,
                "std": 2.0,
                "value_signature": ["a", "b"],
            },
            {
                "name": "region",
                "dtype": "text",
                "missing_rate": 0.4,
                "unique_rate": 0.5,
                "value_signature": ["c"],
            },
        ],
    }

    changes = compare_snapshots(previous, current)
    change_types = {item.change_type for item in changes}

    assert "column_renamed" in change_types
    assert "missing_rate_drift" in change_types
    assert "unique_rate_drift" in change_types
    assert "row_count_drift" in change_types
    assert all("Column " not in item.message for item in changes)
    assert next(item for item in changes if item.change_type == "row_count_drift").message == (
        "数据行数从 100 变为 150。"
    )
    assert next(item for item in changes if item.change_type == "unique_rate_drift").message == (
        "字段 region 的唯一率变化了 40%。"
    )


def test_schema_drift_marks_semantic_model_and_report_stale(tmp_path: Path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"region": "A", "amount": 10}, {"region": "B", "amount": 20}],
    )
    model = repository.save_semantic_model(
        {
            "scope_type": "dataset",
            "scope_id": dataset.id,
            "name": "Sales",
            "status": "published",
            "definition": {},
            "schema_fingerprint": "baseline",
        }
    )
    report_id = repository.save_report(
        dataset_id=dataset.id,
        title="Sales report",
        markdown="# Sales",
        metadata={"question": "How are sales?"},
        job_id=uuid4(),
    )

    repository.replace_raw_record_batches(
        dataset_id=dataset.id,
        batches=iter(
            ([{"region": "A", "sales_amount": 10}, {"region": "B", "sales_amount": 20}],)
        ),
    )

    event = repository.latest_data_drift_event(dataset.id)
    assert event is not None
    assert event["status"] == "critical"
    assert repository.get_semantic_model(model["id"])["status"] == "stale"
    report = repository.get_report(report_id)
    assert report["metadata"]["freshness_status"] == "stale"
    assert report["metadata"]["drift_event_id"] == str(event["id"])


def test_relationship_match_drift_marks_group_edge_stale(tmp_path: Path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    orders = repository.create_dataset(
        name="orders.csv",
        source_type="csv",
        source_metadata={},
    )
    customers = repository.create_dataset(
        name="customers.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=orders.id,
        records=[{"customer_id": value} for value in ("A", "B", "C")],
    )
    repository.append_raw_records(
        dataset_id=customers.id,
        records=[{"customer_id": value} for value in ("A", "B", "C")],
    )
    group = repository.create_dataset_group(
        name="Commerce",
        dataset_ids=(orders.id, customers.id),
    )
    repository.update_dataset_group_relationships(
        group_id=group.id,
        relationships=(
            {
                "left_dataset_id": orders.id,
                "right_dataset_id": customers.id,
                "left_column": "customer_id",
                "right_column": "customer_id",
                "join_type": "left",
                "relationship_type": "many_to_one",
                "enabled": True,
            },
        ),
    )
    service = DataDriftService(repository)
    service.refresh_group_relationships(group.id)

    repository.replace_raw_record_batches(
        dataset_id=customers.id,
        batches=iter(([{"customer_id": value} for value in ("X", "Y", "Z")],)),
    )

    relationship = repository.get_dataset_group(group.id).relationships[0]
    assert relationship["baseline_match_rate"] == 1.0
    assert relationship["last_match_rate"] == 0.0
    assert relationship["freshness_status"] == "stale"
    assert relationship["drift_event_id"]


def test_raw_to_cleaned_transition_establishes_comparable_baseline(tmp_path: Path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="olist_orders_dataset.csv",
        source_type="csv",
        source_metadata={},
    )
    raw_records = [
        {
            "order_id": f"order-{index}",
            "order_purchase_timestamp": f"2017-10-{(index % 28) + 1:02d} 10:56:{index % 60:02d}",
            "order_approved_at": f"2017-10-{(index % 28) + 1:02d} 11:04:{index % 60:02d}",
            "order_delivered_carrier_date": f"2017-10-{(index % 28) + 2:02d} 15:30:{index % 60:02d}",
            "order_delivered_customer_date": f"2017-10-{(index % 28) + 3:02d} 18:20:{index % 60:02d}",
            "order_estimated_delivery_date": f"2017-11-{(index % 28) + 1:02d} 00:00:00",
        }
        for index in range(100)
    ]
    repository.append_raw_records(dataset_id=dataset.id, records=raw_records)
    cleaned = DataCleaningService().clean(
        dataset_id=dataset.id,
        records=raw_records,
        requirement="conservative cleaning",
        use_llm=False,
    )

    repository.save_cleaned_records(dataset_id=dataset.id, records=cleaned.records)

    status = DataDriftService(repository).latest_dataset_status(dataset.id)
    assert status.status == "baseline"
    assert status.snapshot.source == "cleaned"
    assert status.event_id is None
    assert repository.latest_data_drift_event(dataset.id) is None
    purchase = next(
        column
        for column in status.snapshot.columns
        if column.name == "order_purchase_timestamp"
    )
    assert purchase.unique_rate == 1.0
    assert ":56:" in cleaned.records[0]["order_purchase_timestamp"]


def test_source_baseline_does_not_reuse_an_unrelated_older_drift_event(
    tmp_path: Path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"region": "A", "amount": 10}, {"region": "B", "amount": 20}],
    )
    repository.replace_raw_record_batches(
        dataset_id=dataset.id,
        batches=iter(
            ([{"region": "A", "sales_amount": 10}, {"region": "B", "sales_amount": 20}],)
        ),
    )
    assert repository.latest_data_drift_event(dataset.id) is not None

    repository.save_cleaned_records(
        dataset_id=dataset.id,
        records=[{"region": "A", "sales_amount": 10}, {"region": "B", "sales_amount": 20}],
    )

    status = DataDriftService(repository).latest_dataset_status(dataset.id)
    assert status.status == "baseline"
    assert status.snapshot.source == "cleaned"
    assert status.event_id is None
    assert status.changes == ()


def test_cross_source_temporal_precision_warning_remains_visible(
    tmp_path: Path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="olist_orders_dataset.csv",
        source_type="csv",
        source_metadata={},
    )
    rows = [
        {
            "order_id": f"order-{index}",
            "order_purchase_timestamp": f"2017-10-01 10:56:{index:02d}",
        }
        for index in range(10)
    ]
    repository.append_raw_records(dataset_id=dataset.id, records=rows)
    raw_snapshot = repository.latest_data_snapshot(dataset.id, source="raw")
    assert raw_snapshot is not None

    repository.save_cleaned_records(dataset_id=dataset.id, records=rows)
    cleaned_snapshot = repository.latest_data_snapshot(dataset.id, source="cleaned")
    assert cleaned_snapshot is not None
    repository.save_data_drift_event(
        {
            "id": uuid4(),
            "dataset_id": dataset.id,
            "baseline_snapshot_id": raw_snapshot["id"],
            "current_snapshot_id": cleaned_snapshot["id"],
            "status": "critical",
            "changes": [
                {
                    "change_type": "unique_rate_drift",
                    "severity": "warning",
                    "field": "order_purchase_timestamp",
                    "previous_value": 1.0,
                    "current_value": 0.2,
                    "score": 0.8,
                    "message": "时间戳唯一率因清洗丢失时分秒而下降。",
                },
                {
                    "change_type": "column_removed",
                    "severity": "critical",
                    "field": "legacy_code",
                    "previous_field": "legacy_code",
                    "message": "字段 legacy_code 已删除。",
                },
            ],
            "affected_assets": [
                {
                    "asset_type": "report",
                    "asset_id": "unrelated-report",
                    "status": "stale",
                    "reason": "由完整历史事件失效。",
                }
            ],
            "recommended_actions": [
                {
                    "action": "review_schema",
                    "label": "审查字段变化",
                    "reason": "字段已删除。",
                    "requires_authorization": True,
                },
                {
                    "action": "run_cleaning",
                    "label": "重新清洗",
                    "reason": "时间精度下降。",
                    "requires_authorization": True,
                },
            ],
        }
    )
    group = repository.create_dataset_group(
        name="Olist",
        dataset_ids=(dataset.id,),
    )

    service = DataDriftService(repository)
    latest = service.latest_dataset_status(dataset.id)
    rescanned = service.scan_dataset(dataset.id)
    group_status = service.latest_group_status(group.id)

    assert latest.status == "warning"
    assert latest.event_id is not None
    assert len(latest.changes) == 1
    assert latest.changes[0].field == "order_purchase_timestamp"
    assert latest.affected_assets == ()
    assert [action.action for action in latest.recommended_actions] == ["run_cleaning"]
    assert rescanned.status == "warning"
    assert rescanned.event_id == latest.event_id
    assert rescanned.changes == latest.changes
    assert group_status.status == "warning"
    assert group_status.datasets[0].changes == latest.changes

    repository.save_data_drift_event(
        {
            "id": uuid4(),
            "dataset_id": dataset.id,
            "baseline_snapshot_id": raw_snapshot["id"],
            "current_snapshot_id": cleaned_snapshot["id"],
            "status": "warning",
            "changes": [item.model_dump(mode="json") for item in latest.changes],
            "affected_assets": [],
            "recommended_actions": [],
            "created_at": "9999-01-01T00:00:00+00:00",
            "acknowledged_at": "9999-01-01T00:00:01+00:00",
        }
    )
    acknowledged = service.latest_dataset_status(dataset.id)
    assert acknowledged.status == "baseline"
    assert acknowledged.changes == ()


def test_cleaned_snapshots_remain_comparable_after_the_cleaned_baseline(
    tmp_path: Path,
) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"region": "A", "amount": 10}, {"region": "B", "amount": 20}],
    )
    repository.save_cleaned_records(
        dataset_id=dataset.id,
        records=[{"region": "A", "amount": 10}, {"region": "B", "amount": 20}],
    )

    repository.save_cleaned_records(
        dataset_id=dataset.id,
        records=[{"region": "A", "sales_amount": 10}, {"region": "B", "sales_amount": 20}],
    )

    status = DataDriftService(repository).latest_dataset_status(dataset.id)
    assert status.status == "critical"
    assert status.snapshot.source == "cleaned"
    assert status.event_id is not None
    assert "column_renamed" in {change.change_type for change in status.changes}
