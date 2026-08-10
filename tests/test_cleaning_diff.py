from __future__ import annotations

import pytest

from app.analysis.cleaning_diff import build_cleaning_diff_summary

pytestmark = pytest.mark.unit


def test_cleaning_diff_aligns_rows_by_stable_identifier() -> None:
    before = [
        {"order_id": "A", "amount": "10"},
        {"order_id": "B", "amount": "20"},
        {"order_id": "C", "amount": "30"},
    ]
    after = [
        {"order_id": "A", "amount": 10},
        {"order_id": "C", "amount": 35},
    ]

    diff = build_cleaning_diff_summary(
        raw_records=before,
        previous_records=before,
        current_records=after,
    )

    assert diff["removed_rows"] == 1
    assert diff["added_rows"] == 0
    assert diff["changed_rows"] == 2
    assert diff["changed_cells"] == 2
    assert {item["change_type"] for item in diff["sample_diffs"]} == {
        "modified",
        "removed",
    }
    removed = next(item for item in diff["sample_diffs"] if item["change_type"] == "removed")
    assert removed["before"]["order_id"] == "B"


def test_cleaning_diff_reports_modified_cell_samples() -> None:
    diff = build_cleaning_diff_summary(
        raw_records=[{"name": " Alice ", "score": 1}],
        previous_records=[],
        current_records=[{"name": "Alice", "score": 1}],
    )

    assert diff["changed_rows"] == 1
    assert diff["changed_cells"] == 1
    assert diff["sample_diffs"][0]["changes"]["name"] == {
        "before": " Alice ",
        "after": "Alice",
    }
