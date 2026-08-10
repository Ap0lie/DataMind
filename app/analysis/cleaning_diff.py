from __future__ import annotations

import json
import math
from difflib import SequenceMatcher
from typing import Any


def build_cleaning_diff_summary(
    *,
    raw_records: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    sample_limit: int = 20,
) -> dict[str, Any]:
    previous = previous_records or raw_records
    pairs, removed, added = _align_records(previous, current_records)
    samples: list[dict[str, Any]] = []
    changed_rows = 0
    changed_cells = 0

    for before_index, after_index in pairs:
        changes = _cell_changes(previous[before_index], current_records[after_index])
        if not changes:
            continue
        changed_rows += 1
        changed_cells += len(changes)
        _add_sample(
            samples,
            sample_limit,
            {
                "change_type": "modified",
                "row_number": before_index + 1,
                "current_row_number": after_index + 1,
                "changes": changes,
            },
        )

    for index in removed:
        _add_sample(
            samples,
            sample_limit,
            {"change_type": "removed", "row_number": index + 1, "before": previous[index]},
        )
    for index in added:
        _add_sample(
            samples,
            sample_limit,
            {
                "change_type": "added",
                "current_row_number": index + 1,
                "after": current_records[index],
            },
        )

    previous_columns = _columns(previous)
    current_columns = _columns(current_records)
    return {
        "raw_row_count": len(raw_records),
        "previous_row_count": len(previous),
        "current_row_count": len(current_records),
        "added_rows": len(added),
        "removed_rows": len(removed),
        "changed_rows": changed_rows,
        "added_columns": sorted(current_columns - previous_columns),
        "removed_columns": sorted(previous_columns - current_columns),
        "changed_cells": changed_cells,
        "raw_missing_count": _missing_count(raw_records),
        "previous_missing_count": _missing_count(previous),
        "current_missing_count": _missing_count(current_records),
        "sample_diffs": samples,
    }


def _align_records(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    identity = _identity_column(previous, current)
    if identity:
        before = {_identity_value(row.get(identity)): index for index, row in enumerate(previous)}
        after = {_identity_value(row.get(identity)): index for index, row in enumerate(current)}
        common = before.keys() & after.keys()
        pairs = sorted(((before[key], after[key]) for key in common), key=lambda item: item[0])
        return (
            pairs,
            sorted(before[key] for key in before.keys() - after.keys()),
            sorted(after[key] for key in after.keys() - before.keys()),
        )

    matcher = SequenceMatcher(
        a=[_row_fingerprint(row) for row in previous],
        b=[_row_fingerprint(row) for row in current],
        autojunk=False,
    )
    pairs: list[tuple[int, int]] = []
    removed: list[int] = []
    added: list[int] = []
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(range(before_start, before_end), range(after_start, after_end), strict=True))
        elif tag == "delete":
            removed.extend(range(before_start, before_end))
        elif tag == "insert":
            added.extend(range(after_start, after_end))
        else:
            pair_count = min(before_end - before_start, after_end - after_start)
            pairs.extend((before_start + offset, after_start + offset) for offset in range(pair_count))
            removed.extend(range(before_start + pair_count, before_end))
            added.extend(range(after_start + pair_count, after_end))
    return pairs, removed, added


def _identity_column(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> str | None:
    candidates = sorted(
        (
            column
            for column in _columns(previous) & _columns(current)
            if column.casefold() == "id"
            or column.casefold().endswith("_id")
            or column.casefold().startswith("id_")
            or column.casefold() == "uuid"
        ),
        key=lambda column: (column.casefold() != "id", len(column), column),
    )
    for column in candidates:
        before = [_identity_value(row.get(column)) for row in previous]
        after = [_identity_value(row.get(column)) for row in current]
        values = (*before, *after)
        if (
            all(value is not None for value in values)
            and len(before) == len(set(before))
            and len(after) == len(set(after))
        ):
            return column
    return None


def _cell_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        column: {"before": before.get(column), "after": after.get(column)}
        for column in sorted(set(before) | set(after))
        if not _values_equal(before.get(column), after.get(column))
    }


def _values_equal(left: Any, right: Any) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _row_fingerprint(row: dict[str, Any]) -> str:
    return json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if _is_missing(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _identity_value(value: Any) -> str | None:
    return None if _is_missing(value) else str(value).strip().casefold()


def _columns(records: list[dict[str, Any]]) -> set[str]:
    return {str(key) for record in records for key in record}


def _missing_count(records: list[dict[str, Any]]) -> int:
    return sum(_is_missing(value) for record in records for value in record.values())


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _add_sample(samples: list[dict[str, Any]], limit: int, sample: dict[str, Any]) -> None:
    if len(samples) < limit:
        samples.append(sample)
