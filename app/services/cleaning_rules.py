from __future__ import annotations

from datetime import datetime
from typing import Any


def apply_cleaning_rules(
    records: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    output = [dict(record) for record in records]
    issues: list[str] = []
    applied: list[dict[str, Any]] = []
    for index, rule in enumerate(rules, 1):
        if not rule.get("enabled", True):
            continue
        before = output
        try:
            output = _apply_rule(output, rule)
        except ValueError as exc:
            issues.append(f"规则 {index} 失败: {exc}")
            output = before
            continue
        applied.append(rule)
    return output, issues, applied


def _apply_rule(records: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    rule_type = str(rule.get("rule_type") or "")
    column = _optional_text(rule.get("column"))
    if rule_type == "drop_duplicates":
        return _drop_duplicates(records)
    if rule_type in {"fill_missing", "rename_column", "convert_type", "trim_text", "drop_column", "filter_rows"} and not column:
        raise ValueError("缺少 column。")
    if rule_type == "fill_missing":
        return _fill_missing(records, column or "", rule)
    if rule_type == "rename_column":
        return _rename_column(records, column or "", rule)
    if rule_type == "convert_type":
        return _convert_type(records, column or "", rule)
    if rule_type == "trim_text":
        return [
            {key: (value.strip() if key == column and isinstance(value, str) else value) for key, value in record.items()}
            for record in records
        ]
    if rule_type == "drop_column":
        return [{key: value for key, value in record.items() if key != column} for record in records]
    if rule_type == "filter_rows":
        return _filter_rows(records, column or "", rule)
    raise ValueError(f"不支持的规则类型: {rule_type}")


def _drop_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        key = tuple(sorted((str(name), str(value)) for name, value in record.items()))
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def _fill_missing(
    records: list[dict[str, Any]],
    column: str,
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    strategy = str(rule.get("strategy") or "value")
    fill_value = rule.get("value")
    output: list[dict[str, Any]] = []
    for record in records:
        next_record = dict(record)
        if _is_blank(next_record.get(column)):
            if strategy == "drop_row":
                continue
            if strategy == "empty_string":
                next_record[column] = ""
            elif strategy == "zero":
                next_record[column] = 0
            else:
                next_record[column] = fill_value
        output.append(next_record)
    return output


def _rename_column(
    records: list[dict[str, Any]],
    column: str,
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    new_name = _optional_text(rule.get("new_name"))
    if not new_name:
        raise ValueError("缺少 new_name。")
    output: list[dict[str, Any]] = []
    for record in records:
        next_record: dict[str, Any] = {}
        for key, value in record.items():
            next_record[new_name if key == column else key] = value
        output.append(next_record)
    return output


def _convert_type(
    records: list[dict[str, Any]],
    column: str,
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    target_type = str(rule.get("target_type") or "text")
    output: list[dict[str, Any]] = []
    for row_index, record in enumerate(records, 1):
        next_record = dict(record)
        value = next_record.get(column)
        if _is_blank(value):
            output.append(next_record)
            continue
        try:
            next_record[column] = _convert_value(value, target_type)
        except ValueError as exc:
            raise ValueError(f"第 {row_index} 行字段 {column} 转换失败: {exc}") from exc
        output.append(next_record)
    return output


def _filter_rows(
    records: list[dict[str, Any]],
    column: str,
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    operator = str(rule.get("operator") or "equals")
    mode = str(rule.get("mode") or "keep")
    expected = rule.get("value")
    output: list[dict[str, Any]] = []
    for record in records:
        matched = _match(record.get(column), operator, expected)
        if (mode == "keep" and matched) or (mode == "delete" and not matched):
            output.append(record)
    return output


def _match(value: Any, operator: str, expected: Any) -> bool:
    if operator == "blank":
        return _is_blank(value)
    if operator == "not_blank":
        return not _is_blank(value)
    if operator == "equals":
        return str(value) == str(expected)
    if operator == "not_equals":
        return str(value) != str(expected)
    if operator == "contains":
        return str(expected) in str(value)
    if operator == "not_contains":
        return str(expected) not in str(value)
    left = _number_or_none(value)
    right = _number_or_none(expected)
    if left is None or right is None:
        return False
    if operator == "gt":
        return left > right
    if operator == "gte":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "lte":
        return left <= right
    return False


def _convert_value(value: Any, target_type: str) -> Any:
    if target_type == "text":
        return str(value)
    if target_type == "number" or target_type == "float":
        return float(str(value).replace(",", ""))
    if target_type == "integer":
        return int(float(str(value).replace(",", "")))
    if target_type == "boolean":
        lowered = str(value).strip().lower()
        if lowered in {"true", "1", "yes", "y", "是"}:
            return True
        if lowered in {"false", "0", "no", "n", "否"}:
            return False
        raise ValueError(f"无法转为 boolean: {value}")
    if target_type == "date":
        return datetime.fromisoformat(str(value).replace("/", "-")).date().isoformat()
    return value


def _number_or_none(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""
