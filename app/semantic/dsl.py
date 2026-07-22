from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
AGGREGATES = {"sum", "avg", "min", "max", "count", "count_distinct"}
BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
FUNCTIONS = {"coalesce", "nullif", "abs", "round", "date_diff", "date_trunc"}
FILTERS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "between", "is_null", "not_null", "contains"}


class SemanticDslError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledExpression:
    sql: str
    fields: tuple[str, ...]


def compile_expression(
    expression: dict[str, Any],
    *,
    metric_definitions: dict[str, dict[str, Any]] | None = None,
    stack: tuple[str, ...] = (),
    field_resolver: Callable[[str, str], tuple[str, str]] | None = None,
) -> CompiledExpression:
    metrics = metric_definitions or {}
    fields: list[str] = []

    def visit(node: Any) -> str:
        if not isinstance(node, dict):
            raise SemanticDslError("Every DSL node must be an object.")
        op = str(node.get("op") or "").lower()
        if op == "field":
            if node.get("entity_id") and node.get("field_id"):
                if field_resolver is None:
                    raise SemanticDslError("DSL v2 field references require a semantic field resolver.")
                entity, field = field_resolver(str(node["entity_id"]), str(node["field_id"]))
            else:
                entity = _identifier(node.get("entity"), "entity")
                field = _source_identifier(node.get("field"), "field")
            reference = f"{entity}.{field}"
            fields.append(reference)
            return f"{quote_identifier(entity)}.{quote_identifier(field)}"
        if op == "literal":
            return _literal(node.get("value"))
        if op in AGGREGATES:
            if op == "count" and node.get("expr") is None:
                return "COUNT(*)"
            inner = visit(node.get("expr"))
            function = "COUNT" if op == "count_distinct" else op.upper()
            distinct = "DISTINCT " if op == "count_distinct" else ""
            sql = f"{function}({distinct}{inner})"
            filters = node.get("filters") or []
            if filters:
                sql += " FILTER (WHERE " + compile_filters(filters, visit=visit) + ")"
            return sql
        if op in BINARY:
            left = visit(node.get("left"))
            right = visit(node.get("right"))
            if op == "divide":
                right = f"NULLIF({right}, 0)"
            return f"({left} {BINARY[op]} {right})"
        if op == "case":
            branches = node.get("when") or []
            if not isinstance(branches, list) or not branches:
                raise SemanticDslError("case requires at least one when branch.")
            pieces = []
            for branch in branches:
                if not isinstance(branch, dict):
                    raise SemanticDslError("case branches must be objects.")
                pieces.append(f"WHEN {compile_filters([branch.get('condition')], visit=visit)} THEN {visit(branch.get('then'))}")
            return "CASE " + " ".join(pieces) + f" ELSE {visit(node.get('else', {'op': 'literal', 'value': None}))} END"
        if op in FUNCTIONS:
            args = node.get("args") or []
            if not isinstance(args, list):
                raise SemanticDslError(f"{op} args must be a list.")
            return f"{op.upper()}(" + ", ".join(visit(item) for item in args) + ")"
        if op == "metric_ref":
            metric_id = str(node.get("metric_id") or "")
            if metric_id not in metrics:
                raise SemanticDslError(f"Unknown metric_ref: {metric_id}")
            if metric_id in stack:
                raise SemanticDslError("Metric dependency cycle: " + " -> ".join((*stack, metric_id)))
            return compile_expression(
                metrics[metric_id]["formula"],
                metric_definitions=metrics,
                stack=(*stack, metric_id),
                field_resolver=field_resolver,
            ).sql
        raise SemanticDslError(f"Unsupported DSL operation: {op or '<missing>'}")

    sql = visit(expression)
    return CompiledExpression(sql=sql, fields=tuple(dict.fromkeys(fields)))


def compile_filters(filters: Iterable[Any], *, visit: Any) -> str:
    parts: list[str] = []
    symbols = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    for item in filters:
        if not isinstance(item, dict):
            raise SemanticDslError("Filters must be objects.")
        op = str(item.get("op") or "").lower()
        if op not in FILTERS:
            raise SemanticDslError(f"Unsupported filter operation: {op}")
        left = visit(item.get("left"))
        if op in symbols:
            parts.append(f"{left} {symbols[op]} {visit(item.get('right'))}")
        elif op in {"in", "not_in"}:
            values = item.get("values") or []
            if not isinstance(values, list) or not values:
                raise SemanticDslError(f"{op} requires values.")
            parts.append(f"{left} {'NOT IN' if op == 'not_in' else 'IN'} (" + ", ".join(visit({"op": "literal", "value": value}) for value in values) + ")")
        elif op == "between":
            parts.append(f"{left} BETWEEN {visit(item.get('lower'))} AND {visit(item.get('upper'))}")
        elif op in {"is_null", "not_null"}:
            parts.append(f"{left} IS {'NOT ' if op == 'not_null' else ''}NULL")
        else:
            parts.append(f"CAST({left} AS VARCHAR) LIKE '%' || {visit(item.get('right'))} || '%'")
    if not parts:
        raise SemanticDslError("At least one filter is required.")
    return " AND ".join(f"({part})" for part in parts)


def _identifier(value: Any, label: str) -> str:
    text = str(value or "")
    if not IDENTIFIER.fullmatch(text):
        raise SemanticDslError(f"Invalid {label} identifier: {text!r}")
    return text


def _source_identifier(value: Any, label: str) -> str:
    text = str(value or "")
    if not text or len(text) > 512 or any(ord(character) < 32 for character in text):
        raise SemanticDslError(f"Invalid {label} identifier: {text!r}")
    return text


def quote_identifier(value: str) -> str:
    text = _source_identifier(value, "SQL")
    return '"' + text.replace('"', '""') + '"'


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"
