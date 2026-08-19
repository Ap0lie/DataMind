from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.tool_results.contracts import (
    CanonicalFact,
    ToolResultEnvelope,
    ToolResultKind,
    ToolResultStatus,
    ToolResultSummary,
)

_ROW_KEYS = ("rows", "records", "data", "sql_rows")
_ISSUE_KEYS = ("validation_issues", "issues", "errors")
_EVIDENCE_KEYS = ("evidence_id", "evidence_ids", "evidence_refs")


def infer_result_kind(tool_name: str, payload: Any, *, failed: bool = False) -> ToolResultKind:
    name = tool_name.lower()
    if failed or _contains_error(payload):
        return ToolResultKind.ERROR
    if "report" in name or _has_any(payload, ("structured_report", "executive_summary")):
        return ToolResultKind.REPORT
    if "python" in name or _has_any(payload, ("statistics", "python_result", "charts")):
        return ToolResultKind.PYTHON
    if "sql" in name:
        return ToolResultKind.SQL
    if _find_rows(payload) is not None:
        return ToolResultKind.TABLE
    if isinstance(payload, Mapping):
        return ToolResultKind.JSON
    return ToolResultKind.TEXT


def reduce_tool_result(
    envelope: ToolResultEnvelope,
    *,
    max_preview_rows: int = 20,
    max_facts: int = 80,
    max_text_chars: int = 4_000,
) -> ToolResultSummary:
    payload = envelope.payload
    failed = envelope.status == ToolResultStatus.FAILED
    kind = envelope.kind or infer_result_kind(envelope.tool_name, payload, failed=failed)
    rows = _find_rows(payload)
    preview = tuple(_bounded_row(item, max_text_chars=800) for item in (rows or [])[:max_preview_rows])
    schema_fields = _schema_fields(payload, rows)
    row_count = _row_count(payload, rows)
    evidence_ids = _collect_named_values(payload, _EVIDENCE_KEYS, limit=40)
    evidence_ids = tuple(dict.fromkeys((*envelope.evidence_ids, *evidence_ids)))
    validation_issues = _collect_named_values(payload, _ISSUE_KEYS, limit=30)
    error = _extract_error(payload, max_chars=12_000) if failed or kind == ToolResultKind.ERROR else None
    findings = _key_findings(payload, kind=kind, max_chars=max_text_chars)
    facts = tuple(_canonical_facts(payload, evidence_ids=evidence_ids, limit=max_facts))
    omitted: list[str] = []
    if rows is not None and len(rows) > len(preview):
        omitted.append(f"rows_after_{len(preview)}")
    if len(facts) >= max_facts:
        omitted.append("additional_scalar_facts")
    headline = _headline(
        envelope.tool_name,
        kind=kind,
        status=envelope.status,
        row_count=row_count,
        error=error,
    )
    return ToolResultSummary(
        tool_name=envelope.tool_name,
        kind=kind,
        status=envelope.status,
        headline=headline,
        schema_fields=schema_fields,
        row_count=row_count,
        canonical_facts=facts,
        preview=preview,
        key_findings=findings,
        evidence_ids=evidence_ids,
        validation_issues=validation_issues,
        error=error,
        omitted_sections=tuple(omitted),
    )


def summary_for_model(summary: ToolResultSummary) -> str:
    return json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def _headline(
    tool_name: str,
    *,
    kind: ToolResultKind,
    status: ToolResultStatus,
    row_count: int | None,
    error: str | None,
) -> str:
    if status == ToolResultStatus.FAILED:
        return f"{tool_name} failed: {(error or 'unknown error').splitlines()[0][:240]}"
    if row_count is not None:
        return f"{tool_name} produced {row_count} row(s) of {kind.value} evidence."
    return f"{tool_name} produced structured {kind.value} evidence."


def _find_rows(payload: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(payload, Mapping):
        return None
    for key in _ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], Mapping)):
            return value
    for key in ("result", "sql_result", "python_result"):
        nested = payload.get(key)
        found = _find_rows(nested)
        if found is not None:
            return found
    return None


def _row_count(payload: Any, rows: list[Mapping[str, Any]] | None) -> int | None:
    if isinstance(payload, Mapping):
        for key in ("total_rows", "row_count", "rows_count", "total_count"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return len(rows) if rows is not None else None


def _schema_fields(payload: Any, rows: list[Mapping[str, Any]] | None) -> tuple[str, ...]:
    fields: list[str] = []
    if isinstance(payload, Mapping):
        for key in ("columns", "schema_fields", "output_fields"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    name = item.get("name") if isinstance(item, Mapping) else item
                    if isinstance(name, str):
                        fields.append(name)
    if rows:
        fields.extend(str(key) for key in rows[0])
    return tuple(dict.fromkeys(fields))[:100]


def _canonical_facts(
    payload: Any,
    *,
    evidence_ids: tuple[str, ...],
    limit: int,
) -> Iterable[CanonicalFact]:
    evidence_id = evidence_ids[0] if len(evidence_ids) == 1 else None
    emitted = 0

    def visit(value: Any, path: str, depth: int) -> Iterable[CanonicalFact]:
        nonlocal emitted
        if emitted >= limit or depth > 4:
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and len(value) > 500:
                return
            emitted += 1
            yield CanonicalFact(
                path=path or "$",
                value=value,
                value_type=type(value).__name__,
                evidence_id=evidence_id,
            )
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in _ROW_KEYS or emitted >= limit:
                    continue
                yield from visit(nested, f"{path}.{key}" if path else str(key), depth + 1)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, nested in enumerate(value[:10]):
                if emitted >= limit:
                    break
                yield from visit(nested, f"{path}[{index}]", depth + 1)

    yield from visit(payload, "", 0)


def _key_findings(payload: Any, *, kind: ToolResultKind, max_chars: int) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        text = str(payload).strip()
        return (text[:max_chars],) if text else ()
    candidates: list[Any] = []
    for key in ("key_findings", "findings", "insights", "recommendations"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif value:
            candidates.append(value)
    if kind == ToolResultKind.REPORT:
        for key in ("executive_summary", "summary", "markdown"):
            if payload.get(key):
                candidates.append(payload[key])
    output: list[str] = []
    remaining = max_chars
    for item in candidates:
        if isinstance(item, Mapping):
            text = str(item.get("text") or item.get("finding") or item.get("title") or "")
        else:
            text = str(item)
        text = " ".join(text.split())
        if not text or remaining <= 0:
            continue
        clipped = text[: min(1_000, remaining)]
        output.append(clipped)
        remaining -= len(clipped)
        if len(output) >= 12:
            break
    return tuple(output)


def _extract_error(payload: Any, *, max_chars: int) -> str | None:
    if isinstance(payload, Mapping):
        for key in ("error", "message", "detail", "traceback"):
            value = payload.get(key)
            if value:
                return str(value)[:max_chars]
    text = str(payload).strip()
    return text[:max_chars] if text else None


def _collect_named_values(payload: Any, keys: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    output: list[str] = []

    def visit(value: Any, depth: int) -> None:
        if len(output) >= limit or depth > 5:
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in keys:
                    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                        output.extend(str(item) for item in nested[: limit - len(output)])
                    elif nested not in (None, ""):
                        output.append(str(nested))
                elif str(key) in _ROW_KEYS and isinstance(nested, list):
                    visit(nested[:20], depth + 1)
                else:
                    visit(nested, depth + 1)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value[:20]:
                visit(nested, depth + 1)

    visit(payload, 0)
    return tuple(dict.fromkeys(output))[:limit]


def _bounded_row(row: Mapping[str, Any], *, max_text_chars: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in list(row.items())[:40]:
        if isinstance(value, str) and len(value) > max_text_chars:
            output[str(key)] = value[:max_text_chars] + "..."
        elif isinstance(value, (Mapping, list, tuple)):
            output[str(key)] = json.dumps(value, ensure_ascii=False, default=str)[:max_text_chars]
        else:
            output[str(key)] = value
    return output


def _contains_error(payload: Any) -> bool:
    return isinstance(payload, Mapping) and bool(payload.get("error"))


def _has_any(payload: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(payload, Mapping) and any(key in payload for key in keys)
