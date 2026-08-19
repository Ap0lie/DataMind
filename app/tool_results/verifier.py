from __future__ import annotations

import re
from collections.abc import Iterable

from app.tool_results.contracts import ToolResultEnvelope, ToolResultSummary

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:%|‰)?")


def verify_tool_result_summary(
    envelope: ToolResultEnvelope,
    summary: ToolResultSummary,
) -> tuple[bool, tuple[str, ...]]:
    issues: list[str] = []
    if summary.tool_name != envelope.tool_name:
        issues.append("tool_name_mismatch")
    if summary.status != envelope.status:
        issues.append("status_mismatch")
    missing_evidence = set(envelope.evidence_ids) - set(summary.evidence_ids)
    if missing_evidence:
        issues.append("missing_evidence_ids")
    if envelope.status.value == "failed" and not summary.error:
        issues.append("missing_error")
    return not issues, tuple(issues)


def verify_generated_summary_text(
    texts: Iterable[str],
    *,
    source_text: str,
    source_quotes: Iterable[str] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Reject unsupported numeric claims and unverifiable source quotations."""

    issues: list[str] = []
    normalized_source = _normalize(source_text)
    normalized_quotes: list[str] = []
    for quote in source_quotes:
        normalized = _normalize(quote)
        if not normalized or normalized not in normalized_source:
            issues.append("source_quote_not_found")
            continue
        normalized_quotes.append(normalized)
    supported_numbers = set(_NUMBER_RE.findall(normalized_source))
    for text in texts:
        for number in _NUMBER_RE.findall(text):
            if number not in supported_numbers:
                issues.append("unsupported_numeric_claim")
                break
    if any(str(text).strip() for text in texts) and not normalized_quotes:
        issues.append("missing_source_quote")
    return not issues, tuple(dict.fromkeys(issues))


def _normalize(value: str) -> str:
    return " ".join(str(value).split())
