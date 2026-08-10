from __future__ import annotations

from typing import Any

_RELIABILITY_PRIORITY = {
    "rejected": 0,
    "warning": 1,
    "unverified": 2,
    "verified": 3,
}
_DEFAULT_SUMMARIES = {
    "rejected": "统计审查未通过，结论不可作为可靠业务依据。",
    "warning": "统计审查存在警告。",
    "unverified": "未提供统计审查状态。",
    "verified": "DataMind 统计审查已通过。",
}
_TRUNCATION_MARKER = "…（摘要已截断）"
_NUMERIC_TOKEN_CHARACTERS = frozenset("0123456789,.，．%％+-−/")


def report_reliability(value: Any) -> dict[str, str]:
    normalized = _normalized_reliability(value)
    if normalized is not None:
        return normalized
    return {"status": "unverified", "summary": _DEFAULT_SUMMARIES["unverified"]}


def canonical_reliability(*values: Any) -> dict[str, str]:
    """Return the least trustworthy explicit status in one evidence lineage."""

    candidates = [
        normalized
        for value in values
        if (normalized := _normalized_reliability(value)) is not None
    ]
    if not candidates:
        return report_reliability(None)
    return min(candidates, key=lambda item: _RELIABILITY_PRIORITY[item["status"]])


def safe_excerpt(value: Any, *, limit: int = 320) -> str:
    """Build a bounded preview without cutting a sentence, number, or ASCII token."""

    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    if limit <= 0:
        return ""
    if limit <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[:limit]

    cutoff = limit - len(_TRUNCATION_MARKER)
    sentence_end = max(
        (normalized.rfind(character, 0, cutoff + 1) for character in "。！？!?；;"),
        default=-1,
    )
    if sentence_end >= max(1, cutoff // 2):
        cutoff = sentence_end + 1
    elif _splits_numeric_token(normalized, cutoff):
        while cutoff > 0 and normalized[cutoff - 1] in _NUMERIC_TOKEN_CHARACTERS:
            cutoff -= 1
    elif _splits_ascii_token(normalized, cutoff):
        while cutoff > 0 and _is_ascii_token_character(normalized[cutoff - 1]):
            cutoff -= 1

    prefix = normalized[:cutoff].rstrip(" \t,，.;；:")
    return f"{prefix}{_TRUNCATION_MARKER}"


def _normalized_reliability(value: Any) -> dict[str, str] | None:
    verification = value if isinstance(value, dict) else {}
    status = str(verification.get("status") or "").lower()
    requires_replan = bool(verification.get("requires_replan"))
    summary = str(verification.get("summary") or "").strip()
    if status == "failed" or status == "rejected" or requires_replan:
        normalized_status = "rejected"
    elif status == "passed" or status == "verified":
        normalized_status = "verified"
    elif status == "warning":
        normalized_status = "warning"
    elif status == "unverified":
        normalized_status = "unverified"
    else:
        return None
    return {
        "status": normalized_status,
        "summary": summary or _DEFAULT_SUMMARIES[normalized_status],
    }


def _splits_numeric_token(value: str, cutoff: int) -> bool:
    if cutoff <= 0 or cutoff >= len(value):
        return False
    left = value[cutoff - 1]
    right = value[cutoff]
    return (
        left in _NUMERIC_TOKEN_CHARACTERS
        and right in _NUMERIC_TOKEN_CHARACTERS
        and (left.isdigit() or right.isdigit())
    )


def _splits_ascii_token(value: str, cutoff: int) -> bool:
    if cutoff <= 0 or cutoff >= len(value):
        return False
    return _is_ascii_token_character(value[cutoff - 1]) and _is_ascii_token_character(
        value[cutoff]
    )


def _is_ascii_token_character(value: str) -> bool:
    return value.isascii() and (value.isalnum() or value in "_-./")
