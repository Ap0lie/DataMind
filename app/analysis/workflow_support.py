from __future__ import annotations

import json
from typing import Any


def extract_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            nested = extract_json_object(payload)
            if nested is not None:
                return nested
    return None


def string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))


def float_payload(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def require[T](value: T | None, message: str) -> T:
    if value is None:
        raise RuntimeError(message)
    return value
