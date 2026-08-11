from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.settings import Settings

ContextReducer = Callable[[tuple[dict[str, Any], ...], int], tuple[dict[str, Any], ...]]

_STAGE_TOKEN_LIMITS = {
    "cleaning_decide": 4_096,
    "cleaning_execute": 12_288,
    "cleaning": 12_288,
    "planner": 16_384,
    "sql": 16_384,
    "python": 16_384,
    "analysis": 16_384,
    "review": 32_768,
    "report": 32_768,
    "assistant_route": 24_576,
    "assistant_compose": 40_960,
    "assistant": 24_576,
    "default": 16_384,
}

_AGENT_PROFILES = {
    "cleaning_decide": "cleaning_decide",
    "cleaning_execute": "cleaning_execute",
    "planner": "planner",
    "design_framework": "planner",
    "round_plan": "planner",
    "agent_loop": "planner",
    "sql": "sql",
    "python": "python",
    "round_python": "python",
    "python_charts": "python",
    "round_python_charts": "python",
    "reflection": "review",
    "integrate": "review",
    "review": "review",
    "chart_refine": "review",
    "report_decide": "report",
    "report_execute": "report",
    "report": "report",
    "assistant": "assistant_route",
}

_REQUIRED_TEXT_KEYS = {
    "question",
    "requirement",
    "analysis_contract",
    "contract",
    "output_contract",
    "error",
    "errors",
    "validation_error",
}
_CODE_KEYS = {"code", "generated_code", "python_code"}
_EVIDENCE_KEYS = {"evidence", "citations", "evidence_items"}
_JSON_PUNCTUATION = frozenset("{}[],:\\\"")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ERROR_LINE_RE = re.compile(r"(?:line|第)\s*(\d+)", re.IGNORECASE)


class ContextBudgetExceeded(ValueError):
    """Raised before provider invocation when required context cannot fit."""


@dataclass(frozen=True)
class ContextBudget:
    input_tokens: int
    max_chars: int
    output_tokens: int
    safety_tokens: int
    context_window_tokens: int


@dataclass(frozen=True)
class ContextSection:
    name: str
    messages: tuple[dict[str, Any], ...]
    priority: int
    required: bool = False
    reducer: ContextReducer | None = None


@dataclass(frozen=True)
class PromptEnvelope:
    sections: tuple[ContextSection, ...]

    @classmethod
    def from_messages(cls, messages: Iterable[Mapping[str, Any]]) -> PromptEnvelope:
        copied = tuple(dict(message) for message in messages)
        if not copied:
            return cls(())
        last_user = max(
            (index for index, message in enumerate(copied) if message.get("role") == "user"),
            default=-1,
        )
        sections: list[ContextSection] = []
        index = 0
        while index < len(copied):
            message = copied[index]
            group = [message]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                cursor = index + 1
                while cursor < len(copied) and copied[cursor].get("role") == "tool":
                    group.append(copied[cursor])
                    cursor += 1
                index = cursor - 1
            indices = range(index - len(group) + 1, index + 1)
            roles = {str(item.get("role") or "") for item in group}
            required = 0 in indices or last_user in indices or len(copied) - 1 in indices
            if "tool" in roles or any(item.get("tool_calls") for item in group):
                name, priority = "tool_exchange", 5
            elif "system" in roles:
                name, priority = ("system_contract", 0) if required else ("system_context", 2)
            elif last_user in indices:
                name, priority = "current_question", 0
            elif "assistant" in roles:
                name, priority = "assistant_history", 5
            else:
                name, priority = "user_history", 4
            sections.append(
                ContextSection(
                    name=f"{name}_{len(sections)}",
                    messages=tuple(group),
                    priority=priority,
                    required=required,
                )
            )
            index += 1
        return cls(tuple(sections))

    def render(self) -> list[dict[str, Any]]:
        return [dict(message) for section in self.sections for message in section.messages]


@dataclass(frozen=True)
class ContextBudgetReport:
    mode: str
    profile: str
    original_chars: int
    original_tokens: int
    proposed_chars: int
    proposed_tokens: int
    transmitted_chars: int
    transmitted_tokens: int
    budget: ContextBudget
    compressed: bool
    actions: tuple[str, ...]
    suppressed_sections: tuple[str, ...]
    duration_ms: float
    fits: bool

    def as_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "original_chars": self.original_chars,
            "original_tokens": self.original_tokens,
            "proposed_chars": self.proposed_chars,
            "proposed_tokens": self.proposed_tokens,
            "transmitted_chars": self.transmitted_chars,
            "transmitted_tokens": self.transmitted_tokens,
            "input_token_budget": self.budget.input_tokens,
            "character_budget": self.budget.max_chars,
            "output_token_reserve": self.budget.output_tokens,
            "safety_tokens": self.budget.safety_tokens,
            "compressed": self.compressed,
            "actions": list(self.actions),
            "suppressed_sections": list(self.suppressed_sections),
            "duration_ms": round(self.duration_ms, 3),
            "fits": self.fits,
        }


@dataclass(frozen=True)
class PreparedPrompt:
    messages: list[dict[str, Any]]
    report: ContextBudgetReport


class ContextBudgetManager:
    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        context_window_tokens: int,
        max_chars: int,
        safety_ratio: float,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self.context_window_tokens = context_window_tokens
        self.max_chars = max_chars
        self.safety_ratio = safety_ratio

    @classmethod
    def from_settings(cls, settings: Settings) -> ContextBudgetManager:
        return cls(
            enabled=settings.context_budget_enabled,
            mode=settings.context_budget_mode,
            context_window_tokens=settings.llm_context_window_tokens,
            max_chars=settings.llm_prompt_max_chars,
            safety_ratio=settings.context_safety_ratio,
        )

    def prepare(
        self,
        envelope: PromptEnvelope,
        *,
        profile: str,
        output_tokens: int,
    ) -> PreparedPrompt:
        started = time.perf_counter()
        resolved_profile = profile if profile in _STAGE_TOKEN_LIMITS else "default"
        safety_tokens = math.ceil(self.context_window_tokens * self.safety_ratio)
        available = max(1, self.context_window_tokens - output_tokens - safety_tokens)
        budget = ContextBudget(
            input_tokens=min(_STAGE_TOKEN_LIMITS[resolved_profile], available),
            max_chars=self.max_chars,
            output_tokens=output_tokens,
            safety_tokens=safety_tokens,
            context_window_tokens=self.context_window_tokens,
        )
        original = envelope.render()
        original_chars = prompt_text_size(original)
        original_tokens = estimate_prompt_tokens(original)
        actions: tuple[str, ...] = ()
        suppressed: tuple[str, ...] = ()
        fits = _fits(original, budget)
        proposed = original
        if self.enabled:
            try:
                proposed_envelope, actions, suppressed = self._fit(envelope, budget)
                proposed = proposed_envelope.render()
                fits = _fits(proposed, budget)
            except ContextBudgetExceeded:
                fits = False
                if self.mode == "enforce":
                    raise
        transmitted = proposed if self.enabled and self.mode == "enforce" else original
        report = ContextBudgetReport(
            mode=self.mode if self.enabled else "disabled",
            profile=resolved_profile,
            original_chars=original_chars,
            original_tokens=original_tokens,
            proposed_chars=prompt_text_size(proposed),
            proposed_tokens=estimate_prompt_tokens(proposed),
            transmitted_chars=prompt_text_size(transmitted),
            transmitted_tokens=estimate_prompt_tokens(transmitted),
            budget=budget,
            compressed=proposed != original,
            actions=actions,
            suppressed_sections=suppressed,
            duration_ms=(time.perf_counter() - started) * 1000,
            fits=fits,
        )
        return PreparedPrompt(messages=transmitted, report=report)

    def _fit(
        self,
        envelope: PromptEnvelope,
        budget: ContextBudget,
    ) -> tuple[PromptEnvelope, tuple[str, ...], tuple[str, ...]]:
        sections, deduplicated = _deduplicate_sections(envelope.sections)
        actions: list[str] = [f"deduplicated:{count}" for count in (deduplicated,) if count]
        current = PromptEnvelope(sections)
        if _fits(current.render(), budget):
            return current, tuple(actions), ()

        for intensity in range(1, 5):
            updated: list[ContextSection] = []
            for section in current.sections:
                compacted = _reduce_section(section, intensity)
                if compacted.messages != section.messages:
                    actions.append(f"reduced:{section.name}:level_{intensity}")
                updated.append(compacted)
            current = PromptEnvelope(tuple(updated))
            if _fits(current.render(), budget):
                return current, tuple(dict.fromkeys(actions)), ()

        suppressed: list[str] = []
        remaining = list(current.sections)
        optional = sorted(
            (section for section in remaining if not section.required),
            key=lambda section: (section.priority, _section_chars(section)),
            reverse=True,
        )
        for section in optional:
            remaining.remove(section)
            suppressed.append(section.name)
            actions.append(f"suppressed:{section.name}")
            candidate = PromptEnvelope(tuple(remaining))
            if _fits(candidate.render(), budget):
                return candidate, tuple(dict.fromkeys(actions)), tuple(suppressed)

        required = PromptEnvelope(tuple(section for section in remaining if section.required))
        required_chars = prompt_text_size(required.render())
        required_tokens = estimate_prompt_tokens(required.render())
        raise ContextBudgetExceeded(
            "Required LLM context exceeds the configured budget: "
            f"{required_tokens}/{budget.input_tokens} estimated tokens, "
            f"{required_chars}/{budget.max_chars} characters."
        )


def resolve_context_profile(*, agent: str, node_name: str | None, streaming: bool) -> str:
    lowered_agent = agent.strip().lower()
    lowered_node = (node_name or "").strip().lower()
    if lowered_agent == "assistant":
        return "assistant_compose" if streaming or "compose_answer" in lowered_node else "assistant_route"
    if "report" in lowered_node:
        return "report"
    if "review" in lowered_node or "validate" in lowered_node or "integrate" in lowered_node:
        return "review"
    if "cleaning" in lowered_node:
        return _AGENT_PROFILES.get(lowered_agent, "cleaning")
    return _AGENT_PROFILES.get(lowered_agent, "default")


def estimate_prompt_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    return estimate_text_tokens(_messages_text(messages))


def estimate_text_tokens(value: str) -> int:
    if not value:
        return 0
    cjk = len(_CJK_RE.findall(value))
    punctuation = sum(1 for character in value if character in _JSON_PUNCTUATION)
    other = max(0, len(value) - cjk - punctuation)
    return cjk + math.ceil(punctuation / 2) + math.ceil(other / 4)


def prompt_text_size(messages: Iterable[Mapping[str, Any]]) -> int:
    return len(_messages_text(messages))


def _messages_text(messages: Iterable[Mapping[str, Any]]) -> str:
    return "".join(_content_text(message.get("content")) for message in messages)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        if content.get("type") == "image_url":
            return ""
        return "".join(_content_text(value) for value in content.values())
    if isinstance(content, (list, tuple)):
        return "".join(_content_text(item) for item in content)
    return str(content) if content is not None else ""


def _fits(messages: list[dict[str, Any]], budget: ContextBudget) -> bool:
    return (
        prompt_text_size(messages) <= budget.max_chars
        and estimate_prompt_tokens(messages) <= budget.input_tokens
    )


def _deduplicate_sections(
    sections: tuple[ContextSection, ...],
) -> tuple[tuple[ContextSection, ...], int]:
    seen: set[str] = set()
    output: list[ContextSection] = []
    removed = 0
    for section in sections:
        signature = json.dumps(section.messages, ensure_ascii=False, sort_keys=True, default=str)
        if not section.required and signature in seen:
            removed += 1
            continue
        seen.add(signature)
        output.append(section)
    return tuple(output), removed


def _reduce_section(section: ContextSection, intensity: int) -> ContextSection:
    if section.reducer is not None:
        messages = section.reducer(section.messages, intensity)
        return replace(section, messages=messages)
    messages = tuple(
        _reduce_message(message, intensity, required=section.required)
        for message in section.messages
    )
    return replace(section, messages=messages)


def _reduce_message(
    message: dict[str, Any],
    intensity: int,
    *,
    required: bool,
) -> dict[str, Any]:
    role = str(message.get("role") or "")
    if required and role == "system":
        return message
    content = message.get("content")
    if isinstance(content, str):
        compacted = _reduce_text_content(content, intensity, required=required)
    elif isinstance(content, list):
        compacted = [
            item
            if isinstance(item, Mapping) and item.get("type") == "image_url"
            else _reduce_content_item(item, intensity)
            for item in content
        ]
    else:
        compacted = content
    return {**message, "content": compacted}


def _reduce_text_content(value: str, intensity: int, *, required: bool) -> str:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        if required:
            return value
        limits = (4_000, 2_000, 1_000, 500)
        return _truncate_middle(value, limits[intensity - 1])
    error_line = _error_line(value)
    compacted = _compact_json(payload, intensity=intensity, key="", error_line=error_line)
    return json.dumps(compacted, ensure_ascii=False, separators=(",", ":"), default=str)


def _reduce_content_item(value: Any, intensity: int) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _compact_json(item, intensity=intensity, key=str(key), error_line=None)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return _truncate_middle(value, (2_000, 1_000, 500, 240)[intensity - 1])
    return value


def _compact_json(value: Any, *, intensity: int, key: str, error_line: int | None) -> Any:
    lowered = key.casefold()
    string_limit = (2_000, 1_000, 500, 240)[intensity - 1]
    list_limit = (20, 8, 3, 1)[intensity - 1]
    dict_limit = (40, 20, 10, 6)[intensity - 1]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if lowered in _REQUIRED_TEXT_KEYS:
            return value
        if lowered in _CODE_KEYS:
            return _compact_code(value, max_chars=max(2_000, string_limit * 4), error_line=error_line)
        return _truncate_middle(value, string_limit)
    if isinstance(value, Mapping):
        items = list(value.items())
        required_items = [item for item in items if str(item[0]).casefold() in _REQUIRED_TEXT_KEYS]
        optional_items = [item for item in items if item not in required_items]
        selected = [*required_items, *optional_items[: max(0, dict_limit - len(required_items))]]
        return {
            str(item_key): _compact_json(
                item,
                intensity=intensity,
                key=str(item_key),
                error_line=error_line,
            )
            for item_key, item in selected
        }
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if lowered in {"errors", "attempts", "python_attempts"}:
            selected_values = values
        elif lowered in _EVIDENCE_KEYS:
            selected_values = values[:24]
        else:
            selected_values = values[:list_limit]
        return [
            _compact_json(item, intensity=intensity, key=lowered, error_line=error_line)
            for item in selected_values
        ]
    return _truncate_middle(str(value), string_limit)


def _compact_code(value: str, *, max_chars: int, error_line: int | None) -> str:
    if len(value) <= max_chars:
        return value
    lines = value.splitlines()
    selected: set[int] = set(range(min(30, len(lines))))
    selected.update(range(max(0, len(lines) - 10), len(lines)))
    if error_line is not None:
        selected.update(range(max(0, error_line - 16), min(len(lines), error_line + 15)))
    output: list[str] = []
    previous = -2
    for index in sorted(selected):
        if index > previous + 1:
            output.append("# ... [context compressed] ...")
        output.append(lines[index])
        previous = index
    return _truncate_middle("\n".join(output), max_chars)


def _error_line(value: str) -> int | None:
    match = _ERROR_LINE_RE.search(value)
    return int(match.group(1)) if match else None


def _truncate_middle(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head - 30)
    return f"{value[:head].rstrip()}\n... [context compressed] ...\n{value[-tail:].lstrip()}"


def _section_chars(section: ContextSection) -> int:
    return prompt_text_size(section.messages)
