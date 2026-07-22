from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentLoopBenchmarkOutcome:
    case_id: str
    selected_tool: str | None
    expected_tools: frozenset[str]
    legal_call: bool
    recoverable_error: bool = False
    recovered: bool = False
    tool_calls: int = 0
    duplicate_successful_actions: int = 0


@dataclass(frozen=True)
class AgentLoopBenchmarkReport:
    sample_count: int
    legal_call_rate: float
    repair_success_rate: float
    simple_task_mean_tool_calls: float
    duplicate_successful_actions: int
    passed: bool


def evaluate_agent_loop_benchmark(
    outcomes: tuple[AgentLoopBenchmarkOutcome, ...],
    *,
    legal_call_threshold: float = 0.95,
    repair_success_threshold: float = 0.90,
    simple_task_call_limit: float = 4.0,
) -> AgentLoopBenchmarkReport:
    if not outcomes:
        raise ValueError("Agent Loop benchmark requires at least one outcome.")
    legal = sum(
        1
        for item in outcomes
        if item.legal_call
        and (item.selected_tool is None or item.selected_tool in item.expected_tools)
    )
    repairable = tuple(item for item in outcomes if item.recoverable_error)
    repaired = sum(1 for item in repairable if item.recovered)
    legal_rate = legal / len(outcomes)
    repair_rate = repaired / len(repairable) if repairable else 1.0
    mean_calls = sum(item.tool_calls for item in outcomes) / len(outcomes)
    duplicates = sum(item.duplicate_successful_actions for item in outcomes)
    return AgentLoopBenchmarkReport(
        sample_count=len(outcomes),
        legal_call_rate=legal_rate,
        repair_success_rate=repair_rate,
        simple_task_mean_tool_calls=mean_calls,
        duplicate_successful_actions=duplicates,
        passed=(
            legal_rate >= legal_call_threshold
            and repair_rate >= repair_success_threshold
            and mean_calls <= simple_task_call_limit
            and duplicates == 0
        ),
    )
