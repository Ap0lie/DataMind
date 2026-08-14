from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from app.core.settings import get_settings
from app.schemas.analysis import StatisticalVerificationResponse

PLANNER_NODE = "planner"
INTENT_COMPILE_NODE = "intent_compile"
SCOPE_RESOLVE_NODE = "scope_resolve"
CONTRACT_VALIDATE_NODE = "contract_validate"
JOIN_PREPARE_NODE = "join_prepare"
DESIGN_FRAMEWORK_NODE = "design_framework"
SQL_NODE = "sql_agent"
PYTHON_NODE = "python_agent"
ROUND_PREPARE_NODE = "iterative_prepare_rounds"
ROUND_FOUNDATION_NODE = "iterative_round_1"
ROUND_FANOUT_NODE = "iterative_fanout_round"
ROUND_REFLECT_NODE = "iterative_reflect_and_merge"
INTEGRATE_INSIGHTS_NODE = "integrate_insights"
ADVERSARIAL_VALIDATE_NODE = "adversarial_validate"
FORMAT_CHARTS_NODE = "format_charts"
STATISTICAL_VERIFY_NODE = "statistical_verify"
REPORT_NODE = "report_agent"
REPORT_DECIDE_NODE = "report_decide"
REPORT_EXECUTE_NODE = "report_execute"
REPORT_VERIFY_NODE = "report_verify"
REPORT_REPAIR_NODE = "report_repair"
REPORT_FALLBACK_NODE = "report_fallback"
REPORT_COMMIT_NODE = "report_commit"
LOOP_BOOTSTRAP_NODE = "loop_bootstrap"
LOOP_DECIDE_NODE = "loop_decide"
LOOP_EXECUTE_NODE = "loop_execute"
LOOP_OBSERVE_NODE = "loop_observe"
LOOP_VERIFY_NODE = "loop_verify"
LOOP_REPAIR_NODE = "loop_repair"
LOOP_FALLBACK_NODE = "loop_fallback"
LOOP_FINALIZE_NODE = "loop_finalize"
LOOP_ADVERSARIAL_REPAIR_NODE = "loop_adversarial_repair"


def route_after_framework(state: Mapping[str, Any]) -> str:
    if state.get("agent_mode") == "loop":
        return LOOP_BOOTSTRAP_NODE
    return route_after_planner(state)


def route_after_planner(state: Mapping[str, Any]) -> str:
    planned_analysis = state.get("planned_analysis")
    if planned_analysis is None:
        raise RuntimeError("Planner did not produce an analysis plan.")
    return SQL_NODE if planned_analysis.route in {"sql", "hybrid"} else PYTHON_NODE


def route_after_loop_decide(state: Mapping[str, Any]) -> str:
    pending = state.get("loop_pending_call") or {}
    action = pending.get("action")
    if action == "tool_call":
        return LOOP_EXECUTE_NODE
    if action == "retry_decision":
        return LOOP_DECIDE_NODE
    if action == "finish" and state.get("tool_evidence"):
        return LOOP_FINALIZE_NODE
    return LOOP_FALLBACK_NODE


def route_after_loop_verify(state: Mapping[str, Any]) -> str:
    pending = state.get("loop_pending_call") or {}
    action = pending.get("action")
    if action == "continue_sequence":
        return LOOP_DECIDE_NODE
    if action in {"verified", "duplicate_action"} and pending.get("outcome") == "sufficient":
        return LOOP_FINALIZE_NODE
    if action in {"duplicate_action", "repairable"}:
        return LOOP_REPAIR_NODE
    if action in {"fallback", "budget_exhausted"}:
        return LOOP_FALLBACK_NODE
    if loop_budget_exhaustion(state):
        return LOOP_FINALIZE_NODE if state.get("tool_evidence") else LOOP_FALLBACK_NODE
    return LOOP_DECIDE_NODE


def route_after_loop_preflight(state: Mapping[str, Any]) -> str:
    verification = state.get("loop_preflight_verification")
    if verification is None or not verification.requires_replan:
        return INTEGRATE_INSIGHTS_NODE
    can_repair = (
        state.get("adversarial_repair_count", 0) < 1
        and loop_budget_exhaustion(state) is None
        and any(
            check.code in {"request_coverage", "join_grain"}
            for check in verification.checks
            if check.status == "failed"
        )
    )
    if can_repair:
        return LOOP_ADVERSARIAL_REPAIR_NODE
    raise RuntimeError(statistical_failure_message(verification))


def route_after_adversarial_validate(state: Mapping[str, Any]) -> str:
    verification = state.get("statistical_verification")
    if verification is not None and verification.requires_replan:
        if state.get("agent_mode") == "loop" and state.get("adversarial_repair_count", 0) < 1:
            return LOOP_ADVERSARIAL_REPAIR_NODE
        raise RuntimeError(statistical_failure_message(verification))
    if state.get("agent_mode") != "loop" or state.get("adversarial_repair_count", 0) >= 1:
        return (
            REPORT_DECIDE_NODE
            if state.get("agent_mode") == "loop" and get_settings().report_loop_enabled
            else REPORT_NODE
        )
    has_high_issue = any(
        item.severity.lower() in {"high", "critical", "error"}
        for item in state.get("validation_issues", ())
    )
    if has_high_issue:
        return LOOP_ADVERSARIAL_REPAIR_NODE
    return REPORT_DECIDE_NODE if get_settings().report_loop_enabled else REPORT_NODE


def require_reportable_verification(state: Mapping[str, Any]) -> None:
    verification = state.get("statistical_verification")
    if verification is None:
        raise RuntimeError("Statistical verification is missing; report commit was blocked.")
    if verification.requires_replan or verification.status == "failed":
        raise RuntimeError(statistical_failure_message(verification))


def statistical_failure_message(verification: StatisticalVerificationResponse) -> str:
    failures = [check.message for check in verification.checks if check.status == "failed"]
    verdict_failures = [
        f"{verdict.title}: {' '.join(verdict.notes[:3])}"
        for verdict in verification.finding_verdicts
        if verdict.status == "failed" and verdict.notes
    ]
    detail = "; ".join((*failures, *verdict_failures)[:4]) or verification.summary
    return f"Statistical verification failed after replanning; report commit was blocked: {detail}"


def route_after_report_decide(state: Mapping[str, Any]) -> str:
    strategy = state.get("report_strategy")
    if strategy == "evidence_gap":
        return LOOP_BOOTSTRAP_NODE
    if strategy == "rules_fallback":
        return REPORT_FALLBACK_NODE
    return REPORT_EXECUTE_NODE


def route_after_report_verify(state: Mapping[str, Any]) -> str:
    outcome = str((state.get("report_validation") or {}).get("outcome") or "fallback")
    if outcome == "sufficient":
        return REPORT_COMMIT_NODE
    if outcome == "evidence_gap":
        return LOOP_BOOTSTRAP_NODE
    if outcome == "report_issue":
        return REPORT_REPAIR_NODE
    return REPORT_FALLBACK_NODE


def loop_budget_exhaustion(state: Mapping[str, Any]) -> str | None:
    budget = state.get("loop_budget", {})
    if state.get("tool_call_count", 0) >= int(budget.get("max_tool_calls") or 12):
        return "tool_budget_exhausted"
    if state.get("loop_decision_count", 0) >= int(budget.get("max_decisions") or 16):
        return "decision_budget_exhausted"
    if int(budget.get("used_tokens") or 0) >= int(budget.get("max_tokens") or 50_000):
        return "token_budget_exhausted"
    if budget.get("deadline_epoch") and time.time() >= float(budget["deadline_epoch"]):
        return "time_budget_exhausted"
    return None
