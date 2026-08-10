from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

CRITICAL_TASKS = {"DM001", "DM004", "DM005", "DM006"}


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trace_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_trials = 0
    provider_error_trials: list[str] = []
    missing_judge_candidates: list[str] = []
    observed_judge = False
    unreadable_traces: list[str] = []
    terminal_reasons: Counter[str] = Counter()
    recovery_fallback_trials: list[str] = []
    workflow_fallback_trials: list[str] = []
    workflow_errors: Counter[str] = Counter()
    internal_tool_calls = 0
    for item in results:
        task_id = str(item.get("task_id") or "unknown")
        for index, trial in enumerate(item.get("trials") or [], start=1):
            total_trials += 1
            trace_value = trial.get("trace")
            if not trace_value:
                unreadable_traces.append(f"{task_id}#{index}:missing_path")
                continue
            trace_path = Path(str(trace_value))
            terminal_reason = ""
            judge_calls: Any = None
            result: dict[str, Any] = {}
            run_status = ""
            run_failed_for_provider = False
            try:
                with trace_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        event = json.loads(line)
                        if event.get("type") == "audit_snapshot":
                            audit = event.get("audit_data") or {}
                            runs = audit.get("runs") or []
                            if runs and isinstance(runs[-1], dict):
                                run = runs[-1]
                                run_status = str(run.get("status") or "").lower()
                                result = run.get("result") or {}
                                terminal_reason = str(
                                    result.get("loop_terminal_reason") or ""
                                )
                                run_error = str(run.get("error") or "").lower()
                                run_failed_for_provider = (
                                    str(run.get("status") or "").lower() == "failed"
                                    and any(
                                        marker in run_error
                                        for marker in (
                                            "provider failed",
                                            "provider_error",
                                            "api error 429",
                                            "engine_overloaded_error",
                                        )
                                    )
                                )
                        elif event.get("type") == "grading_result":
                            judge_calls = event.get("judge_calls")
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                unreadable_traces.append(
                    f"{task_id}#{index}:{type(exc).__name__}"
                )
                continue
            label = f"{task_id}#{index}"
            if run_failed_for_provider and not terminal_reason:
                terminal_reason = "provider_error"
            if terminal_reason:
                terminal_reasons[terminal_reason] += 1
            if terminal_reason == "provider_error":
                provider_error_trials.append(label)
            if terminal_reason in {"legacy_fallback", "model_requested_fallback"}:
                recovery_fallback_trials.append(label)
            loop_summary = result.get("loop_summary") or {}
            internal_tool_calls += int(loop_summary.get("tool_calls") or 0)
            workflow = result.get("workflow_trace") or []
            used_workflow_fallback = False
            for item in workflow:
                if not isinstance(item, dict):
                    continue
                if item.get("status") == "fallback" or item.get("fallback"):
                    used_workflow_fallback = True
                if item.get("error"):
                    workflow_errors[str(item["error"])] += 1
            if used_workflow_fallback:
                workflow_fallback_trials.append(label)
            if judge_calls:
                observed_judge = True
            elif run_status in {"completed", "failed"}:
                missing_judge_candidates.append(label)
    return {
        "total_trials": total_trials,
        "provider_error_trials": provider_error_trials,
        "missing_judge_trials": missing_judge_candidates if observed_judge else [],
        "unreadable_traces": unreadable_traces,
        "terminal_reasons": dict(terminal_reasons),
        "recovery_fallback_trials": recovery_fallback_trials,
        "workflow_fallback_trials": workflow_fallback_trials,
        "workflow_errors": dict(workflow_errors),
        "internal_tool_calls": internal_tool_calls,
    }


def render(
    results: list[dict[str, Any]],
    source: Path,
    *,
    judge_model: str | None = None,
) -> str:
    diagnostics = _trace_diagnostics(results)
    judge_label = judge_model or os.environ.get(
        "CLAW_EVAL_JUDGE_MODEL_ID", "kimi-k3"
    )
    valid = [item for item in results if not item.get("error") and item.get("trials")]
    suite_score = _average([float(item.get("avg_score") or 0) for item in valid])
    critical_pass = all(
        all(bool(trial.get("passed")) for trial in item.get("trials") or [])
        for item in valid
        if item.get("task_id") in CRITICAL_TASKS
    ) and CRITICAL_TASKS.issubset({str(item.get("task_id")) for item in valid})
    plan_conformant = not any(
        (
            diagnostics["provider_error_trials"],
            diagnostics["missing_judge_trials"],
            diagnostics["unreadable_traces"],
        )
    )
    release_passed = suite_score >= 0.80 and critical_pass and plan_conformant
    lines = [
        "# DataMind claw-eval 评测摘要",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- 被测对象：DataMind 核心分析",
        "- DataMind 内部模型：Kimi K2.6",
        f"- 定性裁判：{judge_label}",
        f"- 数据源：`{source}`",
        f"- 套件平均分：**{suite_score:.3f}**",
        f"- 评测有效性：**{'有效' if plan_conformant else '受污染'}**",
        f"- 关键任务 pass^3：**{'通过' if critical_pass else '未通过'}**",
        f"- 建议发布门槛：**{'通过' if release_passed else '未通过'}**",
        "",
        "## 各任务",
        "",
        "| 任务 | 平均分 | 三次得分 | pass^3 | completion | robustness | communication | safety |",
        "|---|---:|---|:---:|---:|---:|---:|---:|",
    ]
    for item in results:
        task_id = str(item.get("task_id") or "unknown")
        trials = list(item.get("trials") or [])
        scores = [float(trial.get("task_score") or 0) for trial in trials]
        passed_all = bool(trials) and all(bool(trial.get("passed")) for trial in trials)
        dimensions = {
            name: _average([float(trial.get(name) or 0) for trial in trials])
            for name in ("completion", "robustness", "communication", "safety")
        }
        lines.append(
            f"| {task_id} | {float(item.get('avg_score') or 0):.3f} | "
            f"{' / '.join(f'{value:.2f}' for value in scores) or '-'} | "
            f"{'Y' if passed_all else 'N'} | {dimensions['completion']:.2f} | "
            f"{dimensions['robustness']:.2f} | {dimensions['communication']:.2f} | "
            f"{dimensions['safety']:.2f} |"
        )
    errors = [item for item in results if item.get("error")]
    if errors:
        lines.extend(["", "## Harness 错误", ""])
        for item in errors:
            lines.append(f"- {item.get('task_id')}: {item.get('error')}")
    lines.extend(["", "## 基础设施与路由诊断", ""])
    provider_errors = diagnostics["provider_error_trials"]
    if provider_errors:
        lines.extend(
            [
                f"- Agent provider_error：**{len(provider_errors)}/{diagnostics['total_trials']} trials**。",
                "- 受影响 trial 已失败闭合记为 0 分；确定性 fallback 未被当作 Kimi 正式分数。",
                f"- 受影响 trials：{', '.join(provider_errors)}",
            ]
        )
    else:
        lines.append("- Agent provider_error：0，模型路由符合评测要求。")
    terminal_reasons = diagnostics["terminal_reasons"]
    if terminal_reasons:
        terminal_summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(terminal_reasons.items())
        )
        lines.append(f"- Agent loop 终止原因：{terminal_summary}。")
    lines.append(
        f"- DataMind 内部 agent tool calls：**{diagnostics['internal_tool_calls']}**。"
    )
    recovery_trials = diagnostics["recovery_fallback_trials"]
    if recovery_trials:
        lines.append(
            f"- 恢复性 fallback：**{len(recovery_trials)}/{diagnostics['total_trials']} trials**。"
            "这不等同于 provider 路由污染，但说明 agent loop 没有干净结束。"
        )
    workflow_fallbacks = diagnostics["workflow_fallback_trials"]
    if workflow_fallbacks:
        lines.append(
            f"- Planner/workflow fallback：**{len(workflow_fallbacks)}/"
            f"{diagnostics['total_trials']} trials**。"
        )
    if diagnostics["workflow_errors"]:
        error_summary = "; ".join(
            f"{message} ({count})"
            for message, count in sorted(
                diagnostics["workflow_errors"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        lines.append(f"- Workflow 恢复原因：{error_summary}")
    missing_judges = diagnostics["missing_judge_trials"]
    if missing_judges:
        lines.append(
            f"- {judge_label} 裁判结果缺失：**{len(missing_judges)}** 次"
            f"（{', '.join(missing_judges)}）。"
        )
    else:
        lines.append(f"- {judge_label} 裁判结果：未发现缺失。")
    if diagnostics["unreadable_traces"]:
        lines.append(
            "- 无法读取的 traces：" + ", ".join(diagnostics["unreadable_traces"])
        )
    lines.extend(
        [
            "",
            "## 判定口径",
            "",
            "单次 task_score 不低于 0.75 为通过。发布门槛要求套件平均分不低于 0.80，",
            "且 DM001、DM004、DM005、DM006 的三次 trial 全部通过。分数反映的是",
            "DataMind 在 Kimi K2.6 配置下的整体表现，不是 Kimi 的直接模型基准分。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Chinese DataMind claw-eval summary")
    parser.add_argument("--batch-results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("CLAW_EVAL_JUDGE_MODEL_ID", "kimi-k3"),
    )
    args = parser.parse_args()
    source = args.batch_results.resolve()
    results = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("batch_results.json must contain a list")
    output = args.output.resolve() if args.output else source.with_name("datamind_summary.md")
    output.write_text(
        render(results, source, judge_model=args.judge_model), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
