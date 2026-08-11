from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, TypedDict
from uuid import UUID

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.analysis.checkpoints import get_analysis_checkpointer
from app.analysis.cleaning_diff import build_cleaning_diff_summary
from app.analysis.cleaning_sandbox import run_generated_cleaning_analysis
from app.analysis.data_cleaning import (
    _basic_clean_dataframe,
    _extract_cleaning_script,
    _records,
    _validate_cleaning_quality,
)
from app.analysis.model_router import AnalysisModelRouter, MCPAnalysisModelRouter
from app.core.settings import get_settings
from app.harness.node import NodeExecutionHarness, NodeHarnessPolicy
from app.storage.dataset_store import DatasetStoreRepository


class CleaningWorkflowState(TypedDict, total=False):
    job_id: str
    dataset_id: str
    requirement: str
    requested_strategy: str
    decision_count: int
    tool_call_count: int
    iteration: int
    strategy_attempts: dict[str, int]
    selected_strategy: str
    pending_strategy: str
    candidate_artifact_id: str
    candidate_metadata: dict[str, Any]
    failures: list[dict[str, Any]]
    quality: dict[str, Any]
    terminal_reason: str
    started_monotonic: float
    used_tokens: int
    result: dict[str, Any]


_STRATEGY_TOOL = {
    "type": "function",
    "function": {
        "name": "select_cleaning_strategy",
        "description": "Select exactly one safe cleaning strategy based on profile and prior feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["rules", "llm", "hybrid", "fallback"]},
                "reason": {"type": "string"},
            },
            "required": ["strategy", "reason"],
            "additionalProperties": False,
        },
    },
}

_LOCAL_CLEANING_MARKERS = (
    "trim",
    "strip",
    "whitespace",
    "deduplicate",
    "duplicate",
    "missing",
    "null",
    "type conversion",
    "去空格",
    "去重",
    "重复",
    "缺失",
    "空值",
    "类型转换",
    "数字转换",
    "日期格式",
    "通用清洗",
)
_SEMANTIC_CLEANING_MARKERS = (
    "业务语义",
    "语义标准化",
    "同义词",
    "别名",
    "映射",
    "归类",
    "分类",
    "客户等级",
    "标签体系",
    "semantic",
    "taxonomy",
    "category mapping",
)


def local_auto_cleaning_decision(requirement: str) -> tuple[str, str] | None:
    normalized = " ".join(requirement.casefold().split())
    if any(marker in normalized for marker in _SEMANTIC_CLEANING_MARKERS):
        return None
    if not normalized or any(marker in normalized for marker in _LOCAL_CLEANING_MARKERS):
        return "rules", "local_rule_classifier"
    if normalized in {"清洗数据", "自动清洗", "分析前清洗", "clean data"}:
        return "rules", "local_rule_classifier"
    return None


class CleaningWorkflowRunner:
    def __init__(
        self,
        repository: DatasetStoreRepository,
        model_router: AnalysisModelRouter | None = None,
    ) -> None:
        self.repository = repository
        self.model_router = model_router or MCPAnalysisModelRouter()

    def run(
        self,
        *,
        job_id: UUID,
        progress_callback: Callable[[str, int, str], None] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        job = self.repository.get_cleaning_job(job_id)
        graph = self._build_graph(progress_callback, event_callback, cancel_checker)
        config = {"configurable": {"thread_id": f"cleaning:{job.checkpoint_thread_id or job.id}"}}
        initial: CleaningWorkflowState = {
            "job_id": str(job.id), "dataset_id": str(job.dataset_id),
            "requirement": job.requirement, "requested_strategy": job.cleaning_strategy,
            "decision_count": 0, "tool_call_count": 0, "iteration": 0,
            "strategy_attempts": {}, "failures": [], "used_tokens": 0,
            "started_monotonic": time.monotonic(),
        }
        output = graph.invoke(None if resume else initial, config=config)
        result = output.get("result") if isinstance(output, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError("Cleaning workflow did not produce a committed result.")
        return result

    def _build_graph(self, progress, emit, cancel_checker):
        settings = get_settings()

        def check(state: CleaningWorkflowState) -> None:
            if cancel_checker and cancel_checker():
                raise RuntimeError("Cleaning job canceled.")
            if time.monotonic() - float(state.get("started_monotonic") or time.monotonic()) > settings.cleaning_loop_timeout_seconds:
                raise RuntimeError("Cleaning loop time budget exhausted.")

        def notify(stage: str, percent: int, message: str, *, state=None, event_type=None, strategy=None, payload=None):
            if progress:
                progress(stage, percent, message)
            if emit:
                emit({
                    "stage": stage, "status": "completed", "message": message,
                    "event_type": event_type, "iteration": int((state or {}).get("iteration") or 0),
                    "strategy": strategy, "payload": payload or {},
                })

        def bootstrap(state: CleaningWorkflowState) -> dict[str, Any]:
            check(state)
            records = self.repository.read_raw_records(UUID(state["dataset_id"]))
            if not records:
                raise RuntimeError("Dataset has no raw records to clean.")
            profile = _profile_records(records)
            notify("cleaning_bootstrap", 5, "Cleaning scope and quality baseline fixed.", state=state, event_type="cleaning_bootstrap", payload=profile)
            return {"quality": {"baseline": profile}}

        def decide(state: CleaningWorkflowState) -> dict[str, Any]:
            check(state)
            decisions = int(state.get("decision_count") or 0) + 1
            if decisions > settings.cleaning_loop_max_decisions:
                return {"decision_count": decisions, "pending_strategy": "fallback", "terminal_reason": "decision_budget_exhausted"}
            if int(state.get("used_tokens") or 0) >= settings.cleaning_loop_max_tokens:
                return {"decision_count": decisions, "pending_strategy": "fallback", "terminal_reason": "token_budget_exhausted"}
            requested = state.get("requested_strategy") or "auto"
            attempts = state.get("strategy_attempts") or {}
            if requested != "auto":
                strategy = requested if int(attempts.get(requested) or 0) < settings.cleaning_loop_max_strategy_attempts else "fallback"
                reason = "requested_strategy" if strategy != "fallback" else "strategy_attempts_exhausted"
                notify("cleaning_decide", 10, f"Selected cleaning strategy: {strategy}.", state=state, event_type="cleaning_decision", strategy=strategy, payload={"reason": reason})
                return {"decision_count": decisions, "pending_strategy": strategy}
            local_decision = (
                local_auto_cleaning_decision(state.get("requirement") or "")
                if not attempts and not state.get("failures")
                else None
            )
            if local_decision is not None:
                strategy, reason = local_decision
                notify(
                    "cleaning_decide",
                    10,
                    "Local rules selected for a deterministic cleaning task.",
                    state=state,
                    event_type="cleaning_decision",
                    strategy=strategy,
                    payload={"reason": reason, "decision_source": "local_classifier"},
                )
                return {
                    "decision_count": decisions,
                    "pending_strategy": strategy,
                }
            strategy, reason, tokens = self._model_decision(state)
            if int(attempts.get(strategy) or 0) >= settings.cleaning_loop_max_strategy_attempts:
                strategy, reason = "fallback", "selected_strategy_attempts_exhausted"
            notify("cleaning_decide", 10, f"AI selected cleaning strategy: {strategy}.", state=state, event_type="cleaning_decision", strategy=strategy, payload={"reason": reason})
            return {"decision_count": decisions, "pending_strategy": strategy, "used_tokens": int(state.get("used_tokens") or 0) + tokens}

        def execute(state: CleaningWorkflowState) -> dict[str, Any]:
            check(state)
            strategy = state.get("pending_strategy") or "fallback"
            calls = int(state.get("tool_call_count") or 0) + 1
            if calls > settings.cleaning_loop_max_tool_calls or strategy == "fallback":
                return {"pending_strategy": "fallback", "terminal_reason": state.get("terminal_reason") or "tool_budget_exhausted"}
            attempts = dict(state.get("strategy_attempts") or {})
            attempts[strategy] = int(attempts.get(strategy) or 0) + 1
            raw = self.repository.read_raw_records(UUID(state["dataset_id"]))
            try:
                candidate, metadata = self._execute_strategy(strategy, state, raw)
                artifact_id = self.repository.save_artifact(
                    dataset_id=UUID(state["dataset_id"]), artifact_type="cleaning_candidate",
                    content={"job_id": state["job_id"], "strategy": strategy, "records": candidate, "metadata": metadata},
                )
                notify("cleaning_execute", 35, f"Cleaning strategy {strategy} executed in isolation.", state=state, event_type="cleaning_execution", strategy=strategy, payload={"artifact_id": str(artifact_id), "row_count": len(candidate)})
                return {"tool_call_count": calls, "iteration": int(state.get("iteration") or 0) + 1, "strategy_attempts": attempts, "selected_strategy": strategy, "candidate_artifact_id": str(artifact_id), "candidate_metadata": metadata, "pending_strategy": "verify", "used_tokens": int(state.get("used_tokens") or 0) + int(metadata.get("token_usage") or 0)}
            except Exception as exc:
                failures = [*(state.get("failures") or []), {"strategy": strategy, "error": str(exc)[:1000], "iteration": int(state.get("iteration") or 0) + 1}]
                notify("cleaning_observe", 40, f"Cleaning execution failed: {type(exc).__name__}.", state=state, event_type="cleaning_error", strategy=strategy, payload={"error": str(exc)[:500]})
                return {"tool_call_count": calls, "iteration": int(state.get("iteration") or 0) + 1, "strategy_attempts": attempts, "failures": failures, "pending_strategy": "repair"}

        def verify(state: CleaningWorkflowState) -> dict[str, Any]:
            check(state)
            artifact = self.repository.get_artifact(UUID(state["dataset_id"]), UUID(state["candidate_artifact_id"]))
            content = artifact.get("content") or {}
            candidate = content.get("records") or []
            raw = self.repository.read_raw_records(UUID(state["dataset_id"]))
            try:
                fallback = _basic_clean_dataframe(pd.DataFrame(raw))
                frame = pd.DataFrame(candidate)
                _validate_cleaning_quality(fallback_df=fallback, cleaned_df=frame, requirement=state.get("requirement") or "")
                quality = _quality_report(raw, candidate)
                notify("cleaning_verify", 70, "Candidate passed deterministic quality gates.", state=state, event_type="cleaning_validation", strategy=state.get("selected_strategy"), payload=quality)
                return {"quality": quality, "pending_strategy": "commit"}
            except Exception as exc:
                failures = [*(state.get("failures") or []), {"strategy": state.get("selected_strategy"), "error": str(exc)[:1000], "kind": "validation_error"}]
                notify("cleaning_verify", 65, "Candidate failed quality gates and will be repaired or switched.", state=state, event_type="cleaning_validation", strategy=state.get("selected_strategy"), payload={"passed": False, "error": str(exc)[:500]})
                return {"failures": failures, "pending_strategy": "repair"}

        def repair(state: CleaningWorkflowState) -> dict[str, Any]:
            check(state)
            attempts = state.get("strategy_attempts") or {}
            current = state.get("selected_strategy") or state.get("pending_strategy") or "llm"
            next_strategy = current
            if int(attempts.get(current) or 0) >= settings.cleaning_loop_max_strategy_attempts:
                next_strategy = "hybrid" if current == "llm" and int(attempts.get("hybrid") or 0) < settings.cleaning_loop_max_strategy_attempts else "rules"
            if int(state.get("tool_call_count") or 0) >= settings.cleaning_loop_max_tool_calls:
                next_strategy = "fallback"
            notify("cleaning_repair", 55, f"Repairing cleaning plan; next strategy: {next_strategy}.", state=state, event_type="cleaning_repair", strategy=next_strategy, payload={"failure": (state.get("failures") or [{}])[-1]})
            return {"pending_strategy": next_strategy, "requested_strategy": next_strategy if state.get("requested_strategy") != "auto" else "auto"}

        def fallback(state: CleaningWorkflowState) -> dict[str, Any]:
            raw = self.repository.read_raw_records(UUID(state["dataset_id"]))
            candidate = _records(_basic_clean_dataframe(pd.DataFrame(raw)))
            artifact_id = self.repository.save_artifact(
                dataset_id=UUID(state["dataset_id"]), artifact_type="cleaning_candidate",
                content={"job_id": state["job_id"], "strategy": "rules", "records": candidate, "metadata": {"source": "rules_fallback"}},
            )
            notify("rules_fallback", 75, "Cleaning budgets exhausted; conservative rules fallback selected.", state=state, event_type="cleaning_fallback", strategy="rules")
            return {"candidate_artifact_id": str(artifact_id), "selected_strategy": "rules", "terminal_reason": state.get("terminal_reason") or "rules_fallback", "quality": _quality_report(raw, candidate), "pending_strategy": "commit"}

        def commit(state: CleaningWorkflowState) -> dict[str, Any]:
            check(state)
            job = self.repository.get_cleaning_job(UUID(state["job_id"]))
            if job.cancel_requested:
                raise RuntimeError("Cleaning job canceled before commit.")
            artifact = self.repository.get_artifact(UUID(state["dataset_id"]), UUID(state["candidate_artifact_id"]))
            content = artifact.get("content") or {}
            records = [item for item in content.get("records") or [] if isinstance(item, dict)]
            raw = self.repository.read_raw_records(UUID(state["dataset_id"]))
            previous = self.repository.read_cleaned_records(UUID(state["dataset_id"]))
            metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
            strategy = state.get("selected_strategy") or "rules"
            run_id = self.repository.save_cleaning_result(
                dataset_id=UUID(state["dataset_id"]), provider=str(metadata.get("provider") or ("rules" if strategy == "rules" else "model-router")),
                model=str(metadata.get("model") or ("local-basic-cleaner" if strategy == "rules" else "cleaning-agent")),
                prompt=state.get("requirement") or "通用分析前数据清洗",
                result_markdown=str(metadata.get("markdown") or f"## 自主清洗完成\n\n- 最终策略：{strategy}\n"),
                cleaned_dataset={"status": "completed", "source": strategy, "rows": len(records), "columns": len(records[0]) if records else 0, "warnings": [item.get("error") for item in state.get("failures") or []], "records": records},
                raw_summary=_profile_records(raw), previous_summary=_profile_records(previous), current_summary=_profile_records(records),
                diff_summary=build_cleaning_diff_summary(
                    raw_records=raw,
                    previous_records=previous,
                    current_records=records,
                ),
                activate=True,
                job_id=UUID(state["job_id"]),
            )
            self.repository.save_cleaned_records(dataset_id=UUID(state["dataset_id"]), records=records, metadata={"job_id": state["job_id"], "strategy": strategy, "cleaning_run_id": str(run_id)})
            result = {"cleaning_run_id": str(run_id), "selected_strategy": strategy, "terminal_reason": state.get("terminal_reason") or "validated", "quality": state.get("quality") or {}, "preview_records": records[:50], "row_count": len(records), "column_count": len(records[0]) if records else 0, "failures": state.get("failures") or []}
            notify("cleaning_commit", 100, "Validated cleaning version committed and activated.", state=state, event_type="cleaning_commit", strategy=strategy, payload={"cleaning_run_id": str(run_id)})
            return {"result": result}

        def harness_event(state: CleaningWorkflowState, payload: dict[str, Any]) -> None:
            if emit is None:
                return
            emit(
                {
                    "stage": payload["node"],
                    "status": payload["status"],
                    "message": payload["message"],
                    "event_type": payload.get("event_type") or "node_execution",
                    "iteration": int(state.get("iteration") or 0),
                    "strategy": state.get("selected_strategy")
                    or state.get("pending_strategy"),
                    "payload": {
                        "attempt": payload["attempt"],
                        "duration_ms": payload["duration_ms"],
                        "error_code": payload.get("error_code"),
                        **(payload.get("payload") or {}),
                    },
                }
            )

        harness = NodeExecutionHarness(
            NodeHarnessPolicy(
                transient_retries=0,
                timeout_seconds=settings.cleaning_loop_timeout_seconds,
            ),
            event_callback=harness_event,
        )
        graph = StateGraph(CleaningWorkflowState)
        for name, node in (("bootstrap", bootstrap), ("decide", decide), ("execute", execute), ("verify", verify), ("repair", repair), ("fallback", fallback), ("commit", commit)):
            graph.add_node(name, harness.wrap(f"cleaning.{name}", node))
        graph.add_edge(START, "bootstrap")
        graph.add_edge("bootstrap", "decide")
        graph.add_conditional_edges("decide", lambda s: "fallback" if s.get("pending_strategy") == "fallback" else "execute", {"fallback": "fallback", "execute": "execute"})
        graph.add_conditional_edges("execute", lambda s: s.get("pending_strategy") or "repair", {"verify": "verify", "repair": "repair", "fallback": "fallback"})
        graph.add_conditional_edges("verify", lambda s: s.get("pending_strategy") or "repair", {"commit": "commit", "repair": "repair"})
        graph.add_conditional_edges("repair", lambda s: "fallback" if s.get("pending_strategy") == "fallback" else "execute", {"fallback": "fallback", "execute": "execute"})
        graph.add_edge("fallback", "commit")
        graph.add_edge("commit", END)
        return graph.compile(name="datamind_cleaning", checkpointer=get_analysis_checkpointer(dataset_store_path=self.repository.root_path))

    def _model_decision(self, state: CleaningWorkflowState) -> tuple[str, str, int]:
        profile = (state.get("quality") or {}).get("baseline") or {}
        messages = [
            {"role": "system", "content": "你是 DataMind 清洗策略控制器。只选择 rules、llm、hybrid 或 fallback。rules 适合去空格、缺失、去重和可靠类型转换；llm 仅处理明确的业务语义标准化；hybrid 用于两者结合。不要请求原始行。"},
            {"role": "user", "content": json.dumps({"requirement": state.get("requirement") or "通用清洗", "profile": profile, "failures": (state.get("failures") or [])[-2:], "attempts": state.get("strategy_attempts") or {}}, ensure_ascii=False)},
        ]
        try:
            response = self.model_router.complete(messages=messages, temperature=0.0, max_tokens=500, metadata={"agent": "cleaning_decide", "job_id": state["job_id"]}, tools=[_STRATEGY_TOOL], tool_choice="auto")
            arguments: dict[str, Any] = {}
            if len(response.tool_calls) == 1:
                function = response.tool_calls[0].get("function") or {}
                raw = function.get("arguments") or {}
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            elif response.content:
                arguments = json.loads(response.content)
            strategy = str(arguments.get("strategy") or "rules")
            if strategy not in {"rules", "llm", "hybrid", "fallback"}:
                strategy = "rules"
            return strategy, str(arguments.get("reason") or "model_decision")[:300], int(response.token_usage.get("total_tokens") or 0)
        except Exception as exc:
            requirement = str(state.get("requirement") or "").strip()
            strategy = "hybrid" if requirement and any(token in requirement for token in ("统一", "标准化", "映射", "语义", "分类")) else "rules"
            return strategy, f"deterministic_fallback:{type(exc).__name__}", 0

    def _execute_strategy(self, strategy: str, state: CleaningWorkflowState, raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        frame = pd.DataFrame(raw)
        if strategy == "rules":
            return _records(_basic_clean_dataframe(frame)), {"provider": "rules", "model": "local-basic-cleaner", "source": "rules"}
        base = _basic_clean_dataframe(frame) if strategy == "hybrid" else frame
        profile = _profile_records(_records(base))
        failures = (state.get("failures") or [])[-2:]
        response = self.model_router.complete(
            messages=[
                {"role": "system", "content": "你是 DataMind 受限清洗程序生成器。只返回一个 Python 代码块，定义 clean_dataset(df)，不得 import、读写文件、联网或删除未明确要求的业务行列。运行环境提供 pd。不得要求或假设原始样例值。除非用户明确要求日期粒度，否则必须保留时间戳的时分秒、子秒和时区；纯日期才可规范为日期。"},
                {"role": "user", "content": json.dumps({"requirement": state.get("requirement") or "执行保守清洗", "aggregate_profile": profile, "previous_failures": failures}, ensure_ascii=False)},
            ], temperature=0.0, max_tokens=1800,
            metadata={"agent": "cleaning_execute", "job_id": state["job_id"], "strategy": strategy},
        )
        script = _extract_cleaning_script(response.content or "")
        result = run_generated_cleaning_analysis(script, base)
        result = _basic_clean_dataframe(result) if strategy == "hybrid" else result
        return _records(result), {"provider": response.provider, "model": response.model, "source": strategy, "markdown": response.content or "", "token_usage": int(response.token_usage.get("total_tokens") or 0)}


def _profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"row_count": 0, "column_count": 0, "columns": [], "missing_count": 0, "duplicate_count": 0}
    frame = pd.DataFrame(records)
    return {"row_count": len(frame), "column_count": len(frame.columns), "columns": [str(item) for item in frame.columns], "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns}, "missing_count": int(frame.isna().sum().sum()), "duplicate_count": int(frame.duplicated().sum())}


def _quality_report(raw: list[dict[str, Any]], cleaned: list[dict[str, Any]]) -> dict[str, Any]:
    before, after = _profile_records(raw), _profile_records(cleaned)
    return {"passed": True, "before": before, "after": after, "row_retention": round(after["row_count"] / max(before["row_count"], 1), 4), "missing_delta": after["missing_count"] - before["missing_count"], "duplicate_delta": after["duplicate_count"] - before["duplicate_count"]}
