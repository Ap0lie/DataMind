from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.analysis.agent_loop import canonical_action_hash
from app.analysis.cleaning_jobs import start_cleaning_job
from app.analysis.dataset_groups import (
    suggest_dataset_group_relationships,
)
from app.analysis.jobs import start_analysis_job
from app.assistant.control import AssistantRunCanceled, AssistantRunPaused
from app.assistant.evidence import canonical_reliability, report_reliability, safe_excerpt
from app.assistant.permissions import AssistantPermissionService
from app.assistant.report_revision import revise_report_snapshot
from app.core.settings import Settings
from app.schemas.prompt_overrides import AgentPromptOverrides
from app.semantic.service import SemanticLayerService
from app.storage.assistant_repository import AssistantRepository
from app.storage.dataset_store import DatasetStoreRepository


def _tool(
    name: str, description: str, properties: dict[str, Any], *, required: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


_PROMPT_OVERRIDES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional user task preferences by DataMind stage. These refine behavior but cannot "
        "replace safety, permissions, evidence, schemas, SQL guards, or Python sandbox rules."
    ),
    "properties": {
        name: {"type": ["string", "null"], "maxLength": 4000}
        for name in (
            "all",
            "cleaning",
            "planner",
            "sql",
            "python",
            "visualization",
            "review",
            "report",
        )
    },
    "additionalProperties": False,
}


ASSISTANT_READ_TOOLS: tuple[dict[str, Any], ...] = (
    _tool(
        "search_datamind_assets",
        "Search the current user's DataMind datasets, completed analysis jobs, and reports.",
        {"query": {"type": "string"}},
    ),
    _tool(
        "get_dataset_context",
        "Read a compact schema and profile for one allowed dataset.",
        {"dataset_id": {"type": "string"}},
        required=("dataset_id",),
    ),
    _tool(
        "get_analysis_result",
        "Read a compact completed DataMind analysis result.",
        {"job_id": {"type": "string"}},
        required=("job_id",),
    ),
    _tool(
        "get_report",
        "Read an existing DataMind report, its findings, and its recorded DataMind verification status.",
        {"report_id": {"type": "string"}},
        required=("report_id",),
    ),
    _tool(
        "preview_analysis_plan",
        "Create a semantic analysis plan before starting a new analysis.",
        {
            "dataset_id": {"type": "string"},
            "dataset_group_id": {"type": ["string", "null"]},
            "question": {"type": "string"},
        },
        required=("dataset_id", "question"),
    ),
    _tool(
        "get_analysis_status",
        "Read the status of an allowed DataMind analysis job.",
        {"job_id": {"type": "string"}},
        required=("job_id",),
    ),
    _tool(
        "get_cleaning_status",
        "Read one cleaning job status.",
        {"dataset_id": {"type": "string"}, "job_id": {"type": "string"}},
        required=("dataset_id", "job_id"),
    ),
    _tool(
        "suggest_relationships",
        "Read validated relationship suggestions for a dataset group.",
        {"dataset_group_id": {"type": "string"}},
        required=("dataset_group_id",),
    ),
    _tool(
        "validate_semantic_model",
        "Validate a semantic draft without publishing it.",
        {"model_id": {"type": "string"}},
        required=("model_id",),
    ),
)

ASSISTANT_WRITE_TOOLS: tuple[dict[str, Any], ...] = (
    _tool(
        "start_analysis",
        "Start a DataMind analysis Loop and wait for its result. Use prompt_overrides when the user requests stage-specific analysis, chart, review, or report behavior.",
        {
            "dataset_id": {"type": "string"},
            "dataset_group_id": {"type": ["string", "null"]},
            "question": {"type": "string"},
            "planner_decision_id": {"type": ["string", "null"]},
            "prompt_overrides": _PROMPT_OVERRIDES_SCHEMA,
        },
        required=("dataset_id", "question"),
    ),
    _tool(
        "start_cleaning",
        "Start an autonomous cleaning Loop and wait for its validated version. Use prompt_overrides.cleaning for user-specific cleaning preferences.",
        {
            "dataset_id": {"type": "string"},
            "requirement": {"type": "string"},
            "cleaning_strategy": {"type": "string", "enum": ["auto", "rules", "llm", "hybrid"]},
            "prompt_overrides": _PROMPT_OVERRIDES_SCHEMA,
        },
        required=("dataset_id",),
    ),
    _tool(
        "activate_cleaning_version",
        "Activate a validated cleaning version.",
        {"dataset_id": {"type": "string"}, "run_id": {"type": "string"}},
        required=("dataset_id", "run_id"),
    ),
    _tool(
        "rollback_cleaning_version",
        "Rollback to an earlier validated cleaning version.",
        {"dataset_id": {"type": "string"}, "run_id": {"type": "string"}},
        required=("dataset_id", "run_id"),
    ),
    _tool(
        "update_column_metadata",
        "Update one column type, role, or description.",
        {
            "dataset_id": {"type": "string"},
            "column_name": {"type": "string"},
            "override_type": {"type": ["string", "null"]},
            "role": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
        },
        required=("dataset_id", "column_name"),
    ),
    _tool(
        "save_relationship_plan",
        "Save a validated dataset-group relationship plan.",
        {
            "dataset_group_id": {"type": "string"},
            "relationships": {"type": "array", "items": {"type": "object"}},
        },
        required=("dataset_group_id", "relationships"),
    ),
    _tool(
        "cancel_analysis",
        "Request cooperative cancellation of an analysis job.",
        {"job_id": {"type": "string"}},
        required=("job_id",),
    ),
    _tool(
        "retry_analysis",
        "Retry a failed or canceled analysis job with the original context.",
        {"job_id": {"type": "string"}},
        required=("job_id",),
    ),
    _tool(
        "rename_report",
        "Rename a report without altering its evidence.",
        {"report_id": {"type": "string"}, "title": {"type": "string"}},
        required=("report_id", "title"),
    ),
    _tool(
        "revise_report",
        "Create a new DataMind report version from an existing report. Preserve the original and use stage prompt overrides to satisfy the user's requested content and chart changes.",
        {
            "report_id": {"type": "string"},
            "instruction": {"type": "string", "maxLength": 4000},
            "prompt_overrides": _PROMPT_OVERRIDES_SCHEMA,
        },
        required=("report_id", "instruction"),
    ),
    _tool(
        "create_semantic_draft",
        "Create a semantic-model draft for an authorized dataset or group.",
        {
            "scope_type": {"type": "string", "enum": ["dataset", "dataset_group"]},
            "scope_id": {"type": "string"},
            "name": {"type": ["string", "null"]},
        },
        required=("scope_type", "scope_id"),
    ),
    _tool(
        "update_semantic_draft",
        "Update a semantic draft using its current revision.",
        {
            "model_id": {"type": "string"},
            "revision": {"type": "integer"},
            "name": {"type": ["string", "null"]},
            "definition": {"type": "object"},
        },
        required=("model_id", "revision", "definition"),
    ),
    _tool(
        "publish_semantic_model",
        "Publish a semantic draft only after strict validation.",
        {"model_id": {"type": "string"}},
        required=("model_id",),
    ),
    _tool(
        "soft_delete_asset",
        "Move an authorized asset to the 30-day recycle bin. Requires separate confirmation.",
        {
            "asset_type": {
                "type": "string",
                "enum": ["dataset", "dataset_group", "report", "semantic_model"],
            },
            "asset_id": {"type": "string"},
        },
        required=("asset_type", "asset_id"),
    ),
    _tool(
        "restore_asset",
        "Restore an authorized asset from the recycle bin.",
        {
            "asset_type": {
                "type": "string",
                "enum": ["dataset", "dataset_group", "report", "semantic_model"],
            },
            "asset_id": {"type": "string"},
        },
        required=("asset_type", "asset_id"),
    ),
)

ASSISTANT_TOOLS = (*ASSISTANT_READ_TOOLS, *ASSISTANT_WRITE_TOOLS)


def _prompt_overrides(arguments: dict[str, Any]) -> dict[str, str]:
    return AgentPromptOverrides.from_value(arguments.get("prompt_overrides")).as_dict()


def assistant_tools_for_mode(mode: str) -> tuple[dict[str, Any], ...]:
    return ASSISTANT_TOOLS if mode == "execute" else ASSISTANT_READ_TOOLS


class AssistantConfirmationRequired(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("Analysis plan requires confirmation.")
        self.payload = payload


class AssistantToolRuntime:
    """Server-scoped DataMind tools with capability checks for every mutation."""

    def __init__(
        self,
        *,
        store: DatasetStoreRepository,
        assistant_store: AssistantRepository,
        settings: Settings,
        run_id: UUID,
        conversation: dict[str, Any],
        event: Callable[..., None],
    ) -> None:
        self.store = store
        self.assistant_store = assistant_store
        self.settings = settings
        self.run_id = run_id
        self.conversation = conversation
        self.event = event
        self.evidence: dict[str, dict[str, Any]] = {}
        self.allowed_jobs: set[UUID] = set()
        self.permissions = AssistantPermissionService(store=store, assistant_store=assistant_store)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        run = self.assistant_store.get_run(self.run_id)
        available = assistant_tools_for_mode(run.execution_mode)
        if tool_name not in {item["function"]["name"] for item in available}:
            raise PermissionError("Tool is not available to the DataMind assistant.")
        authorization = self.permissions.authorize_tool(
            tool_name=tool_name,
            arguments=arguments,
            conversation=self.conversation,
            execution_mode=run.execution_mode,
        )
        if authorization is not None:
            self.event(
                event_type="permission.checked",
                status="completed",
                message=f"已授权 {authorization.capability}",
                tool_name=tool_name,
                payload={
                    "asset_type": authorization.asset_type,
                    "asset_id": str(authorization.asset_id),
                    "capability": authorization.capability,
                },
            )
            return self._execute_audited(tool_name, arguments, authorization)
        self.event(
            event_type="tool.started",
            status="running",
            message=f"正在执行 {tool_name}",
            tool_name=tool_name,
        )
        result = getattr(self, f"_tool_{tool_name}")(arguments)
        compact = _compact(result)
        self.event(
            event_type="tool.completed",
            status="completed",
            message=f"{tool_name} 已完成",
            tool_name=tool_name,
            payload={"summary": _summary(compact)},
        )
        return compact

    def _execute_audited(
        self, tool_name: str, arguments: dict[str, Any], authorization: Any
    ) -> dict[str, Any]:
        arguments_hash = canonical_action_hash(tool_name, arguments)
        idempotency_key = hashlib.sha256(
            f"{self.assistant_store.user_id}:{self.run_id}:{tool_name}:{arguments_hash}".encode()
        ).hexdigest()
        previous = self.assistant_store.get_action_by_idempotency_key(idempotency_key)
        if previous and previous["status"] == "completed":
            return _compact(
                previous["result"]
                | {"idempotent_replay": True, "action_id": str(previous["action_id"])}
            )
        action = previous or self.assistant_store.create_action(
            run_id=self.run_id,
            conversation_id=UUID(str(self.conversation["conversation_id"])),
            grant_id=authorization.grant_id,
            tool_name=tool_name,
            arguments_hash=arguments_hash,
            idempotency_key=idempotency_key,
            asset_type=authorization.asset_type,
            asset_id=authorization.asset_id,
        )
        self.assistant_store.update_run(
            self.run_id,
            current_action_id=action["action_id"],
            required_permission=authorization.capability,
        )
        self.event(
            event_type="action.planned",
            status="running",
            message=f"准备执行 {tool_name}",
            tool_name=tool_name,
            payload={
                "action_id": str(action["action_id"]),
                "asset_type": authorization.asset_type,
                "asset_id": str(authorization.asset_id),
            },
        )
        try:
            raw = getattr(self, f"_tool_{tool_name}")(arguments)
            before = raw.pop("_before_state", {}) if isinstance(raw, dict) else {}
            after = raw.pop("_after_state", {}) if isinstance(raw, dict) else {}
            reversible = bool(raw.pop("_reversible", False)) if isinstance(raw, dict) else False
            compact = _compact(raw)
            self.assistant_store.complete_action(
                action["action_id"],
                result=compact,
                before_state=before,
                after_state=after,
                reversible=reversible,
            )
            self.event(
                event_type="action.completed",
                status="completed",
                message=f"{tool_name} 已完成",
                tool_name=tool_name,
                payload={"action_id": str(action["action_id"]), "summary": _summary(compact)},
            )
            return compact | {"action_id": str(action["action_id"])}
        except Exception as exc:
            self.assistant_store.fail_action(action["action_id"], str(exc))
            raise

    def auto_retrieve(self, query: str) -> tuple[dict[str, Any], ...]:
        started = time.perf_counter()
        normalized_query = query.strip()
        if not normalized_query:
            return ()
        explicit_follow_up = _is_report_follow_up(normalized_query)
        reports: list[dict[str, Any]] = [
            report
            for report in self.store.list_reports(
                query=normalized_query,
                limit=20,
                include_content=False,
            )
            if _report_matches_query(report, normalized_query)
        ]
        if not reports:
            seen_report_ids: set[str] = set()
            for term in _report_retrieval_terms(normalized_query):
                for report in self.store.list_reports(
                    query=term,
                    limit=20,
                    include_content=False,
                ):
                    report_id = str(report.get("report_id") or report.get("id") or "")
                    if not report_id or report_id in seen_report_ids:
                        continue
                    if not _report_matches_query(report, normalized_query):
                        continue
                    seen_report_ids.add(report_id)
                    reports.append(report)
        if not reports and explicit_follow_up:
            reports = list(self.store.list_reports(limit=20, include_content=False))
        matched_query = bool(reports)
        items: list[dict[str, Any]] = []
        for report in _latest_report_candidates(reports, limit=3):
            try:
                items.append(
                    self._tool_get_report(
                        {"report_id": str(report.get("report_id") or report["id"])}
                    )
                )
            except RuntimeError:
                continue
        duration_ms = round((time.perf_counter() - started) * 1000)
        self.event(
            event_type="retrieval.completed",
            status="completed",
            message=f"已检索到 {len(items)} 份可引用报告。",
            payload={
                "report_count": len(items),
                "duration_ms": duration_ms,
                "matched_query": matched_query,
            },
        )
        return tuple(items)

    def _tool_search_datamind_assets(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip().lower()
        datasets = [item for item in self.store.list_datasets() if self._dataset_allowed(item.id)]
        reports = [
            item
            for item in self.store.list_reports(
                query=query or None,
                limit=20,
                include_content=False,
            )
            if self._report_allowed(item)
        ]
        jobs = [
            item
            for item in self.store.list_analysis_jobs(limit=100)
            if self._dataset_allowed(item.dataset_id) and item.status == "completed"
        ]

        def matches(*values: Any) -> bool:
            if not query:
                return True
            haystack = " ".join(str(value or "").lower() for value in values)
            if query in haystack:
                return True
            compact_query = "".join(character for character in query if character.isalnum())
            terms = {
                compact_query[index : index + 2] for index in range(max(0, len(compact_query) - 1))
            }
            return any(term and term in haystack for term in terms)

        return {
            "datasets": [
                {
                    "dataset_id": str(item.id),
                    "name": item.name,
                    "status": item.status,
                    "source_type": item.source_type,
                }
                for item in datasets
                if matches(item.name, item.source_type)
            ][:20],
            "reports": [
                {
                    "report_id": str(item["id"]),
                    "dataset_id": str(item["dataset_id"]),
                    "title": item["title"],
                    "question": item.get("question"),
                    "version": int(item.get("version") or 1),
                    "created_at": _display_timestamp(
                        item.get("created_at"), self.settings.display_timezone
                    ),
                }
                for item in reports
            ][:20],
            "analysis_jobs": [
                {
                    "job_id": str(item.id),
                    "dataset_id": str(item.dataset_id),
                    "question": item.question,
                    "report_id": str(item.report_id) if item.report_id else None,
                    "completed_at": _display_timestamp(
                        item.completed_at, self.settings.display_timezone
                    ),
                }
                for item in jobs
                if matches(item.question)
            ][:20],
        }

    def _tool_get_dataset_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        self._require_dataset(dataset_id)
        dataset = self.store.get_dataset(dataset_id)
        columns = self.store.list_column_metadata(dataset_id)
        samples = self.store.sample_analysis_records(dataset_id, limit=5)
        if columns:
            column_context = [
                {
                    "name": item["column_name"],
                    "type": item.get("override_type") or item.get("inferred_type"),
                    "role": item.get("role"),
                    "description": item.get("description"),
                }
                for item in columns
            ]
        else:
            names = sorted({str(key) for record in samples for key in record})
            column_context = [
                {"name": name, "type": "unknown", "role": "unknown", "description": ""}
                for name in names
            ]
        result = {
            "dataset_id": str(dataset_id),
            "name": dataset.name,
            "status": dataset.status,
            "row_count": self.store.count_analysis_records(dataset_id),
            "columns": column_context,
            "sample_records": samples,
        }
        self._add_evidence(
            "dataset",
            dataset_id,
            dataset.name,
            f"{result['row_count']} rows; {len(result['columns'])} columns",
            dataset_id,
            facts={
                "datasets_used": [dataset.name],
                "row_count": result["row_count"],
            },
        )
        return result

    def _tool_get_report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        report_id = UUID(str(arguments["report_id"]))
        report = self.store.get_report(report_id)
        if not self._report_allowed(report):
            raise RuntimeError("Report is outside the conversation scope.")
        dataset_id = UUID(str(report["dataset_id"]))
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        structured = (
            metadata.get("structured_report")
            if isinstance(metadata.get("structured_report"), dict)
            else {}
        )
        multi_dataset = (
            metadata.get("multi_dataset_context")
            if isinstance(metadata.get("multi_dataset_context"), dict)
            else {}
        )
        dataset_names: list[str] = []
        primary_dataset = (
            multi_dataset.get("primary_dataset")
            if isinstance(multi_dataset.get("primary_dataset"), dict)
            else {}
        )
        additional_datasets = (
            multi_dataset.get("additional_datasets")
            if isinstance(multi_dataset.get("additional_datasets"), list)
            else []
        )
        total_row_count = 0
        for item in (primary_dataset, *additional_datasets):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name and name not in dataset_names:
                dataset_names.append(name)
            row_count = item.get("row_count")
            if isinstance(row_count, int) and not isinstance(row_count, bool):
                total_row_count += max(0, row_count)
        if not dataset_names:
            try:
                dataset_names.append(self.store.get_dataset(dataset_id).name)
                total_row_count = self.store.count_analysis_records(dataset_id)
            except RuntimeError:
                pass
        report_verification = (
            metadata.get("statistical_verification")
            if isinstance(metadata.get("statistical_verification"), dict)
            else {}
        )
        lineage_verification = self._report_lineage_verification(report_id, report)
        reliability = canonical_reliability(report_verification, lineage_verification)
        statistical_verification = _compact_statistical_verification(
            _least_reliable_verification(report_verification, lineage_verification)
        )
        join_context = _compact_report_join_context(metadata)
        chart_context = _compact_report_chart_context(structured.get("charts"))
        result = {
            "report_id": str(report_id),
            "dataset_id": str(dataset_id),
            "title": report["title"],
            "version": int(report.get("version") or 1),
            "question": report.get("question") or metadata.get("question"),
            "datasets_used": dataset_names,
            "executive_summary": structured.get("executive_summary")
            or safe_excerpt(report.get("markdown"), limit=1600),
            "analysis_context": str(structured.get("analysis_context") or "").strip(),
            "analysis_contract": metadata.get("analysis_contract")
            if isinstance(metadata.get("analysis_contract"), dict)
            else None,
            "statistical_verification": statistical_verification,
            "join_context": join_context,
            "chart_context": chart_context,
            "key_findings": (structured.get("key_findings") or [])[:8],
            "validation_issues": (structured.get("validation_issues") or [])[:4],
            "recommended_next_steps": (structured.get("recommended_next_steps") or [])[:5],
            "reliability": reliability,
            "created_at": _display_timestamp(
                report.get("created_at"), self.settings.display_timezone
            ),
        }
        self._add_evidence(
            "report",
            report_id,
            str(report["title"]),
            safe_excerpt(result["executive_summary"]),
            dataset_id,
            artifact_role=str(arguments.get("_artifact_role") or "evidence"),
            reliability=reliability,
            facts={
                "datasets_used": dataset_names,
                "row_count": total_row_count,
                "executive_summary": result["executive_summary"],
                "analysis_context": result["analysis_context"],
                "analysis_contract": result["analysis_contract"],
                "statistical_verification": statistical_verification,
                "join_context": join_context,
                "chart_context": chart_context,
                "key_findings": [
                    str(item.get("content") or "").strip()
                    for item in result["key_findings"][:3]
                    if isinstance(item, dict) and str(item.get("content") or "").strip()
                ],
            },
        )
        return result

    def _tool_get_analysis_result(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job_id = UUID(str(arguments["job_id"]))
        job = self.store.get_analysis_job(job_id)
        self._require_dataset(job.dataset_id)
        if job.status != "completed" or not job.result:
            raise RuntimeError("Analysis result is not available.")
        result = _compact_analysis(job.result)
        self.allowed_jobs.add(job_id)
        report = (
            self._tool_get_report(
                {"report_id": str(job.report_id), "_artifact_role": "deliverable"}
            )
            if job.report_id is not None
            else None
        )
        lineage_reliability = canonical_reliability(
            result.get("reliability"),
            report.get("reliability") if report else None,
        )
        result["reliability"] = lineage_reliability
        if report is not None:
            report["reliability"] = lineage_reliability
            report_evidence = self.evidence.get(f"report:{job.report_id}")
            if report_evidence is not None:
                report_evidence["reliability"] = lineage_reliability
        self._add_evidence(
            "analysis_job",
            job_id,
            job.question,
            safe_excerpt(result.get("executive_summary")),
            job.dataset_id,
            reliability=lineage_reliability,
        )
        return result | {
            "job_id": str(job_id),
            "dataset_id": str(job.dataset_id),
            "report_id": str(job.report_id) if job.report_id else None,
            "report": report,
        }

    def _tool_preview_analysis_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        self._require_dataset(dataset_id)
        group_id = (
            UUID(str(arguments["dataset_group_id"])) if arguments.get("dataset_group_id") else None
        )
        if group_id:
            self._require_group(group_id, dataset_id)
        decision = SemanticLayerService(self.store).create_planner_decision(
            dataset_id=dataset_id, dataset_group_id=group_id, question=str(arguments["question"])
        )
        return {
            "planner_decision_id": str(decision["id"]),
            "dataset_id": str(dataset_id),
            "dataset_group_id": str(group_id) if group_id else None,
            "semantic_plan": decision.get("semantic_plan") or {},
            "component_scores": decision.get("component_scores") or {},
            "confidence": decision.get("calibrated_confidence"),
            "confidence_level": decision.get("confidence_level"),
            "requires_confirmation": bool(decision.get("requires_confirmation")),
        }

    def _tool_start_analysis(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        self._require_dataset(dataset_id)
        group_id = (
            UUID(str(arguments["dataset_group_id"])) if arguments.get("dataset_group_id") else None
        )
        group = self._require_group(group_id, dataset_id) if group_id else None
        question = str(arguments["question"]).strip()
        decision_id = (
            UUID(str(arguments["planner_decision_id"]))
            if arguments.get("planner_decision_id")
            else None
        )
        decision = (
            self.store.get_planner_decision(decision_id)
            if decision_id
            else SemanticLayerService(self.store).create_planner_decision(
                dataset_id=dataset_id, dataset_group_id=group_id, question=question
            )
        )
        pending = self.assistant_store.get_run(self.run_id).pending_confirmation
        accepted_decision = (
            str(pending.get("planner_decision_id") or "") == str(decision["id"])
            and pending.get("accepted") is True
        )
        if bool(decision.get("requires_confirmation")) and not (
            bool(decision.get("confirmed")) or accepted_decision
        ):
            raise AssistantConfirmationRequired(
                {
                    "planner_decision_id": str(decision["id"]),
                    "dataset_id": str(dataset_id),
                    "dataset_group_id": str(group_id) if group_id else None,
                    "question": question,
                    "confidence": decision.get("calibrated_confidence"),
                    "confidence_level": decision.get("confidence_level"),
                    "semantic_plan": decision.get("semantic_plan") or {},
                }
            )
        if accepted_decision:
            self.store.save_planner_feedback(
                decision_id=UUID(str(decision["id"])), action="accepted", corrected_plan={}
            )
            decision = self.store.get_planner_decision(UUID(str(decision["id"])))
        additional = tuple(
            item for item in (group.dataset_ids if group else ()) if item != dataset_id
        )
        relationships = tuple(group.relationships if group else ())
        multimodal = self._message_multimodal_inputs()
        job = self.store.create_analysis_job(
            dataset_id=dataset_id,
            dataset_group_id=group_id,
            additional_dataset_ids=additional,
            relationship_plan=relationships,
            question=question,
            prompt_overrides=_prompt_overrides(arguments),
            multimodal_inputs=multimodal,
            agent_mode="loop" if self.settings.agent_loop_enabled else "legacy",
        )
        self.store.attach_planner_decision_to_job(job_id=job.id, decision=decision)
        self.allowed_jobs.add(job.id)
        self.assistant_store.bind_analysis_job(self.run_id, job.id)
        start_analysis_job(
            job_id=job.id,
            user_id=self.assistant_store.user_id,
            dataset_store_path=self.settings.dataset_store_path,
        )
        return self._wait_for_analysis(job.id)

    def _tool_get_analysis_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job_id = UUID(str(arguments["job_id"]))
        job = self.store.get_analysis_job(job_id)
        self._require_dataset(job.dataset_id)
        return {
            "job_id": str(job.id),
            "status": job.status,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "error": job.error,
            "report_id": str(job.report_id) if job.report_id else None,
        }

    def _tool_start_cleaning(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        self._require_dataset(dataset_id)
        job = self.store.create_cleaning_job(
            dataset_id=dataset_id,
            requirement=str(arguments.get("requirement") or ""),
            cleaning_strategy=str(arguments.get("cleaning_strategy") or "auto"),
            prompt_overrides=_prompt_overrides(arguments),
        )
        start_cleaning_job(
            job_id=job.id,
            user_id=self.assistant_store.user_id,
            dataset_store_path=self.settings.dataset_store_path,
        )
        result = self._wait_for_cleaning(dataset_id, job.id)
        return result | {"_reversible": False}

    def _tool_get_cleaning_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        self._require_dataset(dataset_id)
        job = self.store.get_cleaning_job(UUID(str(arguments["job_id"])))
        if job.dataset_id != dataset_id:
            raise RuntimeError("Cleaning job is outside the selected dataset.")
        return {
            "job_id": str(job.id),
            "dataset_id": str(dataset_id),
            "status": job.status,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "cleaning_run_id": str(job.cleaning_run_id) if job.cleaning_run_id else None,
            "error": job.error,
        }

    def _tool_activate_cleaning_version(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._activate_cleaning(arguments)

    def _tool_rollback_cleaning_version(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._activate_cleaning(arguments)

    def _activate_cleaning(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        run_id = UUID(str(arguments["run_id"]))
        runs = self.store.list_cleaning_runs(dataset_id)
        previous = next((item for item in runs if item.get("is_active")), None)
        activated = self.store.activate_cleaning_run(dataset_id=dataset_id, run_id=run_id)
        return {
            "dataset_id": str(dataset_id),
            "cleaning_run_id": str(run_id),
            "version": activated.get("version"),
            "status": "active",
            "_before_state": {"run_id": str(previous["id"]) if previous else None},
            "_after_state": {"run_id": str(run_id)},
            "_reversible": previous is not None,
        }

    def _tool_update_column_metadata(self, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = UUID(str(arguments["dataset_id"]))
        column_name = str(arguments["column_name"])
        before = next(
            (
                item
                for item in self.store.list_column_metadata(dataset_id)
                if item["column_name"] == column_name
            ),
            {},
        )
        updated = self.store.update_column_metadata(
            dataset_id=dataset_id,
            column_name=column_name,
            override_type=arguments.get("override_type"),
            role=arguments.get("role"),
            description=arguments.get("description"),
        )
        return {
            "dataset_id": str(dataset_id),
            "column": updated,
            "_before_state": before,
            "_after_state": updated,
            "_reversible": bool(before),
        }

    def _tool_suggest_relationships(self, arguments: dict[str, Any]) -> dict[str, Any]:
        group_id = UUID(str(arguments["dataset_group_id"]))
        group = self.store.get_dataset_group(group_id)
        if any(not self._dataset_allowed(item) for item in group.dataset_ids):
            raise RuntimeError("Dataset group is outside the conversation scope.")
        suggestions = suggest_dataset_group_relationships(self.store, group_id=group_id)
        return {
            "dataset_group_id": str(group_id),
            "candidates": [item.model_dump(mode="json") for item in suggestions.candidates],
            "validation_issues": list(suggestions.validation_issues),
        }

    def _tool_save_relationship_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        group_id = UUID(str(arguments["dataset_group_id"]))
        group = self.store.get_dataset_group(group_id)
        if group.metadata.get("recycle_missing_dataset_ids"):
            raise ValueError("Restore recycled datasets before changing relationships.")
        before = [dict(item) for item in group.relationships]
        relationships = tuple(
            dict(item) for item in arguments.get("relationships") or [] if isinstance(item, dict)
        )
        updated = self.store.update_dataset_group_relationships(
            group_id=group_id, relationships=relationships
        )
        return {
            "dataset_group_id": str(group_id),
            "relationship_count": len(updated.relationships),
            "relationships": list(updated.relationships),
            "_before_state": {"relationships": before},
            "_after_state": {"relationships": list(updated.relationships)},
            "_reversible": True,
        }

    def _tool_cancel_analysis(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job = self.store.request_analysis_job_cancel(UUID(str(arguments["job_id"])))
        return {"job_id": str(job.id), "status": job.status}

    def _tool_retry_analysis(self, arguments: dict[str, Any]) -> dict[str, Any]:
        original = self.store.get_analysis_job(UUID(str(arguments["job_id"])))
        if original.status in {"queued", "running", "cancel_requested"}:
            raise ValueError("A running analysis job cannot be retried.")
        job = self.store.create_analysis_job(
            dataset_id=original.dataset_id,
            dataset_group_id=original.dataset_group_id,
            additional_dataset_ids=original.additional_dataset_ids,
            join_plan=original.join_plan,
            relationship_plan=original.relationship_plan,
            question=original.question,
            prompt_overrides=original.prompt_overrides,
            multimodal_inputs=original.multimodal_inputs,
            retry_of=original.id,
            agent_mode=original.agent_mode,
        )
        self.assistant_store.bind_analysis_job(self.run_id, job.id)
        start_analysis_job(
            job_id=job.id,
            user_id=self.assistant_store.user_id,
            dataset_store_path=self.settings.dataset_store_path,
        )
        return self._wait_for_analysis(job.id)

    def _tool_rename_report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        report_id = UUID(str(arguments["report_id"]))
        before = self.store.get_report(report_id)
        updated = self.store.update_report(report_id=report_id, title=str(arguments["title"]))
        return {
            "report_id": str(report_id),
            "title": updated["title"],
            "_before_state": {"title": before["title"]},
            "_after_state": {"title": updated["title"]},
            "_reversible": True,
        }

    def _tool_revise_report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        report_id = UUID(str(arguments["report_id"]))
        report = self.store.get_report(report_id)
        dataset_id = UUID(str(report["dataset_id"]))
        self._require_dataset(dataset_id)
        instruction = str(arguments.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("Report revision instruction cannot be empty.")
        revision = revise_report_snapshot(
            report=report,
            report_id=report_id,
            instruction=instruction,
        )
        revised_report_id = self.store.save_report(
            dataset_id=dataset_id,
            title=revision.title,
            markdown=revision.markdown,
            metadata=revision.metadata,
        )
        result = self._tool_get_report(
            {"report_id": str(revised_report_id), "_artifact_role": "deliverable"}
        )
        return {
            **result,
            "source_report_id": str(report_id),
            "revision_instruction": instruction,
            "analysis_rerun": False,
            "evidence_frozen": True,
            "source_evidence_fingerprint": revision.source_evidence_fingerprint,
            "_after_state": {"created_report_id": str(revised_report_id)},
            "_reversible": True,
        }

    def _tool_create_semantic_draft(self, arguments: dict[str, Any]) -> dict[str, Any]:
        model = SemanticLayerService(self.store).create_draft(
            scope_type=str(arguments["scope_type"]),
            scope_id=UUID(str(arguments["scope_id"])),
            name=str(arguments.get("name") or "Kimi semantic draft"),
        )
        return {
            "model_id": str(model["id"]),
            "name": model["name"],
            "version": model["version"],
            "revision": model["revision"],
            "status": model["status"],
        }

    def _tool_update_semantic_draft(self, arguments: dict[str, Any]) -> dict[str, Any]:
        model_id = UUID(str(arguments["model_id"]))
        before = self.store.get_semantic_model(model_id)
        updated = SemanticLayerService(self.store).update_draft(
            model_id,
            revision=int(arguments["revision"]),
            name=str(arguments["name"]) if arguments.get("name") else None,
            definition=dict(arguments["definition"]),
        )
        return {
            "model_id": str(model_id),
            "revision": updated["revision"],
            "status": updated["status"],
            "_before_state": {
                "revision": before["revision"],
                "name": before["name"],
                "definition": before["definition"],
            },
            "_after_state": {
                "revision": updated["revision"],
                "name": updated["name"],
                "definition": updated["definition"],
            },
            "_reversible": True,
        }

    def _tool_validate_semantic_model(self, arguments: dict[str, Any]) -> dict[str, Any]:
        model_id = UUID(str(arguments["model_id"]))
        model = self.store.get_semantic_model(model_id)
        return {"model_id": str(model_id)} | SemanticLayerService(self.store).validate(model)

    def _tool_publish_semantic_model(self, arguments: dict[str, Any]) -> dict[str, Any]:
        model = SemanticLayerService(self.store).publish(UUID(str(arguments["model_id"])))
        return {
            "model_id": str(model["id"]),
            "version": model["version"],
            "status": model["status"],
            "validation": model.get("validation") or {},
            "_reversible": False,
        }

    def _tool_soft_delete_asset(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action_hash = canonical_action_hash("soft_delete_asset", arguments)
        pending = self.assistant_store.get_run(self.run_id).pending_confirmation
        accepted = (
            pending.get("confirmation_type") == "soft_delete"
            and pending.get("action_hash") == action_hash
            and pending.get("accepted") is True
        )
        if not accepted:
            raise AssistantConfirmationRequired(
                {
                    "confirmation_type": "soft_delete",
                    "action_hash": action_hash,
                    "asset_type": str(arguments["asset_type"]),
                    "asset_id": str(arguments["asset_id"]),
                    "message": "该操作会把资产移入 30 天回收站。",
                }
            )
        recycled = self.store.soft_delete_asset(
            asset_type=str(arguments["asset_type"]), asset_id=UUID(str(arguments["asset_id"]))
        )
        self.event(
            event_type="asset.recycled",
            status="completed",
            message="资产已移入回收站。",
            tool_name="soft_delete_asset",
            payload={
                "asset_type": recycled["asset_type"],
                "asset_id": str(recycled["asset_id"]),
                "purge_after": recycled["purge_after"],
            },
        )
        return {**recycled, "asset_id": str(recycled["asset_id"]), "_reversible": True}

    def _tool_restore_asset(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.store.restore_asset(
            asset_type=str(arguments["asset_type"]), asset_id=UUID(str(arguments["asset_id"]))
        )
        return {**result, "asset_id": str(result["asset_id"]), "_reversible": False}

    def _wait_for_cleaning(self, dataset_id: UUID, job_id: UUID) -> dict[str, Any]:
        cursor = 0
        deadline = time.monotonic() + self.settings.assistant_timeout_seconds
        while time.monotonic() < deadline:
            if self.assistant_store.cancel_requested(self.run_id):
                self.store.request_cleaning_job_cancel(job_id)
                raise AssistantRunCanceled("Assistant run was canceled.")
            if self.assistant_store.pause_requested(self.run_id):
                raise AssistantRunPaused("Assistant run was paused.")
            for event in self.store.list_cleaning_job_events(job_id, after_sequence=cursor):
                cursor = max(cursor, int(event.get("sequence") or 0))
                self.event(
                    event_type="analysis.progress",
                    status=str(event.get("status") or "running"),
                    message=str(event.get("message") or "清洗进行中"),
                    tool_name="cleaning",
                    payload={
                        "job_id": str(job_id),
                        "progress": event.get("progress"),
                        "stage": event.get("stage"),
                    },
                )
            job = self.store.get_cleaning_job(job_id)
            if job.status == "completed":
                return {
                    "job_id": str(job.id),
                    "dataset_id": str(dataset_id),
                    "status": job.status,
                    "cleaning_run_id": str(job.cleaning_run_id) if job.cleaning_run_id else None,
                    "loop_summary": job.loop_summary or {},
                }
            if job.status in {"failed", "canceled", "interrupted"}:
                raise RuntimeError(job.error or f"Cleaning job ended with status {job.status}.")
            time.sleep(0.7)
        return {
            "job_id": str(job_id),
            "dataset_id": str(dataset_id),
            "status": "running",
            "message": "Cleaning continues in the background.",
        }

    def _wait_for_analysis(self, job_id: UUID) -> dict[str, Any]:
        cursor = 0
        deadline = time.monotonic() + self.settings.assistant_timeout_seconds
        while time.monotonic() < deadline:
            if self.assistant_store.cancel_requested(self.run_id):
                self.store.request_analysis_job_cancel(job_id)
                raise AssistantRunCanceled("Assistant run was canceled.")
            if self.assistant_store.pause_requested(self.run_id):
                raise AssistantRunPaused("Assistant run was paused.")
            for event in self.store.list_analysis_job_events(job_id, after_sequence=cursor):
                cursor = max(cursor, int(event.get("sequence") or 0))
                self.event(
                    event_type="analysis.progress",
                    status=str(event.get("status") or "running"),
                    message=str(event.get("message") or "分析进行中"),
                    tool_name=str(event.get("node") or "analysis"),
                    payload={
                        "job_id": str(job_id),
                        "progress": event.get("progress"),
                        "stage": event.get("stage") or event.get("node"),
                    },
                )
            job = self.store.get_analysis_job(job_id)
            if job.status == "completed":
                return self._tool_get_analysis_result({"job_id": str(job_id)})
            if job.status in {"failed", "canceled", "interrupted"}:
                raise RuntimeError(job.error or f"Analysis job ended with status {job.status}.")
            time.sleep(0.7)
        return {
            "job_id": str(job_id),
            "status": "running",
            "message": "Analysis continues in the background.",
        }

    def _message_multimodal_inputs(self) -> tuple[dict[str, Any], ...]:
        run = self.assistant_store.get_run(self.run_id)
        attachments = self.assistant_store.list_message_attachments(run.user_message_id)
        result = []
        for item in attachments:
            if str(item.get("attachment_kind") or "image") != "image":
                continue
            path = self.assistant_store.attachment_path(UUID(str(item["id"])))
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            result.append(
                {
                    "kind": "image",
                    "title": item["file_name"],
                    "description": "Kimi 对话中上传的分析辅助图片。",
                    "media_type": item["media_type"],
                    "data_url": f"data:{item['media_type']};base64,{encoded}",
                }
            )
        return tuple(result)

    def _dataset_allowed(self, dataset_id: UUID) -> bool:
        scope_type, scope_id = self.conversation["scope_type"], self.conversation.get("scope_id")
        if scope_type == "auto":
            return True
        if scope_type == "dataset":
            return dataset_id == scope_id
        if scope_type == "dataset_group":
            return dataset_id in self.store.get_dataset_group(scope_id).dataset_ids
        if scope_type == "report":
            return dataset_id == UUID(str(self.store.get_report(scope_id)["dataset_id"]))
        return False

    def _report_allowed(self, report: dict[str, Any]) -> bool:
        if not self._dataset_allowed(UUID(str(report["dataset_id"]))):
            return False
        if self.conversation["scope_type"] != "report":
            return True
        scope_id = UUID(str(self.conversation["scope_id"]))
        if UUID(str(report["id"])) == scope_id:
            return True
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        revision = (
            metadata.get("report_revision")
            if isinstance(metadata.get("report_revision"), dict)
            else {}
        )
        return str(revision.get("source_report_id") or "") == str(scope_id)

    def _report_lineage_verification(
        self,
        report_id: UUID,
        report: dict[str, Any],
    ) -> dict[str, Any] | None:
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        revision = (
            metadata.get("report_revision")
            if isinstance(metadata.get("report_revision"), dict)
            else {}
        )
        lineage_report_ids = {str(report_id)}
        if revision.get("source_report_id"):
            lineage_report_ids.add(str(revision["source_report_id"]))
        for job in self.store.list_analysis_jobs(
            dataset_id=UUID(str(report["dataset_id"])),
            limit=200,
        ):
            if job.report_id is None or str(job.report_id) not in lineage_report_ids:
                continue
            result = job.result if isinstance(job.result, dict) else {}
            verification = result.get("statistical_verification")
            return verification if isinstance(verification, dict) else None
        return None

    def _require_dataset(self, dataset_id: UUID) -> Any:
        dataset = self.store.get_dataset(dataset_id)
        if not self._dataset_allowed(dataset_id):
            raise RuntimeError("Dataset is outside the conversation scope.")
        return dataset

    def _require_group(self, group_id: UUID, dataset_id: UUID) -> Any:
        group = self.store.get_dataset_group(group_id)
        if dataset_id not in group.dataset_ids or any(
            not self._dataset_allowed(item) for item in group.dataset_ids
        ):
            raise RuntimeError("Dataset group is outside the conversation scope.")
        return group

    def _add_evidence(
        self,
        source_type: str,
        source_id: UUID,
        label: str,
        excerpt: str,
        dataset_id: UUID,
        *,
        artifact_role: str = "evidence",
        reliability: dict[str, Any] | None = None,
        facts: dict[str, Any] | None = None,
    ) -> None:
        if artifact_role == "deliverable":
            for item in self.evidence.values():
                if item.get("artifact_role") == "deliverable":
                    item["artifact_role"] = "evidence"
        key = f"{source_type}:{source_id}"
        self.evidence[key] = {
            "source_type": source_type,
            "source_id": str(source_id),
            "label": label,
            "excerpt": safe_excerpt(excerpt),
            "dataset_id": str(dataset_id),
            "artifact_role": artifact_role,
            "reliability": reliability
            or {"status": "unverified", "summary": "未提供统计审查状态。"},
        }
        if facts:
            self.evidence[key]["facts"] = facts


_REPORT_BAR_CATEGORY_DISPLAY_LIMIT = 24
_RELIABILITY_RANK = {
    "rejected": 0,
    "warning": 1,
    "unverified": 2,
    "verified": 3,
}


def _least_reliable_verification(*values: Any) -> dict[str, Any]:
    candidates = [
        value
        for value in values
        if isinstance(value, dict)
        and (str(value.get("status") or "").strip() or value.get("requires_replan") is True)
    ]
    if not candidates:
        return {}
    return min(
        candidates,
        key=lambda value: _RELIABILITY_RANK[report_reliability(value)["status"]],
    )


def _bounded_report_metadata(value: Any, *, depth: int = 0) -> Any:
    """Keep report evidence useful without injecting unbounded report payloads."""

    if depth >= 4:
        return safe_excerpt(value, limit=400)
    if isinstance(value, dict):
        return {
            str(key): _bounded_report_metadata(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, list | tuple):
        return [_bounded_report_metadata(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, str):
        return safe_excerpt(value, limit=800)
    return value


def _compact_statistical_verification(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    checks = value.get("checks") if isinstance(value.get("checks"), list) else []
    return {
        "status": value.get("status"),
        "summary": safe_excerpt(value.get("summary"), limit=800),
        "requires_replan": bool(value.get("requires_replan")),
        "numeric_evidence_coverage": value.get("numeric_evidence_coverage"),
        "checks": [
            {
                "code": check.get("code"),
                "status": check.get("status"),
                "message": safe_excerpt(check.get("message"), limit=500),
                "details": _bounded_report_metadata(check.get("details") or {}),
            }
            for check in checks[:16]
            if isinstance(check, dict)
        ],
    }


def _compact_report_join_context(metadata: dict[str, Any]) -> dict[str, Any] | None:
    summary = metadata.get("join_summary") if isinstance(metadata.get("join_summary"), dict) else {}
    lineage = (
        metadata.get("analysis_lineage")
        if isinstance(metadata.get("analysis_lineage"), dict)
        else {}
    )
    graph = (
        lineage.get("relationship_graph")
        if isinstance(lineage.get("relationship_graph"), dict)
        else {}
    )
    grain_plan = lineage.get("grain_plan") if isinstance(lineage.get("grain_plan"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    names_by_entity = {
        str(node.get("entity_id")): str(node.get("name") or node.get("entity_id"))
        for node in nodes
        if isinstance(node, dict) and node.get("entity_id")
    }
    join_path = grain_plan.get("join_path") if isinstance(grain_plan.get("join_path"), list) else []
    executed_joins = summary.get("joins") if isinstance(summary.get("joins"), list) else []
    paths: list[dict[str, Any]] = []
    for index, edge in enumerate(join_path[:12]):
        if not isinstance(edge, dict):
            continue
        executed = executed_joins[index] if index < len(executed_joins) else {}
        if not isinstance(executed, dict):
            executed = {}
        paths.append(
            {
                "left_dataset": names_by_entity.get(
                    str(edge.get("left_entity_id")), str(edge.get("left_entity_id") or "")
                ),
                "left_column": executed.get("left_column"),
                "right_dataset": names_by_entity.get(
                    str(edge.get("right_entity_id")),
                    str(executed.get("right_dataset_name") or edge.get("right_entity_id") or ""),
                ),
                "right_column": executed.get("right_column"),
                "cardinality": edge.get("cardinality"),
                "join_type": edge.get("join_type") or executed.get("join_type"),
                "row_expansion_ratio": executed.get("row_expansion_ratio"),
                "before_rows": executed.get("before_rows"),
                "after_rows": executed.get("after_rows"),
                "risk_note": safe_excerpt(edge.get("risk_note"), limit=400),
            }
        )
    if not paths:
        paths = [
            _bounded_report_metadata(item) for item in executed_joins[:12] if isinstance(item, dict)
        ]
    if not summary and not paths and not grain_plan:
        return None
    return {
        "mode": summary.get("mode"),
        "dataset_count": summary.get("dataset_count"),
        "joined_dataset_count": summary.get("joined_dataset_count"),
        "joined_row_count": summary.get("joined_row_count"),
        "joined_column_count": summary.get("joined_column_count"),
        "row_expansion_ratio": summary.get("row_expansion_ratio"),
        "skipped_join_count": summary.get("skipped_join_count"),
        "metric_grain": _bounded_report_metadata(grain_plan.get("metric_grain") or []),
        "safe": grain_plan.get("safe"),
        "warnings": _bounded_report_metadata(grain_plan.get("warnings") or []),
        "paths": paths,
    }


def _compact_report_chart_context(value: Any) -> list[dict[str, Any]]:
    charts = value if isinstance(value, list | tuple) else []
    result: list[dict[str, Any]] = []
    for chart in charts[:8]:
        if not isinstance(chart, dict):
            continue
        chart_type = str(chart.get("chart_type") or "")
        spec = chart.get("spec") if isinstance(chart.get("spec"), dict) else {}
        data = chart.get("data") if isinstance(chart.get("data"), list) else []
        context = {
            "title": chart.get("title"),
            "chart_type": chart_type,
            "x": spec.get("x") or spec.get("names"),
            "y": spec.get("y") or spec.get("values"),
            "data_point_count": len(data),
            "denominator_scope": spec.get("denominator_scope"),
            "displayed_category_count": spec.get("displayed_category_count"),
            "excluded_category_count": spec.get("excluded_category_count"),
            "explanation": safe_excerpt(chart.get("explanation"), limit=1000),
        }
        if chart_type == "bar" and len(data) > _REPORT_BAR_CATEGORY_DISPLAY_LIMIT:
            context.update(
                {
                    "display_rule": "top_n_by_y_descending",
                    "displayed_category_count": _REPORT_BAR_CATEGORY_DISPLAY_LIMIT,
                    "excluded_category_count": len(data) - _REPORT_BAR_CATEGORY_DISPLAY_LIMIT,
                    "denominator_scope": "not_applicable_for_bar_chart",
                }
            )
        result.append(context)
    return result


def _compact_analysis(result: dict[str, Any]) -> dict[str, Any]:
    structured = (
        result.get("structured_report") if isinstance(result.get("structured_report"), dict) else {}
    )
    python_result = (
        result.get("python_result") if isinstance(result.get("python_result"), dict) else {}
    )
    sql_result = result.get("sql_result") if isinstance(result.get("sql_result"), dict) else {}
    reliability = report_reliability(result.get("statistical_verification"))
    return {
        "question": result.get("question"),
        "executive_summary": structured.get("executive_summary"),
        "key_findings": (structured.get("key_findings") or result.get("final_insights") or [])[:8],
        "validation_issues": (
            structured.get("validation_issues") or result.get("validation_issues") or []
        )[:4],
        "recommended_next_steps": (structured.get("recommended_next_steps") or [])[:5],
        "sql_rows": (sql_result.get("rows") or [])[:20],
        "python_statistics": python_result.get("statistics") or {},
        "report_id": result.get("report_id"),
        "statistical_verification": result.get("statistical_verification"),
        "reliability": reliability,
    }


def _latest_report_candidates(
    reports: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    latest: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    ordered = sorted(
        reports,
        key=lambda report: (
            str(report.get("created_at") or ""),
            int(report.get("version") or 0),
            str(report.get("report_id") or report.get("id") or ""),
        ),
        reverse=True,
    )
    for report in ordered:
        question = _normalized_report_identity(_report_question(report))
        title = _normalized_report_identity(report.get("title"))
        # Re-running the same user question supersedes the older result even if
        # the corrected planner chose a different primary dataset.  Dataset ID
        # is only part of the fallback identity when no question was persisted.
        key = (
            ("question", question)
            if question
            else (
                "title",
                str(report.get("dataset_id") or ""),
                title or str(report.get("report_id") or report.get("id") or ""),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        latest.append(report)
        if len(latest) >= limit:
            break
    return tuple(latest)


def _normalized_report_identity(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def _report_question(report: dict[str, Any]) -> str:
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    return str(report.get("question") or metadata.get("question") or "").strip()


def _report_matches_query(report: dict[str, Any], query: str) -> bool:
    """Match report topics, never incidental words in report bodies or verification metadata."""

    topic = " ".join(
        value
        for value in (_report_question(report), str(report.get("title") or "").strip())
        if value
    )
    normalized_query = _normalized_report_identity(query)
    normalized_topic = _normalized_report_identity(topic)
    if not normalized_query or not normalized_topic:
        return False
    query_terms = set(_report_retrieval_terms(query))
    if not query_terms:
        return False
    if normalized_query in normalized_topic:
        return True

    topic_terms = set(_report_retrieval_terms(topic))
    overlap = query_terms & topic_terms
    return bool(overlap)


_RETRIEVAL_STOP_TERMS = {
    "please",
    "could",
    "would",
    "what",
    "when",
    "where",
    "which",
    "with",
    "about",
    "show",
    "tell",
    "如何",
    "怎么",
    "怎样",
    "请问",
    "一下",
    "给出",
    "完整",
    "结论",
    "报告",
    "结果",
    "分析",
    "统计",
    "审查",
    "通过",
    "回答",
    "发送",
    "按钮",
    "测试",
    "问题",
    "用户",
    "解释",
    "含义",
}


def _report_retrieval_terms(query: str) -> tuple[str, ...]:
    """Return topic-bearing terms without falling back to unrelated recent reports."""

    normalized = query.casefold()
    terms: list[str] = []
    for word in re.findall(r"[a-z0-9_]{3,}", normalized):
        if word not in _RETRIEVAL_STOP_TERMS:
            terms.append(word)
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        for index in range(max(len(segment) - 1, 0)):
            term = segment[index : index + 2]
            if term not in _RETRIEVAL_STOP_TERMS:
                terms.append(term)
    return tuple(dict.fromkeys(terms))


def _is_report_follow_up(query: str) -> bool:
    normalized = query.casefold()
    return any(
        marker in normalized
        for marker in (
            "上述",
            "上面",
            "刚才",
            "刚完成",
            "最新报告",
            "最近报告",
            "这份报告",
            "该报告",
            "给出完整结论",
            "完整结论",
            "完整结果",
            "继续上面的分析",
            "补全上面的回答",
            "complete conclusion",
            "the report",
            "previous result",
        )
    )


def _display_timestamp(value: Any, timezone_name: str) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            timezone = UTC
        return parsed.astimezone(timezone).isoformat(timespec="seconds")
    except ValueError:
        return text


def _compact(value: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= 50_000:
        return value
    result = dict(value)
    for key in (
        "sample_records",
        "sql_rows",
        "key_findings",
        "datasets",
        "reports",
        "analysis_jobs",
    ):
        if isinstance(result.get(key), list):
            result[key] = result[key][:8]
    result["truncated"] = True
    return result


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "dataset_id",
            "job_id",
            "report_id",
            "status",
            "title",
            "row_count",
            "confidence_level",
        )
        if key in value
    }
