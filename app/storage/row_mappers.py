from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import UUID

from app.storage.models import (
    StoredAnalysisJob,
    StoredCleaningJob,
    StoredDataset,
    StoredDatasetGroup,
)


def stored_dataset_from_row(row: sqlite3.Row) -> StoredDataset:
    source_metadata = json_loads(row["source_metadata"], {})
    return StoredDataset(
        id=UUID(str(row["id"])),
        user_id=str(row["user_id"] if "user_id" in row.keys() else "default"),
        name=str(row["name"]),
        source_type=str(row["source_type"]),
        status=str(row["status"]),
        source_metadata=source_metadata if isinstance(source_metadata, dict) else {},
        created_at=str(row["created_at"]) if row["created_at"] else None,
        updated_at=str(row["updated_at"]) if row["updated_at"] else None,
    )


def stored_dataset_group_from_row(row: sqlite3.Row) -> StoredDatasetGroup:
    dataset_ids = json_loads(row["dataset_ids"], [])
    relationships = json_loads(row["relationships"], [])
    metadata = json_loads(row["metadata"], {})
    return StoredDatasetGroup(
        id=UUID(str(row["id"])),
        user_id=str(row["user_id"] if "user_id" in row.keys() else "default"),
        name=str(row["name"]),
        description=str(row["description"] if "description" in row.keys() else ""),
        dataset_ids=(
            tuple(UUID(str(item)) for item in dataset_ids if item)
            if isinstance(dataset_ids, list)
            else ()
        ),
        relationships=(
            tuple(item for item in relationships if isinstance(item, dict))
            if isinstance(relationships, list)
            else ()
        ),
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=str(row["created_at"]) if row["created_at"] else None,
        updated_at=str(row["updated_at"]) if row["updated_at"] else None,
    )


def stored_cleaning_job_from_row(row: sqlite3.Row) -> StoredCleaningJob:
    events = json_loads(row["events"], [])
    result = json_loads(row["result"], None)
    return StoredCleaningJob(
        id=UUID(str(row["id"])),
        user_id=str(row["user_id"]),
        dataset_id=UUID(str(row["dataset_id"])),
        requirement=str(row["requirement"] or ""),
        prompt_overrides=json_loads(
            row["prompt_overrides"] if "prompt_overrides" in row.keys() else "{}",
            {},
        ),
        cleaning_strategy=str(row["cleaning_strategy"] or "auto"),
        status=str(row["status"]),
        progress=bounded_progress(row["progress"]),
        current_stage=str(row["current_stage"]),
        events=(
            tuple(item for item in events if isinstance(item, dict))
            if isinstance(events, list)
            else ()
        ),
        selected_strategy=optional_text(row["selected_strategy"]),
        loop_summary=json_loads(row["loop_summary"], {}),
        terminal_reason=optional_text(row["terminal_reason"]),
        result=result if isinstance(result, dict) else None,
        error=optional_text(row["error"]),
        cleaning_run_id=UUID(str(row["cleaning_run_id"])) if row["cleaning_run_id"] else None,
        retry_of=UUID(str(row["retry_of"])) if row["retry_of"] else None,
        cancel_requested=bool(row["cancel_requested"]),
        broker_task_id=optional_text(row["broker_task_id"]),
        attempt_count=int(row["attempt_count"] or 0),
        lease_owner=optional_text(row["lease_owner"]),
        lease_expires_at=optional_text(row["lease_expires_at"]),
        heartbeat_at=optional_text(row["heartbeat_at"]),
        checkpoint_thread_id=optional_text(row["checkpoint_thread_id"]) or str(row["id"]),
        created_at=optional_text(row["created_at"]),
        updated_at=optional_text(row["updated_at"]),
        started_at=optional_text(row["started_at"]),
        completed_at=optional_text(row["completed_at"]),
    )


def cleaning_job_event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sequence": int(row["sequence"]),
        "stage": str(row["stage"]),
        "status": str(row["status"]),
        "progress": bounded_progress(row["progress"]),
        "message": str(row["message"]),
        "event_type": optional_text(row["event_type"]),
        "iteration": int(row["iteration"]) if row["iteration"] is not None else None,
        "strategy": optional_text(row["strategy"]),
        "payload": json_loads(row["payload"], {}),
        "created_at": str(row["created_at"]),
    }


def stored_analysis_job_from_row(row: sqlite3.Row) -> StoredAnalysisJob:
    multimodal_inputs = json_loads(row["multimodal_inputs"], [])
    dataset_group_id = row["dataset_group_id"] if "dataset_group_id" in row.keys() else None
    additional_dataset_ids = json_loads(
        row["additional_dataset_ids"] if "additional_dataset_ids" in row.keys() else "[]",
        [],
    )
    join_plan = json_loads(
        row["join_plan"] if "join_plan" in row.keys() else "[]",
        [],
    )
    relationship_plan = json_loads(
        row["relationship_plan"] if "relationship_plan" in row.keys() else "[]",
        [],
    )
    events = json_loads(row["events"], [])
    result = json_loads(row["result"], None)
    return StoredAnalysisJob(
        id=UUID(str(row["id"])),
        user_id=str(row["user_id"] if "user_id" in row.keys() else "default"),
        dataset_id=UUID(str(row["dataset_id"])),
        dataset_group_id=UUID(str(dataset_group_id)) if dataset_group_id else None,
        additional_dataset_ids=(
            tuple(
                UUID(str(item))
                for item in additional_dataset_ids
                if item and str(item) != str(row["dataset_id"])
            )
            if isinstance(additional_dataset_ids, list)
            else ()
        ),
        join_plan=(
            tuple(item for item in join_plan if isinstance(item, dict))
            if isinstance(join_plan, list)
            else ()
        ),
        relationship_plan=(
            tuple(item for item in relationship_plan if isinstance(item, dict))
            if isinstance(relationship_plan, list)
            else ()
        ),
        question=str(row["question"]),
        prompt_overrides=json_loads(
            row["prompt_overrides"] if "prompt_overrides" in row.keys() else "{}",
            {},
        ),
        multimodal_inputs=(
            tuple(item for item in multimodal_inputs if isinstance(item, dict))
            if isinstance(multimodal_inputs, list)
            else ()
        ),
        status=str(row["status"]),
        progress=bounded_progress(row["progress"]),
        current_stage=str(row["current_stage"]),
        events=(
            tuple(item for item in events if isinstance(item, dict))
            if isinstance(events, list)
            else ()
        ),
        result=result if isinstance(result, dict) else None,
        error=str(row["error"]) if row["error"] is not None else None,
        report_id=UUID(str(row["report_id"])) if row["report_id"] else None,
        retry_of=UUID(str(row["retry_of"])) if row["retry_of"] else None,
        cancel_requested=bool(row["cancel_requested"]),
        broker_task_id=(
            str(row["broker_task_id"])
            if "broker_task_id" in row.keys() and row["broker_task_id"]
            else None
        ),
        attempt_count=int(row["attempt_count"] if "attempt_count" in row.keys() else 0),
        lease_owner=(
            str(row["lease_owner"])
            if "lease_owner" in row.keys() and row["lease_owner"]
            else None
        ),
        lease_expires_at=(
            str(row["lease_expires_at"])
            if "lease_expires_at" in row.keys() and row["lease_expires_at"]
            else None
        ),
        heartbeat_at=(
            str(row["heartbeat_at"])
            if "heartbeat_at" in row.keys() and row["heartbeat_at"]
            else None
        ),
        checkpoint_thread_id=(
            str(row["checkpoint_thread_id"])
            if "checkpoint_thread_id" in row.keys() and row["checkpoint_thread_id"]
            else str(row["id"])
        ),
        planner_decision_id=(
            UUID(str(row["planner_decision_id"]))
            if "planner_decision_id" in row.keys() and row["planner_decision_id"]
            else None
        ),
        semantic_model_id=(
            UUID(str(row["semantic_model_id"]))
            if "semantic_model_id" in row.keys() and row["semantic_model_id"]
            else None
        ),
        semantic_model_version=(
            int(row["semantic_model_version"])
            if "semantic_model_version" in row.keys()
            and row["semantic_model_version"] is not None
            else None
        ),
        agent_mode=str(
            row["agent_mode"]
            if "agent_mode" in row.keys() and row["agent_mode"]
            else "legacy"
        ),
        loop_summary=(
            json_loads(row["loop_summary"], {}) if "loop_summary" in row.keys() else {}
        ),
        loop_terminal_reason=(
            str(row["loop_terminal_reason"])
            if "loop_terminal_reason" in row.keys() and row["loop_terminal_reason"]
            else None
        ),
        report_strategy=(
            str(row["report_strategy"])
            if "report_strategy" in row.keys() and row["report_strategy"]
            else None
        ),
        report_revision_count=(
            int(row["report_revision_count"])
            if "report_revision_count" in row.keys()
            and row["report_revision_count"] is not None
            else 0
        ),
        report_terminal_reason=(
            str(row["report_terminal_reason"])
            if "report_terminal_reason" in row.keys() and row["report_terminal_reason"]
            else None
        ),
        created_at=str(row["created_at"]) if row["created_at"] else None,
        updated_at=str(row["updated_at"]) if row["updated_at"] else None,
        started_at=str(row["started_at"]) if row["started_at"] else None,
        completed_at=str(row["completed_at"]) if row["completed_at"] else None,
    )


def semantic_model_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": UUID(str(row["id"])),
        "user_id": str(row["user_id"]),
        "scope_type": str(row["scope_type"]),
        "scope_id": UUID(str(row["scope_id"])),
        "name": str(row["name"]),
        "version": int(row["version"]),
        "revision": int(row["revision"]),
        "status": str(row["status"]),
        "source": str(row["source"]),
        "parent_model_id": (
            UUID(str(row["parent_model_id"])) if row["parent_model_id"] else None
        ),
        "definition": json_loads(row["definition"], {}),
        "schema_fingerprint": str(row["schema_fingerprint"] or ""),
        "validation": json_loads(row["validation"], {}),
        "created_at": str(row["created_at"]) if row["created_at"] else None,
        "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
        "published_at": str(row["published_at"]) if row["published_at"] else None,
    }


def data_snapshot_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": UUID(str(row["id"])),
        "user_id": str(row["user_id"]),
        "dataset_id": UUID(str(row["dataset_id"])),
        "source": str(row["source"]),
        "row_count": int(row["row_count"]),
        "sample_size": int(row["sample_size"]),
        "fingerprint": str(row["fingerprint"]),
        "profile": json_loads(row["profile"], {}),
        "created_at": str(row["created_at"]),
    }


def data_drift_event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": UUID(str(row["id"])),
        "user_id": str(row["user_id"]),
        "dataset_id": UUID(str(row["dataset_id"])),
        "baseline_snapshot_id": UUID(str(row["baseline_snapshot_id"])),
        "current_snapshot_id": UUID(str(row["current_snapshot_id"])),
        "status": str(row["status"]),
        "changes": json_loads(row["changes"], []),
        "affected_assets": json_loads(row["affected_assets"], []),
        "recommended_actions": json_loads(row["recommended_actions"], []),
        "created_at": str(row["created_at"]),
        "acknowledged_at": optional_text(row["acknowledged_at"]),
    }


def cleaning_run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "user_id": row["user_id"] if "user_id" in row.keys() else "default",
        "version": int(row["version"] if "version" in row.keys() else 1),
        "is_active": bool(row["is_active"] if "is_active" in row.keys() else 0),
        "created_at": row["created_at"],
        "provider": row["provider"],
        "model": row["model"],
        "prompt": row["prompt"],
        "result_markdown": row["result_markdown"],
        "cleaned_dataset": json_loads(row["cleaned_dataset"], {}),
        "raw_summary": json_loads(row["raw_summary"], {}),
        "previous_summary": json_loads(row["previous_summary"], {}),
        "current_summary": json_loads(row["current_summary"], {}),
        "diff_summary": json_loads(row["diff_summary"], {}),
        "job_id": row["job_id"] if "job_id" in row.keys() else None,
    }


def column_metadata_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "column_name": row["column_name"],
        "inferred_type": row["inferred_type"],
        "override_type": row["override_type"],
        "role": row["role"],
        "description": row["description"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def report_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata = json_loads(row["metadata"], {})
    question = row["question"] if "question" in row.keys() else None
    if question and isinstance(metadata, dict) and "question" not in metadata:
        metadata["question"] = str(question)
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "created_at": row["created_at"],
        "updated_at": (
            row["updated_at"] if "updated_at" in row.keys() else row["created_at"]
        ),
        "version": int(row["version"] if "version" in row.keys() else 1),
        "title": row["title"],
        "markdown": row["markdown"],
        "metadata": metadata,
    }


def analysis_job_event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    token_usage = json_loads(row["token_usage"], {}) if "token_usage" in row.keys() else {}
    node = str(row["node"])
    return {
        "sequence": int(row["sequence"]),
        "node": node,
        "stage": node,
        "status": str(row["status"]),
        "progress": bounded_progress(row["progress"]),
        "attempt": int(row["attempt"]),
        "duration_ms": float(row["duration_ms"]) if row["duration_ms"] is not None else None,
        "provider": str(row["provider"]) if row["provider"] else None,
        "model": str(row["model"]) if row["model"] else None,
        "token_usage": token_usage if isinstance(token_usage, dict) else {},
        "error_code": str(row["error_code"]) if row["error_code"] else None,
        "event_type": (
            str(row["event_type"])
            if "event_type" in row.keys() and row["event_type"]
            else None
        ),
        "iteration": (
            int(row["iteration"])
            if "iteration" in row.keys() and row["iteration"] is not None
            else None
        ),
        "tool_name": (
            str(row["tool_name"])
            if "tool_name" in row.keys() and row["tool_name"]
            else None
        ),
        "repair_of_sequence": (
            int(row["repair_of_sequence"])
            if "repair_of_sequence" in row.keys()
            and row["repair_of_sequence"] is not None
            else None
        ),
        "payload": json_loads(row["payload"], {}) if "payload" in row.keys() else {},
        "message": str(row["message"]),
        "created_at": str(row["created_at"]),
    }


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def json_loads(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return fallback


def bounded_progress(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return max(0, min(number, 100))
