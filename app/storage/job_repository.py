from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from app.storage.models import StoredAnalysisJob, StoredCleaningJob
from app.storage.repository_utils import now_iso as _now_iso
from app.storage.repository_utils import (
    validate_relationship_graph_structure as _validate_relationship_graph_structure,
)
from app.storage.row_mappers import (
    analysis_job_event_from_row as _analysis_job_event_from_row,
)
from app.storage.row_mappers import bounded_progress as _bounded_progress
from app.storage.row_mappers import (
    cleaning_job_event_from_row as _cleaning_job_event_from_row,
)
from app.storage.row_mappers import (
    stored_analysis_job_from_row as _stored_analysis_job_from_row,
)
from app.storage.row_mappers import (
    stored_cleaning_job_from_row as _stored_cleaning_job_from_row,
)


class JobRepositoryMixin:
    def create_cleaning_job(
        self,
        *,
        dataset_id: UUID,
        requirement: str = "",
        cleaning_strategy: str = "auto",
        prompt_overrides: dict[str, str] | None = None,
        retry_of: UUID | None = None,
    ) -> StoredCleaningJob:
        self.get_dataset(dataset_id)
        if cleaning_strategy not in {"auto", "rules", "llm", "hybrid"}:
            raise ValueError("Unsupported cleaning strategy.")
        job_id = uuid4()
        now = _now_iso()
        message = "Cleaning job queued."
        event = _job_event(stage="queued", progress=0, message=message, status="queued")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cleaning_jobs (
                    id, user_id, dataset_id, requirement, prompt_overrides, cleaning_strategy,
                    status, progress, current_stage, events, loop_summary,
                    retry_of, cancel_requested, attempt_count, checkpoint_thread_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    self._user_id,
                    str(dataset_id),
                    requirement,
                    json.dumps(prompt_overrides or {}, ensure_ascii=False),
                    cleaning_strategy,
                    "queued",
                    0,
                    "queued",
                    json.dumps([event], ensure_ascii=False),
                    "{}",
                    str(retry_of) if retry_of else None,
                    0,
                    0,
                    str(job_id),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO cleaning_job_events (
                    job_id, sequence, stage, status, progress, message, created_at
                ) VALUES (?, 1, 'queued', 'queued', 0, ?, ?)
                """,
                (str(job_id), message, event["created_at"]),
            )
        return self.get_cleaning_job(job_id)

    def get_cleaning_job(self, job_id: UUID) -> StoredCleaningJob:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cleaning_jobs WHERE id = ? AND user_id = ?",
                (str(job_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Cleaning job was not found: {job_id}")
        return self._hydrate_cleaning_job_events(_stored_cleaning_job_from_row(row))

    def list_cleaning_jobs(
        self, *, dataset_id: UUID | None = None, limit: int = 50
    ) -> tuple[StoredCleaningJob, ...]:
        limit = max(1, min(limit, 200))
        with self._connect() as connection:
            if dataset_id is None:
                rows = connection.execute(
                    "SELECT * FROM cleaning_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (self._user_id, limit),
                ).fetchall()
            else:
                self.get_dataset(dataset_id)
                rows = connection.execute(
                    "SELECT * FROM cleaning_jobs WHERE user_id = ? AND dataset_id = ? ORDER BY created_at DESC LIMIT ?",
                    (self._user_id, str(dataset_id), limit),
                ).fetchall()
        return tuple(
            self._hydrate_cleaning_job_events(_stored_cleaning_job_from_row(row)) for row in rows
        )

    def update_cleaning_job(
        self,
        job_id: UUID,
        *,
        status: str | None = None,
        progress: int | None = None,
        current_stage: str | None = None,
        event_message: str | None = None,
        selected_strategy: str | None = None,
        loop_summary: dict[str, Any] | None = None,
        terminal_reason: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        cleaning_run_id: UUID | None = None,
        cancel_requested: bool | None = None,
        broker_task_id: str | None = None,
        increment_attempt: bool = False,
        lease_owner: str | None = None,
        lease_expires_at: str | None = None,
        heartbeat_at: str | None = None,
        started: bool = False,
        completed: bool = False,
    ) -> StoredCleaningJob:
        job = self.get_cleaning_job(job_id)
        now = _now_iso()
        next_status = status or job.status
        next_progress = _bounded_progress(progress if progress is not None else job.progress)
        next_stage = current_stage or job.current_stage
        next_events = list(job.events)
        if event_message is not None or status is not None or current_stage is not None:
            next_events.append(
                _job_event(
                    stage=next_stage,
                    progress=next_progress,
                    message=event_message or next_stage,
                    status=next_status,
                )
            )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cleaning_jobs SET
                    status = ?, progress = ?, current_stage = ?, events = ?,
                    selected_strategy = ?, loop_summary = ?, terminal_reason = ?,
                    result = ?, error = ?, cleaning_run_id = ?, cancel_requested = ?,
                    broker_task_id = ?, attempt_count = ?, lease_owner = ?,
                    lease_expires_at = ?, heartbeat_at = ?, updated_at = ?,
                    started_at = ?, completed_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    next_status,
                    next_progress,
                    next_stage,
                    json.dumps(next_events, ensure_ascii=False, default=str),
                    selected_strategy if selected_strategy is not None else job.selected_strategy,
                    json.dumps(
                        loop_summary if loop_summary is not None else (job.loop_summary or {}),
                        ensure_ascii=False,
                        default=str,
                    ),
                    terminal_reason if terminal_reason is not None else job.terminal_reason,
                    json.dumps(result, ensure_ascii=False, default=str)
                    if result is not None
                    else (
                        json.dumps(job.result, ensure_ascii=False, default=str)
                        if job.result is not None
                        else None
                    ),
                    error if error is not None else job.error,
                    str(cleaning_run_id or job.cleaning_run_id)
                    if (cleaning_run_id or job.cleaning_run_id)
                    else None,
                    int(cancel_requested if cancel_requested is not None else job.cancel_requested),
                    broker_task_id if broker_task_id is not None else job.broker_task_id,
                    job.attempt_count + (1 if increment_attempt else 0),
                    lease_owner if lease_owner is not None else job.lease_owner,
                    lease_expires_at if lease_expires_at is not None else job.lease_expires_at,
                    heartbeat_at if heartbeat_at is not None else job.heartbeat_at,
                    now,
                    now if started and job.started_at is None else job.started_at,
                    now if completed else job.completed_at,
                    str(job_id),
                    self._user_id,
                ),
            )
        if event_message is not None or status is not None or current_stage is not None:
            self.append_cleaning_job_event(
                job_id,
                stage=next_stage,
                status=next_status,
                message=event_message or next_stage,
            )
        return self.get_cleaning_job(job_id)

    def append_cleaning_job_event(
        self,
        job_id: UUID,
        *,
        stage: str,
        status: str,
        message: str,
        event_type: str | None = None,
        iteration: int | None = None,
        strategy: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job = self.get_cleaning_job(job_id)
        now = _now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM cleaning_job_events WHERE job_id = ?",
                    (str(job_id),),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO cleaning_job_events (
                    job_id, sequence, stage, status, progress, message,
                    event_type, iteration, strategy, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    sequence,
                    stage,
                    status,
                    job.progress,
                    message,
                    event_type,
                    iteration,
                    strategy,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                    now,
                ),
            )
        return self.list_cleaning_job_events(job_id, after_sequence=sequence - 1, limit=1)[0]

    def list_cleaning_job_events(
        self, job_id: UUID, *, after_sequence: int = 0, limit: int = 500
    ) -> tuple[dict[str, Any], ...]:
        self.get_cleaning_job(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cleaning_job_events WHERE job_id = ? AND sequence > ? ORDER BY sequence ASC LIMIT ?",
                (str(job_id), max(0, after_sequence), max(1, min(limit, 2000))),
            ).fetchall()
        return tuple(_cleaning_job_event_from_row(row) for row in rows)

    def _hydrate_cleaning_job_events(self, job: StoredCleaningJob) -> StoredCleaningJob:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cleaning_job_events WHERE job_id = ? ORDER BY sequence ASC",
                (str(job.id),),
            ).fetchall()
        return (
            replace(job, events=tuple(_cleaning_job_event_from_row(row) for row in rows))
            if rows
            else job
        )

    def claim_cleaning_job(
        self, job_id: UUID, *, worker_id: str, lease_seconds: int
    ) -> StoredCleaningJob | None:
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=max(30, lease_seconds))).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lock_clause = (
                " FOR UPDATE" if getattr(connection, "dialect_name", "") == "postgresql" else ""
            )
            row = connection.execute(
                f"SELECT * FROM cleaning_jobs WHERE id = ? AND user_id = ?{lock_clause}",
                (str(job_id), self._user_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Cleaning job was not found: {job_id}")
            job = _stored_cleaning_job_from_row(row)
            expired = bool(job.lease_expires_at and job.lease_expires_at < now.isoformat())
            claimable = job.status in {"queued", "interrupted"} or (
                job.status == "running" and expired
            )
            if job.cancel_requested or not claimable:
                return None
            connection.execute(
                """
                UPDATE cleaning_jobs SET status = 'running', current_stage = 'cleaning_bootstrap',
                    progress = 1, attempt_count = attempt_count + 1, lease_owner = ?,
                    lease_expires_at = ?, heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    worker_id,
                    expires,
                    now.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                    str(job_id),
                    self._user_id,
                ),
            )
        return self.update_cleaning_job(
            job_id, event_message="Cleaning job started.", heartbeat_at=now.isoformat()
        )

    def heartbeat_cleaning_job(
        self, job_id: UUID, *, worker_id: str, lease_seconds: int
    ) -> StoredCleaningJob:
        now = datetime.now(UTC)
        return self.update_cleaning_job(
            job_id,
            lease_owner=worker_id,
            lease_expires_at=(now + timedelta(seconds=max(30, lease_seconds))).isoformat(),
            heartbeat_at=now.isoformat(),
        )

    def request_cleaning_job_cancel(self, job_id: UUID) -> StoredCleaningJob:
        job = self.get_cleaning_job(job_id)
        if job.status == "queued":
            return self.update_cleaning_job(
                job_id,
                status="canceled",
                current_stage="canceled",
                event_message="Cleaning job canceled before it started.",
                cancel_requested=True,
                completed=True,
            )
        if job.status == "running":
            return self.update_cleaning_job(
                job_id,
                status="cancel_requested",
                event_message="Cancellation requested; the active version remains unchanged.",
                cancel_requested=True,
            )
        return job

    def list_all_recoverable_cleaning_jobs(
        self, *, limit: int = 500
    ) -> tuple[StoredCleaningJob, ...]:
        now = _now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cleaning_jobs WHERE cancel_requested = 0 AND (
                    status IN ('queued', 'interrupted') OR
                    (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
                ) ORDER BY created_at ASC LIMIT ?
                """,
                (now, max(1, min(limit, 2000))),
            ).fetchall()
        return tuple(_stored_cleaning_job_from_row(row) for row in rows)

    def mark_interrupted_cleaning_jobs(self) -> None:
        now = _now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cleaning_jobs SET status = 'interrupted', current_stage = 'interrupted',
                    error = 'Cleaning job was interrupted by a service restart.',
                    updated_at = ?, completed_at = ?
                WHERE status IN ('queued', 'running', 'cancel_requested')
                """,
                (now, now),
            )

    def create_analysis_job(
        self,
        *,
        dataset_id: UUID,
        question: str,
        prompt_overrides: dict[str, str] | None = None,
        dataset_group_id: UUID | None = None,
        additional_dataset_ids: tuple[UUID, ...] = (),
        join_plan: tuple[dict[str, Any], ...] = (),
        relationship_plan: tuple[dict[str, Any], ...] = (),
        multimodal_inputs: tuple[dict[str, Any], ...] = (),
        retry_of: UUID | None = None,
        agent_mode: str = "legacy",
    ) -> StoredAnalysisJob:
        self.get_dataset(dataset_id)
        if dataset_group_id is not None:
            group = self.get_dataset_group(dataset_group_id)
            if dataset_id not in group.dataset_ids:
                raise RuntimeError("Primary dataset must belong to the dataset group.")
        for additional_dataset_id in additional_dataset_ids:
            if additional_dataset_id != dataset_id:
                self.get_dataset(additional_dataset_id)
        for plan in (join_plan, relationship_plan):
            if not plan:
                continue
            for relationship in plan:
                left_dataset_id = relationship.get("left_dataset_id")
                right_dataset_id = relationship.get("right_dataset_id")
                if left_dataset_id:
                    self.get_dataset(UUID(str(left_dataset_id)))
                if right_dataset_id:
                    self.get_dataset(UUID(str(right_dataset_id)))
                self._validate_relationship_columns(relationship)
                if dataset_group_id is not None and (
                    str(left_dataset_id) not in {str(item) for item in group.dataset_ids}
                    or str(right_dataset_id) not in {str(item) for item in group.dataset_ids}
                ):
                    raise ValueError(
                        "Relationship datasets must belong to the selected dataset group."
                    )
            _validate_relationship_graph_structure(plan, expected_root=dataset_id)
        job_id = uuid4()
        now = _now_iso()
        events = (
            _job_event(
                stage="queued",
                progress=0,
                message="Analysis job queued.",
                status="queued",
            ),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    id, user_id, dataset_id, dataset_group_id, additional_dataset_ids,
                    join_plan, relationship_plan, question, prompt_overrides, multimodal_inputs, status,
                    progress, current_stage, events, result, error, report_id,
                    retry_of, cancel_requested, broker_task_id, attempt_count,
                    lease_owner, lease_expires_at, heartbeat_at, checkpoint_thread_id,
                    created_at, updated_at, started_at, completed_at,
                    agent_mode, loop_summary, loop_terminal_reason,
                    report_strategy, report_revision_count, report_terminal_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    self._user_id,
                    str(dataset_id),
                    str(dataset_group_id) if dataset_group_id else None,
                    json.dumps([str(item) for item in additional_dataset_ids], ensure_ascii=False),
                    json.dumps(list(join_plan), ensure_ascii=False, default=str),
                    json.dumps(list(relationship_plan), ensure_ascii=False, default=str),
                    question,
                    json.dumps(prompt_overrides or {}, ensure_ascii=False),
                    json.dumps(list(multimodal_inputs), ensure_ascii=False, default=str),
                    "queued",
                    0,
                    "queued",
                    json.dumps(list(events), ensure_ascii=False, default=str),
                    None,
                    None,
                    None,
                    str(retry_of) if retry_of else None,
                    0,
                    None,
                    0,
                    None,
                    None,
                    None,
                    str(job_id),
                    now,
                    now,
                    None,
                    None,
                    agent_mode,
                    "{}",
                    None,
                    None,
                    0,
                    None,
                ),
            )
            connection.execute(
                """
                INSERT INTO analysis_job_events (
                    job_id, sequence, node, status, progress, attempt,
                    message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    1,
                    "queued",
                    "queued",
                    0,
                    0,
                    "Analysis job queued.",
                    events[0]["created_at"],
                ),
            )
        return self.get_analysis_job(job_id)

    def _validate_relationship_columns(self, relationship: dict[str, Any]) -> None:
        left_dataset_id = UUID(str(relationship.get("left_dataset_id") or ""))
        right_dataset_id = UUID(str(relationship.get("right_dataset_id") or ""))
        left_column = str(relationship.get("left_column") or "")
        right_column = str(relationship.get("right_column") or "")
        if left_dataset_id == right_dataset_id:
            raise ValueError("Relationship cannot join a dataset to itself.")
        left_records = self.sample_analysis_records(left_dataset_id, limit=100)
        right_records = self.sample_analysis_records(right_dataset_id, limit=100)
        if not left_records or not right_records:
            raise ValueError("Relationship datasets must contain analysis records.")
        left_columns = {str(column) for record in left_records for column in record}
        right_columns = {str(column) for record in right_records for column in record}
        if left_column not in left_columns:
            raise ValueError(f"Relationship column was not found: {left_column}")
        if right_column not in right_columns:
            raise ValueError(f"Relationship column was not found: {right_column}")

    def get_analysis_job(self, job_id: UUID) -> StoredAnalysisJob:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE id = ? AND user_id = ?",
                (str(job_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Analysis job was not found: {job_id}")
        return self._hydrate_job_events(_stored_analysis_job_from_row(row))

    def list_analysis_jobs(
        self,
        *,
        dataset_id: UUID | None = None,
        limit: int = 50,
    ) -> tuple[StoredAnalysisJob, ...]:
        limit = max(1, min(limit, 200))
        with self._connect() as connection:
            if dataset_id is None:
                rows = connection.execute(
                    """
                    SELECT * FROM analysis_jobs
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self._user_id, limit),
                ).fetchall()
            else:
                self.get_dataset(dataset_id)
                rows = connection.execute(
                    """
                    SELECT * FROM analysis_jobs
                    WHERE user_id = ? AND dataset_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (self._user_id, str(dataset_id), limit),
                ).fetchall()
        return tuple(self._hydrate_job_events(_stored_analysis_job_from_row(row)) for row in rows)

    def update_analysis_job(
        self,
        job_id: UUID,
        *,
        status: str | None = None,
        progress: int | None = None,
        current_stage: str | None = None,
        event_message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        report_id: UUID | None = None,
        cancel_requested: bool | None = None,
        started: bool = False,
        completed: bool = False,
        broker_task_id: str | None = None,
        increment_attempt: bool = False,
        lease_owner: str | None = None,
        lease_expires_at: str | None = None,
        heartbeat_at: str | None = None,
        loop_summary: dict[str, Any] | None = None,
        loop_terminal_reason: str | None = None,
        report_strategy: str | None = None,
        report_revision_count: int | None = None,
        report_terminal_reason: str | None = None,
    ) -> StoredAnalysisJob:
        job = self.get_analysis_job(job_id)
        now = _now_iso()
        next_status = status or job.status
        next_progress = _bounded_progress(progress if progress is not None else job.progress)
        next_stage = current_stage or job.current_stage
        next_events = list(job.events)
        if event_message is not None or status is not None or current_stage is not None:
            next_events.append(
                _job_event(
                    stage=next_stage,
                    progress=next_progress,
                    message=event_message or next_stage,
                    status=next_status,
                )
            )
        next_attempt_count = job.attempt_count + (1 if increment_attempt else 0)
        next_broker_task_id = broker_task_id if broker_task_id is not None else job.broker_task_id
        next_lease_owner = lease_owner if lease_owner is not None else job.lease_owner
        next_lease_expires_at = (
            lease_expires_at if lease_expires_at is not None else job.lease_expires_at
        )
        next_heartbeat_at = heartbeat_at if heartbeat_at is not None else job.heartbeat_at
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status = ?, progress = ?, current_stage = ?, events = ?,
                    result = ?, error = ?, report_id = ?, cancel_requested = ?,
                    broker_task_id = ?, attempt_count = ?, lease_owner = ?,
                    lease_expires_at = ?, heartbeat_at = ?,
                    updated_at = ?, started_at = ?, completed_at = ?,
                    loop_summary = ?, loop_terminal_reason = ?, report_strategy = ?,
                    report_revision_count = ?, report_terminal_reason = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    next_status,
                    next_progress,
                    next_stage,
                    json.dumps(next_events, ensure_ascii=False, default=str),
                    (
                        json.dumps(result, ensure_ascii=False, default=str)
                        if result is not None
                        else (
                            json.dumps(job.result, ensure_ascii=False, default=str)
                            if job.result is not None
                            else None
                        )
                    ),
                    error if error is not None else job.error,
                    str(report_id or job.report_id) if (report_id or job.report_id) else None,
                    int(cancel_requested if cancel_requested is not None else job.cancel_requested),
                    next_broker_task_id,
                    next_attempt_count,
                    next_lease_owner,
                    next_lease_expires_at,
                    next_heartbeat_at,
                    now,
                    now if started and job.started_at is None else job.started_at,
                    now if completed else job.completed_at,
                    json.dumps(
                        loop_summary if loop_summary is not None else (job.loop_summary or {}),
                        ensure_ascii=False,
                        default=str,
                    ),
                    loop_terminal_reason
                    if loop_terminal_reason is not None
                    else job.loop_terminal_reason,
                    report_strategy if report_strategy is not None else job.report_strategy,
                    report_revision_count
                    if report_revision_count is not None
                    else job.report_revision_count,
                    report_terminal_reason
                    if report_terminal_reason is not None
                    else job.report_terminal_reason,
                    str(job_id),
                    self._user_id,
                ),
            )
            if event_message is not None or status is not None or current_stage is not None:
                sequence = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM analysis_job_events WHERE job_id = ?",
                        (str(job_id),),
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    INSERT INTO analysis_job_events (
                        job_id, sequence, node, status, progress, attempt,
                        message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(job_id),
                        sequence,
                        next_stage,
                        next_status,
                        next_progress,
                        next_attempt_count,
                        event_message or next_stage,
                        now,
                    ),
                )
        return self.get_analysis_job(job_id)

    def set_analysis_job_broker_task(self, job_id: UUID, broker_task_id: str) -> StoredAnalysisJob:
        return self.update_analysis_job(job_id, broker_task_id=broker_task_id)

    def claim_analysis_job(
        self,
        job_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> StoredAnalysisJob | None:
        now = datetime.now(UTC)
        expires_at = (now + timedelta(seconds=max(30, lease_seconds))).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lock_clause = (
                " FOR UPDATE" if getattr(connection, "dialect_name", "") == "postgresql" else ""
            )
            row = connection.execute(
                f"SELECT * FROM analysis_jobs WHERE id = ? AND user_id = ?{lock_clause}",
                (str(job_id), self._user_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Analysis job was not found: {job_id}")
            job = _stored_analysis_job_from_row(row)
            lease_expired = bool(job.lease_expires_at and job.lease_expires_at < now.isoformat())
            claimable = job.status in {"queued", "interrupted"} or (
                job.status == "running" and lease_expired
            )
            if not claimable or job.cancel_requested:
                return None
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'running', current_stage = 'starting', progress = 1,
                    attempt_count = attempt_count + 1, lease_owner = ?,
                    lease_expires_at = ?, heartbeat_at = ?, started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    worker_id,
                    expires_at,
                    now.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                    str(job_id),
                    self._user_id,
                ),
            )
        return self.update_analysis_job(
            job_id,
            event_message="Analysis job started.",
            heartbeat_at=now.isoformat(),
        )

    def heartbeat_analysis_job(
        self,
        job_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> StoredAnalysisJob:
        now = datetime.now(UTC)
        return self.update_analysis_job(
            job_id,
            lease_owner=worker_id,
            lease_expires_at=(now + timedelta(seconds=max(30, lease_seconds))).isoformat(),
            heartbeat_at=now.isoformat(),
        )

    def list_recoverable_analysis_jobs(self, *, limit: int = 100) -> tuple[StoredAnalysisJob, ...]:
        now = _now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_jobs
                WHERE user_id = ? AND cancel_requested = 0
                  AND (
                    status = 'queued'
                    OR (status IN ('running', 'interrupted') AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
                  )
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (self._user_id, now, max(1, min(limit, 500))),
            ).fetchall()
        return tuple(self._hydrate_job_events(_stored_analysis_job_from_row(row)) for row in rows)

    def list_all_recoverable_analysis_jobs(
        self,
        *,
        limit: int = 500,
    ) -> tuple[StoredAnalysisJob, ...]:
        now = _now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_jobs
                WHERE cancel_requested = 0
                  AND (
                    status = 'queued'
                    OR status = 'interrupted'
                    OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
                  )
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, max(1, min(limit, 2000))),
            ).fetchall()
        return tuple(_stored_analysis_job_from_row(row) for row in rows)

    def list_analysis_job_events(
        self,
        job_id: UUID,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> tuple[dict[str, Any], ...]:
        self.get_analysis_job(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_job_events
                WHERE job_id = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (str(job_id), max(0, after_sequence), max(1, min(limit, 2000))),
            ).fetchall()
        return tuple(_analysis_job_event_from_row(row) for row in rows)

    def append_analysis_job_event(
        self,
        job_id: UUID,
        *,
        node: str,
        status: str,
        message: str,
        attempt: int = 0,
        duration_ms: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        token_usage: dict[str, int] | None = None,
        error_code: str | None = None,
        event_type: str | None = None,
        iteration: int | None = None,
        tool_name: str | None = None,
        repair_of_sequence: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job = self.get_analysis_job(job_id)
        now = _now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE analysis_jobs SET updated_at = updated_at WHERE id = ? AND user_id = ?",
                (str(job_id), self._user_id),
            )
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM analysis_job_events WHERE job_id = ?",
                    (str(job_id),),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO analysis_job_events (
                    job_id, sequence, node, status, progress, attempt, duration_ms,
                    provider, model, token_usage, error_code, message, created_at,
                    event_type, iteration, tool_name, repair_of_sequence, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    sequence,
                    node,
                    status,
                    job.progress,
                    max(0, attempt),
                    duration_ms,
                    provider,
                    model,
                    json.dumps(token_usage or {}, ensure_ascii=False),
                    error_code,
                    message,
                    now,
                    event_type,
                    iteration,
                    tool_name,
                    repair_of_sequence,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                ),
            )
            legacy_events = [
                *job.events,
                {
                    "sequence": sequence,
                    "node": node,
                    "stage": node,
                    "status": status,
                    "progress": job.progress,
                    "attempt": max(0, attempt),
                    "duration_ms": duration_ms,
                    "provider": provider,
                    "model": model,
                    "token_usage": token_usage or {},
                    "error_code": error_code,
                    "event_type": event_type,
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "repair_of_sequence": repair_of_sequence,
                    "payload": payload or {},
                    "message": message,
                    "created_at": now,
                },
            ]
            connection.execute(
                "UPDATE analysis_jobs SET events = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (
                    json.dumps(legacy_events, ensure_ascii=False, default=str),
                    now,
                    str(job_id),
                    self._user_id,
                ),
            )
        return self.list_analysis_job_events(job_id, after_sequence=sequence - 1, limit=1)[0]

    def _hydrate_job_events(self, job: StoredAnalysisJob) -> StoredAnalysisJob:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analysis_job_events WHERE job_id = ? ORDER BY sequence ASC",
                (str(job.id),),
            ).fetchall()
        if not rows:
            return job
        return replace(job, events=tuple(_analysis_job_event_from_row(row) for row in rows))

    def request_analysis_job_cancel(self, job_id: UUID) -> StoredAnalysisJob:
        job = self.get_analysis_job(job_id)
        if job.status == "queued":
            return self.update_analysis_job(
                job_id,
                status="canceled",
                progress=job.progress,
                current_stage="canceled",
                event_message="Analysis job canceled before it started.",
                cancel_requested=True,
                completed=True,
            )
        if job.status == "running":
            return self.update_analysis_job(
                job_id,
                status="cancel_requested",
                current_stage=job.current_stage,
                event_message="Cancellation requested. The job will stop at the next safe checkpoint.",
                cancel_requested=True,
            )
        return job

    def mark_interrupted_analysis_jobs(self) -> None:
        now = _now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_jobs
                WHERE status IN ('queued', 'running', 'cancel_requested')
                """
            ).fetchall()
            for row in rows:
                job = _stored_analysis_job_from_row(row)
                events = list(job.events)
                events.append(
                    _job_event(
                        stage="interrupted",
                        progress=job.progress,
                        message="Analysis job was interrupted by a service restart.",
                        status="interrupted",
                    )
                )
                connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status = 'interrupted', current_stage = 'interrupted',
                        events = ?, error = ?, updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(events, ensure_ascii=False, default=str),
                        "Analysis job was interrupted by a service restart.",
                        now,
                        now,
                        str(job.id),
                    ),
                )




def _job_event(
    *,
    stage: str,
    progress: int,
    message: str,
    status: str,
) -> dict[str, Any]:
    return {
        "sequence": 0,
        "node": stage,
        "stage": stage,
        "progress": _bounded_progress(progress),
        "message": message,
        "status": status,
        "created_at": _now_iso(),
    }
