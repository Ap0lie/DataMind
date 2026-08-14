from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar
from uuid import UUID

from app.storage.auth_repository import (
    AuthRepositoryMixin,
    normalize_login_name,
    normalize_user_id,
)
from app.storage.data_reliability_repository import DataReliabilityRepositoryMixin
from app.storage.dataset_group_repository import DatasetGroupRepositoryMixin
from app.storage.dataset_repository import DatasetRepositoryMixin
from app.storage.job_repository import JobRepositoryMixin
from app.storage.models import (
    StoredAnalysisJob,
    StoredCleaningJob,
    StoredDataset,
    StoredDatasetGroup,
)
from app.storage.recycle_repository import AssetRecycleRepositoryMixin
from app.storage.report_repository import ReportRepositoryMixin
from app.storage.repository_utils import now_iso as _now_iso
from app.storage.row_mappers import (
    bounded_progress as _bounded_progress,
)
from app.storage.row_mappers import (
    json_loads as _json_loads,
)
from app.storage.row_mappers import (
    optional_text as _optional_text,
)
from app.storage.semantic_repository import SemanticRepositoryMixin

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetStoreRepository",
    "StoredAnalysisJob",
    "StoredCleaningJob",
    "StoredDataset",
    "StoredDatasetGroup",
]


class DatasetStoreRepository(
    AuthRepositoryMixin,
    ReportRepositoryMixin,
    AssetRecycleRepositoryMixin,
    DataReliabilityRepositoryMixin,
    SemanticRepositoryMixin,
    DatasetGroupRepositoryMixin,
    DatasetRepositoryMixin,
    JobRepositoryMixin,
):
    """Local JSON-file dataset store for the DataMind v1 workflow."""

    _initialization_lock = Lock()
    _initialized_stores: ClassVar[set[str]] = set()

    def __init__(self, root_path: str, user_id: str = "default") -> None:
        self._root = Path(root_path)
        self._user_id = normalize_user_id(user_id)
        from app.core.settings import get_settings

        self._database_url = get_settings().database_url
        self._root.mkdir(parents=True, exist_ok=True)
        self._db_path = (
            self._root.parent / "datamind.db"
            if self._root.name == "datasets"
            else self._root / "datamind.db"
        )
        self._initialize_store_once()

    def _initialize_store_once(self) -> None:
        key = self._database_url or str(self._db_path.resolve())
        store_exists = bool(self._database_url) or self._db_path.exists()
        if key in self._initialized_stores and store_exists:
            return
        with self._initialization_lock:
            store_exists = bool(self._database_url) or self._db_path.exists()
            if key in self._initialized_stores and store_exists:
                return
            self._initialize_database()
            self._migrate_json_store()
            self._initialized_stores.add(key)

    @property
    def root_path(self) -> str:
        return str(self._root)

    @property
    def user_id(self) -> str:
        return self._user_id


    def _dataset_dir(self, dataset_id: UUID) -> Path:
        return self._root / str(dataset_id)

    def _require_dataset_dir(self, dataset_id: UUID) -> Path:
        self.get_dataset(dataset_id)
        dataset_dir = self._dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _update_dataset(self, dataset_id: UUID, **updates: Any) -> None:
        self.get_dataset(dataset_id)
        allowed = {"name", "source_type", "status", "source_metadata"}
        assignments = [f"{key} = ?" for key in updates if key in allowed]
        values = [
            json.dumps(value, ensure_ascii=False, default=str)
            if key == "source_metadata"
            else value
            for key, value in updates.items()
            if key in allowed
        ]
        assignments.append("updated_at = ?")
        values.append(_now_iso())
        values.append(str(dataset_id))
        with self._connect() as connection:
            connection.execute(
                f"UPDATE datasets SET {', '.join(assignments)} WHERE id = ?",
                tuple(values),
            )

    def _replace_records(
        self,
        *,
        dataset_id: UUID,
        source: str,
        records: list[dict[str, Any]],
    ) -> None:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM records WHERE dataset_id = ? AND source = ?",
                (str(dataset_id), source),
            )
            connection.executemany(
                """
                INSERT INTO records (dataset_id, source, row_number, record)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        str(dataset_id),
                        source,
                        index + 1,
                        json.dumps(record, ensure_ascii=False, default=str),
                    )
                    for index, record in enumerate(records)
                ],
            )

    def _connect(self) -> Any:
        if self._database_url:
            from app.storage.sqlalchemy_compat import SQLAlchemyConnectionAdapter

            return SQLAlchemyConnectionAdapter(self._database_url)
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def ping(self) -> bool:
        with self._connect() as connection:
            return bool(connection.execute("SELECT 1").fetchone()[0])

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    login_name_normalized TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    csrf_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    absolute_expires_at TEXT NOT NULL,
                    revoked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS records (
                    dataset_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    row_number INTEGER NOT NULL,
                    record TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, source, row_number)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    file_name TEXT,
                    content TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cleaning_runs (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    version INTEGER NOT NULL DEFAULT 1,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    result_markdown TEXT NOT NULL,
                    cleaned_dataset TEXT NOT NULL,
                    raw_summary TEXT NOT NULL DEFAULT '{}',
                    previous_summary TEXT NOT NULL DEFAULT '{}',
                    current_summary TEXT NOT NULL DEFAULT '{}',
                    diff_summary TEXT NOT NULL DEFAULT '{}',
                    job_id TEXT
                );

                CREATE TABLE IF NOT EXISTS cleaning_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    dataset_id TEXT NOT NULL,
                    requirement TEXT NOT NULL DEFAULT '',
                    prompt_overrides TEXT NOT NULL DEFAULT '{}',
                    cleaning_strategy TEXT NOT NULL DEFAULT 'auto',
                    selected_strategy TEXT,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_stage TEXT NOT NULL DEFAULT 'queued',
                    events TEXT NOT NULL DEFAULT '[]',
                    loop_summary TEXT NOT NULL DEFAULT '{}',
                    terminal_reason TEXT,
                    result TEXT,
                    error TEXT,
                    cleaning_run_id TEXT,
                    retry_of TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    broker_task_id TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    checkpoint_thread_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS cleaning_job_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL,
                    event_type TEXT,
                    iteration INTEGER,
                    strategy TEXT,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS dataset_columns (
                    dataset_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    column_name TEXT NOT NULL,
                    inferred_type TEXT NOT NULL DEFAULT 'text',
                    override_type TEXT,
                    role TEXT NOT NULL DEFAULT 'dimension',
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, user_id, column_name)
                );

                CREATE TABLE IF NOT EXISTS dataset_groups (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    dataset_ids TEXT NOT NULL DEFAULT '[]',
                    relationships TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS charts (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chart_type TEXT NOT NULL,
                    chart_spec TEXT NOT NULL,
                    chart_data TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    title TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    question TEXT,
                    job_id TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    dataset_id TEXT NOT NULL,
                    dataset_group_id TEXT,
                    additional_dataset_ids TEXT NOT NULL DEFAULT '[]',
                    join_plan TEXT NOT NULL DEFAULT '[]',
                    relationship_plan TEXT NOT NULL DEFAULT '[]',
                    question TEXT NOT NULL,
                    prompt_overrides TEXT NOT NULL DEFAULT '{}',
                    multimodal_inputs TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_stage TEXT NOT NULL DEFAULT 'queued',
                    events TEXT NOT NULL DEFAULT '[]',
                    result TEXT,
                    error TEXT,
                    report_id TEXT,
                    retry_of TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    broker_task_id TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    checkpoint_thread_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                    ,agent_mode TEXT NOT NULL DEFAULT 'legacy'
                    ,loop_summary TEXT NOT NULL DEFAULT '{}'
                    ,loop_terminal_reason TEXT
                    ,report_strategy TEXT
                    ,report_revision_count INTEGER NOT NULL DEFAULT 0
                    ,report_terminal_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis_job_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    duration_ms REAL,
                    provider TEXT,
                    model TEXT,
                    token_usage TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT,
                    iteration INTEGER,
                    tool_name TEXT,
                    repair_of_sequence INTEGER,
                    payload TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (job_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS semantic_models (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL,
                    source TEXT NOT NULL, parent_model_id TEXT, definition TEXT NOT NULL DEFAULT '{}',
                    schema_fingerprint TEXT NOT NULL DEFAULT '', validation TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, published_at TEXT
                );

                CREATE TABLE IF NOT EXISTS semantic_embedding_cache (
                    user_id TEXT NOT NULL, model_revision TEXT NOT NULL, text_hash TEXT NOT NULL,
                    vector TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, model_revision, text_hash)
                );

                CREATE TABLE IF NOT EXISTS data_snapshots (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    sample_size INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    profile TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS data_drift_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    baseline_snapshot_id TEXT NOT NULL,
                    current_snapshot_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    changes TEXT NOT NULL DEFAULT '[]',
                    affected_assets TEXT NOT NULL DEFAULT '[]',
                    recommended_actions TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT
                );

                CREATE TABLE IF NOT EXISTS planner_decisions (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
                    dataset_group_id TEXT, question TEXT NOT NULL, semantic_model_id TEXT,
                    semantic_model_version INTEGER, semantic_source TEXT NOT NULL DEFAULT 'legacy',
                    semantic_plan TEXT NOT NULL DEFAULT '{}', component_scores TEXT NOT NULL DEFAULT '{}',
                    raw_confidence REAL NOT NULL, calibrated_confidence REAL NOT NULL,
                    confidence_level TEXT NOT NULL, requires_confirmation INTEGER NOT NULL DEFAULT 0,
                    confirmed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS planner_feedback (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, decision_id TEXT NOT NULL,
                    action TEXT NOT NULL, corrected_plan TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS planner_calibrators (
                    id TEXT PRIMARY KEY, user_id TEXT, version INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL, breakpoints TEXT NOT NULL DEFAULT '[]',
                    metrics TEXT NOT NULL DEFAULT '{}', active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_dataset_id ON artifacts(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_cleaning_runs_dataset_id ON cleaning_runs(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_cleaning_jobs_user ON cleaning_jobs(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_cleaning_jobs_dataset ON cleaning_jobs(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_cleaning_job_events_job ON cleaning_job_events(job_id);
                CREATE INDEX IF NOT EXISTS idx_charts_dataset_id ON charts(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_reports_dataset_id ON reports(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_jobs_dataset_id ON analysis_jobs(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_job_events_job_id ON analysis_job_events(job_id);
                CREATE INDEX IF NOT EXISTS idx_dataset_groups_user_id ON dataset_groups(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash);
                CREATE INDEX IF NOT EXISTS idx_semantic_models_scope ON semantic_models(user_id, scope_type, scope_id);
                CREATE INDEX IF NOT EXISTS idx_data_snapshots_dataset ON data_snapshots(user_id, dataset_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_data_drift_events_dataset ON data_drift_events(user_id, dataset_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_planner_decisions_user ON planner_decisions(user_id, created_at);
                """
            )
            _ensure_column(connection, "datasets", "user_id", "TEXT NOT NULL DEFAULT 'default'")
            _ensure_column(connection, "users", "login_name_normalized", "TEXT")
            existing_login_names = {
                str(row["login_name_normalized"])
                for row in connection.execute(
                    """
                    SELECT login_name_normalized
                    FROM users
                    WHERE login_name_normalized IS NOT NULL
                    """
                ).fetchall()
                if row["login_name_normalized"]
            }
            for row in connection.execute(
                """
                SELECT user_id,display_name
                FROM users
                WHERE login_name_normalized IS NULL OR login_name_normalized=''
                """
            ).fetchall():
                candidate = normalize_login_name(str(row["display_name"]))
                if not candidate:
                    candidate = str(row["user_id"])
                if candidate in existing_login_names:
                    candidate = f"{candidate}#{row['user_id']}"
                connection.execute(
                    "UPDATE users SET login_name_normalized=? WHERE user_id=?",
                    (candidate, str(row["user_id"])),
                )
                existing_login_names.add(candidate)
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                uq_users_login_name_normalized
                ON users(login_name_normalized)
                """
            )
            for table in ("datasets", "dataset_groups", "reports", "semantic_models"):
                _ensure_column(connection, table, "deleted_at", "TEXT")
                _ensure_column(connection, table, "purge_after", "TEXT")
                _ensure_column(connection, table, "deleted_by_batch_id", "TEXT")
            _ensure_column(
                connection, "cleaning_runs", "user_id", "TEXT NOT NULL DEFAULT 'default'"
            )
            _ensure_column(connection, "cleaning_runs", "version", "INTEGER NOT NULL DEFAULT 1")
            _ensure_column(connection, "cleaning_runs", "is_active", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(connection, "cleaning_runs", "raw_summary", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(
                connection, "cleaning_runs", "previous_summary", "TEXT NOT NULL DEFAULT '{}'"
            )
            _ensure_column(
                connection, "cleaning_runs", "current_summary", "TEXT NOT NULL DEFAULT '{}'"
            )
            _ensure_column(
                connection, "cleaning_runs", "diff_summary", "TEXT NOT NULL DEFAULT '{}'"
            )
            _ensure_column(connection, "cleaning_runs", "job_id", "TEXT")
            _ensure_column(
                connection,
                "cleaning_jobs",
                "prompt_overrides",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_cleaning_runs_job_id ON cleaning_runs(job_id)"
            )
            _ensure_column(connection, "reports", "user_id", "TEXT NOT NULL DEFAULT 'default'")
            _ensure_column(connection, "reports", "updated_at", "TEXT")
            _ensure_column(connection, "reports", "version", "INTEGER NOT NULL DEFAULT 1")
            _ensure_column(connection, "reports", "question", "TEXT")
            _ensure_column(connection, "reports", "job_id", "TEXT")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_job_id ON reports(job_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_reports_user_created_at "
                "ON reports(user_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_reports_user_dataset_created_at "
                "ON reports(user_id, dataset_id, created_at DESC)"
            )
            _ensure_column(
                connection,
                "analysis_jobs",
                "user_id",
                "TEXT NOT NULL DEFAULT 'default'",
            )
            _ensure_column(
                connection,
                "analysis_jobs",
                "prompt_overrides",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            _ensure_column(
                connection,
                "analysis_jobs",
                "additional_dataset_ids",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            _ensure_column(
                connection,
                "analysis_jobs",
                "dataset_group_id",
                "TEXT",
            )
            _ensure_column(
                connection,
                "analysis_jobs",
                "join_plan",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            _ensure_column(
                connection,
                "analysis_jobs",
                "relationship_plan",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            _ensure_column(connection, "analysis_jobs", "broker_task_id", "TEXT")
            _ensure_column(
                connection, "analysis_jobs", "attempt_count", "INTEGER NOT NULL DEFAULT 0"
            )
            _ensure_column(connection, "analysis_jobs", "lease_owner", "TEXT")
            _ensure_column(connection, "analysis_jobs", "lease_expires_at", "TEXT")
            _ensure_column(connection, "analysis_jobs", "heartbeat_at", "TEXT")
            _ensure_column(connection, "analysis_jobs", "checkpoint_thread_id", "TEXT")
            _ensure_column(connection, "analysis_jobs", "planner_decision_id", "TEXT")
            _ensure_column(connection, "analysis_jobs", "semantic_model_id", "TEXT")
            _ensure_column(connection, "analysis_jobs", "semantic_model_version", "INTEGER")
            _ensure_column(
                connection, "analysis_jobs", "agent_mode", "TEXT NOT NULL DEFAULT 'legacy'"
            )
            _ensure_column(
                connection, "analysis_jobs", "loop_summary", "TEXT NOT NULL DEFAULT '{}'"
            )
            _ensure_column(connection, "analysis_jobs", "loop_terminal_reason", "TEXT")
            _ensure_column(connection, "analysis_jobs", "report_strategy", "TEXT")
            _ensure_column(
                connection, "analysis_jobs", "report_revision_count", "INTEGER NOT NULL DEFAULT 0"
            )
            _ensure_column(connection, "analysis_jobs", "report_terminal_reason", "TEXT")
            _ensure_column(connection, "analysis_job_events", "event_type", "TEXT")
            _ensure_column(connection, "analysis_job_events", "iteration", "INTEGER")
            _ensure_column(connection, "analysis_job_events", "tool_name", "TEXT")
            _ensure_column(connection, "analysis_job_events", "repair_of_sequence", "INTEGER")
            _ensure_column(
                connection, "analysis_job_events", "payload", "TEXT NOT NULL DEFAULT '{}'"
            )
            connection.execute(
                """
                UPDATE reports
                SET updated_at = created_at
                WHERE updated_at IS NULL
                """
            )
            report_rows = connection.execute(
                "SELECT id, metadata FROM reports WHERE question IS NULL"
            ).fetchall()
            for report_row in report_rows:
                metadata = _json_loads(report_row["metadata"], {})
                question = metadata.get("question") if isinstance(metadata, dict) else None
                connection.execute(
                    "UPDATE reports SET question = ? WHERE id = ?",
                    (_optional_text(question), str(report_row["id"])),
                )
            self._backfill_analysis_job_events(connection)

    @staticmethod
    def _backfill_analysis_job_events(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT id, events FROM analysis_jobs
            WHERE NOT EXISTS (
                SELECT 1 FROM analysis_job_events WHERE analysis_job_events.job_id = analysis_jobs.id
            )
            """
        ).fetchall()
        for row in rows:
            events = _json_loads(row["events"], [])
            for index, event in enumerate(events if isinstance(events, list) else (), start=1):
                if not isinstance(event, dict):
                    continue
                connection.execute(
                    """
                    INSERT INTO analysis_job_events (
                        job_id, sequence, node, status, progress, attempt, message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (job_id, sequence) DO NOTHING
                    """,
                    (
                        str(row["id"]),
                        index,
                        str(event.get("node") or event.get("stage") or "unknown"),
                        str(event.get("status") or "unknown"),
                        _bounded_progress(event.get("progress")),
                        int(event.get("attempt") or 0),
                        str(event.get("message") or ""),
                        str(event.get("created_at") or _now_iso()),
                    ),
                )

    def _migrate_json_store(self) -> None:
        if not self._root.exists():
            return
        for dataset_path in sorted(self._root.glob("*/dataset.json")):
            payload = self._read_json(dataset_path)
            dataset_id = UUID(str(payload["id"]))
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM datasets WHERE id = ?",
                    (str(dataset_id),),
                ).fetchone()
                if exists:
                    continue
                connection.execute(
                    """
                    INSERT INTO datasets (
                        id, user_id, name, source_type, status, source_metadata, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(dataset_id),
                        self._user_id,
                        str(payload["name"]),
                        str(payload["source_type"]),
                        str(payload["status"]),
                        json.dumps(payload.get("source_metadata") or {}, ensure_ascii=False),
                        str(payload.get("created_at") or _now_iso()),
                        str(payload.get("updated_at") or _now_iso()),
                    ),
                )
            raw_records = self._read_jsonl_records(dataset_id, "raw_records.jsonl")
            cleaned_records = self._read_jsonl_records(dataset_id, "cleaned_records.jsonl")
            if raw_records:
                self._replace_records(dataset_id=dataset_id, source="raw", records=raw_records)
            if cleaned_records:
                self._replace_records(
                    dataset_id=dataset_id,
                    source="cleaned",
                    records=cleaned_records,
                )
            self._migrate_records_folder(dataset_id, "artifacts")
            self._migrate_records_folder(dataset_id, "cleaning_runs")
            self._migrate_records_folder(dataset_id, "charts")
            self._migrate_records_folder(dataset_id, "reports")

    def _migrate_records_folder(self, dataset_id: UUID, folder: str) -> None:
        folder_path = self._dataset_dir(dataset_id) / folder
        if not folder_path.exists():
            return
        for path in sorted(folder_path.glob("*.json")):
            payload = self._read_json(path)
            record_id = UUID(str(payload["id"]))
            with self._connect() as connection:
                table_exists = connection.execute(
                    f"SELECT 1 FROM {folder} WHERE id = ?",
                    (str(record_id),),
                ).fetchone()
                if table_exists:
                    continue
            if folder == "artifacts":
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO artifacts (
                            id, dataset_id, created_at, artifact_type, file_name, content
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(record_id),
                            str(dataset_id),
                            str(payload.get("created_at") or _now_iso()),
                            str(payload.get("artifact_type") or "artifact"),
                            payload.get("file_name"),
                            json.dumps(payload.get("content") or {}, ensure_ascii=False),
                        ),
                    )
            elif folder == "cleaning_runs":
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO cleaning_runs (
                            id, dataset_id, user_id, version, is_active, created_at,
                            provider, model, prompt, result_markdown, cleaned_dataset,
                            raw_summary, previous_summary, current_summary, diff_summary
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(record_id),
                            str(dataset_id),
                            self._user_id,
                            1,
                            0,
                            str(payload.get("created_at") or _now_iso()),
                            str(payload.get("provider") or "unknown"),
                            str(payload.get("model") or "unknown"),
                            str(payload.get("prompt") or ""),
                            str(payload.get("result_markdown") or ""),
                            json.dumps(payload.get("cleaned_dataset") or {}, ensure_ascii=False),
                            "{}",
                            "{}",
                            "{}",
                            "{}",
                        ),
                    )
            elif folder == "charts":
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO charts (
                            id, dataset_id, created_at, title, chart_type, chart_spec, chart_data
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(record_id),
                            str(dataset_id),
                            str(payload.get("created_at") or _now_iso()),
                            str(payload.get("title") or "Chart"),
                            str(payload.get("chart_type") or "bar"),
                            json.dumps(payload.get("chart_spec") or {}, ensure_ascii=False),
                            json.dumps(payload.get("chart_data") or [], ensure_ascii=False),
                        ),
                    )
            elif folder == "reports":
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO reports (
                            id, dataset_id, user_id, created_at, updated_at, version,
                            title, markdown, metadata
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(record_id),
                            str(dataset_id),
                            self._user_id,
                            str(payload.get("created_at") or _now_iso()),
                            str(
                                payload.get("updated_at") or payload.get("created_at") or _now_iso()
                            ),
                            1,
                            str(payload.get("title") or "DataMind 分析报告"),
                            str(payload.get("markdown") or ""),
                            json.dumps(payload.get("metadata") or {}, ensure_ascii=False),
                        ),
                    )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def _ensure_column(
    connection: Any,
    table: str,
    column: str,
    definition: str,
) -> None:
    if hasattr(connection, "column_names"):
        columns = connection.column_names(table)
    else:
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
