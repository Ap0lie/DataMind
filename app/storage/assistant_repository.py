from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar
from uuid import UUID, uuid4

from app.core.settings import get_settings
from app.storage.dataset_store import DatasetStoreRepository

logger = logging.getLogger(__name__)


class AssistantConversationIdempotencyConflict(RuntimeError):
    """The same creation intent was replayed with incompatible state."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(str(value)) if value not in (None, "") else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


@dataclass(frozen=True)
class StoredAssistantRun:
    id: UUID
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    status: str
    current_stage: str
    analysis_job_id: UUID | None
    pending_confirmation: dict[str, Any]
    execution_mode: str
    execution_plan: dict[str, Any]
    current_action_id: UUID | None
    required_permission: str | None
    error: str | None
    cancel_requested: bool
    broker_task_id: str | None
    attempt_count: int
    lease_owner: str | None
    lease_expires_at: str | None
    heartbeat_at: str | None
    checkpoint_thread_id: str | None
    last_event_sequence: int
    created_at: str
    updated_at: str
    completed_at: str | None


class AssistantRepository:
    """User-scoped persistence for Kimi conversations and runs."""

    _initialization_lock = Lock()
    _initialized_stores: ClassVar[set[str]] = set()

    def __init__(self, root_path: str, *, user_id: str) -> None:
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        settings = get_settings()
        self.database_url = settings.database_url
        self.db_path = self.root.parent / "datamind.db" if self.root.name == "datasets" else self.root / "datamind.db"
        DatasetStoreRepository(root_path, user_id=user_id)
        self._initialize_store_once(environment=settings.environment)

    @property
    def attachment_root(self) -> Path:
        path = self.root.parent / "assistant-attachments" if self.root.name == "datasets" else self.root / "assistant-attachments"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _connect(self) -> Any:
        if self.database_url:
            from app.storage.sqlalchemy_compat import SQLAlchemyConnectionAdapter

            return SQLAlchemyConnectionAdapter(self.database_url)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_store_once(self, *, environment: str) -> None:
        key = self.database_url or str(self.db_path.resolve())
        if key in self._initialized_stores:
            return
        with self._initialization_lock:
            if key in self._initialized_stores:
                return
            if self.database_url and environment.lower() == "production":
                self._verify_migrated_schema()
            else:
                self._initialize()
            self._initialized_stores.add(key)

    def _verify_migrated_schema(self) -> None:
        """Fail fast in production; Alembic owns PostgreSQL schema changes."""
        with self._connect() as connection:
            connection.execute(
                """
                SELECT next_event_sequence
                FROM assistant_runs
                WHERE 1 = 0
                """
            )
            connection.execute(
                """
                SELECT id
                FROM assistant_permission_grants
                WHERE 1 = 0
                """
            )
            connection.execute(
                """
                SELECT idempotency_key, request_fingerprint
                FROM assistant_conversations
                WHERE 1 = 0
                """
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(_ASSISTANT_SCHEMA)
            for table, column, definition in (
                ("assistant_runs", "execution_mode", "TEXT NOT NULL DEFAULT 'ask'"),
                ("assistant_runs", "execution_plan", "TEXT NOT NULL DEFAULT '{}'"),
                ("assistant_runs", "current_action_id", "TEXT"),
                ("assistant_runs", "required_permission", "TEXT"),
                ("assistant_runs", "next_event_sequence", "INTEGER NOT NULL DEFAULT 1"),
                ("assistant_attachments", "attachment_kind", "TEXT NOT NULL DEFAULT 'image'"),
                ("assistant_attachments", "import_status", "TEXT"),
                ("assistant_attachments", "dataset_id", "TEXT"),
                ("assistant_attachments", "import_batch_id", "TEXT"),
                ("assistant_conversations", "summary_through_message_id", "TEXT"),
                ("assistant_conversations", "summary_version", "INTEGER NOT NULL DEFAULT 0"),
                ("assistant_conversations", "summary_updated_at", "TEXT"),
                ("assistant_conversations", "summary_payload", "TEXT NOT NULL DEFAULT '{}'"),
                ("assistant_conversations", "idempotency_key", "TEXT"),
                ("assistant_conversations", "request_fingerprint", "TEXT"),
            ):
                _ensure_column(connection, table, column, definition)
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_assistant_conversation_idempotency
                   ON assistant_conversations(user_id,idempotency_key)"""
            )
            connection.execute(
                """
                UPDATE assistant_runs
                SET next_event_sequence = COALESCE(
                    (
                        SELECT MAX(event.sequence) + 1
                        FROM assistant_run_events event
                        WHERE event.run_id = assistant_runs.id
                    ),
                    1
                )
                WHERE next_event_sequence <= COALESCE(
                    (
                        SELECT MAX(event.sequence)
                        FROM assistant_run_events event
                        WHERE event.run_id = assistant_runs.id
                    ),
                    0
                )
                """
            )

    def create_conversation(
        self,
        *,
        title: str,
        scope_type: str,
        scope_id: UUID | None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        normalized_title = _normalize_conversation_title(title)
        normalized_key = idempotency_key.strip() if idempotency_key else None
        fingerprint = request_fingerprint or _conversation_request_fingerprint(
            title=normalized_title,
            scope_type=scope_type,
            scope_id=scope_id,
        )
        if normalized_key:
            existing = self._resolve_idempotent_conversation(
                normalized_key, fingerprint
            )
            if existing is not None:
                return existing

        self._validate_scope(scope_type, scope_id)
        conversation_id = uuid4()
        now = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO assistant_conversations
                       (id,user_id,title,scope_type,scope_id,summary,deleted_at,created_at,
                        updated_at,last_message_at,idempotency_key,request_fingerprint)
                       VALUES (?,?,?,?,?,'',NULL,?,?,NULL,?,?)""",
                    (
                        str(conversation_id),
                        self.user_id,
                        normalized_title,
                        scope_type,
                        str(scope_id) if scope_id else None,
                        now,
                        now,
                        normalized_key,
                        fingerprint if normalized_key else None,
                    ),
                )
        except Exception as exc:
            if not normalized_key or not _is_integrity_error(exc):
                raise
            existing = self._resolve_idempotent_conversation(
                normalized_key, fingerprint
            )
            if existing is None:
                raise
            return existing
        return self.get_conversation(conversation_id)

    def _resolve_idempotent_conversation(
        self, idempotency_key: str, request_fingerprint: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT c.*,
                   (SELECT id FROM assistant_runs r WHERE r.conversation_id=c.id AND r.user_id=c.user_id
                    AND r.status IN ('queued','running','pause_requested','paused','awaiting_confirmation') ORDER BY r.created_at DESC LIMIT 1) active_run_id,
                   (SELECT status FROM assistant_runs r WHERE r.conversation_id=c.id AND r.user_id=c.user_id
                    AND r.status IN ('queued','running','pause_requested','paused','awaiting_confirmation') ORDER BY r.created_at DESC LIMIT 1) active_run_status
                   FROM assistant_conversations c
                   WHERE c.user_id=? AND c.idempotency_key=?""",
                (self.user_id, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:12]
        if str(row["request_fingerprint"] or "") != request_fingerprint:
            logger.warning(
                "Assistant conversation idempotency conflict key_hash=%s user_id=%s",
                key_hash,
                self.user_id,
            )
            raise AssistantConversationIdempotencyConflict(
                "Idempotency key conflict: request content does not match the original creation."
            )
        if row["deleted_at"]:
            logger.info(
                "Assistant conversation idempotency replay rejected for deleted resource key_hash=%s user_id=%s",
                key_hash,
                self.user_id,
            )
            raise AssistantConversationIdempotencyConflict(
                "Idempotency key belongs to a deleted conversation."
            )
        logger.debug(
            "Assistant conversation idempotency replay key_hash=%s user_id=%s",
            key_hash,
            self.user_id,
        )
        return self._conversation(row)

    def list_conversations(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT c.*,
                   (SELECT id FROM assistant_runs r WHERE r.conversation_id=c.id AND r.user_id=c.user_id
                    AND r.status IN ('queued','running','pause_requested','paused','awaiting_confirmation') ORDER BY r.created_at DESC LIMIT 1) active_run_id,
                   (SELECT status FROM assistant_runs r WHERE r.conversation_id=c.id AND r.user_id=c.user_id
                    AND r.status IN ('queued','running','pause_requested','paused','awaiting_confirmation') ORDER BY r.created_at DESC LIMIT 1) active_run_status
                   FROM assistant_conversations c WHERE c.user_id=? AND c.deleted_at IS NULL
                   ORDER BY COALESCE(c.last_message_at,c.created_at) DESC""",
                (self.user_id,),
            ).fetchall()
        return tuple(self._conversation(row) for row in rows)

    def get_conversation(self, conversation_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT c.*,
                   (SELECT id FROM assistant_runs r WHERE r.conversation_id=c.id AND r.user_id=c.user_id
                    AND r.status IN ('queued','running','pause_requested','paused','awaiting_confirmation') ORDER BY r.created_at DESC LIMIT 1) active_run_id,
                   (SELECT status FROM assistant_runs r WHERE r.conversation_id=c.id AND r.user_id=c.user_id
                    AND r.status IN ('queued','running','pause_requested','paused','awaiting_confirmation') ORDER BY r.created_at DESC LIMIT 1) active_run_status
                   FROM assistant_conversations c WHERE c.id=? AND c.user_id=? AND c.deleted_at IS NULL""",
                (str(conversation_id), self.user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Assistant conversation was not found.")
        return self._conversation(row)

    def update_conversation(self, conversation_id: UUID, *, title: str | None = None, scope_type: str | None = None, scope_id: UUID | None = None) -> dict[str, Any]:
        current = self.get_conversation(conversation_id)
        next_scope = scope_type or current["scope_type"]
        next_scope_id = scope_id if scope_type is not None else current["scope_id"]
        self._validate_scope(next_scope, next_scope_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE assistant_conversations SET title=?,scope_type=?,scope_id=?,updated_at=? WHERE id=? AND user_id=?",
                (title.strip() if title else current["title"], next_scope, str(next_scope_id) if next_scope_id else None, _now(), str(conversation_id), self.user_id),
            )
        return self.get_conversation(conversation_id)

    def update_conversation_summary(
        self,
        conversation_id: UUID,
        *,
        summary: str,
        through_message_id: UUID,
        summary_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cursor = self.get_message(through_message_id)
        if cursor["conversation_id"] != conversation_id:
            raise RuntimeError("Assistant summary cursor does not belong to this conversation.")
        now = _now()
        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE assistant_conversations
                SET summary=?, summary_payload=?, summary_through_message_id=?,
                    summary_version=summary_version+1,
                    summary_updated_at=?, updated_at=?
                WHERE id=? AND user_id=? AND deleted_at IS NULL
                """,
                (
                    summary,
                    _json(summary_payload or {}),
                    str(through_message_id),
                    now,
                    now,
                    str(conversation_id),
                    self.user_id,
                ),
            )
        if result.rowcount != 1:
            raise RuntimeError("Assistant conversation was not found.")
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: UUID) -> None:
        self.get_conversation(conversation_id)
        now = _now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM assistant_runs
                WHERE conversation_id=? AND user_id=?
                  AND status IN (
                    'queued','running','pause_requested','paused',
                    'awaiting_confirmation','interrupted'
                  )
                """,
                (str(conversation_id), self.user_id),
            ).fetchall()
        for row in rows:
            self.request_cancel(UUID(str(row["id"])))
        with self._connect() as connection:
            connection.execute("UPDATE assistant_conversations SET deleted_at=?,updated_at=? WHERE id=? AND user_id=?", (now, now, str(conversation_id), self.user_id))

    def create_message(self, *, conversation_id: UUID, role: str, content: str, status: str = "completed", provider: str | None = None, model: str | None = None, citations: tuple[dict[str, Any], ...] = (), metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        message_id = uuid4()
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO assistant_messages
                   (id,conversation_id,user_id,role,content,status,provider,model,token_usage,citations,metadata,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (str(message_id), str(conversation_id), self.user_id, role, content, status, provider, model, "{}", _json(citations), _json(metadata or {}), now),
            )
            title = conversation["title"]
            if role == "user" and title == "新对话":
                title = " ".join(content.strip().split())[:36] or title
            connection.execute("UPDATE assistant_conversations SET title=?,last_message_at=?,updated_at=? WHERE id=? AND user_id=?", (title, now, now, str(conversation_id), self.user_id))
        return self.get_message(message_id)

    def get_message(self, message_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assistant_messages WHERE id=? AND user_id=?", (str(message_id), self.user_id)).fetchone()
        if row is None:
            raise RuntimeError("Assistant message was not found.")
        return self._message(row)

    def list_messages(self, conversation_id: UUID, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        self.get_conversation(conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM assistant_messages WHERE conversation_id=? AND user_id=? ORDER BY created_at ASC LIMIT ?",
                (str(conversation_id), self.user_id, max(1, min(limit, 500))),
            ).fetchall()
        return tuple(self._message(row) for row in rows)

    def list_messages_after(
        self,
        conversation_id: UUID,
        *,
        after_message_id: UUID | None,
        limit: int = 500,
    ) -> tuple[dict[str, Any], ...]:
        self.get_conversation(conversation_id)
        parameters: list[Any] = [str(conversation_id), self.user_id]
        cursor_filter = ""
        if after_message_id is not None:
            cursor = self.get_message(after_message_id)
            if cursor["conversation_id"] != conversation_id:
                raise RuntimeError("Assistant summary cursor does not belong to this conversation.")
            cursor_filter = " AND created_at > ?"
            parameters.append(cursor["created_at"])
        parameters.append(max(1, min(limit, 1_000)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assistant_messages
                WHERE conversation_id=? AND user_id=?
                """
                + cursor_filter
                + " ORDER BY created_at ASC LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return tuple(self._message(row) for row in rows)

    def update_message(self, message_id: UUID, *, content: str, status: str, provider: str | None = None, model: str | None = None, citations: tuple[dict[str, Any], ...] = (), token_usage: dict[str, int] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.get_message(message_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE assistant_messages SET content=?,status=?,provider=?,model=?,citations=?,token_usage=?,metadata=? WHERE id=? AND user_id=?",
                (content, status, provider, model, _json(citations), _json(token_usage or {}), _json(metadata or {}), str(message_id), self.user_id),
            )
        return self.get_message(message_id)

    def save_attachment(self, *, conversation_id: UUID, file_name: str, media_type: str, content: bytes, width: int, height: int) -> dict[str, Any]:
        self.get_conversation(conversation_id)
        import hashlib

        attachment_id = uuid4()
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[media_type]
        user_directory = hashlib.sha256(self.user_id.encode("utf-8")).hexdigest()[:24]
        relative = Path(user_directory) / str(conversation_id) / f"{attachment_id}{suffix}"
        target = (self.attachment_root / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO assistant_attachments
                   (id,conversation_id,message_id,user_id,file_name,media_type,size_bytes,sha256,width,height,storage_path,created_at)
                   VALUES (?,?,NULL,?,?,?,?,?,?,?,?,?)""",
                (str(attachment_id), str(conversation_id), self.user_id, file_name, media_type, len(content), hashlib.sha256(content).hexdigest(), width, height, str(relative).replace("\\", "/"), now),
            )
        return self.get_attachment(attachment_id)

    def save_attachment_file(
        self,
        *,
        conversation_id: UUID,
        file_name: str,
        media_type: str,
        source_path: Path,
        attachment_kind: str,
        width: int = 0,
        height: int = 0,
    ) -> dict[str, Any]:
        self.get_conversation(conversation_id)
        attachment_id = uuid4()
        suffix = Path(file_name).suffix.lower() or ".bin"
        user_directory = hashlib.sha256(self.user_id.encode("utf-8")).hexdigest()[:24]
        relative = Path(user_directory) / str(conversation_id) / f"{attachment_id}{suffix}"
        target = (self.attachment_root / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with source_path.open("rb") as source, target.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
                destination.write(chunk)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO assistant_attachments
                   (id,conversation_id,message_id,user_id,file_name,media_type,size_bytes,sha256,width,height,storage_path,created_at,attachment_kind,import_status,dataset_id,import_batch_id)
                   VALUES (?,?,NULL,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)""",
                (str(attachment_id), str(conversation_id), self.user_id, file_name, media_type, size, digest.hexdigest(), width, height, str(relative).replace("\\", "/"), now, attachment_kind, "uploaded" if attachment_kind == "data_file" else None),
            )
        return self.get_attachment(attachment_id)

    def attach_to_message(self, *, message_id: UUID, attachment_ids: tuple[UUID, ...]) -> None:
        message = self.get_message(message_id)
        with self._connect() as connection:
            for attachment_id in attachment_ids:
                result = connection.execute("UPDATE assistant_attachments SET message_id=? WHERE id=? AND conversation_id=? AND user_id=? AND message_id IS NULL", (str(message_id), str(attachment_id), str(message["conversation_id"]), self.user_id))
                if result.rowcount != 1:
                    raise RuntimeError("Assistant attachment was not found or is already used.")

    def get_attachment(self, attachment_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assistant_attachments WHERE id=? AND user_id=?", (str(attachment_id), self.user_id)).fetchone()
        if row is None:
            raise RuntimeError("Assistant attachment was not found.")
        return {key: row[key] for key in row.keys()}

    def list_message_attachments(self, message_id: UUID) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM assistant_attachments WHERE message_id=? AND user_id=? ORDER BY created_at", (str(message_id), self.user_id)).fetchall()
        return tuple({key: row[key] for key in row.keys()} for row in rows)

    def attachment_path(self, attachment_id: UUID) -> Path:
        item = self.get_attachment(attachment_id)
        path = (self.attachment_root / str(item["storage_path"])).resolve()
        if self.attachment_root.resolve() not in path.parents:
            raise RuntimeError("Invalid assistant attachment path.")
        return path

    def create_run(self, *, conversation_id: UUID, user_message_id: UUID, assistant_message_id: UUID, execution_mode: str = "ask") -> StoredAssistantRun:
        self.get_conversation(conversation_id)
        run_id = uuid4()
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO assistant_runs
                   (id,conversation_id,user_id,user_message_id,assistant_message_id,status,current_stage,analysis_job_id,pending_confirmation,error,cancel_requested,broker_task_id,attempt_count,lease_owner,lease_expires_at,heartbeat_at,checkpoint_thread_id,created_at,updated_at,completed_at,execution_mode,execution_plan,current_action_id,required_permission)
                   VALUES (?,?,?,?,?,'queued','queued',NULL,'{}',NULL,0,NULL,0,NULL,NULL,NULL,?,?,?,NULL,?,'{}',NULL,NULL)""",
                (str(run_id), str(conversation_id), self.user_id, str(user_message_id), str(assistant_message_id), str(run_id), now, now, execution_mode),
            )
        self.append_event(run_id, event_type="run.started", status="queued", message="Kimi 已收到问题。")
        return self.get_run(run_id)

    def get_run(self, run_id: UUID) -> StoredAssistantRun:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT r.*,(SELECT COALESCE(MAX(sequence),0) FROM assistant_run_events e WHERE e.run_id=r.id) last_event_sequence
                   FROM assistant_runs r WHERE r.id=? AND r.user_id=?""",
                (str(run_id), self.user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Assistant run was not found.")
        return StoredAssistantRun(
            id=UUID(str(row["id"])), conversation_id=UUID(str(row["conversation_id"])), user_message_id=UUID(str(row["user_message_id"])), assistant_message_id=UUID(str(row["assistant_message_id"])), status=str(row["status"]), current_stage=str(row["current_stage"]), analysis_job_id=UUID(str(row["analysis_job_id"])) if row["analysis_job_id"] else None, pending_confirmation=_loads(row["pending_confirmation"], {}), execution_mode=str(row["execution_mode"] or "ask"), execution_plan=_loads(row["execution_plan"], {}), current_action_id=UUID(str(row["current_action_id"])) if row["current_action_id"] else None, required_permission=str(row["required_permission"]) if row["required_permission"] else None, error=str(row["error"]) if row["error"] else None, cancel_requested=bool(row["cancel_requested"]), broker_task_id=str(row["broker_task_id"]) if row["broker_task_id"] else None, attempt_count=int(row["attempt_count"] or 0), lease_owner=str(row["lease_owner"]) if row["lease_owner"] else None, lease_expires_at=str(row["lease_expires_at"]) if row["lease_expires_at"] else None, heartbeat_at=str(row["heartbeat_at"]) if row["heartbeat_at"] else None, checkpoint_thread_id=str(row["checkpoint_thread_id"]) if row["checkpoint_thread_id"] else None, last_event_sequence=int(row["last_event_sequence"] or 0), created_at=str(row["created_at"]), updated_at=str(row["updated_at"]), completed_at=str(row["completed_at"]) if row["completed_at"] else None,
        )

    def claim_run(
        self,
        run_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> StoredAssistantRun | None:
        now = datetime.now(UTC)
        expires_at = (now + timedelta(seconds=max(30, lease_seconds))).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_runs
                SET status='running',
                    current_stage=CASE WHEN current_stage='resuming' THEN 'resuming' ELSE 'starting' END,
                    attempt_count=attempt_count+1, lease_owner=?,
                    lease_expires_at=?, heartbeat_at=?,
                    checkpoint_thread_id=COALESCE(checkpoint_thread_id,id),
                    updated_at=?
                WHERE id=? AND user_id=? AND cancel_requested=0
                  AND (
                    status IN ('queued','interrupted')
                    OR (
                        status='running'
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at < ?
                    )
                  )
                """,
                (
                    worker_id,
                    expires_at,
                    now.isoformat(),
                    now.isoformat(),
                    str(run_id),
                    self.user_id,
                    now.isoformat(),
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_run(run_id)

    def heartbeat_run(
        self,
        run_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        now = datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_runs
                SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE id=? AND user_id=? AND status='running' AND lease_owner=?
                """,
                (
                    now.isoformat(),
                    (now + timedelta(seconds=max(30, lease_seconds))).isoformat(),
                    now.isoformat(),
                    str(run_id),
                    self.user_id,
                    worker_id,
                ),
            )
        return cursor.rowcount > 0

    def release_run_lease(self, run_id: UUID, *, worker_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_runs
                SET lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    updated_at=?
                WHERE id=? AND user_id=? AND lease_owner=?
                """,
                (_now(), str(run_id), self.user_id, worker_id),
            )

    def list_all_recoverable_runs(
        self,
        *,
        limit: int = 500,
    ) -> tuple[dict[str, Any], ...]:
        now = _now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id,user_id,status,broker_task_id,updated_at,lease_expires_at
                FROM assistant_runs
                WHERE cancel_requested=0
                  AND (
                    status IN ('queued','interrupted')
                    OR (
                        status='running'
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at < ?
                    )
                  )
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, max(1, min(limit, 2000))),
            ).fetchall()
        return tuple(
            {
                "run_id": UUID(str(row["id"])),
                "user_id": str(row["user_id"]),
                "status": str(row["status"]),
                "broker_task_id": (
                    str(row["broker_task_id"]) if row["broker_task_id"] else None
                ),
                "updated_at": str(row["updated_at"]),
                "lease_expires_at": (
                    str(row["lease_expires_at"]) if row["lease_expires_at"] else None
                ),
            }
            for row in rows
        )

    def update_run(self, run_id: UUID, *, status: str | None = None, current_stage: str | None = None, analysis_job_id: UUID | None = None, pending_confirmation: dict[str, Any] | None = None, execution_plan: dict[str, Any] | None = None, current_action_id: UUID | None = None, required_permission: str | None = None, error: str | None = None, completed: bool = False) -> StoredAssistantRun:
        current = self.get_run(run_id)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """UPDATE assistant_runs SET status=?,current_stage=?,analysis_job_id=?,pending_confirmation=?,execution_plan=?,current_action_id=?,required_permission=?,error=?,updated_at=?,completed_at=? WHERE id=? AND user_id=?""",
                (status or current.status, current_stage or current.current_stage, str(analysis_job_id or current.analysis_job_id) if (analysis_job_id or current.analysis_job_id) else None, _json(pending_confirmation if pending_confirmation is not None else current.pending_confirmation), _json(execution_plan if execution_plan is not None else current.execution_plan), str(current_action_id) if current_action_id else (str(current.current_action_id) if current.current_action_id else None), required_permission if required_permission is not None else current.required_permission, error, now, now if completed else current.completed_at, str(run_id), self.user_id),
            )
        return self.get_run(run_id)

    def request_cancel(self, run_id: UUID) -> StoredAssistantRun:
        current = self.get_run(run_id)
        if current.status in {"completed", "failed", "canceled"}:
            return current
        now = _now()
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE assistant_runs
                SET cancel_requested=1,status='canceled',current_stage='canceled',
                    completed_at=?,updated_at=?
                WHERE id=? AND user_id=?
                  AND status NOT IN ('completed','failed','canceled')
                RETURNING id
                """,
                (
                    now,
                    now,
                    str(run_id),
                    self.user_id,
                ),
            ).fetchone()
        if changed is None:
            return self.get_run(run_id)
        self.update_message(
            current.assistant_message_id,
            content="已结束本次 Kimi 任务，已完成的步骤和事件仍会保留。",
            status="canceled",
            metadata={"canceled": True},
        )
        self.append_event(
            run_id,
            event_type="run.canceled",
            status="canceled",
            message="Kimi 任务已结束。",
        )
        return self.get_run(run_id)

    def complete_run_answer(
        self,
        run_id: UUID,
        *,
        content: str,
        provider: str | None,
        model: str | None,
        citations: tuple[dict[str, Any], ...],
        token_usage: dict[str, int],
        metadata: dict[str, Any],
        event_payload: dict[str, Any],
    ) -> bool:
        """Atomically commit an answer only while the run is still active."""
        now = _now()
        with self._connect() as connection:
            run = connection.execute(
                """
                UPDATE assistant_runs
                SET status='completed',current_stage='complete',
                    completed_at=?,updated_at=?
                WHERE id=? AND user_id=? AND status='running'
                  AND cancel_requested=0
                RETURNING assistant_message_id
                """,
                (now, now, str(run_id), self.user_id),
            ).fetchone()
            if run is None:
                return False
            connection.execute(
                """
                UPDATE assistant_messages
                SET content=?,status='completed',provider=?,model=?,
                    citations=?,token_usage=?,metadata=?
                WHERE id=? AND user_id=?
                """,
                (
                    content,
                    provider,
                    model,
                    _json(citations),
                    _json(token_usage),
                    _json(metadata),
                    str(run["assistant_message_id"]),
                    self.user_id,
                ),
            )
            event = connection.execute(
                """
                UPDATE assistant_runs
                SET next_event_sequence=next_event_sequence+1
                WHERE id=? AND user_id=?
                RETURNING next_event_sequence-1 AS sequence
                """,
                (str(run_id), self.user_id),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO assistant_run_events
                    (run_id,sequence,event_type,status,message,tool_name,payload,created_at)
                VALUES (?,?,'message.completed','completed',?,NULL,?,?)
                """,
                (
                    str(run_id),
                    int(event["sequence"]),
                    "Kimi 已完成回答。",
                    _json(event_payload),
                    now,
                ),
            )
        return True

    def request_pause(self, run_id: UUID) -> StoredAssistantRun:
        current = self.get_run(run_id)
        if current.status in {"paused", "pause_requested"}:
            return current
        if current.status not in {"queued", "running", "interrupted"}:
            raise RuntimeError("Assistant run cannot be paused in its current state.")
        immediate = current.status in {"queued", "interrupted"}
        status = "paused" if immediate else "pause_requested"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_runs
                SET status=?,current_stage=?,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    status,
                    "paused" if immediate else current.current_stage,
                    _now(),
                    str(run_id),
                    self.user_id,
                ),
            )
        self.append_event(
            run_id,
            event_type="run.paused" if immediate else "run.pause_requested",
            status=status,
            message="Kimi 任务已暂停。" if immediate else "将在当前安全步骤结束后暂停。",
        )
        return self.get_run(run_id)

    def mark_paused(self, run_id: UUID) -> StoredAssistantRun:
        current = self.get_run(run_id)
        if current.status == "paused":
            return current
        paused = self.update_run(run_id, status="paused", current_stage="paused")
        self.append_event(
            run_id,
            event_type="run.paused",
            status="paused",
            message="Kimi 任务已暂停，可稍后继续。",
        )
        return paused

    def resume_run(self, run_id: UUID) -> StoredAssistantRun:
        current = self.get_run(run_id)
        if current.status != "paused":
            raise RuntimeError("Assistant run is not paused.")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_runs
                SET status='queued',current_stage='resuming',cancel_requested=0,
                    broker_task_id=NULL,lease_owner=NULL,lease_expires_at=NULL,
                    heartbeat_at=NULL,error=NULL,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (_now(), str(run_id), self.user_id),
            )
        self.append_event(
            run_id,
            event_type="run.resumed",
            status="queued",
            message="Kimi 任务正在从已保存进度继续。",
        )
        return self.get_run(run_id)

    def pause_requested(self, run_id: UUID) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM assistant_runs WHERE id=? AND user_id=?",
                (str(run_id), self.user_id),
            ).fetchone()
        return bool(row and str(row["status"]) in {"pause_requested", "paused"})

    def cancel_requested(self, run_id: UUID) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT cancel_requested FROM assistant_runs WHERE id=? AND user_id=?", (str(run_id), self.user_id)).fetchone()
        return bool(row and row["cancel_requested"])

    def append_event(self, run_id: UUID, *, event_type: str, status: str, message: str, tool_name: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.get_run(run_id) if event_type != "run.started" else None
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                """
                UPDATE assistant_runs
                SET next_event_sequence=next_event_sequence+1
                WHERE id=? AND user_id=?
                RETURNING next_event_sequence-1 AS sequence
                """,
                (str(run_id), self.user_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("Assistant run was not found.")
            sequence = int(row["sequence"])
            connection.execute("INSERT INTO assistant_run_events (run_id,sequence,event_type,status,message,tool_name,payload,created_at) VALUES (?,?,?,?,?,?,?,?)", (str(run_id), sequence, event_type, status, message, tool_name, _json(payload or {}), now))
        return {"sequence": sequence, "event_type": event_type, "status": status, "message": message, "tool_name": tool_name, "payload": payload or {}, "created_at": now}

    def list_events(self, run_id: UUID, *, after_sequence: int = 0) -> tuple[dict[str, Any], ...]:
        self.get_run(run_id)
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM assistant_run_events WHERE run_id=? AND sequence>? ORDER BY sequence", (str(run_id), after_sequence)).fetchall()
        return tuple({"sequence": int(row["sequence"]), "event_type": str(row["event_type"]), "status": str(row["status"]), "message": str(row["message"]), "tool_name": str(row["tool_name"]) if row["tool_name"] else None, "payload": _loads(row["payload"], {}), "created_at": str(row["created_at"])} for row in rows)

    def set_broker_task(self, run_id: UUID, broker_task_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE assistant_runs SET broker_task_id=?,updated_at=? WHERE id=? AND user_id=?", (broker_task_id, _now(), str(run_id), self.user_id))

    def bind_analysis_job(self, run_id: UUID, job_id: UUID) -> None:
        self.update_run(run_id, analysis_job_id=job_id, current_stage="analysis")

    def confirm_run(self, run_id: UUID, *, accepted: bool) -> StoredAssistantRun:
        run = self.get_run(run_id)
        if run.status != "awaiting_confirmation":
            raise RuntimeError("Assistant run is not awaiting confirmation.")
        pending = dict(run.pending_confirmation)
        pending["accepted"] = accepted
        return self.update_run(run_id, status="queued", current_stage="confirmation_resolved", pending_confirmation=pending)

    def save_permission_grant(self, *, asset_type: str, asset_id: UUID, capabilities: tuple[str, ...]) -> dict[str, Any]:
        grant_id = uuid4()
        now = _now()
        normalized = tuple(sorted(set(capabilities)))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM assistant_permission_grants WHERE user_id=? AND asset_type=? AND asset_id=?",
                (self.user_id, asset_type, str(asset_id)),
            ).fetchone()
            if existing:
                grant_id = UUID(str(existing["id"]))
                connection.execute(
                    "UPDATE assistant_permission_grants SET capabilities=?,status='active',created_at=?,revoked_at=NULL WHERE id=? AND user_id=?",
                    (_json(normalized), now, str(grant_id), self.user_id),
                )
            else:
                connection.execute(
                    "INSERT INTO assistant_permission_grants (id,user_id,asset_type,asset_id,capabilities,status,created_at,revoked_at) VALUES (?,?,?,?,?,'active',?,NULL)",
                    (str(grant_id), self.user_id, asset_type, str(asset_id), _json(normalized), now),
                )
        return self.get_permission_grant(grant_id)

    def get_permission_grant(self, grant_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assistant_permission_grants WHERE id=? AND user_id=?",
                (str(grant_id), self.user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Assistant permission grant was not found.")
        return _grant(row)

    def list_permission_grants(self, *, active_only: bool = True) -> tuple[dict[str, Any], ...]:
        suffix = " AND status='active'" if active_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM assistant_permission_grants WHERE user_id=?{suffix} ORDER BY created_at DESC",
                (self.user_id,),
            ).fetchall()
        return tuple(_grant(row) for row in rows)

    def revoke_permission_grant(self, grant_id: UUID) -> dict[str, Any]:
        self.get_permission_grant(grant_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE assistant_permission_grants SET status='revoked',revoked_at=? WHERE id=? AND user_id=?",
                (_now(), str(grant_id), self.user_id),
            )
        return self.get_permission_grant(grant_id)

    def create_action(
        self,
        *,
        run_id: UUID | None,
        conversation_id: UUID | None,
        grant_id: UUID | None,
        tool_name: str,
        arguments_hash: str,
        idempotency_key: str,
        asset_type: str | None,
        asset_id: UUID | None,
    ) -> dict[str, Any]:
        action_id = uuid4()
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO assistant_action_log
                   (id,user_id,run_id,conversation_id,grant_id,tool_name,arguments_hash,idempotency_key,status,asset_type,asset_id,before_state,after_state,result,reversible,undone_at,error,created_at,completed_at)
                   VALUES (?,?,?,?,?,?,?,?, 'running',?,?, '{}','{}','{}',0,NULL,NULL,?,NULL)""",
                (str(action_id), self.user_id, str(run_id) if run_id else None, str(conversation_id) if conversation_id else None, str(grant_id) if grant_id else None, tool_name, arguments_hash, idempotency_key, asset_type, str(asset_id) if asset_id else None, now),
            )
        return self.get_action(action_id)

    def get_action(self, action_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assistant_action_log WHERE id=? AND user_id=?", (str(action_id), self.user_id)).fetchone()
        if row is None:
            raise RuntimeError("Assistant action was not found.")
        return _action(row)

    def get_action_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assistant_action_log WHERE idempotency_key=? AND user_id=?", (key, self.user_id)).fetchone()
        return _action(row) if row else None

    def complete_action(self, action_id: UUID, *, result: dict[str, Any], before_state: dict[str, Any], after_state: dict[str, Any], reversible: bool) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                "UPDATE assistant_action_log SET status='completed',result=?,before_state=?,after_state=?,reversible=?,completed_at=?,error=NULL WHERE id=? AND user_id=?",
                (_json(result), _json(before_state), _json(after_state), int(reversible), _now(), str(action_id), self.user_id),
            )
        return self.get_action(action_id)

    def fail_action(self, action_id: UUID, error: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("UPDATE assistant_action_log SET status='failed',error=?,completed_at=? WHERE id=? AND user_id=?", (error[:2000], _now(), str(action_id), self.user_id))
        return self.get_action(action_id)

    def mark_action_undone(self, action_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("UPDATE assistant_action_log SET status='undone',undone_at=? WHERE id=? AND user_id=?", (_now(), str(action_id), self.user_id))
        return self.get_action(action_id)

    def list_actions(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM assistant_action_log WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (self.user_id, max(1, min(limit, 500)))).fetchall()
        return tuple(_action(row) for row in rows)

    def deliverable_report_id_for_message(self, message_id: UUID) -> UUID | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT action.*
                FROM assistant_action_log AS action
                JOIN assistant_runs AS run ON run.id = action.run_id
                WHERE action.user_id = ? AND run.user_id = ?
                  AND run.assistant_message_id = ? AND action.status = 'completed'
                  AND action.tool_name IN ('start_analysis', 'retry_analysis', 'revise_report')
                ORDER BY action.completed_at DESC, action.created_at DESC
                """,
                (self.user_id, self.user_id, str(message_id)),
            ).fetchall()
        for row in rows:
            action = _action(row)
            for payload in (action.get("after_state") or {}, action.get("result") or {}):
                report_id = payload.get("created_report_id") or payload.get("report_id")
                if report_id:
                    return UUID(str(report_id))
        return None

    def create_import_batch(self, *, conversation_id: UUID, attachment_ids: tuple[UUID, ...], preview: dict[str, Any]) -> dict[str, Any]:
        self.get_conversation(conversation_id)
        batch_id = uuid4()
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO assistant_import_batches (id,user_id,conversation_id,attachment_ids,status,preview,dataset_ids,dataset_group_id,error,created_at,updated_at,completed_at) VALUES (?,?,?,?,'previewed',?,'[]',NULL,NULL,?,?,NULL)",
                (str(batch_id), self.user_id, str(conversation_id), _json(attachment_ids), _json(preview), now, now),
            )
            for attachment_id in attachment_ids:
                connection.execute("UPDATE assistant_attachments SET import_batch_id=?,import_status='previewed' WHERE id=? AND user_id=?", (str(batch_id), str(attachment_id), self.user_id))
        return self.get_import_batch(batch_id)

    def get_import_batch(self, batch_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM assistant_import_batches WHERE id=? AND user_id=?", (str(batch_id), self.user_id)).fetchone()
        if row is None:
            raise RuntimeError("Assistant import batch was not found.")
        return _import_batch(row)

    def complete_import_batch(self, batch_id: UUID, *, dataset_ids: tuple[UUID, ...], dataset_group_id: UUID | None, attachment_dataset_ids: dict[UUID, UUID]) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE assistant_import_batches SET status='completed',dataset_ids=?,dataset_group_id=?,updated_at=?,completed_at=?,error=NULL WHERE id=? AND user_id=?",
                (_json(dataset_ids), str(dataset_group_id) if dataset_group_id else None, now, now, str(batch_id), self.user_id),
            )
            for attachment_id, dataset_id in attachment_dataset_ids.items():
                connection.execute("UPDATE assistant_attachments SET import_status='completed',dataset_id=? WHERE id=? AND user_id=?", (str(dataset_id), str(attachment_id), self.user_id))
        return self.get_import_batch(batch_id)

    def fail_import_batch(self, batch_id: UUID, error: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("UPDATE assistant_import_batches SET status='failed',error=?,updated_at=? WHERE id=? AND user_id=?", (error[:2000], _now(), str(batch_id), self.user_id))
        return self.get_import_batch(batch_id)

    def _validate_scope(self, scope_type: str, scope_id: UUID | None) -> None:
        if scope_type == "auto":
            if scope_id is not None:
                raise RuntimeError("Auto assistant scope cannot have a scope id.")
            return
        if scope_id is None:
            raise RuntimeError("Assistant scope requires a scope id.")
        store = DatasetStoreRepository(str(self.root), user_id=self.user_id)
        if scope_type == "dataset":
            store.get_dataset(scope_id)
        elif scope_type == "dataset_group":
            store.get_dataset_group(scope_id)
        elif scope_type == "report":
            store.get_report(scope_id)
        else:
            raise RuntimeError("Unsupported assistant scope.")

    @staticmethod
    def _conversation(row: Any) -> dict[str, Any]:
        keys = set(row.keys())
        return {"conversation_id": UUID(str(row["id"])), "title": str(row["title"]), "scope_type": str(row["scope_type"]), "scope_id": UUID(str(row["scope_id"])) if row["scope_id"] else None, "summary": str(row["summary"] or ""), "summary_payload": _loads(row["summary_payload"], {}) if "summary_payload" in keys else {}, "summary_through_message_id": UUID(str(row["summary_through_message_id"])) if "summary_through_message_id" in keys and row["summary_through_message_id"] else None, "summary_version": int(row["summary_version"] or 0) if "summary_version" in keys else 0, "summary_updated_at": str(row["summary_updated_at"]) if "summary_updated_at" in keys and row["summary_updated_at"] else None, "active_run_id": UUID(str(row["active_run_id"])) if row["active_run_id"] else None, "active_run_status": str(row["active_run_status"]) if row["active_run_status"] else None, "created_at": str(row["created_at"]), "updated_at": str(row["updated_at"]), "last_message_at": str(row["last_message_at"]) if row["last_message_at"] else None}

    @staticmethod
    def _message(row: Any) -> dict[str, Any]:
        return {"message_id": UUID(str(row["id"])), "conversation_id": UUID(str(row["conversation_id"])), "role": str(row["role"]), "content": str(row["content"]), "status": str(row["status"]), "provider": str(row["provider"]) if row["provider"] else None, "model": str(row["model"]) if row["model"] else None, "token_usage": _loads(row["token_usage"], {}), "citations": tuple(_loads(row["citations"], [])), "metadata": _loads(row["metadata"], {}), "created_at": str(row["created_at"])}


_ASSISTANT_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_conversations (id TEXT PRIMARY KEY,user_id TEXT NOT NULL,title TEXT NOT NULL,scope_type TEXT NOT NULL,scope_id TEXT,summary TEXT NOT NULL DEFAULT '',summary_payload TEXT NOT NULL DEFAULT '{}',summary_through_message_id TEXT,summary_version INTEGER NOT NULL DEFAULT 0,summary_updated_at TEXT,deleted_at TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,last_message_at TEXT,idempotency_key TEXT,request_fingerprint TEXT);
CREATE TABLE IF NOT EXISTS assistant_messages (id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,user_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,status TEXT NOT NULL,provider TEXT,model TEXT,token_usage TEXT NOT NULL DEFAULT '{}',citations TEXT NOT NULL DEFAULT '[]',metadata TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assistant_runs (id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,user_id TEXT NOT NULL,user_message_id TEXT NOT NULL,assistant_message_id TEXT NOT NULL,status TEXT NOT NULL,current_stage TEXT NOT NULL,analysis_job_id TEXT,pending_confirmation TEXT NOT NULL DEFAULT '{}',error TEXT,cancel_requested INTEGER NOT NULL DEFAULT 0,broker_task_id TEXT,attempt_count INTEGER NOT NULL DEFAULT 0,lease_owner TEXT,lease_expires_at TEXT,heartbeat_at TEXT,checkpoint_thread_id TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,completed_at TEXT,execution_mode TEXT NOT NULL DEFAULT 'ask',execution_plan TEXT NOT NULL DEFAULT '{}',current_action_id TEXT,required_permission TEXT,next_event_sequence INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS assistant_run_events (run_id TEXT NOT NULL,sequence INTEGER NOT NULL,event_type TEXT NOT NULL,status TEXT NOT NULL,message TEXT NOT NULL,tool_name TEXT,payload TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL,PRIMARY KEY(run_id,sequence));
CREATE TABLE IF NOT EXISTS assistant_attachments (id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,message_id TEXT,user_id TEXT NOT NULL,file_name TEXT NOT NULL,media_type TEXT NOT NULL,size_bytes INTEGER NOT NULL,sha256 TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,storage_path TEXT NOT NULL,created_at TEXT NOT NULL,attachment_kind TEXT NOT NULL DEFAULT 'image',import_status TEXT,dataset_id TEXT,import_batch_id TEXT);
CREATE TABLE IF NOT EXISTS assistant_permission_grants (id TEXT PRIMARY KEY,user_id TEXT NOT NULL,asset_type TEXT NOT NULL,asset_id TEXT NOT NULL,capabilities TEXT NOT NULL DEFAULT '[]',status TEXT NOT NULL DEFAULT 'active',created_at TEXT NOT NULL,revoked_at TEXT);
CREATE TABLE IF NOT EXISTS assistant_action_log (id TEXT PRIMARY KEY,user_id TEXT NOT NULL,run_id TEXT,conversation_id TEXT,grant_id TEXT,tool_name TEXT NOT NULL,arguments_hash TEXT NOT NULL,idempotency_key TEXT NOT NULL,status TEXT NOT NULL,asset_type TEXT,asset_id TEXT,before_state TEXT NOT NULL DEFAULT '{}',after_state TEXT NOT NULL DEFAULT '{}',result TEXT NOT NULL DEFAULT '{}',reversible INTEGER NOT NULL DEFAULT 0,undone_at TEXT,error TEXT,created_at TEXT NOT NULL,completed_at TEXT);
CREATE TABLE IF NOT EXISTS assistant_import_batches (id TEXT PRIMARY KEY,user_id TEXT NOT NULL,conversation_id TEXT NOT NULL,attachment_ids TEXT NOT NULL DEFAULT '[]',status TEXT NOT NULL,preview TEXT NOT NULL DEFAULT '{}',dataset_ids TEXT NOT NULL DEFAULT '[]',dataset_group_id TEXT,error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,completed_at TEXT);
CREATE INDEX IF NOT EXISTS idx_assistant_conversations_user ON assistant_conversations(user_id,last_message_at);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation ON assistant_messages(conversation_id,created_at);
CREATE INDEX IF NOT EXISTS idx_assistant_runs_conversation ON assistant_runs(conversation_id,created_at);
CREATE INDEX IF NOT EXISTS idx_assistant_events_run ON assistant_run_events(run_id,sequence);
CREATE INDEX IF NOT EXISTS idx_assistant_attachments_message ON assistant_attachments(message_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_assistant_grant_active ON assistant_permission_grants(user_id,asset_type,asset_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_assistant_action_idempotency ON assistant_action_log(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_assistant_actions_user ON assistant_action_log(user_id,created_at);
CREATE INDEX IF NOT EXISTS idx_assistant_imports_conversation ON assistant_import_batches(conversation_id,created_at);
"""


def _ensure_column(connection: Any, table: str, column: str, definition: str) -> None:
    if hasattr(connection, "column_names"):
        columns = connection.column_names(table)
    else:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _normalize_conversation_title(title: str) -> str:
    return " ".join(title.split()) or "新对话"


def _conversation_request_fingerprint(
    *, title: str, scope_type: str, scope_id: UUID | None
) -> str:
    payload = {
        "scope_id": str(scope_id) if scope_id else None,
        "scope_type": scope_type.strip().lower(),
        "title": _normalize_conversation_title(title),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_integrity_error(error: Exception) -> bool:
    if isinstance(error, sqlite3.IntegrityError):
        return True
    try:
        from sqlalchemy.exc import IntegrityError
    except ImportError:
        return False
    return isinstance(error, IntegrityError)


def _grant(row: Any) -> dict[str, Any]:
    return {"grant_id": UUID(str(row["id"])), "asset_type": str(row["asset_type"]), "asset_id": UUID(str(row["asset_id"])), "capabilities": tuple(_loads(row["capabilities"], [])), "status": str(row["status"]), "created_at": str(row["created_at"]), "revoked_at": str(row["revoked_at"]) if row["revoked_at"] else None}


def _action(row: Any) -> dict[str, Any]:
    return {"action_id": UUID(str(row["id"])), "run_id": UUID(str(row["run_id"])) if row["run_id"] else None, "conversation_id": UUID(str(row["conversation_id"])) if row["conversation_id"] else None, "grant_id": UUID(str(row["grant_id"])) if row["grant_id"] else None, "tool_name": str(row["tool_name"]), "arguments_hash": str(row["arguments_hash"]), "idempotency_key": str(row["idempotency_key"]), "status": str(row["status"]), "asset_type": str(row["asset_type"]) if row["asset_type"] else None, "asset_id": UUID(str(row["asset_id"])) if row["asset_id"] else None, "before_state": _loads(row["before_state"], {}), "after_state": _loads(row["after_state"], {}), "result": _loads(row["result"], {}), "reversible": bool(row["reversible"]), "undone_at": str(row["undone_at"]) if row["undone_at"] else None, "error": str(row["error"]) if row["error"] else None, "created_at": str(row["created_at"]), "completed_at": str(row["completed_at"]) if row["completed_at"] else None}


def _import_batch(row: Any) -> dict[str, Any]:
    return {"batch_id": UUID(str(row["id"])), "conversation_id": UUID(str(row["conversation_id"])), "attachment_ids": tuple(UUID(str(item)) for item in _loads(row["attachment_ids"], [])), "status": str(row["status"]), "preview": _loads(row["preview"], {}), "dataset_ids": tuple(UUID(str(item)) for item in _loads(row["dataset_ids"], [])), "dataset_group_id": UUID(str(row["dataset_group_id"])) if row["dataset_group_id"] else None, "error": str(row["error"]) if row["error"] else None, "created_at": str(row["created_at"]), "updated_at": str(row["updated_at"]), "completed_at": str(row["completed_at"]) if row["completed_at"] else None}
