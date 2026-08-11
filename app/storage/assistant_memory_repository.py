from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Protocol
from uuid import UUID, uuid4

from app.core.settings import get_settings


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(str(value)) if value not in (None, "") else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


class AssistantMemoryStore(Protocol):
    """Storage contract kept small enough for a future LangGraph BaseStore adapter."""

    def get(self, memory_id: UUID, *, include_recycled: bool = True) -> dict[str, Any]: ...

    def list(self, **filters: Any) -> tuple[dict[str, Any], ...]: ...

    def save(self, **memory: Any) -> dict[str, Any]: ...


class AssistantMemoryRepository:
    """User-scoped persistence for trustworthy, versioned Assistant memory."""

    _initialization_lock = Lock()
    _initialized_stores: ClassVar[set[str]] = set()

    def __init__(self, root_path: str, *, user_id: str) -> None:
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        settings = get_settings()
        self.database_url = settings.database_url
        self.db_path = (
            self.root.parent / "datamind.db"
            if self.root.name == "datasets"
            else self.root / "datamind.db"
        )
        self._initialize_store_once(environment=settings.environment)

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
                with self._connect() as connection:
                    connection.execute("SELECT id FROM assistant_memories WHERE 1=0")
            else:
                with self._connect() as connection:
                    connection.executescript(_MEMORY_SCHEMA)
                    _ensure_v2_columns(connection)
                    _ensure_v3_columns(connection)
            self._initialized_stores.add(key)

    def save(
        self,
        *,
        memory_type: str,
        scope_type: str,
        scope_id: UUID | None,
        normalized_key: str,
        content: str,
        explicit: bool,
        confidence: float,
        status: str,
        pinned: bool = False,
        source_conversation_id: UUID | None = None,
        source_message_id: UUID | None = None,
        memory_kind: str = "semantic",
        subject_key: str | None = None,
        structured_value: dict[str, Any] | None = None,
        entity_key: str | None = None,
        predicate: str = "value",
        typed_value: dict[str, Any] | None = None,
        unit: str | None = None,
        application_policy: str = "relevant",
        source_kind: str = "user_message",
        source_job_id: UUID | None = None,
        correction: bool = False,
    ) -> dict[str, Any]:
        scope_key = str(scope_id) if scope_id else "user"
        subject = str(subject_key or normalized_key).strip()
        now = _now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assistant_memories
                WHERE user_id=? AND scope_type=? AND scope_key=?
                  AND memory_type=? AND subject_key=?
                ORDER BY version DESC,created_at DESC
                """,
                (self.user_id, scope_type, scope_key, memory_type, subject),
            ).fetchall()
            active = next((row for row in rows if str(row["status"]) == "active"), None)
            predecessor = active or next(
                (
                    row
                    for row in rows
                    if str(row["status"]) in {"dormant", "stale"}
                ),
                None,
            )
            equivalent = next(
                (
                    row
                    for row in rows
                    if str(row["status"]) in {"active", "pending"}
                    and _same_value(row, content, structured_value or {})
                ),
                None,
            )
            if equivalent is not None:
                memory_id = UUID(str(equivalent["id"]))
                self._merge_source(
                    connection,
                    row=equivalent,
                    source_conversation_id=source_conversation_id,
                    source_message_id=source_message_id,
                    explicit=explicit,
                    confidence=confidence,
                    pinned=pinned,
                    now=now,
                )
                return self._get_with_connection(connection, memory_id)

            next_version = max((int(row["version"] or 1) for row in rows), default=0) + 1
            memory_id = uuid4()
            pending_conflict = status == "pending" and active is not None
            supersedes_id = (
                UUID(str(predecessor["id"]))
                if predecessor is not None and not pending_conflict
                else None
            )
            stored_key = (
                f"{normalized_key}::pending::{str(memory_id)[:8]}"
                if pending_conflict
                else normalized_key
            )
            if supersedes_id is not None:
                self._retire_active(
                    connection,
                    predecessor,
                    memory_id,
                    now,
                    correction=correction,
                )
            sources = [str(source_message_id)] if source_message_id else []
            connection.execute(
                """
                INSERT INTO assistant_memories (
                    id,user_id,scope_type,scope_id,scope_key,memory_type,
                    normalized_key,subject_key,entity_key,predicate,content,
                    structured_value,typed_value,unit,memory_kind,
                    version,supersedes_id,superseded_by_id,valid_from,valid_to,
                    application_policy,source_kind,source_job_id,
                    source_conversation_id,source_message_id,source_message_ids,
                    explicit,confidence,status,pinned,last_used_at,recycle_from_status,
                    deleted_at,purge_after,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,NULL,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,?,?)
                """,
                (
                    str(memory_id),
                    self.user_id,
                    scope_type,
                    str(scope_id) if scope_id else None,
                    scope_key,
                    memory_type,
                    stored_key,
                    subject,
                    str(entity_key or subject),
                    str(predicate or "value"),
                    content,
                    json.dumps(structured_value or {}, ensure_ascii=False, default=str),
                    json.dumps(typed_value or structured_value or {}, ensure_ascii=False, default=str),
                    unit,
                    memory_kind,
                    next_version,
                    str(supersedes_id) if supersedes_id else None,
                    now,
                    application_policy,
                    source_kind,
                    str(source_job_id) if source_job_id else None,
                    str(source_conversation_id) if source_conversation_id else None,
                    str(source_message_id) if source_message_id else None,
                    json.dumps(sources, ensure_ascii=False),
                    bool(explicit),
                    max(0.0, min(1.0, float(confidence))),
                    status,
                    bool(pinned),
                    now,
                    now,
                ),
            )
            return self._get_with_connection(connection, memory_id)

    def _merge_source(
        self,
        connection: Any,
        *,
        row: Any,
        source_conversation_id: UUID | None,
        source_message_id: UUID | None,
        explicit: bool,
        confidence: float,
        pinned: bool,
        now: str,
    ) -> None:
        sources = list(_loads(row["source_message_ids"], []))
        if source_message_id and str(source_message_id) not in sources:
            sources.append(str(source_message_id))
        connection.execute(
            """
            UPDATE assistant_memories
            SET source_conversation_id=COALESCE(?,source_conversation_id),
                source_message_id=COALESCE(?,source_message_id),source_message_ids=?,
                explicit=CASE WHEN explicit OR ? THEN TRUE ELSE FALSE END,
                confidence=CASE WHEN confidence>? THEN confidence ELSE ? END,
                pinned=CASE WHEN pinned OR ? THEN TRUE ELSE FALSE END,updated_at=?
            WHERE id=? AND user_id=?
            """,
            (
                str(source_conversation_id) if source_conversation_id else None,
                str(source_message_id) if source_message_id else None,
                json.dumps(sources, ensure_ascii=False),
                bool(explicit),
                float(confidence),
                float(confidence),
                bool(pinned),
                now,
                str(row["id"]),
                self.user_id,
            ),
        )

    def _retire_active(
        self,
        connection: Any,
        row: Any,
        replacement_id: UUID,
        now: str,
        *,
        correction: bool = False,
    ) -> None:
        revision_key = f"{row['subject_key']}::v{int(row['version'] or 1)}::{str(row['id'])[:8]}"
        connection.execute(
            """
            UPDATE assistant_memories
            SET normalized_key=?,status='superseded',superseded_by_id=?,valid_to=?,
                correction_count=correction_count+?,updated_at=?
            WHERE id=? AND user_id=?
            """,
            (
                revision_key,
                str(replacement_id),
                now,
                1 if correction else 0,
                now,
                str(row["id"]),
                self.user_id,
            ),
        )

    def get(self, memory_id: UUID, *, include_recycled: bool = True) -> dict[str, Any]:
        with self._connect() as connection:
            return self._get_with_connection(connection, memory_id, include_recycled)

    def _get_with_connection(
        self,
        connection: Any,
        memory_id: UUID,
        include_recycled: bool = True,
    ) -> dict[str, Any]:
        source_join, source_field = _conversation_source_sql(connection)
        row = connection.execute(
            f"""
            SELECT m.*,{source_field} source_conversation_deleted
            FROM assistant_memories m {source_join}
            WHERE m.id=? AND m.user_id=?
            """
            + ("" if include_recycled else " AND m.status!='recycled'"),
            (str(memory_id), self.user_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Assistant memory was not found.")
        return _memory(row)

    def list(
        self,
        *,
        scope_type: str | None = None,
        scope_id: UUID | None = None,
        memory_type: str | None = None,
        memory_kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> tuple[dict[str, Any], ...]:
        filters = ["m.user_id=?"]
        parameters: list[Any] = [self.user_id]
        for column, value in (
            ("m.scope_type", scope_type),
            ("m.scope_id", str(scope_id) if scope_id else None),
            ("m.memory_type", memory_type),
            ("m.memory_kind", memory_kind),
            ("m.status", status),
        ):
            if value is not None:
                filters.append(f"{column}=?")
                parameters.append(value)
        needle = str(query or "").strip().lower()
        if needle:
            filters.append(
                "(LOWER(m.content) LIKE ? OR LOWER(m.normalized_key) LIKE ? OR LOWER(m.subject_key) LIKE ?)"
            )
            parameters.extend((f"%{needle}%", f"%{needle}%", f"%{needle}%"))
        parameters.append(max(1, min(limit, 500)))
        with self._connect() as connection:
            source_join, source_field = _conversation_source_sql(connection)
            rows = connection.execute(
                f"""
                SELECT m.*,{source_field} source_conversation_deleted
                FROM assistant_memories m {source_join}
                WHERE {" AND ".join(filters)}
                ORDER BY m.pinned DESC,m.updated_at DESC LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return tuple(_memory(row) for row in rows)

    def history(self, memory_id: UUID) -> tuple[dict[str, Any], ...]:
        current = self.get(memory_id)
        with self._connect() as connection:
            source_join, source_field = _conversation_source_sql(connection)
            rows = connection.execute(
                f"""
                SELECT m.*,{source_field} source_conversation_deleted
                FROM assistant_memories m {source_join}
                WHERE m.user_id=? AND m.scope_type=? AND m.scope_key=?
                  AND m.memory_type=? AND m.subject_key=?
                ORDER BY m.version DESC,m.created_at DESC
                """,
                (
                    self.user_id,
                    current["scope_type"],
                    str(current["scope_id"] or "user"),
                    current["memory_type"],
                    current["subject_key"],
                ),
            ).fetchall()
        return tuple(_memory(row) for row in rows)

    def update(
        self,
        memory_id: UUID,
        *,
        memory_type: str | None = None,
        normalized_key: str | None = None,
        content: str | None = None,
        pinned: bool | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        current = self.get(memory_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_memories
                SET memory_type=?,normalized_key=?,content=?,pinned=?,status=?,
                    recycle_from_status=NULL,deleted_at=NULL,purge_after=NULL,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    memory_type or current["memory_type"],
                    normalized_key or current["normalized_key"],
                    content if content is not None else current["content"],
                    bool(pinned if pinned is not None else current["pinned"]),
                    status or current["status"],
                    _now(),
                    str(memory_id),
                    self.user_id,
                ),
            )
        return self.get(memory_id)

    def confirm(self, memory_id: UUID) -> dict[str, Any]:
        current = self.get(memory_id)
        if current["status"] != "pending":
            return current
        now = _now()
        with self._connect() as connection:
            active = connection.execute(
                """
                SELECT * FROM assistant_memories
                WHERE user_id=? AND scope_type=? AND scope_key=? AND memory_type=?
                  AND subject_key=? AND status='active' AND id<>?
                ORDER BY version DESC LIMIT 1
                """,
                (
                    self.user_id,
                    current["scope_type"],
                    str(current["scope_id"] or "user"),
                    current["memory_type"],
                    current["subject_key"],
                    str(memory_id),
                ),
            ).fetchone()
            if active is not None:
                self._retire_active(connection, active, memory_id, now)
            connection.execute(
                """
                UPDATE assistant_memories
                SET normalized_key=?,status='active',supersedes_id=?,valid_from=?,valid_to=NULL,
                    deleted_at=NULL,purge_after=NULL,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    current["subject_key"],
                    str(active["id"]) if active is not None else None,
                    now,
                    now,
                    str(memory_id),
                    self.user_id,
                ),
            )
        return self.get(memory_id)

    def reactivate(self, memory_id: UUID) -> dict[str, Any]:
        current = self.get(memory_id)
        return self.save(
            memory_type=current["memory_type"],
            scope_type=current["scope_type"],
            scope_id=current["scope_id"],
            normalized_key=current["subject_key"],
            subject_key=current["subject_key"],
            entity_key=current["entity_key"],
            predicate=current["predicate"],
            content=current["content"],
            structured_value=current["structured_value"],
            typed_value=current["typed_value"],
            unit=current["unit"],
            memory_kind=current["memory_kind"],
            explicit=True,
            confidence=1.0,
            status="active",
            pinned=current["pinned"],
            source_kind="reactivated",
            application_policy=current["application_policy"],
        )

    def mark_stale(self, memory_id: UUID, *, reason: str) -> dict[str, Any]:
        current = self.get(memory_id)
        structured = dict(current["structured_value"])
        structured["stale_reason"] = reason
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_memories
                SET status='stale',structured_value=?,valid_to=?,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    json.dumps(structured, ensure_ascii=False, default=str),
                    _now(),
                    _now(),
                    str(memory_id),
                    self.user_id,
                ),
            )
        return self.get(memory_id)

    def mark_used(self, memory_ids: tuple[UUID, ...]) -> None:
        if not memory_ids:
            return
        now = _now()
        with self._connect() as connection:
            for memory_id in memory_ids:
                connection.execute(
                    """
                    UPDATE assistant_memories SET last_used_at=?,updated_at=?
                    WHERE id=? AND user_id=? AND status='active'
                    """,
                    (now, now, str(memory_id), self.user_id),
                )

    def record_usage(
        self,
        *,
        run_id: UUID,
        memory: dict[str, Any],
        assistant_message_id: UUID | None = None,
        retrieval_rank: int | None = None,
        final_selected: bool = True,
        suppression_reason: str | None = None,
    ) -> dict[str, Any]:
        return self.record_usage_batch(
            run_id=run_id,
            assistant_message_id=assistant_message_id,
            entries=(
                {
                    "memory": memory,
                    "retrieval_rank": retrieval_rank,
                    "final_selected": final_selected,
                    "suppression_reason": suppression_reason,
                },
            ),
        )[0]

    def record_usage_batch(
        self,
        *,
        run_id: UUID,
        entries: tuple[dict[str, Any], ...],
        assistant_message_id: UUID | None = None,
    ) -> tuple[dict[str, Any], ...]:
        if not entries:
            return ()
        now = _now()
        parameters = []
        memory_ids = []
        for entry in entries:
            memory = dict(entry["memory"])
            memory_id = UUID(str(memory["memory_id"]))
            scores = dict(memory.get("score_breakdown") or {})
            relevance_score = float(memory.get("relevance_score") or 0)
            utility_score = float(memory.get("utility_score") or 0.5)
            final_score = float(memory.get("final_score") or relevance_score)
            memory_ids.append(memory_id)
            parameters.append(
                (
                    str(uuid4()),
                    self.user_id,
                    str(run_id),
                    str(memory_id),
                    final_score,
                    float(scores.get("lexical") or 0),
                    float(scores.get("embedding") or 0),
                    float(scores.get("scope") or 0),
                    float(scores.get("recency") or 0),
                    str(memory.get("recall_reason") or "relevant context"),
                    str(memory["scope_type"]),
                    str(assistant_message_id) if assistant_message_id else None,
                    entry.get("retrieval_rank"),
                    bool(entry.get("final_selected", True)),
                    relevance_score,
                    utility_score,
                    final_score,
                    entry.get("suppression_reason"),
                    False,
                    now,
                )
            )
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO assistant_memory_usage
                    (id,user_id,run_id,memory_id,score,lexical_score,embedding_score,
                     scope_score,recency_score,reason,scope_type,assistant_message_id,
                     retrieval_rank,final_selected,relevance_score,utility_score,final_score,
                     suppression_reason,outcome_recorded,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (run_id,memory_id) DO UPDATE SET
                    score=excluded.score,lexical_score=excluded.lexical_score,
                    embedding_score=excluded.embedding_score,scope_score=excluded.scope_score,
                    recency_score=excluded.recency_score,reason=excluded.reason,
                    assistant_message_id=excluded.assistant_message_id,
                    retrieval_rank=excluded.retrieval_rank,
                    final_selected=excluded.final_selected,
                    relevance_score=excluded.relevance_score,
                    utility_score=excluded.utility_score,final_score=excluded.final_score,
                    suppression_reason=excluded.suppression_reason
                """,
                parameters,
            )
            rows = connection.execute(
                "SELECT * FROM assistant_memory_usage WHERE user_id=? AND run_id=?",
                (self.user_id, str(run_id)),
            ).fetchall()
        by_memory = {UUID(str(row["memory_id"])): _usage(row) for row in rows}
        return tuple(by_memory[memory_id] for memory_id in memory_ids)

    def list_usage(
        self,
        *,
        run_id: UUID,
        include_suppressed: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assistant_memory_usage
                WHERE user_id=? AND run_id=?
                """
                + ("" if include_suppressed else " AND final_selected IS TRUE")
                + " ORDER BY final_selected DESC,final_score DESC,created_at",
                (self.user_id, str(run_id)),
            ).fetchall()
        return tuple(_usage(row) for row in rows)

    def record_feedback(
        self,
        *,
        usage_id: UUID,
        feedback: str,
        reason: str | None,
        auto_dormancy: bool,
        dormancy_threshold: float,
        dormancy_min_feedback: int,
        wrong_feedback_limit: int,
    ) -> dict[str, Any]:
        if feedback not in {"helpful", "irrelevant", "wrong"}:
            raise ValueError("Unsupported assistant memory feedback.")
        now = _now()
        with self._connect() as connection:
            usage = connection.execute(
                """
                SELECT * FROM assistant_memory_usage
                WHERE id=? AND user_id=? AND final_selected IS TRUE
                """,
                (str(usage_id), self.user_id),
            ).fetchone()
            if usage is None:
                raise RuntimeError("Assistant memory usage was not found.")
            feedback_id = uuid4()
            connection.execute(
                """
                INSERT INTO assistant_memory_feedback
                    (id,user_id,usage_id,memory_id,run_id,feedback,reason,source,
                     created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,'user',?,?)
                ON CONFLICT (user_id,usage_id) DO UPDATE SET
                    feedback=excluded.feedback,reason=excluded.reason,
                    source='user',updated_at=excluded.updated_at
                """,
                (
                    str(feedback_id),
                    self.user_id,
                    str(usage_id),
                    str(usage["memory_id"]),
                    str(usage["run_id"]),
                    feedback,
                    str(reason or "").strip() or None,
                    now,
                    now,
                ),
            )
            memory = self._refresh_memory_utility(
                connection,
                UUID(str(usage["memory_id"])),
                auto_dormancy=auto_dormancy,
                dormancy_threshold=dormancy_threshold,
                dormancy_min_feedback=dormancy_min_feedback,
                wrong_feedback_limit=wrong_feedback_limit,
            )
            row = connection.execute(
                """
                SELECT * FROM assistant_memory_feedback
                WHERE user_id=? AND usage_id=?
                """,
                (self.user_id, str(usage_id)),
            ).fetchone()
        return _feedback(row, memory)

    def record_validated_reuse(self, *, run_id: UUID) -> int:
        now = _now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assistant_memory_usage
                WHERE user_id=? AND run_id=? AND final_selected IS TRUE
                  AND outcome_recorded IS FALSE
                """,
                (self.user_id, str(run_id)),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE assistant_memory_usage
                    SET outcome_recorded=TRUE WHERE id=? AND user_id=?
                    """,
                    (str(row["id"]), self.user_id),
                )
                connection.execute(
                    """
                    UPDATE assistant_memories
                    SET validated_reuse_count=validated_reuse_count+1,
                        last_validated_at=?,updated_at=?
                    WHERE id=? AND user_id=?
                    """,
                    (now, now, str(row["memory_id"]), self.user_id),
                )
                self._refresh_memory_utility(
                    connection,
                    UUID(str(row["memory_id"])),
                    auto_dormancy=False,
                )
        return len(rows)

    def _refresh_memory_utility(
        self,
        connection: Any,
        memory_id: UUID,
        *,
        auto_dormancy: bool,
        dormancy_threshold: float = 0.25,
        dormancy_min_feedback: int = 3,
        wrong_feedback_limit: int = 2,
    ) -> dict[str, Any]:
        row = connection.execute(
            "SELECT * FROM assistant_memories WHERE id=? AND user_id=?",
            (str(memory_id), self.user_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Assistant memory was not found.")
        counts = {"helpful": 0, "irrelevant": 0, "wrong": 0}
        feedback_rows = connection.execute(
            """
            SELECT feedback,COUNT(*) count FROM assistant_memory_feedback
            WHERE user_id=? AND memory_id=? GROUP BY feedback
            """,
            (self.user_id, str(memory_id)),
        ).fetchall()
        for item in feedback_rows:
            if str(item["feedback"]) in counts:
                counts[str(item["feedback"])] = int(item["count"] or 0)
        validated = int(row["validated_reuse_count"] or 0)
        corrections = int(row["correction_count"] or 0)
        positive = 2 + counts["helpful"] * 2 + validated
        negative = 2 + counts["irrelevant"] + counts["wrong"] * 3 + corrections * 2
        utility = round(positive / max(1, positive + negative), 4)
        feedback_count = sum(counts.values())
        status = str(row["status"])
        dormant_reason = row["dormant_reason"]
        should_sleep = (
            auto_dormancy
            and status == "active"
            and not bool(row["pinned"])
            and (
                counts["wrong"] >= wrong_feedback_limit
                or (
                    feedback_count >= dormancy_min_feedback
                    and utility < dormancy_threshold
                )
            )
        )
        if should_sleep:
            status = "dormant"
            dormant_reason = "多次反馈表明这条记忆无关或错误"
        connection.execute(
            """
            UPDATE assistant_memories
            SET helpful_count=?,irrelevant_count=?,wrong_count=?,feedback_count=?,
                utility_score=?,status=?,dormant_reason=?,updated_at=?
            WHERE id=? AND user_id=?
            """,
            (
                counts["helpful"],
                counts["irrelevant"],
                counts["wrong"],
                feedback_count,
                utility,
                status,
                dormant_reason,
                _now(),
                str(memory_id),
                self.user_id,
            ),
        )
        return self._get_with_connection(connection, memory_id)

    def wake(self, memory_id: UUID) -> dict[str, Any]:
        current = self.get(memory_id)
        if current["status"] != "dormant":
            raise RuntimeError("Only dormant memory can be woken.")
        return self.save(
            memory_type=current["memory_type"],
            scope_type=current["scope_type"],
            scope_id=current["scope_id"],
            normalized_key=current["subject_key"],
            subject_key=current["subject_key"],
            entity_key=current["entity_key"],
            predicate=current["predicate"],
            content=current["content"],
            structured_value=current["structured_value"],
            typed_value=current["typed_value"],
            unit=current["unit"],
            memory_kind=current["memory_kind"],
            explicit=True,
            confidence=1.0,
            status="active",
            pinned=current["pinned"],
            source_kind="woken",
            application_policy=current["application_policy"],
        )

    def effectiveness(self, *, shadow_mode: bool) -> dict[str, Any]:
        memories = self.list(limit=500)
        with self._connect() as connection:
            usage = connection.execute(
                """
                SELECT COUNT(*) total,
                       SUM(CASE WHEN final_selected IS TRUE THEN 1 ELSE 0 END) selected
                FROM assistant_memory_usage WHERE user_id=?
                """,
                (self.user_id,),
            ).fetchone()
            feedback_rows = connection.execute(
                """
                SELECT feedback,COUNT(*) count FROM assistant_memory_feedback
                WHERE user_id=? GROUP BY feedback
                """,
                (self.user_id,),
            ).fetchall()
            suppression_rows = connection.execute(
                """
                SELECT suppression_reason,COUNT(*) count FROM assistant_memory_usage
                WHERE user_id=? AND final_selected IS FALSE
                  AND suppression_reason IS NOT NULL
                GROUP BY suppression_reason
                """,
                (self.user_id,),
            ).fetchall()
        feedback_counts = {"helpful": 0, "irrelevant": 0, "wrong": 0}
        for row in feedback_rows:
            feedback_counts[str(row["feedback"])] = int(row["count"] or 0)
        suppression_counts = {
            str(row["suppression_reason"]): int(row["count"] or 0)
            for row in suppression_rows
        }
        return {
            "total_memories": len(memories),
            "active_memories": sum(item["status"] == "active" for item in memories),
            "dormant_memories": sum(item["status"] == "dormant" for item in memories),
            "low_quality_memories": sum(item["utility_score"] < 0.35 for item in memories),
            "never_used_memories": sum(item["last_used_at"] is None for item in memories),
            "usage_count": int(usage["total"] or 0),
            "selected_usage_count": int(usage["selected"] or 0),
            "average_utility": round(
                sum(item["utility_score"] for item in memories) / max(1, len(memories)),
                4,
            ),
            "feedback_counts": feedback_counts,
            "suppression_counts": suppression_counts,
            "shadow_mode": shadow_mode,
        }

    def get_settings(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assistant_memory_settings WHERE user_id=?",
                (self.user_id,),
            ).fetchone()
        return {
            "enabled": bool(row["enabled"]) if row is not None else True,
            "updated_at": str(row["updated_at"]) if row is not None else None,
        }

    def update_settings(self, *, enabled: bool) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_memory_settings (user_id,enabled,updated_at)
                VALUES (?,?,?) ON CONFLICT (user_id) DO UPDATE SET
                    enabled=excluded.enabled,updated_at=excluded.updated_at
                """,
                (self.user_id, bool(enabled), now),
            )
        return self.get_settings()

    def create_maintenance_job(
        self,
        *,
        run_id: UUID,
        conversation_id: UUID,
        user_message_id: UUID,
        assistant_message_id: UUID,
        analysis_job_id: UUID | None = None,
    ) -> dict[str, Any]:
        job_id = uuid4()
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_memory_maintenance_jobs
                    (id,user_id,run_id,conversation_id,user_message_id,assistant_message_id,
                     analysis_job_id,status,attempt_count,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,'queued',0,?,?)
                ON CONFLICT (user_message_id) DO NOTHING
                """,
                (
                    str(job_id),
                    self.user_id,
                    str(run_id),
                    str(conversation_id),
                    str(user_message_id),
                    str(assistant_message_id),
                    str(analysis_job_id) if analysis_job_id else None,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM assistant_memory_maintenance_jobs
                WHERE user_id=? AND user_message_id=?
                """,
                (self.user_id, str(user_message_id)),
            ).fetchone()
        return _maintenance_job(row)

    def get_maintenance_job(self, job_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assistant_memory_maintenance_jobs WHERE id=? AND user_id=?",
                (str(job_id), self.user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Assistant memory maintenance job was not found.")
        return _maintenance_job(row)

    def get_maintenance_job_for_run(self, run_id: UUID) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM assistant_memory_maintenance_jobs
                WHERE user_id=? AND run_id=? ORDER BY created_at DESC LIMIT 1
                """,
                (self.user_id, str(run_id)),
            ).fetchone()
        return _maintenance_job(row) if row is not None else None

    def claim_maintenance_job(
        self,
        job_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=max(30, lease_seconds))).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                UPDATE assistant_memory_maintenance_jobs
                SET status='running',attempt_count=attempt_count+1,lease_owner=?,
                    lease_expires_at=?,updated_at=?
                WHERE id=? AND user_id=? AND (
                    status='queued' OR (status='running' AND lease_expires_at<?)
                ) RETURNING *
                """,
                (
                    worker_id,
                    expires,
                    now.isoformat(),
                    str(job_id),
                    self.user_id,
                    now.isoformat(),
                ),
            ).fetchone()
        return _maintenance_job(row) if row is not None else None

    def finish_maintenance_job(self, job_id: UUID, *, error: str | None = None) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_memory_maintenance_jobs
                SET status=?,error=?,lease_owner=NULL,lease_expires_at=NULL,
                    completed_at=?,updated_at=? WHERE id=? AND user_id=?
                """,
                (
                    "failed" if error else "completed",
                    error,
                    now,
                    now,
                    str(job_id),
                    self.user_id,
                ),
            )

    def set_maintenance_broker_task(self, job_id: UUID, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_memory_maintenance_jobs SET broker_task_id=?,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (task_id, _now(), str(job_id), self.user_id),
            )

    def list_all_recoverable_maintenance_jobs(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM assistant_memory_maintenance_jobs
                WHERE status='queued' OR (status='running' AND lease_expires_at<?)
                ORDER BY created_at
                """,
                (_now(),),
            ).fetchall()
        return tuple(_maintenance_job(row) for row in rows)

    def recycle(self, memory_id: UUID, *, retention_days: int) -> dict[str, Any]:
        current = self.get(memory_id)
        if current["status"] == "recycled":
            return current
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assistant_memories
                SET recycle_from_status=?,status='recycled',deleted_at=?,purge_after=?,updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    current["status"],
                    now.isoformat(),
                    (now + timedelta(days=retention_days)).isoformat(),
                    now.isoformat(),
                    str(memory_id),
                    self.user_id,
                ),
            )
        return self.get(memory_id)

    def restore(self, memory_id: UUID) -> dict[str, Any]:
        current = self.get(memory_id)
        if current["status"] != "recycled":
            raise RuntimeError("Assistant memory is not recycled.")
        if current.get("recycle_from_status") in {"superseded", "stale"}:
            return self.reactivate(memory_id)
        return self.update(
            memory_id,
            status=current["recycle_from_status"] or ("active" if current["explicit"] else "pending"),
        )

    def recycle_stale(
        self,
        *,
        active_days: int,
        pending_days: int,
        retention_days: int,
        superseded_days: int = 180,
    ) -> int:
        now = datetime.now(UTC)
        active_before = (now - timedelta(days=active_days)).isoformat()
        pending_before = (now - timedelta(days=pending_days)).isoformat()
        superseded_before = (now - timedelta(days=superseded_days)).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM assistant_memories
                WHERE user_id=? AND pinned IS FALSE AND (
                    (status='active' AND COALESCE(last_used_at,updated_at) < ?)
                    OR (status='pending' AND updated_at < ?)
                    OR (status IN ('superseded','stale','dormant') AND COALESCE(valid_to,updated_at) < ?)
                )
                """,
                (self.user_id, active_before, pending_before, superseded_before),
            ).fetchall()
        memory_ids = [UUID(str(row["id"])) for row in rows]
        for memory_id in memory_ids:
            self.recycle(memory_id, retention_days=retention_days)
        return len(memory_ids)

    def purge_expired(self) -> int:
        with self._connect() as connection:
            result = connection.execute(
                """
                DELETE FROM assistant_memories
                WHERE user_id=? AND status='recycled' AND purge_after IS NOT NULL AND purge_after<=?
                """,
                (self.user_id, _now()),
            )
        return int(result.rowcount or 0)

    def list_user_ids(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT user_id FROM assistant_memories ORDER BY user_id"
            ).fetchall()
        return tuple(str(row["user_id"]) for row in rows)


def _memory(row: Any) -> dict[str, Any]:
    keys = set(row.keys())
    return {
        "memory_id": UUID(str(row["id"])),
        "memory_kind": str(row["memory_kind"] or "semantic"),
        "memory_type": str(row["memory_type"]),
        "scope_type": str(row["scope_type"]),
        "scope_id": UUID(str(row["scope_id"])) if row["scope_id"] else None,
        "normalized_key": str(row["normalized_key"]),
        "subject_key": str(row["subject_key"] or row["normalized_key"]),
        "entity_key": str(row["entity_key"] or row["subject_key"] or row["normalized_key"]),
        "predicate": str(row["predicate"] or "value"),
        "content": str(row["content"]),
        "structured_value": _loads(row["structured_value"], {}),
        "typed_value": _loads(row["typed_value"], {}),
        "unit": str(row["unit"]) if row["unit"] else None,
        "version": int(row["version"] or 1),
        "supersedes_id": UUID(str(row["supersedes_id"])) if row["supersedes_id"] else None,
        "superseded_by_id": UUID(str(row["superseded_by_id"])) if row["superseded_by_id"] else None,
        "application_policy": str(row["application_policy"] or "relevant"),
        "source_kind": str(row["source_kind"] or "user_message"),
        "source_job_id": UUID(str(row["source_job_id"])) if row["source_job_id"] else None,
        "source_conversation_id": UUID(str(row["source_conversation_id"])) if row["source_conversation_id"] else None,
        "source_message_id": UUID(str(row["source_message_id"])) if row["source_message_id"] else None,
        "source_message_ids": tuple(_loads(row["source_message_ids"], [])),
        "source_conversation_deleted": bool(row["source_conversation_deleted"])
        if "source_conversation_deleted" in keys
        else False,
        "explicit": bool(row["explicit"]),
        "confidence": float(row["confidence"]),
        "status": str(row["status"]),
        "pinned": bool(row["pinned"]),
        "last_used_at": str(row["last_used_at"]) if row["last_used_at"] else None,
        "utility_score": float(row["utility_score"] or 0.5),
        "helpful_count": int(row["helpful_count"] or 0),
        "irrelevant_count": int(row["irrelevant_count"] or 0),
        "wrong_count": int(row["wrong_count"] or 0),
        "correction_count": int(row["correction_count"] or 0),
        "validated_reuse_count": int(row["validated_reuse_count"] or 0),
        "feedback_count": int(row["feedback_count"] or 0),
        "last_validated_at": str(row["last_validated_at"])
        if row["last_validated_at"]
        else None,
        "dormant_reason": str(row["dormant_reason"]) if row["dormant_reason"] else None,
        "valid_from": str(row["valid_from"]) if row["valid_from"] else None,
        "valid_to": str(row["valid_to"]) if row["valid_to"] else None,
        "recycle_from_status": str(row["recycle_from_status"])
        if row["recycle_from_status"]
        else None,
        "deleted_at": str(row["deleted_at"]) if row["deleted_at"] else None,
        "purge_after": str(row["purge_after"]) if row["purge_after"] else None,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _usage(row: Any) -> dict[str, Any]:
    return {
        "usage_id": UUID(str(row["id"])),
        "run_id": UUID(str(row["run_id"])),
        "memory_id": UUID(str(row["memory_id"])),
        "assistant_message_id": UUID(str(row["assistant_message_id"]))
        if row["assistant_message_id"]
        else None,
        "retrieval_rank": int(row["retrieval_rank"])
        if row["retrieval_rank"] is not None
        else None,
        "final_selected": bool(row["final_selected"]),
        "score": float(row["score"]),
        "relevance_score": float(row["relevance_score"] or 0),
        "utility_score": float(row["utility_score"] or 0.5),
        "final_score": float(row["final_score"] or row["score"]),
        "lexical_score": float(row["lexical_score"]),
        "embedding_score": float(row["embedding_score"]),
        "scope_score": float(row["scope_score"]),
        "recency_score": float(row["recency_score"]),
        "reason": str(row["reason"]),
        "suppression_reason": str(row["suppression_reason"])
        if row["suppression_reason"]
        else None,
        "scope_type": str(row["scope_type"]),
        "created_at": str(row["created_at"]),
    }


def _feedback(row: Any, memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "feedback_id": UUID(str(row["id"])),
        "usage_id": UUID(str(row["usage_id"])),
        "memory_id": UUID(str(row["memory_id"])),
        "run_id": UUID(str(row["run_id"])),
        "feedback": str(row["feedback"]),
        "reason": str(row["reason"]) if row["reason"] else None,
        "utility_score": memory["utility_score"],
        "memory_status": memory["status"],
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _maintenance_job(row: Any) -> dict[str, Any]:
    return {
        "job_id": UUID(str(row["id"])),
        "user_id": str(row["user_id"]),
        "run_id": UUID(str(row["run_id"])),
        "conversation_id": UUID(str(row["conversation_id"])),
        "user_message_id": UUID(str(row["user_message_id"])),
        "assistant_message_id": UUID(str(row["assistant_message_id"])),
        "analysis_job_id": UUID(str(row["analysis_job_id"])) if row["analysis_job_id"] else None,
        "status": str(row["status"]),
        "attempt_count": int(row["attempt_count"] or 0),
        "broker_task_id": str(row["broker_task_id"]) if row["broker_task_id"] else None,
        "lease_owner": str(row["lease_owner"]) if row["lease_owner"] else None,
        "lease_expires_at": str(row["lease_expires_at"]) if row["lease_expires_at"] else None,
        "error": str(row["error"]) if row["error"] else None,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "completed_at": str(row["completed_at"]) if row["completed_at"] else None,
    }


def _same_value(row: Any, content: str, structured_value: dict[str, Any]) -> bool:
    left = json.dumps(
        _loads(row["structured_value"], {}), ensure_ascii=False, sort_keys=True, default=str
    )
    right = json.dumps(structured_value, ensure_ascii=False, sort_keys=True, default=str)
    return " ".join(str(row["content"]).casefold().split()) == " ".join(
        content.casefold().split()
    ) and left == right


def _conversation_source_sql(connection: Any) -> tuple[str, str]:
    try:
        connection.execute("SELECT id FROM assistant_conversations WHERE 1=0")
    except Exception:
        return "", "0"
    return (
        "LEFT JOIN assistant_conversations c ON c.id=m.source_conversation_id AND c.user_id=m.user_id",
        "CASE WHEN c.deleted_at IS NOT NULL THEN 1 ELSE 0 END",
    )


def _ensure_v2_columns(connection: Any) -> None:
    columns = (
        connection.column_names("assistant_memories")
        if hasattr(connection, "column_names")
        else {str(row[1]) for row in connection.execute("PRAGMA table_info(assistant_memories)")}
    )
    additions = {
        "memory_kind": "TEXT NOT NULL DEFAULT 'semantic'",
        "subject_key": "TEXT NOT NULL DEFAULT ''",
        "structured_value": "TEXT NOT NULL DEFAULT '{}'",
        "version": "INTEGER NOT NULL DEFAULT 1",
        "supersedes_id": "TEXT",
        "superseded_by_id": "TEXT",
        "valid_from": "TEXT",
        "valid_to": "TEXT",
        "application_policy": "TEXT NOT NULL DEFAULT 'relevant'",
        "source_kind": "TEXT NOT NULL DEFAULT 'user_message'",
        "source_job_id": "TEXT",
    }
    for name, definition in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE assistant_memories ADD COLUMN {name} {definition}")
    connection.execute(
        "UPDATE assistant_memories SET subject_key=normalized_key WHERE subject_key='' OR subject_key IS NULL"
    )
    connection.execute(
        "UPDATE assistant_memories SET valid_from=created_at WHERE valid_from IS NULL"
    )


def _ensure_v3_columns(connection: Any) -> None:
    memory_columns = (
        connection.column_names("assistant_memories")
        if hasattr(connection, "column_names")
        else {str(row[1]) for row in connection.execute("PRAGMA table_info(assistant_memories)")}
    )
    memory_additions = {
        "entity_key": "TEXT NOT NULL DEFAULT ''",
        "predicate": "TEXT NOT NULL DEFAULT 'value'",
        "typed_value": "TEXT NOT NULL DEFAULT '{}'",
        "unit": "TEXT",
        "utility_score": "REAL NOT NULL DEFAULT 0.5",
        "helpful_count": "INTEGER NOT NULL DEFAULT 0",
        "irrelevant_count": "INTEGER NOT NULL DEFAULT 0",
        "wrong_count": "INTEGER NOT NULL DEFAULT 0",
        "correction_count": "INTEGER NOT NULL DEFAULT 0",
        "validated_reuse_count": "INTEGER NOT NULL DEFAULT 0",
        "feedback_count": "INTEGER NOT NULL DEFAULT 0",
        "last_validated_at": "TEXT",
        "dormant_reason": "TEXT",
    }
    for name, definition in memory_additions.items():
        if name not in memory_columns:
            connection.execute(f"ALTER TABLE assistant_memories ADD COLUMN {name} {definition}")
    connection.execute(
        "UPDATE assistant_memories SET entity_key=subject_key WHERE entity_key='' OR entity_key IS NULL"
    )
    connection.execute(
        "UPDATE assistant_memories SET typed_value=structured_value WHERE typed_value='{}' OR typed_value IS NULL"
    )

    usage_columns = (
        connection.column_names("assistant_memory_usage")
        if hasattr(connection, "column_names")
        else {str(row[1]) for row in connection.execute("PRAGMA table_info(assistant_memory_usage)")}
    )
    usage_additions = {
        "assistant_message_id": "TEXT",
        "retrieval_rank": "INTEGER",
        "final_selected": "INTEGER NOT NULL DEFAULT 1",
        "relevance_score": "REAL NOT NULL DEFAULT 0",
        "utility_score": "REAL NOT NULL DEFAULT 0.5",
        "final_score": "REAL NOT NULL DEFAULT 0",
        "suppression_reason": "TEXT",
        "outcome_recorded": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in usage_additions.items():
        if name not in usage_columns:
            connection.execute(f"ALTER TABLE assistant_memory_usage ADD COLUMN {name} {definition}")
    connection.execute(
        "UPDATE assistant_memory_usage SET relevance_score=score,final_score=score WHERE final_score=0"
    )


_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_memories (
    id TEXT PRIMARY KEY,user_id TEXT NOT NULL,scope_type TEXT NOT NULL,scope_id TEXT,
    scope_key TEXT NOT NULL,memory_type TEXT NOT NULL,normalized_key TEXT NOT NULL,
    subject_key TEXT NOT NULL DEFAULT '',entity_key TEXT NOT NULL DEFAULT '',
    predicate TEXT NOT NULL DEFAULT 'value',content TEXT NOT NULL,
    structured_value TEXT NOT NULL DEFAULT '{}',typed_value TEXT NOT NULL DEFAULT '{}',
    unit TEXT,memory_kind TEXT NOT NULL DEFAULT 'semantic',
    version INTEGER NOT NULL DEFAULT 1,supersedes_id TEXT,superseded_by_id TEXT,
    valid_from TEXT,valid_to TEXT,application_policy TEXT NOT NULL DEFAULT 'relevant',
    source_kind TEXT NOT NULL DEFAULT 'user_message',source_job_id TEXT,
    source_conversation_id TEXT,source_message_id TEXT,
    source_message_ids TEXT NOT NULL DEFAULT '[]',explicit INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,status TEXT NOT NULL,pinned INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,utility_score REAL NOT NULL DEFAULT 0.5,
    helpful_count INTEGER NOT NULL DEFAULT 0,irrelevant_count INTEGER NOT NULL DEFAULT 0,
    wrong_count INTEGER NOT NULL DEFAULT 0,correction_count INTEGER NOT NULL DEFAULT 0,
    validated_reuse_count INTEGER NOT NULL DEFAULT 0,feedback_count INTEGER NOT NULL DEFAULT 0,
    last_validated_at TEXT,dormant_reason TEXT,recycle_from_status TEXT,
    deleted_at TEXT,purge_after TEXT,
    created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
    UNIQUE(user_id,scope_type,scope_key,memory_type,normalized_key)
);
CREATE INDEX IF NOT EXISTS idx_assistant_memories_user_status
ON assistant_memories(user_id,status,pinned,updated_at);
CREATE INDEX IF NOT EXISTS idx_assistant_memories_scope
ON assistant_memories(user_id,scope_type,scope_key,status);
CREATE TABLE IF NOT EXISTS assistant_memory_settings (
    user_id TEXT PRIMARY KEY,enabled INTEGER NOT NULL DEFAULT 1,updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assistant_memory_usage (
    id TEXT PRIMARY KEY,user_id TEXT NOT NULL,run_id TEXT NOT NULL,memory_id TEXT NOT NULL,
    score REAL NOT NULL,lexical_score REAL NOT NULL,embedding_score REAL NOT NULL,
    scope_score REAL NOT NULL,recency_score REAL NOT NULL,reason TEXT NOT NULL,
    scope_type TEXT NOT NULL,assistant_message_id TEXT,retrieval_rank INTEGER,
    final_selected INTEGER NOT NULL DEFAULT 1,relevance_score REAL NOT NULL DEFAULT 0,
    utility_score REAL NOT NULL DEFAULT 0.5,final_score REAL NOT NULL DEFAULT 0,
    suppression_reason TEXT,outcome_recorded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,UNIQUE(run_id,memory_id)
);
CREATE INDEX IF NOT EXISTS idx_assistant_memory_usage_run
ON assistant_memory_usage(user_id,run_id,created_at);
CREATE TABLE IF NOT EXISTS assistant_memory_feedback (
    id TEXT PRIMARY KEY,user_id TEXT NOT NULL,usage_id TEXT NOT NULL,memory_id TEXT NOT NULL,
    run_id TEXT NOT NULL,feedback TEXT NOT NULL,reason TEXT,source TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(user_id,usage_id)
);
CREATE INDEX IF NOT EXISTS idx_assistant_memory_feedback_memory
ON assistant_memory_feedback(user_id,memory_id,updated_at);
CREATE TABLE IF NOT EXISTS assistant_memory_maintenance_jobs (
    id TEXT PRIMARY KEY,user_id TEXT NOT NULL,run_id TEXT NOT NULL,conversation_id TEXT NOT NULL,
    user_message_id TEXT NOT NULL UNIQUE,assistant_message_id TEXT NOT NULL,analysis_job_id TEXT,
    status TEXT NOT NULL,attempt_count INTEGER NOT NULL DEFAULT 0,broker_task_id TEXT,
    lease_owner TEXT,lease_expires_at TEXT,error TEXT,created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_assistant_memory_maintenance_status
ON assistant_memory_maintenance_jobs(status,lease_expires_at,updated_at);
"""
