from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar
from uuid import UUID, uuid4

from app.core.settings import get_settings
from app.tool_results.artifacts import archive_json_payload, load_json_payload
from app.tool_results.context import build_tool_context_bundle
from app.tool_results.contracts import (
    ToolContextBundle,
    ToolResultArtifact,
    ToolResultChunkSummary,
    ToolResultEnvelope,
    ToolResultKind,
    ToolResultProjection,
    ToolResultStatus,
    ToolResultSummary,
)
from app.tool_results.distiller import (
    DeterministicToolResultDistiller,
    ToolResultDistiller,
)
from app.tool_results.projections import ProjectionPolicy, build_tool_result_projection
from app.tool_results.reducers import infer_result_kind


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ToolResultRepository:
    """User-scoped short-term archive for complete tool outputs.

    Payloads are gzip files, not long-term memories. The database stores only
    ownership, lifecycle and distillation metadata.
    """

    _initialization_lock = Lock()
    _initialized_stores: ClassVar[set[str]] = set()

    def __init__(
        self,
        root_path: str,
        *,
        user_id: str,
        artifact_root: str | None = None,
    ) -> None:
        settings = get_settings()
        self.user_id = user_id
        self.database_url = settings.database_url
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root.parent / "datamind.db" if root.name == "datasets" else root / "datamind.db"
        configured_artifact_root = Path(artifact_root or settings.tool_artifact_path)
        if artifact_root is not None or configured_artifact_root.is_absolute():
            self.artifact_root = configured_artifact_root
        elif root.name == "datasets":
            self.artifact_root = root.parent / configured_artifact_root.name
        else:
            self.artifact_root = root / configured_artifact_root.name
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = settings.tool_artifact_max_bytes
        self.ttl_days = settings.tool_artifact_ttl_days
        self.failed_ttl_days = settings.tool_artifact_failed_ttl_days
        self.context_target_chars = settings.tool_context_target_chars
        self.context_max_chars = settings.tool_context_max_chars
        self.continuation_enabled = settings.tool_continuation_enabled
        self.continuation_max_calls = settings.tool_continuation_max_calls
        self.continuation_max_chars = settings.tool_continuation_max_chars
        self.continuation_scan_max_bytes = settings.tool_continuation_scan_max_bytes
        self._initialize_once(environment=settings.environment)

    def _connect(self) -> Any:
        if self.database_url:
            from app.storage.sqlalchemy_compat import SQLAlchemyConnectionAdapter

            return SQLAlchemyConnectionAdapter(self.database_url)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_once(self, *, environment: str) -> None:
        key = self.database_url or str(self.db_path.resolve())
        if key in self._initialized_stores:
            return
        with self._initialization_lock:
            if key in self._initialized_stores:
                return
            if self.database_url and environment.lower() == "production":
                with self._connect() as connection:
                    connection.execute("SELECT id FROM tool_result_artifacts WHERE 1=0")
            else:
                with self._connect() as connection:
                    connection.executescript(_TOOL_RESULT_SCHEMA)
                    _ensure_sqlite_chunk_columns(connection)
            self._initialized_stores.add(key)

    def archive(self, envelope: ToolResultEnvelope) -> ToolResultArtifact:
        archived = archive_json_payload(
            envelope.payload,
            root=self.artifact_root,
            max_bytes=self.max_bytes,
        )
        existing = self._find_existing(
            run_id=envelope.run_id,
            action_hash=envelope.action_hash,
            payload_sha256=archived.payload_sha256,
        )
        if existing is not None:
            return existing

        kind = envelope.kind or infer_result_kind(
            envelope.tool_name,
            envelope.payload,
            failed=envelope.status == ToolResultStatus.FAILED,
        )
        retention = str(envelope.metadata.get("retention_policy") or "default")
        ttl_days = self.failed_ttl_days if envelope.status == ToolResultStatus.FAILED else self.ttl_days
        expires_at = None if retention == "report_evidence" else (
            datetime.now(UTC) + timedelta(days=ttl_days)
        ).isoformat()
        artifact_id = uuid4()
        now = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO tool_result_artifacts (
                        id,user_id,run_id,tool_name,action_hash,payload_sha256,status,
                        result_kind,content_type,size_bytes,compressed_size_bytes,
                        storage_path,metadata,expires_at,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(artifact_id),
                        self.user_id,
                        str(envelope.run_id),
                        envelope.tool_name,
                        envelope.action_hash,
                        archived.payload_sha256,
                        envelope.status.value,
                        kind.value,
                        envelope.content_type,
                        archived.size_bytes,
                        archived.compressed_size_bytes,
                        archived.storage_path,
                        json.dumps(envelope.metadata, ensure_ascii=False, default=str),
                        expires_at,
                        now,
                        now,
                    ),
                )
        except Exception:
            existing = self._find_existing(
                run_id=envelope.run_id,
                action_hash=envelope.action_hash,
                payload_sha256=archived.payload_sha256,
            )
            if existing is not None:
                return existing
            raise
        return self.get(artifact_id)

    def archive_and_summarize(
        self,
        envelope: ToolResultEnvelope,
        *,
        distiller: ToolResultDistiller | None = None,
    ) -> ToolContextBundle:
        artifact = self.archive(envelope)
        if distiller is not None and not isinstance(
            distiller, DeterministicToolResultDistiller
        ):
            try:
                existing = self.get_summary(artifact.artifact_id)
            except RuntimeError:
                existing = None
            if existing is not None and existing.summary_version >= 2:
                return build_tool_context_bundle(
                    artifact,
                    existing,
                    max_context_chars=self._context_limit(artifact),
                )
        result = (distiller or DeterministicToolResultDistiller()).distill(
            envelope,
            artifact_id=artifact.artifact_id,
        )
        summary_id = self.save_summary(
            result.summary,
            provider=result.provider,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        for chunk in result.chunks:
            self.save_chunk_summary(
                artifact.artifact_id,
                chunk,
                summary_id=summary_id,
            )
        return build_tool_context_bundle(
            artifact,
            result.summary,
            max_context_chars=self._context_limit(artifact),
            attempts=result.attempts,
        )

    def save_summary(
        self,
        summary: ToolResultSummary,
        *,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> UUID:
        if summary.artifact_id is None:
            raise ValueError("A summary must reference an archived tool result.")
        now = _now()
        payload = json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, default=str)
        with self._connect() as connection:
            existing = connection.execute(
                """SELECT id FROM tool_result_summaries
                   WHERE artifact_id=? AND summary_version=?""",
                (str(summary.artifact_id), summary.summary_version),
            ).fetchone()
            if existing is not None:
                connection.execute(
                    """UPDATE tool_result_summaries SET
                       summary=?,provider=?,model=?,input_tokens=?,output_tokens=?,
                       verified=?,verification_issues=?,updated_at=? WHERE id=?""",
                    (
                        payload,
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        bool(summary.verified),
                        json.dumps(summary.warnings, ensure_ascii=False),
                        now,
                        str(existing["id"]),
                    ),
                )
                return UUID(str(existing["id"]))
            summary_id = uuid4()
            connection.execute(
                """
                INSERT INTO tool_result_summaries (
                    id,artifact_id,summary_version,summary,provider,model,
                    input_tokens,output_tokens,verified,verification_issues,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(summary_id),
                    str(summary.artifact_id),
                    summary.summary_version,
                    payload,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    bool(summary.verified),
                    json.dumps(summary.warnings, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return summary_id

    def save_chunk_summary(
        self,
        artifact_id: UUID,
        chunk: ToolResultChunkSummary,
        *,
        summary_id: UUID,
    ) -> None:
        self.get(artifact_id)
        now = _now()
        payload = json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, default=str)
        idempotency_key = f"{artifact_id}:{chunk.chunk_index}:{chunk.content_sha256}"
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM tool_result_summary_chunks WHERE artifact_id=? AND chunk_index=?",
                (str(artifact_id), chunk.chunk_index),
            ).fetchone()
            values = (
                chunk.section,
                chunk.content_sha256,
                "completed" if chunk.verified else "rejected",
                str(summary_id),
                idempotency_key,
                payload,
                chunk.provider,
                chunk.model,
                chunk.input_tokens,
                chunk.output_tokens,
                bool(chunk.verified),
                json.dumps(chunk.verification_issues, ensure_ascii=False),
                None if chunk.verified else ",".join(chunk.verification_issues),
                now,
            )
            if existing is not None:
                connection.execute(
                    """UPDATE tool_result_summary_chunks SET
                       section=?,content_sha256=?,status=?,summary_id=?,idempotency_key=?,
                       summary=?,provider=?,model=?,input_tokens=?,output_tokens=?,verified=?,
                       verification_issues=?,error=?,updated_at=? WHERE id=?""",
                    (*values, str(existing["id"])),
                )
                return
            connection.execute(
                """INSERT INTO tool_result_summary_chunks (
                   id,artifact_id,chunk_index,section,content_sha256,status,summary_id,
                   idempotency_key,summary,provider,model,input_tokens,output_tokens,
                   verified,verification_issues,error,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid4()),
                    str(artifact_id),
                    chunk.chunk_index,
                    *values[:-1],
                    now,
                    values[-1],
                ),
            )

    def get(self, artifact_id: UUID) -> ToolResultArtifact:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tool_result_artifacts WHERE id=? AND user_id=?",
                (str(artifact_id), self.user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("Tool result artifact not found.")
        return _artifact_from_row(row)

    def load_payload(self, artifact_id: UUID) -> Any:
        artifact = self.get(artifact_id)
        return load_json_payload(root=self.artifact_root, storage_path=artifact.storage_path)

    def get_summary(self, artifact_id: UUID) -> ToolResultSummary:
        self.get(artifact_id)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT summary FROM tool_result_summaries
                   WHERE artifact_id=? ORDER BY summary_version DESC LIMIT 1""",
                (str(artifact_id),),
            ).fetchone()
        if row is None:
            raise RuntimeError("Tool result summary not found.")
        return ToolResultSummary.model_validate_json(str(row["summary"]))

    def list_chunk_summaries(
        self, artifact_id: UUID
    ) -> tuple[ToolResultChunkSummary, ...]:
        self.get(artifact_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT summary FROM tool_result_summary_chunks
                   WHERE artifact_id=? ORDER BY chunk_index""",
                (str(artifact_id),),
            ).fetchall()
        return tuple(
            ToolResultChunkSummary.model_validate_json(str(row["summary"]))
            for row in rows
            if row["summary"]
        )

    def model_context(self, artifact_id: UUID) -> dict[str, Any]:
        artifact = self.get(artifact_id)
        summary = self.get_summary(artifact_id)
        bundle = build_tool_context_bundle(
            artifact,
            summary,
            max_context_chars=self._context_limit(artifact),
        )
        return {
            "tool_result_artifact_id": str(artifact_id),
            "summary": bundle.summary.model_dump(mode="json"),
            "continuation_available": bool(
                artifact.size_bytes > bundle.context_size_bytes
                or bundle.summary.omitted_sections
            ),
        }

    def project_context(
        self,
        artifact_id: UUID,
        *,
        run_id: UUID,
        query: str,
    ) -> ToolResultProjection:
        if not self.continuation_enabled:
            raise RuntimeError("Tool result continuation is disabled.")
        artifact = self.get(artifact_id)
        if artifact.run_id != run_id:
            raise RuntimeError("Tool result artifact is outside the current run.")
        normalized_query = " ".join(query.split())[:1_000]
        if not normalized_query:
            raise ValueError("A continuation query is required.")
        query_hash = hashlib.sha256(normalized_query.casefold().encode()).hexdigest()
        existing = self._get_projection(artifact_id, run_id=run_id, query_hash=query_hash)
        if existing is not None:
            return existing
        if self.continuation_max_calls <= 0:
            raise RuntimeError("Tool result continuation is disabled.")
        if self._projection_count(artifact_id, run_id=run_id) >= self.continuation_max_calls:
            raise RuntimeError("Tool result continuation limit reached for this artifact.")

        projection = build_tool_result_projection(
            artifact_id=artifact_id,
            storage_root=self.artifact_root,
            storage_path=artifact.storage_path,
            artifact_size_bytes=artifact.size_bytes,
            summary=self.get_summary(artifact_id),
            chunks=self.list_chunk_summaries(artifact_id),
            query=normalized_query,
            policy=ProjectionPolicy(
                max_chars=self.continuation_max_chars,
                scan_max_bytes=self.continuation_scan_max_bytes,
            ),
        )
        return self._save_projection(projection, run_id=run_id)

    def retain_for_report(
        self,
        artifact_ids: tuple[UUID, ...],
        *,
        report_id: UUID,
    ) -> int:
        if not artifact_ids:
            return 0
        now = _now()
        retained = 0
        with self._connect() as connection:
            report = connection.execute(
                "SELECT 1 FROM reports WHERE id=? AND user_id=? LIMIT 1",
                (str(report_id), self.user_id),
            ).fetchone()
            if report is None:
                raise RuntimeError("Report not found for tool-result retention.")
            for artifact_id in dict.fromkeys(artifact_ids):
                row = connection.execute(
                    "SELECT metadata FROM tool_result_artifacts WHERE id=? AND user_id=?",
                    (str(artifact_id), self.user_id),
                ).fetchone()
                if row is None:
                    continue
                metadata = json.loads(str(row["metadata"] or "{}"))
                metadata.update(
                    {
                        "retention_policy": "report_evidence",
                        "report_id": str(report_id),
                    }
                )
                connection.execute(
                    """UPDATE tool_result_artifacts
                       SET metadata=?,expires_at=NULL,updated_at=?
                       WHERE id=? AND user_id=?""",
                    (
                        json.dumps(metadata, ensure_ascii=False, default=str),
                        now,
                        str(artifact_id),
                        self.user_id,
                    ),
                )
                retained += 1
        return retained

    def _get_projection(
        self,
        artifact_id: UUID,
        *,
        run_id: UUID,
        query_hash: str,
    ) -> ToolResultProjection | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT id,projection FROM tool_result_projections
                   WHERE artifact_id=? AND user_id=? AND run_id=? AND query_hash=?""",
                (str(artifact_id), self.user_id, str(run_id), query_hash),
            ).fetchone()
        if row is None:
            return None
        return ToolResultProjection.model_validate_json(str(row["projection"])).model_copy(
            update={"projection_id": UUID(str(row["id"]))}
        )

    def _projection_count(self, artifact_id: UUID, *, run_id: UUID) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS count FROM tool_result_projections
                   WHERE artifact_id=? AND user_id=? AND run_id=?""",
                (str(artifact_id), self.user_id, str(run_id)),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def _save_projection(
        self,
        projection: ToolResultProjection,
        *,
        run_id: UUID,
    ) -> ToolResultProjection:
        projection_id = uuid4()
        stored = projection.model_copy(update={"projection_id": projection_id})
        payload = json.dumps(stored.model_dump(mode="json"), ensure_ascii=False, default=str)
        now = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO tool_result_projections (
                       id,artifact_id,user_id,run_id,query_hash,projection,
                       selected_paths,context_size_bytes,scanned_bytes,truncated,
                       created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(projection_id),
                        str(projection.artifact_id),
                        self.user_id,
                        str(run_id),
                        projection.query_hash,
                        payload,
                        json.dumps(projection.selected_paths, ensure_ascii=False),
                        projection.context_size_bytes,
                        projection.scanned_bytes,
                        projection.truncated,
                        now,
                        now,
                    ),
                )
        except Exception:
            existing = self._get_projection(
                projection.artifact_id,
                run_id=run_id,
                query_hash=projection.query_hash,
            )
            if existing is not None:
                return existing
            raise
        return stored

    def _context_limit(self, artifact: ToolResultArtifact) -> int:
        if artifact.status == ToolResultStatus.FAILED or artifact.kind in {
            ToolResultKind.REPORT,
            ToolResultKind.ERROR,
        }:
            return self.context_max_chars
        return self.context_target_chars

    def purge_expired(self, *, at: datetime | None = None) -> int:
        cutoff = (at or datetime.now(UTC)).isoformat()
        with self._connect() as connection:
            expired = connection.execute(
                """SELECT id,storage_path FROM tool_result_artifacts
                   WHERE user_id=? AND expires_at IS NOT NULL AND expires_at<=?""",
                (self.user_id, cutoff),
            ).fetchall()
            retained = connection.execute(
                """SELECT id,storage_path,metadata FROM tool_result_artifacts
                   WHERE user_id=? AND expires_at IS NULL""",
                (self.user_id,),
            ).fetchall()
            orphaned = []
            for row in retained:
                metadata = json.loads(str(row["metadata"] or "{}"))
                report_id = metadata.get("report_id")
                if metadata.get("retention_policy") != "report_evidence" or not report_id:
                    continue
                report = connection.execute(
                    "SELECT 1 FROM reports WHERE id=? AND user_id=? LIMIT 1",
                    (str(report_id), self.user_id),
                ).fetchone()
                if report is None:
                    orphaned.append(row)
            rows = (*expired, *orphaned)
            for row in rows:
                connection.execute(
                    "DELETE FROM tool_result_projections WHERE artifact_id=?",
                    (str(row["id"]),),
                )
                connection.execute(
                    "DELETE FROM tool_result_summary_chunks WHERE artifact_id=?",
                    (str(row["id"]),),
                )
                connection.execute(
                    "DELETE FROM tool_result_summaries WHERE artifact_id=?",
                    (str(row["id"]),),
                )
                connection.execute(
                    "DELETE FROM tool_result_artifacts WHERE id=? AND user_id=?",
                    (str(row["id"]), self.user_id),
                )
        for row in rows:
            with self._connect() as connection:
                referenced = connection.execute(
                    "SELECT 1 FROM tool_result_artifacts WHERE storage_path=? LIMIT 1",
                    (str(row["storage_path"]),),
                ).fetchone()
            if referenced is None:
                (self.artifact_root / str(row["storage_path"])).unlink(missing_ok=True)
        return len(rows)

    def purge_orphan_files(self, *, older_than: datetime | None = None) -> int:
        cutoff = older_than or (datetime.now(UTC) - timedelta(days=1))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT storage_path FROM tool_result_artifacts"
            ).fetchall()
        referenced = {str(row["storage_path"]) for row in rows}
        removed = 0
        for path in self.artifact_root.rglob("*.json.gz"):
            relative = path.relative_to(self.artifact_root).as_posix()
            modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            if relative not in referenced and modified <= cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def metrics(self) -> dict[str, int | float]:
        with self._connect() as connection:
            artifacts = connection.execute(
                """SELECT COUNT(*) AS count,COALESCE(SUM(size_bytes),0) AS source,
                          COALESCE(SUM(compressed_size_bytes),0) AS compressed
                   FROM tool_result_artifacts WHERE user_id=?""",
                (self.user_id,),
            ).fetchone()
            projections = connection.execute(
                """SELECT COUNT(*) AS count,COALESCE(SUM(context_size_bytes),0) AS context,
                          COALESCE(SUM(scanned_bytes),0) AS scanned
                   FROM tool_result_projections WHERE user_id=?""",
                (self.user_id,),
            ).fetchone()
        source = int(artifacts["source"] if artifacts is not None else 0)
        context = int(projections["context"] if projections is not None else 0)
        return {
            "artifacts": int(artifacts["count"] if artifacts is not None else 0),
            "source_bytes": source,
            "compressed_bytes": int(artifacts["compressed"] if artifacts is not None else 0),
            "projections": int(projections["count"] if projections is not None else 0),
            "projection_context_bytes": context,
            "projection_scanned_bytes": int(
                projections["scanned"] if projections is not None else 0
            ),
            "projection_ratio": context / source if source else 0.0,
        }

    def list_user_ids(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT user_id FROM tool_result_artifacts ORDER BY user_id"
            ).fetchall()
        return tuple(str(row["user_id"]) for row in rows)

    def _find_existing(
        self,
        *,
        run_id: UUID,
        action_hash: str,
        payload_sha256: str,
    ) -> ToolResultArtifact | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM tool_result_artifacts
                WHERE user_id=? AND run_id=? AND action_hash=? AND payload_sha256=?
                """,
                (self.user_id, str(run_id), action_hash, payload_sha256),
            ).fetchone()
        return _artifact_from_row(row) if row is not None else None


def _artifact_from_row(row: Any) -> ToolResultArtifact:
    try:
        metadata = json.loads(str(row["metadata"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    return ToolResultArtifact(
        artifact_id=UUID(str(row["id"])),
        run_id=UUID(str(row["run_id"])),
        tool_name=str(row["tool_name"]),
        action_hash=str(row["action_hash"]),
        payload_sha256=str(row["payload_sha256"]),
        status=ToolResultStatus(str(row["status"])),
        kind=ToolResultKind(str(row["result_kind"])),
        content_type=str(row["content_type"]),
        size_bytes=int(row["size_bytes"]),
        compressed_size_bytes=int(row["compressed_size_bytes"]),
        storage_path=str(row["storage_path"]),
        expires_at=str(row["expires_at"]) if row["expires_at"] else None,
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=str(row["created_at"]),
    )


_TOOL_RESULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_result_artifacts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    result_kind TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    compressed_size_bytes INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id,run_id,action_hash,payload_sha256)
);
CREATE INDEX IF NOT EXISTS idx_tool_result_artifacts_user_run
ON tool_result_artifacts(user_id,run_id);
CREATE INDEX IF NOT EXISTS idx_tool_result_artifacts_sha
ON tool_result_artifacts(payload_sha256);
CREATE INDEX IF NOT EXISTS idx_tool_result_artifacts_expires
ON tool_result_artifacts(expires_at);

CREATE TABLE IF NOT EXISTS tool_result_summaries (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    summary_version INTEGER NOT NULL,
    summary TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    verified INTEGER NOT NULL DEFAULT 0,
    verification_issues TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(artifact_id) REFERENCES tool_result_artifacts(id) ON DELETE CASCADE,
    UNIQUE(artifact_id,summary_version)
);
CREATE INDEX IF NOT EXISTS idx_tool_result_summaries_artifact_version
ON tool_result_summaries(artifact_id,summary_version);

CREATE TABLE IF NOT EXISTS tool_result_summary_chunks (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    summary_id TEXT,
    idempotency_key TEXT NOT NULL,
    summary TEXT,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    verified INTEGER NOT NULL DEFAULT 0,
    verification_issues TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(artifact_id) REFERENCES tool_result_artifacts(id) ON DELETE CASCADE,
    FOREIGN KEY(summary_id) REFERENCES tool_result_summaries(id) ON DELETE SET NULL,
    UNIQUE(artifact_id,chunk_index),
    UNIQUE(idempotency_key)
);

CREATE TABLE IF NOT EXISTS tool_result_projections (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    projection TEXT NOT NULL,
    selected_paths TEXT NOT NULL DEFAULT '[]',
    context_size_bytes INTEGER NOT NULL,
    scanned_bytes INTEGER NOT NULL,
    truncated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(artifact_id) REFERENCES tool_result_artifacts(id) ON DELETE CASCADE,
    UNIQUE(user_id,run_id,artifact_id,query_hash)
);
CREATE INDEX IF NOT EXISTS idx_tool_result_projections_user_run
ON tool_result_projections(user_id,run_id);
CREATE INDEX IF NOT EXISTS idx_tool_result_projections_artifact
ON tool_result_projections(artifact_id);
"""


def _ensure_sqlite_chunk_columns(connection: sqlite3.Connection) -> None:
    existing = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(tool_result_summary_chunks)")
    }
    additions = {
        "summary": "TEXT",
        "provider": "TEXT",
        "model": "TEXT",
        "input_tokens": "INTEGER",
        "output_tokens": "INTEGER",
        "verified": "INTEGER NOT NULL DEFAULT 0",
        "verification_issues": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, definition in additions.items():
        if name not in existing:
            connection.execute(
                f"ALTER TABLE tool_result_summary_chunks ADD COLUMN {name} {definition}"
            )
