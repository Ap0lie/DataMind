from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

from app.storage.repository_utils import now_iso as _now_iso
from app.storage.row_mappers import (
    json_loads as _json_loads,
)
from app.storage.row_mappers import (
    optional_text as _optional_text,
)
from app.storage.row_mappers import (
    semantic_model_from_row as _semantic_model_from_row,
)


class SemanticRepositoryMixin:
    def mark_semantic_models_stale(
        self,
        *,
        dataset_id: UUID,
        group_ids: tuple[UUID, ...],
        drift_event_id: UUID,
        reason: str,
    ) -> tuple[UUID, ...]:
        scopes = {("dataset", str(dataset_id))}
        scopes.update(("dataset_group", str(group_id)) for group_id in group_ids)
        changed: list[UUID] = []
        now = _now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM semantic_models
                WHERE user_id=? AND status='published' AND deleted_at IS NULL
                """,
                (self._user_id,),
            ).fetchall()
            for row in rows:
                if (str(row["scope_type"]), str(row["scope_id"])) not in scopes:
                    continue
                validation = _json_loads(row["validation"], {})
                validation["drift_event_id"] = str(drift_event_id)
                validation["stale_reason"] = reason
                connection.execute(
                    """
                    UPDATE semantic_models
                    SET status='stale', validation=?, updated_at=?
                    WHERE id=? AND user_id=?
                    """,
                    (
                        json.dumps(validation, ensure_ascii=False),
                        now,
                        str(row["id"]),
                        self._user_id,
                    ),
                )
                changed.append(UUID(str(row["id"])))
        return tuple(changed)


    def save_semantic_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = _now_iso()
        model_id = UUID(str(payload.get("id") or uuid4()))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO semantic_models (
                    id, user_id, scope_type, scope_id, name, version, revision, status,
                    source, parent_model_id, definition, schema_fingerprint, validation,
                    created_at, updated_at, published_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name, revision = excluded.revision,
                    status = excluded.status, definition = excluded.definition,
                    schema_fingerprint = excluded.schema_fingerprint,
                    validation = excluded.validation, updated_at = excluded.updated_at,
                    published_at = excluded.published_at
                """,
                (
                    str(model_id),
                    self._user_id,
                    str(payload["scope_type"]),
                    str(payload["scope_id"]),
                    str(payload.get("name") or "Semantic model"),
                    int(payload.get("version") or 1),
                    int(payload.get("revision") or 1),
                    str(payload.get("status") or "draft"),
                    str(payload.get("source") or "auto"),
                    _optional_text(payload.get("parent_model_id")),
                    json.dumps(payload.get("definition") or {}, ensure_ascii=False, default=str),
                    str(payload.get("schema_fingerprint") or ""),
                    json.dumps(payload.get("validation") or {}, ensure_ascii=False, default=str),
                    str(payload.get("created_at") or now),
                    now,
                    _optional_text(payload.get("published_at")),
                ),
            )
        return self.get_semantic_model(model_id)


    def get_semantic_embedding_cache(
        self, *, model_revision: str, text_hashes: tuple[str, ...]
    ) -> dict[str, tuple[float, ...]]:
        if not text_hashes:
            return {}
        placeholders = ",".join("?" for _ in text_hashes)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT text_hash, vector FROM semantic_embedding_cache WHERE user_id = ? AND model_revision = ? AND text_hash IN ({placeholders})",
                (self._user_id, model_revision, *text_hashes),
            ).fetchall()
        return {
            str(row["text_hash"]): tuple(float(value) for value in _json_loads(row["vector"], []))
            for row in rows
        }


    def save_semantic_embedding_cache(
        self, *, model_revision: str, vectors: dict[str, tuple[float, ...]]
    ) -> None:
        now = _now_iso()
        with self._connect() as connection:
            for text_hash, vector in vectors.items():
                connection.execute(
                    """INSERT INTO semantic_embedding_cache (user_id, model_revision, text_hash, vector, created_at)
                       VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, model_revision, text_hash)
                       DO UPDATE SET vector = excluded.vector, created_at = excluded.created_at""",
                    (self._user_id, model_revision, text_hash, json.dumps(vector), now),
                )


    def get_semantic_model(self, model_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM semantic_models WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (str(model_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Semantic model was not found: {model_id}")
        return _semantic_model_from_row(row)


    def list_semantic_models(
        self, *, scope_type: str, scope_id: UUID
    ) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM semantic_models
                   WHERE user_id = ? AND scope_type = ? AND scope_id = ? AND deleted_at IS NULL
                   ORDER BY version DESC, created_at DESC""",
                (self._user_id, scope_type, str(scope_id)),
            ).fetchall()
        return tuple(_semantic_model_from_row(row) for row in rows)


    def save_planner_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision_id = UUID(str(payload.get("id") or uuid4()))
        created_at = str(payload.get("created_at") or _now_iso())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO planner_decisions (
                    id, user_id, dataset_id, dataset_group_id, question, semantic_model_id,
                    semantic_model_version, semantic_source, semantic_plan, component_scores,
                    raw_confidence, calibrated_confidence, confidence_level,
                    requires_confirmation, confirmed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(decision_id),
                    self._user_id,
                    str(payload["dataset_id"]),
                    _optional_text(payload.get("dataset_group_id")),
                    str(payload["question"]),
                    _optional_text(payload.get("semantic_model_id")),
                    payload.get("semantic_model_version"),
                    str(payload.get("semantic_source") or "legacy"),
                    json.dumps(payload.get("semantic_plan") or {}, ensure_ascii=False, default=str),
                    json.dumps(
                        payload.get("component_scores") or {}, ensure_ascii=False, default=str
                    ),
                    float(payload.get("raw_confidence") or 0),
                    float(payload.get("calibrated_confidence") or 0),
                    str(payload.get("confidence_level") or "low"),
                    int(bool(payload.get("requires_confirmation"))),
                    int(bool(payload.get("confirmed"))),
                    created_at,
                ),
            )
        return self.get_planner_decision(decision_id)


    def get_planner_decision(self, decision_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM planner_decisions WHERE id = ? AND user_id = ?",
                (str(decision_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Planner decision was not found: {decision_id}")
        return {key: row[key] for key in row.keys()} | {
            "semantic_plan": _json_loads(row["semantic_plan"], {}),
            "component_scores": _json_loads(row["component_scores"], {}),
        }


    def save_planner_feedback(
        self, *, decision_id: UUID, action: str, corrected_plan: dict[str, Any]
    ) -> dict[str, Any]:
        decision = self.get_planner_decision(decision_id)
        feedback_id = uuid4()
        now = _now_iso()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO planner_feedback (id, user_id, decision_id, action, corrected_plan, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(feedback_id),
                    self._user_id,
                    str(decision_id),
                    action,
                    json.dumps(corrected_plan, ensure_ascii=False, default=str),
                    now,
                ),
            )
            if action in {"accepted", "edited"}:
                connection.execute(
                    "UPDATE planner_decisions SET confirmed = 1 WHERE id = ? AND user_id = ?",
                    (str(decision_id), self._user_id),
                )
        return {
            "id": feedback_id,
            "decision_id": UUID(str(decision["id"])),
            "action": action,
            "created_at": now,
        }


    def planner_training_samples(self) -> tuple[tuple[float, int], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT d.raw_confidence, f.action FROM planner_feedback f
                   JOIN planner_decisions d ON d.id = f.decision_id
                   WHERE f.user_id = ? ORDER BY f.created_at""",
                (self._user_id,),
            ).fetchall()
        return tuple(
            (float(row["raw_confidence"]), 1 if str(row["action"]) == "accepted" else 0)
            for row in rows
        )


    def active_planner_calibrator(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM planner_calibrators WHERE (user_id = ? OR user_id IS NULL) AND active = 1 ORDER BY CASE WHEN user_id = ? THEN 0 ELSE 1 END, version DESC LIMIT 1",
                (self._user_id, self._user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": UUID(str(row["id"])),
            "user_id": row["user_id"],
            "version": int(row["version"]),
            "sample_count": int(row["sample_count"]),
            "breakpoints": _json_loads(row["breakpoints"], []),
            "metrics": _json_loads(row["metrics"], {}),
        }


    def save_planner_calibrator(
        self,
        *,
        breakpoints: tuple[tuple[float, float], ...],
        sample_count: int,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        now = _now_iso()
        calibrator_id = uuid4()
        with self._connect() as connection:
            connection.execute(
                "UPDATE planner_calibrators SET active = 0 WHERE user_id = ?", (self._user_id,)
            )
            row = connection.execute(
                "SELECT MAX(version) AS version FROM planner_calibrators WHERE user_id = ?",
                (self._user_id,),
            ).fetchone()
            version = int(row["version"] or 0) + 1
            connection.execute(
                "INSERT INTO planner_calibrators (id, user_id, version, sample_count, breakpoints, metrics, active, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    str(calibrator_id),
                    self._user_id,
                    version,
                    sample_count,
                    json.dumps(breakpoints),
                    json.dumps(metrics),
                    now,
                ),
            )
        return {
            "id": calibrator_id,
            "version": version,
            "sample_count": sample_count,
            "breakpoints": breakpoints,
            "metrics": metrics,
        }


    def attach_planner_decision_to_job(self, *, job_id: UUID, decision: dict[str, Any]) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE analysis_jobs SET planner_decision_id = ?, semantic_model_id = ?,
                   semantic_model_version = ?, updated_at = ? WHERE id = ? AND user_id = ?""",
                (
                    str(decision["id"]),
                    _optional_text(decision.get("semantic_model_id")),
                    decision.get("semantic_model_version"),
                    _now_iso(),
                    str(job_id),
                    self._user_id,
                ),
            )
        if cursor.rowcount == 0:
            raise RuntimeError(f"Analysis job was not found: {job_id}")
