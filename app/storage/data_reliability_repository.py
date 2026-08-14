from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

from app.storage.repository_utils import now_iso as _now_iso
from app.storage.row_mappers import (
    data_drift_event_from_row as _data_drift_event_from_row,
)
from app.storage.row_mappers import (
    data_snapshot_from_row as _data_snapshot_from_row,
)
from app.storage.row_mappers import (
    json_loads as _json_loads,
)
from app.storage.row_mappers import (
    optional_text as _optional_text,
)


class DataReliabilityRepositoryMixin:
    def save_data_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = UUID(str(payload.get("id") or uuid4()))
        created_at = str(payload.get("created_at") or _now_iso())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO data_snapshots (
                    id, user_id, dataset_id, source, row_count, sample_size,
                    fingerprint, profile, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(snapshot_id),
                    self._user_id,
                    str(payload["dataset_id"]),
                    str(payload.get("source") or "raw"),
                    int(payload.get("row_count") or 0),
                    int(payload.get("sample_size") or 0),
                    str(payload["fingerprint"]),
                    json.dumps(payload.get("profile") or {}, ensure_ascii=False),
                    created_at,
                ),
            )
        return self.get_data_snapshot(snapshot_id)


    def get_data_snapshot(self, snapshot_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM data_snapshots WHERE id=? AND user_id=?",
                (str(snapshot_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Data snapshot was not found: {snapshot_id}")
        return _data_snapshot_from_row(row)


    def latest_data_snapshot(
        self,
        dataset_id: UUID,
        *,
        source: str | None = None,
    ) -> dict[str, Any] | None:
        self.get_dataset(dataset_id)
        source_filter = " AND source=?" if source is not None else ""
        parameters: tuple[object, ...] = (str(dataset_id), self._user_id)
        if source is not None:
            parameters += (source,)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM data_snapshots
                WHERE dataset_id=? AND user_id=?{source_filter}
                ORDER BY created_at DESC LIMIT 1
                """,
                parameters,
            ).fetchone()
        return _data_snapshot_from_row(row) if row is not None else None


    def save_data_drift_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = UUID(str(payload.get("id") or uuid4()))
        created_at = str(payload.get("created_at") or _now_iso())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO data_drift_events (
                    id, user_id, dataset_id, baseline_snapshot_id,
                    current_snapshot_id, status, changes, affected_assets,
                    recommended_actions, created_at, acknowledged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_id),
                    self._user_id,
                    str(payload["dataset_id"]),
                    str(payload["baseline_snapshot_id"]),
                    str(payload["current_snapshot_id"]),
                    str(payload.get("status") or "stable"),
                    json.dumps(payload.get("changes") or [], ensure_ascii=False),
                    json.dumps(payload.get("affected_assets") or [], ensure_ascii=False),
                    json.dumps(payload.get("recommended_actions") or [], ensure_ascii=False),
                    created_at,
                    _optional_text(payload.get("acknowledged_at")),
                ),
            )
        return self.get_data_drift_event(event_id)


    def get_data_drift_event(self, event_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM data_drift_events WHERE id=? AND user_id=?",
                (str(event_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Data drift event was not found: {event_id}")
        return _data_drift_event_from_row(row)


    def latest_data_drift_event(self, dataset_id: UUID) -> dict[str, Any] | None:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM data_drift_events
                WHERE dataset_id=? AND user_id=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (str(dataset_id), self._user_id),
            ).fetchone()
        return _data_drift_event_from_row(row) if row is not None else None


    def list_data_drift_events(
        self,
        *,
        dataset_id: UUID,
        limit: int = 50,
    ) -> tuple[dict[str, Any], ...]:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM data_drift_events
                WHERE dataset_id=? AND user_id=?
                ORDER BY created_at DESC LIMIT ?
                """,
                (str(dataset_id), self._user_id, max(1, min(limit, 200))),
            ).fetchall()
        return tuple(_data_drift_event_from_row(row) for row in rows)


    def replace_dataset_group_relationship_states(
        self,
        *,
        group_id: UUID,
        relationships: tuple[dict[str, Any], ...],
    ) -> None:
        self.get_dataset_group(group_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE dataset_groups
                SET relationships=?, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    json.dumps(relationships, ensure_ascii=False, default=str),
                    _now_iso(),
                    str(group_id),
                    self._user_id,
                ),
            )


    def mark_reports_stale(
        self,
        *,
        dataset_id: UUID,
        group_ids: tuple[UUID, ...],
        drift_event_id: UUID,
        reason: str,
    ) -> tuple[UUID, ...]:
        group_text = {str(group_id) for group_id in group_ids}
        changed: list[UUID] = []
        now = _now_iso()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM reports
                WHERE user_id=? AND deleted_at IS NULL
                """,
                (self._user_id,),
            ).fetchall()
            for row in rows:
                metadata = _json_loads(row["metadata"], {})
                additional_ids = {
                    str(item) for item in metadata.get("additional_dataset_ids") or ()
                }
                report_group_id = str(metadata.get("dataset_group_id") or "")
                if (
                    str(row["dataset_id"]) != str(dataset_id)
                    and str(dataset_id) not in additional_ids
                    and report_group_id not in group_text
                ):
                    continue
                metadata.update(
                    {
                        "freshness_status": "stale",
                        "stale_reason": reason,
                        "drift_event_id": str(drift_event_id),
                        "stale_at": now,
                    }
                )
                connection.execute(
                    """
                    UPDATE reports SET metadata=?, updated_at=?
                    WHERE id=? AND user_id=?
                    """,
                    (
                        json.dumps(metadata, ensure_ascii=False, default=str),
                        now,
                        str(row["id"]),
                        self._user_id,
                    ),
                )
                changed.append(UUID(str(row["id"])))
        return tuple(changed)
