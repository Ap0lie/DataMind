from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

from app.storage.models import StoredDatasetGroup
from app.storage.repository_utils import dedupe_uuids as _dedupe_uuids
from app.storage.repository_utils import now_iso as _now_iso
from app.storage.repository_utils import recycle_times as _recycle_times
from app.storage.repository_utils import (
    validate_relationship_graph_structure as _validate_relationship_graph_structure,
)
from app.storage.row_mappers import (
    stored_dataset_group_from_row as _stored_dataset_group_from_row,
)


class DatasetGroupRepositoryMixin:
    def create_dataset_group(
        self,
        *,
        name: str,
        dataset_ids: tuple[UUID, ...],
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StoredDatasetGroup:
        deduped_ids = _dedupe_uuids(dataset_ids)
        if not deduped_ids:
            raise RuntimeError("Dataset group requires at least one dataset.")
        for dataset_id in deduped_ids:
            self.get_dataset(dataset_id)
        group_id = uuid4()
        now = _now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_groups (
                    id, user_id, name, description, dataset_ids, relationships,
                    metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(group_id),
                    self._user_id,
                    name,
                    description,
                    json.dumps([str(item) for item in deduped_ids], ensure_ascii=False),
                    "[]",
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    now,
                    now,
                ),
            )
        return self.get_dataset_group(group_id)

    def list_dataset_groups(self) -> tuple[StoredDatasetGroup, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM dataset_groups WHERE user_id = ? AND deleted_at IS NULL ORDER BY created_at DESC",
                (self._user_id,),
            ).fetchall()
        return tuple(self.get_dataset_group(UUID(str(row["id"]))) for row in rows)

    def get_dataset_group(self, group_id: UUID) -> StoredDatasetGroup:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_groups WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (str(group_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Dataset group was not found: {group_id}")
        group = _stored_dataset_group_from_row(row)
        active_ids: list[UUID] = []
        missing_ids: list[UUID] = []
        for dataset_id in group.dataset_ids:
            try:
                self.get_dataset(dataset_id)
                active_ids.append(dataset_id)
            except RuntimeError:
                missing_ids.append(dataset_id)
        if not missing_ids:
            return group
        active_text = {str(item) for item in active_ids}
        relationships = tuple(
            item
            for item in group.relationships
            if str(item.get("left_dataset_id")) in active_text
            and str(item.get("right_dataset_id")) in active_text
        )
        return StoredDatasetGroup(
            id=group.id,
            user_id=group.user_id,
            name=group.name,
            description=group.description,
            dataset_ids=tuple(active_ids),
            relationships=relationships,
            metadata=group.metadata
            | {"recycle_missing_dataset_ids": [str(item) for item in missing_ids]},
            created_at=group.created_at,
            updated_at=group.updated_at,
        )

    def dataset_group_contains_dataset(
        self,
        *,
        group_id: UUID,
        dataset_id: UUID,
        include_recycled: bool = False,
    ) -> bool:
        deleted_filter = "" if include_recycled else " AND deleted_at IS NULL"
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT dataset_ids FROM dataset_groups WHERE id=? AND user_id=?{deleted_filter}",
                (str(group_id), self._user_id),
            ).fetchone()
        if row is None:
            return False
        try:
            dataset_ids = json.loads(str(row["dataset_ids"] or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            dataset_ids = []
        return str(dataset_id) in {str(item) for item in dataset_ids}

    def update_dataset_group_relationships(
        self,
        *,
        group_id: UUID,
        relationships: tuple[dict[str, Any], ...],
    ) -> StoredDatasetGroup:
        group = self.get_dataset_group(group_id)
        valid_ids = {str(dataset_id) for dataset_id in group.dataset_ids}
        cleaned: list[dict[str, Any]] = []
        for relationship in relationships:
            left_dataset_id = str(relationship.get("left_dataset_id") or "")
            right_dataset_id = str(relationship.get("right_dataset_id") or "")
            if left_dataset_id not in valid_ids or right_dataset_id not in valid_ids:
                raise RuntimeError("Relationship datasets must belong to the dataset group.")
            if left_dataset_id == right_dataset_id:
                raise ValueError("Relationship cannot join a dataset to itself.")
            self._validate_relationship_columns(relationship)
            cleaned.append(dict(relationship))
        _validate_relationship_graph_structure(tuple(cleaned), expected_root=None)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE dataset_groups
                SET relationships = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    json.dumps(cleaned, ensure_ascii=False, default=str),
                    _now_iso(),
                    str(group_id),
                    self._user_id,
                ),
            )
        return self.get_dataset_group(group_id)

    def delete_dataset_group(
        self, group_id: UUID, *, delete_datasets: bool = False
    ) -> tuple[UUID, ...]:
        group = self.get_dataset_group(group_id)
        batch_id = uuid4()
        now, purge_after = _recycle_times()
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_groups SET deleted_at=?,purge_after=?,deleted_by_batch_id=?,updated_at=? WHERE id=? AND user_id=?",
                (now, purge_after, str(batch_id), now, str(group_id), self._user_id),
            )
        if delete_datasets:
            for dataset_id in group.dataset_ids:
                self._soft_delete_dataset(
                    dataset_id, batch_id=batch_id, now=now, purge_after=purge_after
                )
            return group.dataset_ids
        return ()

    def hard_delete_dataset_group(
        self, group_id: UUID, *, delete_datasets: bool = False
    ) -> tuple[UUID, ...]:
        group = self.get_dataset_group(group_id)
        deleted_dataset_ids: tuple[UUID, ...] = ()
        if delete_datasets:
            deleted_dataset_ids = group.dataset_ids
            self._delete_datasets_batch(group.dataset_ids, dataset_group_id=group_id)
            return deleted_dataset_ids
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM dataset_groups WHERE id = ? AND user_id = ?",
                (str(group_id), self._user_id),
            )
        return deleted_dataset_ids
