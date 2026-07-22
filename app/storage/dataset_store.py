from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar
from uuid import UUID, uuid4

from app.storage.auth_repository import AuthRepositoryMixin, normalize_user_id
from app.storage.models import (
    StoredAnalysisJob,
    StoredCleaningJob,
    StoredDataset,
    StoredDatasetGroup,
)
from app.storage.row_mappers import (
    analysis_job_event_from_row as _analysis_job_event_from_row,
)
from app.storage.row_mappers import (
    bounded_progress as _bounded_progress,
)
from app.storage.row_mappers import (
    cleaning_job_event_from_row as _cleaning_job_event_from_row,
)
from app.storage.row_mappers import (
    cleaning_run_from_row as _cleaning_run_from_row,
)
from app.storage.row_mappers import (
    column_metadata_from_row as _column_metadata_from_row,
)
from app.storage.row_mappers import (
    json_loads as _json_loads,
)
from app.storage.row_mappers import (
    optional_text as _optional_text,
)
from app.storage.row_mappers import (
    report_from_row as _report_from_row,
)
from app.storage.row_mappers import (
    semantic_model_from_row as _semantic_model_from_row,
)
from app.storage.row_mappers import (
    stored_analysis_job_from_row as _stored_analysis_job_from_row,
)
from app.storage.row_mappers import (
    stored_cleaning_job_from_row as _stored_cleaning_job_from_row,
)
from app.storage.row_mappers import (
    stored_dataset_from_row as _stored_dataset_from_row,
)
from app.storage.row_mappers import (
    stored_dataset_group_from_row as _stored_dataset_group_from_row,
)


class DatasetStoreRepository(AuthRepositoryMixin):
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

    def create_dataset(
        self,
        *,
        name: str,
        source_type: str,
        source_metadata: dict[str, Any],
    ) -> StoredDataset:
        dataset = StoredDataset(
            id=uuid4(),
            user_id=self._user_id,
            name=name,
            source_type=source_type,
            status="imported",
            source_metadata=source_metadata,
        )
        now = _now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    id, user_id, name, source_type, status, source_metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(dataset.id),
                    dataset.user_id,
                    dataset.name,
                    dataset.source_type,
                    dataset.status,
                    json.dumps(source_metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return StoredDataset(
            id=dataset.id,
            user_id=dataset.user_id,
            name=dataset.name,
            source_type=dataset.source_type,
            status=dataset.status,
            source_metadata=source_metadata,
            created_at=now,
            updated_at=now,
        )

    def list_datasets(self) -> tuple[StoredDataset, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets WHERE user_id = ? AND deleted_at IS NULL ORDER BY created_at DESC",
                (self._user_id,),
            ).fetchall()
        return tuple(_stored_dataset_from_row(row) for row in rows)

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

    def get_dataset(self, dataset_id: UUID) -> StoredDataset:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (str(dataset_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Dataset was not found: {dataset_id}")
        return _stored_dataset_from_row(row)

    def delete_dataset(self, dataset_id: UUID) -> None:
        self.get_dataset(dataset_id)
        now, purge_after = _recycle_times()
        self._soft_delete_dataset(dataset_id, batch_id=uuid4(), now=now, purge_after=purge_after)

    def _soft_delete_dataset(
        self, dataset_id: UUID, *, batch_id: UUID, now: str, purge_after: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE datasets SET deleted_at=?,purge_after=?,deleted_by_batch_id=?,updated_at=? WHERE id=? AND user_id=? AND deleted_at IS NULL",
                (now, purge_after, str(batch_id), now, str(dataset_id), self._user_id),
            )
            connection.execute(
                "UPDATE reports SET deleted_at=?,purge_after=?,deleted_by_batch_id=? WHERE dataset_id=? AND user_id=? AND deleted_at IS NULL",
                (now, purge_after, str(batch_id), str(dataset_id), self._user_id),
            )

    def hard_delete_dataset(self, dataset_id: UUID) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT id FROM datasets WHERE id=? AND user_id=?", (str(dataset_id), self._user_id)
            ).fetchone()
            if exists is None:
                raise RuntimeError(f"Dataset was not found: {dataset_id}")
            for table in (
                "records",
                "artifacts",
                "cleaning_runs",
                "dataset_columns",
                "charts",
                "reports",
                "analysis_jobs",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE dataset_id = ?",
                    (str(dataset_id),),
                )
            rows = connection.execute(
                "SELECT * FROM dataset_groups WHERE user_id = ?",
                (self._user_id,),
            ).fetchall()
            for row in rows:
                group = _stored_dataset_group_from_row(row)
                if dataset_id not in group.dataset_ids:
                    continue
                remaining_ids = tuple(item for item in group.dataset_ids if item != dataset_id)
                if not remaining_ids:
                    connection.execute("DELETE FROM dataset_groups WHERE id = ?", (str(group.id),))
                    continue
                remaining_relationships = tuple(
                    relationship
                    for relationship in group.relationships
                    if str(relationship.get("left_dataset_id")) != str(dataset_id)
                    and str(relationship.get("right_dataset_id")) != str(dataset_id)
                )
                connection.execute(
                    """
                    UPDATE dataset_groups
                    SET dataset_ids = ?, relationships = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps([str(item) for item in remaining_ids], ensure_ascii=False),
                        json.dumps(list(remaining_relationships), ensure_ascii=False, default=str),
                        _now_iso(),
                        str(group.id),
                    ),
                )
            connection.execute("DELETE FROM datasets WHERE id = ?", (str(dataset_id),))
        dataset_dir = self._dataset_dir(dataset_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

    def _delete_datasets_batch(
        self,
        dataset_ids: tuple[UUID, ...],
        *,
        dataset_group_id: UUID | None = None,
    ) -> None:
        dataset_ids = _dedupe_uuids(dataset_ids)
        if not dataset_ids:
            return
        for dataset_id in dataset_ids:
            self.get_dataset(dataset_id)

        dataset_id_values = [str(dataset_id) for dataset_id in dataset_ids]
        placeholders = ", ".join("?" for _ in dataset_id_values)
        deleted = set(dataset_ids)
        deleted_text = {str(dataset_id) for dataset_id in dataset_ids}
        with self._connect() as connection:
            for table in (
                "records",
                "artifacts",
                "cleaning_runs",
                "dataset_columns",
                "charts",
                "reports",
                "analysis_jobs",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE dataset_id IN ({placeholders})",
                    dataset_id_values,
                )
            if dataset_group_id is not None:
                connection.execute(
                    "DELETE FROM dataset_groups WHERE id = ? AND user_id = ?",
                    (str(dataset_group_id), self._user_id),
                )
            rows = connection.execute(
                "SELECT * FROM dataset_groups WHERE user_id = ?",
                (self._user_id,),
            ).fetchall()
            now = _now_iso()
            for row in rows:
                group = _stored_dataset_group_from_row(row)
                remaining_ids = tuple(
                    dataset_id for dataset_id in group.dataset_ids if dataset_id not in deleted
                )
                if len(remaining_ids) == len(group.dataset_ids):
                    continue
                if not remaining_ids:
                    connection.execute("DELETE FROM dataset_groups WHERE id = ?", (str(group.id),))
                    continue
                remaining_relationships = tuple(
                    relationship
                    for relationship in group.relationships
                    if str(relationship.get("left_dataset_id")) not in deleted_text
                    and str(relationship.get("right_dataset_id")) not in deleted_text
                )
                connection.execute(
                    """
                    UPDATE dataset_groups
                    SET dataset_ids = ?, relationships = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps([str(item) for item in remaining_ids], ensure_ascii=False),
                        json.dumps(list(remaining_relationships), ensure_ascii=False, default=str),
                        now,
                        str(group.id),
                    ),
                )
            connection.execute(
                f"DELETE FROM datasets WHERE user_id = ? AND id IN ({placeholders})",
                [self._user_id, *dataset_id_values],
            )

        for dataset_id in dataset_ids:
            dataset_dir = self._dataset_dir(dataset_id)
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)

    def append_raw_records(self, *, dataset_id: UUID, records: list[dict[str, Any]]) -> int:
        self._require_dataset_dir(dataset_id)
        if not records:
            return 0

        self._replace_records(dataset_id=dataset_id, source="raw", records=records)
        return len(records)

    def save_cleaned_records(
        self,
        *,
        dataset_id: UUID,
        records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        self.get_dataset(dataset_id)
        self._replace_records(dataset_id=dataset_id, source="cleaned", records=records)
        self.save_artifact(
            dataset_id=dataset_id,
            artifact_type="cleaning_metadata",
            content={
                "record_count": len(records),
                "updated_at": _now_iso(),
                "metadata": metadata or {},
            },
        )
        self._update_dataset(dataset_id, status="cleaned")
        return len(records)

    def read_raw_records(self, dataset_id: UUID) -> list[dict[str, Any]]:
        return self._read_records_file(dataset_id, "raw_records.jsonl")

    def read_cleaned_records(self, dataset_id: UUID) -> list[dict[str, Any]]:
        return self._read_records_file(dataset_id, "cleaned_records.jsonl")

    def read_analysis_records(self, dataset_id: UUID) -> list[dict[str, Any]]:
        cleaned_records = self.read_cleaned_records(dataset_id)
        return cleaned_records or self.read_raw_records(dataset_id)

    def sample_analysis_records(
        self, dataset_id: UUID, *, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Read a bounded analysis sample, preferring cleaned records when available."""
        if limit <= 0:
            return []
        cleaned_records = self._read_records_file(dataset_id, "cleaned_records.jsonl", limit=limit)
        return cleaned_records or self._read_records_file(
            dataset_id, "raw_records.jsonl", limit=limit
        )

    def sample_analysis_column_values(
        self,
        dataset_id: UUID,
        *,
        column_name: str,
        limit: int = 500,
    ) -> set[str]:
        """Return a bounded, deterministic sample of distinct scalar values.

        Relationship inference must compare the same part of a key domain on
        both sides of a join. Sampling the first N *rows* independently makes a
        valid foreign key look unrelated whenever the source files use
        different row orders. Sorting distinct values in the database keeps the
        Python result bounded while making samples comparable across datasets.
        """
        self.get_dataset(dataset_id)
        if limit <= 0 or not column_name:
            return set()
        source = (
            "cleaned"
            if self._count_records(dataset_id, source="cleaned")
            else "raw"
        )
        with self._connect() as connection:
            dialect = getattr(connection, "dialect_name", "sqlite")
            if dialect == "postgresql":
                expression = "jsonb_extract_path_text(CAST(record AS jsonb), ?)"
                path_parameter = column_name
            else:
                expression = "CAST(json_extract(record, ?) AS TEXT)"
                path_parameter = f"$.{json.dumps(column_name, ensure_ascii=False)}"
            rows = connection.execute(
                f"""
                SELECT DISTINCT lower(trim({expression})) AS value
                FROM records
                WHERE dataset_id = ? AND source = ?
                  AND {expression} IS NOT NULL
                  AND trim({expression}) <> ''
                ORDER BY value
                LIMIT ?
                """,
                (
                    path_parameter,
                    str(dataset_id),
                    source,
                    path_parameter,
                    path_parameter,
                    limit,
                ),
            ).fetchall()
        return {str(row["value"]) for row in rows if row["value"] is not None}

    def count_analysis_records(self, dataset_id: UUID) -> int:
        """Return the active record count without materialising the records in Python."""
        cleaned_count = self._count_records(dataset_id, source="cleaned")
        return cleaned_count or self._count_records(dataset_id, source="raw")

    def _read_records_file(
        self,
        dataset_id: UUID,
        file_name: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.get_dataset(dataset_id)
        source = "cleaned" if file_name.startswith("cleaned") else "raw"
        if limit is not None and limit <= 0:
            return []
        query = """
            SELECT record FROM records
            WHERE dataset_id = ? AND source = ?
            ORDER BY row_number
        """
        parameters: tuple[object, ...] = (str(dataset_id), source)
        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            record for row in rows if isinstance((record := json.loads(str(row["record"]))), dict)
        ]

    def _count_records(self, dataset_id: UUID, *, source: str) -> int:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS record_count FROM records
                WHERE dataset_id = ? AND source = ?
                """,
                (str(dataset_id), source),
            ).fetchone()
        return int(row["record_count"]) if row is not None else 0

    def _read_jsonl_records(self, dataset_id: UUID, file_name: str) -> list[dict[str, Any]]:
        dataset_dir = self._dataset_dir(dataset_id)
        path = dataset_dir / file_name
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                record = payload.get("record")
                if isinstance(record, dict):
                    records.append(record)
        return records

    def preview_raw_records(self, dataset_id: UUID, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._read_records_file(dataset_id, "raw_records.jsonl", limit=limit)

    def preview_cleaned_records(self, dataset_id: UUID, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._read_records_file(dataset_id, "cleaned_records.jsonl", limit=limit)

    def preview_analysis_records(
        self, dataset_id: UUID, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        return self.sample_analysis_records(dataset_id, limit=limit)

    def save_artifact(
        self,
        *,
        dataset_id: UUID,
        artifact_type: str,
        content: dict[str, Any],
        file_name: str | None = None,
        artifact_id: UUID | None = None,
        if_absent: bool = False,
    ) -> UUID:
        self.get_dataset(dataset_id)
        record_id = artifact_id or uuid4()
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO artifacts (
                    id, dataset_id, created_at, artifact_type, file_name, content
                )
                VALUES (?, ?, ?, ?, ?, ?)
                {"ON CONFLICT(id) DO NOTHING" if if_absent else ""}
                """,
                (
                    str(record_id),
                    str(dataset_id),
                    _now_iso(),
                    artifact_type,
                    file_name,
                    json.dumps(content, ensure_ascii=False, default=str),
                ),
            )
        return record_id

    def get_artifact(self, dataset_id: UUID, artifact_id: UUID) -> dict[str, Any]:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE id = ? AND dataset_id = ?
                """,
                (str(artifact_id), str(dataset_id)),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Artifact was not found: {artifact_id}")
        content = _json_loads(row["content"], {})
        return {
            "id": str(row["id"]),
            "dataset_id": str(row["dataset_id"]),
            "artifact_type": str(row["artifact_type"]),
            "file_name": str(row["file_name"]) if row["file_name"] else None,
            "content": content if isinstance(content, dict) else {},
            "created_at": str(row["created_at"]),
        }

    def save_cleaning_result(
        self,
        *,
        dataset_id: UUID,
        provider: str,
        model: str,
        prompt: str,
        result_markdown: str,
        cleaned_dataset: dict[str, Any],
        raw_summary: dict[str, Any] | None = None,
        previous_summary: dict[str, Any] | None = None,
        current_summary: dict[str, Any] | None = None,
        diff_summary: dict[str, Any] | None = None,
        activate: bool = True,
        job_id: UUID | None = None,
    ) -> UUID:
        self.get_dataset(dataset_id)
        record_id = uuid4()
        with self._connect() as connection:
            if job_id is not None:
                existing = connection.execute(
                    "SELECT id FROM cleaning_runs WHERE job_id = ? AND user_id = ?",
                    (str(job_id), self._user_id),
                ).fetchone()
                if existing is not None:
                    return UUID(str(existing["id"]))
            version_row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM cleaning_runs
                WHERE dataset_id = ? AND user_id = ?
                """,
                (str(dataset_id), self._user_id),
            ).fetchone()
            version = int(version_row["next_version"] if version_row else 1)
            if activate:
                connection.execute(
                    """
                    UPDATE cleaning_runs
                    SET is_active = 0
                    WHERE dataset_id = ? AND user_id = ?
                    """,
                    (str(dataset_id), self._user_id),
                )
            connection.execute(
                """
                INSERT INTO cleaning_runs (
                    id, dataset_id, user_id, version, is_active, created_at, provider, model,
                    prompt, result_markdown, cleaned_dataset, raw_summary, previous_summary,
                    current_summary, diff_summary, job_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record_id),
                    str(dataset_id),
                    self._user_id,
                    version,
                    1 if activate else 0,
                    _now_iso(),
                    provider,
                    model,
                    prompt,
                    result_markdown,
                    json.dumps(cleaned_dataset, ensure_ascii=False, default=str),
                    json.dumps(raw_summary or {}, ensure_ascii=False, default=str),
                    json.dumps(previous_summary or {}, ensure_ascii=False, default=str),
                    json.dumps(current_summary or {}, ensure_ascii=False, default=str),
                    json.dumps(diff_summary or {}, ensure_ascii=False, default=str),
                    str(job_id) if job_id else None,
                ),
            )
        return record_id

    def list_cleaning_runs(self, dataset_id: UUID) -> tuple[dict[str, Any], ...]:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cleaning_runs
                WHERE dataset_id = ? AND user_id = ?
                ORDER BY version DESC, created_at DESC
                """,
                (str(dataset_id), self._user_id),
            ).fetchall()
        runs = [_cleaning_run_from_row(row) for row in rows]
        return tuple(runs)

    def get_cleaning_run(self, dataset_id: UUID, run_id: UUID) -> dict[str, Any]:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM cleaning_runs
                WHERE id = ? AND dataset_id = ? AND user_id = ?
                """,
                (str(run_id), str(dataset_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Cleaning run was not found: {run_id}")
        return _cleaning_run_from_row(row)

    def activate_cleaning_run(self, *, dataset_id: UUID, run_id: UUID) -> dict[str, Any]:
        run = self.get_cleaning_run(dataset_id, run_id)
        cleaned_dataset = run.get("cleaned_dataset")
        records = cleaned_dataset.get("records") if isinstance(cleaned_dataset, dict) else None
        if not isinstance(records, list):
            raise RuntimeError("Cleaning run does not contain restorable records.")
        clean_records = [record for record in records if isinstance(record, dict)]
        self._replace_records(dataset_id=dataset_id, source="cleaned", records=clean_records)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cleaning_runs
                SET is_active = 0
                WHERE dataset_id = ? AND user_id = ?
                """,
                (str(dataset_id), self._user_id),
            )
            connection.execute(
                """
                UPDATE cleaning_runs
                SET is_active = 1
                WHERE id = ? AND dataset_id = ? AND user_id = ?
                """,
                (str(run_id), str(dataset_id), self._user_id),
            )
        self._update_dataset(dataset_id, status="cleaned")
        return self.get_cleaning_run(dataset_id, run_id)

    def list_column_metadata(self, dataset_id: UUID) -> tuple[dict[str, Any], ...]:
        self.get_dataset(dataset_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dataset_columns
                WHERE dataset_id = ? AND user_id = ?
                ORDER BY column_name
                """,
                (str(dataset_id), self._user_id),
            ).fetchall()
        return tuple(_column_metadata_from_row(row) for row in rows)

    def save_column_metadata(
        self,
        *,
        dataset_id: UUID,
        columns: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        self.get_dataset(dataset_id)
        now = _now_iso()
        with self._connect() as connection:
            for column in columns:
                column_name = str(column.get("column_name") or "").strip()
                if not column_name:
                    continue
                existing = connection.execute(
                    """
                    SELECT created_at FROM dataset_columns
                    WHERE dataset_id = ? AND user_id = ? AND column_name = ?
                    """,
                    (str(dataset_id), self._user_id, column_name),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO dataset_columns (
                        dataset_id, user_id, column_name, inferred_type, override_type,
                        role, description, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (dataset_id, user_id, column_name) DO UPDATE SET
                        inferred_type = excluded.inferred_type,
                        override_type = excluded.override_type,
                        role = excluded.role,
                        description = excluded.description,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(dataset_id),
                        self._user_id,
                        column_name,
                        str(column.get("inferred_type") or "text"),
                        _optional_text(column.get("override_type")),
                        str(column.get("role") or "dimension"),
                        str(column.get("description") or ""),
                        str(existing["created_at"]) if existing else now,
                        now,
                    ),
                )
        return self.list_column_metadata(dataset_id)

    def update_column_metadata(
        self,
        *,
        dataset_id: UUID,
        column_name: str,
        inferred_type: str | None = None,
        override_type: str | None = None,
        role: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        self.get_dataset(dataset_id)
        now = _now_iso()
        current = {item["column_name"]: item for item in self.list_column_metadata(dataset_id)}
        existing = current.get(column_name, {})
        payload = {
            "column_name": column_name,
            "inferred_type": inferred_type
            if inferred_type is not None
            else existing.get("inferred_type", "text"),
            "override_type": override_type
            if override_type is not None
            else existing.get("override_type"),
            "role": role if role is not None else existing.get("role", "dimension"),
            "description": description
            if description is not None
            else existing.get("description", ""),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_columns (
                    dataset_id, user_id, column_name, inferred_type, override_type,
                    role, description, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, user_id, column_name) DO UPDATE SET
                    inferred_type = excluded.inferred_type,
                    override_type = excluded.override_type,
                    role = excluded.role,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """,
                (
                    str(dataset_id),
                    self._user_id,
                    column_name,
                    str(payload["inferred_type"] or "text"),
                    _optional_text(payload["override_type"]),
                    str(payload["role"] or "dimension"),
                    str(payload["description"] or ""),
                    str(existing.get("created_at") or now),
                    now,
                ),
            )
        return {item["column_name"]: item for item in self.list_column_metadata(dataset_id)}[
            column_name
        ]

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

    def save_chart(
        self,
        *,
        dataset_id: UUID,
        title: str,
        chart_type: str,
        chart_spec: dict[str, Any],
        chart_data: list[dict[str, Any]],
    ) -> UUID:
        self.get_dataset(dataset_id)
        record_id = uuid4()
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
                    _now_iso(),
                    title,
                    chart_type,
                    json.dumps(chart_spec, ensure_ascii=False, default=str),
                    json.dumps(chart_data, ensure_ascii=False, default=str),
                ),
            )
        return record_id

    def save_report(
        self,
        *,
        dataset_id: UUID,
        title: str,
        markdown: str,
        metadata: dict[str, Any],
        job_id: UUID | None = None,
    ) -> UUID:
        self.get_dataset(dataset_id)
        record_id = uuid4()
        with self._connect() as connection:
            if job_id is not None:
                existing = connection.execute(
                    "SELECT id FROM reports WHERE job_id = ? AND user_id = ?",
                    (str(job_id), self._user_id),
                ).fetchone()
                if existing is not None:
                    return UUID(str(existing["id"]))
            version_row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM reports
                WHERE dataset_id = ? AND user_id = ?
                """,
                (str(dataset_id), self._user_id),
            ).fetchone()
            version = int(version_row["next_version"] if version_row else 1)
            now = _now_iso()
            connection.execute(
                """
                INSERT INTO reports (
                    id, dataset_id, user_id, created_at, updated_at, version,
                    title, markdown, metadata, question, job_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record_id),
                    str(dataset_id),
                    self._user_id,
                    now,
                    now,
                    version,
                    title,
                    markdown,
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    _optional_text(metadata.get("question")),
                    str(job_id) if job_id else None,
                ),
            )
        return record_id

    def list_reports(
        self,
        dataset_id: UUID | None = None,
        *,
        query: str | None = None,
        limit: int | None = None,
        include_content: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        needs_content = include_content or bool(query)
        columns = (
            "*"
            if needs_content
            else (
                "id, dataset_id, user_id, created_at, updated_at, version, title, "
                "'' AS markdown, '{}' AS metadata, question"
            )
        )
        sql_limit = " LIMIT ?" if limit is not None and not query else ""
        with self._connect() as connection:
            if dataset_id is None:
                parameters: tuple[object, ...] = (self._user_id,)
                if sql_limit:
                    parameters += (max(limit or 0, 0),)
                rows = connection.execute(
                    f"SELECT {columns} FROM reports "
                    f"WHERE user_id = ? AND deleted_at IS NULL ORDER BY created_at DESC{sql_limit}",
                    parameters,
                ).fetchall()
            else:
                self.get_dataset(dataset_id)
                parameters = (str(dataset_id), self._user_id)
                if sql_limit:
                    parameters += (max(limit or 0, 0),)
                rows = connection.execute(
                    f"SELECT {columns} FROM reports "
                    f"WHERE dataset_id = ? AND user_id = ? AND deleted_at IS NULL "
                    f"ORDER BY created_at DESC{sql_limit}",
                    parameters,
                ).fetchall()
        reports = [_report_from_row(row) for row in rows]
        if query:
            needle = query.lower().strip()
            reports = [
                report
                for report in reports
                if needle in str(report.get("title", "")).lower()
                or needle in str(report.get("markdown", "")).lower()
                or needle in json.dumps(report.get("metadata", {}), ensure_ascii=False).lower()
            ]
        if limit is not None:
            reports = reports[: max(limit, 0)]
        return tuple(reports)

    def get_report(self, report_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (str(report_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Report was not found: {report_id}")
        return _report_from_row(row)

    def update_report(self, *, report_id: UUID, title: str | None = None) -> dict[str, Any]:
        existing = self.get_report(report_id)
        next_title = title.strip() if title is not None else str(existing.get("title") or "")
        if not next_title:
            raise RuntimeError("Report title cannot be empty.")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE reports
                SET title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (next_title, _now_iso(), str(report_id), self._user_id),
            )
        return self.get_report(report_id)

    def list_report_versions(self, dataset_id: UUID) -> tuple[dict[str, Any], ...]:
        self.get_dataset(dataset_id)
        return self.list_reports(dataset_id=dataset_id, include_content=False)

    def delete_report(self, report_id: UUID) -> None:
        self.get_report(report_id)
        now, purge_after = _recycle_times()
        with self._connect() as connection:
            connection.execute(
                "UPDATE reports SET deleted_at=?,purge_after=?,deleted_by_batch_id=? WHERE id=? AND user_id=?",
                (now, purge_after, str(uuid4()), str(report_id), self._user_id),
            )

    def hard_delete_report(self, report_id: UUID) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM reports WHERE id = ? AND user_id = ?",
                (str(report_id), self._user_id),
            )
        if cursor.rowcount == 0:
            raise RuntimeError(f"Report was not found: {report_id}")

    def soft_delete_asset(self, *, asset_type: str, asset_id: UUID) -> dict[str, Any]:
        if asset_type == "dataset":
            name = self.get_dataset(asset_id).name
            self.delete_dataset(asset_id)
        elif asset_type == "dataset_group":
            name = self.get_dataset_group(asset_id).name
            self.delete_dataset_group(asset_id)
        elif asset_type == "report":
            name = str(self.get_report(asset_id).get("title") or "Report")
            self.delete_report(asset_id)
        elif asset_type == "semantic_model":
            model = self.get_semantic_model(asset_id)
            if str(model.get("status")) != "draft":
                raise ValueError("Only unpublished semantic drafts can be recycled.")
            name = str(model.get("name") or "Semantic model")
            now, purge_after = _recycle_times()
            with self._connect() as connection:
                connection.execute(
                    "UPDATE semantic_models SET deleted_at=?,purge_after=?,deleted_by_batch_id=? WHERE id=? AND user_id=?",
                    (now, purge_after, str(uuid4()), str(asset_id), self._user_id),
                )
        else:
            raise ValueError("Unsupported recycled asset type.")
        return next(
            item
            for item in self.list_recycled_assets()
            if item["asset_type"] == asset_type and item["asset_id"] == asset_id
        ) | {"name": name}

    def restore_asset(self, *, asset_type: str, asset_id: UUID) -> dict[str, Any]:
        table = {
            "dataset": "datasets",
            "dataset_group": "dataset_groups",
            "report": "reports",
            "semantic_model": "semantic_models",
        }.get(asset_type)
        if table is None:
            raise ValueError("Unsupported recycled asset type.")
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE id=? AND user_id=? AND deleted_at IS NOT NULL",
                (str(asset_id), self._user_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("Recycled asset was not found.")
            batch_id = row["deleted_by_batch_id"]
            connection.execute(
                f"UPDATE {table} SET deleted_at=NULL,purge_after=NULL,deleted_by_batch_id=NULL WHERE id=? AND user_id=?",
                (str(asset_id), self._user_id),
            )
            if asset_type == "dataset" and batch_id:
                connection.execute(
                    "UPDATE reports SET deleted_at=NULL,purge_after=NULL,deleted_by_batch_id=NULL WHERE dataset_id=? AND user_id=? AND deleted_by_batch_id=?",
                    (str(asset_id), self._user_id, batch_id),
                )
        return {"asset_type": asset_type, "asset_id": asset_id, "restored": True}

    def list_recycled_assets(self) -> tuple[dict[str, Any], ...]:
        specs = (
            ("dataset", "datasets", "name"),
            ("dataset_group", "dataset_groups", "name"),
            ("report", "reports", "title"),
            ("semantic_model", "semantic_models", "name"),
        )
        assets: list[dict[str, Any]] = []
        with self._connect() as connection:
            for asset_type, table, name_column in specs:
                rows = connection.execute(
                    f"SELECT id,{name_column} name,deleted_at,purge_after FROM {table} WHERE user_id=? AND deleted_at IS NOT NULL",
                    (self._user_id,),
                ).fetchall()
                assets.extend(
                    {
                        "asset_type": asset_type,
                        "asset_id": UUID(str(row["id"])),
                        "name": str(row["name"]),
                        "deleted_at": str(row["deleted_at"]),
                        "purge_after": str(row["purge_after"]),
                    }
                    for row in rows
                )
        return tuple(sorted(assets, key=lambda item: item["deleted_at"], reverse=True))

    def purge_expired_assets(self, *, now: str | None = None) -> int:
        cutoff = now or _now_iso()
        purged = 0
        with self._connect() as connection:
            reports = connection.execute(
                "SELECT id FROM reports WHERE user_id=? AND purge_after IS NOT NULL AND purge_after<=?",
                (self._user_id, cutoff),
            ).fetchall()
            semantics = connection.execute(
                "SELECT id FROM semantic_models WHERE user_id=? AND purge_after IS NOT NULL AND purge_after<=?",
                (self._user_id, cutoff),
            ).fetchall()
            groups = connection.execute(
                "SELECT id FROM dataset_groups WHERE user_id=? AND purge_after IS NOT NULL AND purge_after<=?",
                (self._user_id, cutoff),
            ).fetchall()
            datasets = connection.execute(
                "SELECT id FROM datasets WHERE user_id=? AND purge_after IS NOT NULL AND purge_after<=?",
                (self._user_id, cutoff),
            ).fetchall()
        for row in reports:
            self.hard_delete_report(UUID(str(row["id"])))
            purged += 1
        with self._connect() as connection:
            for row in semantics:
                connection.execute(
                    "DELETE FROM semantic_models WHERE id=? AND user_id=?",
                    (str(row["id"]), self._user_id),
                )
                purged += 1
            for row in groups:
                connection.execute(
                    "DELETE FROM dataset_groups WHERE id=? AND user_id=?",
                    (str(row["id"]), self._user_id),
                )
                purged += 1
        for row in datasets:
            self.hard_delete_dataset(UUID(str(row["id"])))
            purged += 1
        return purged

    def list_asset_user_ids(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT user_id FROM datasets
                   UNION SELECT user_id FROM dataset_groups
                   UNION SELECT user_id FROM reports
                   UNION SELECT user_id FROM semantic_models"""
            ).fetchall()
        return tuple(sorted({str(row["user_id"]) for row in rows if row["user_id"]}))

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
                CREATE INDEX IF NOT EXISTS idx_planner_decisions_user ON planner_decisions(user_id, created_at);
                """
            )
            _ensure_column(connection, "datasets", "user_id", "TEXT NOT NULL DEFAULT 'default'")
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



def _dedupe_uuids(values: tuple[UUID, ...]) -> tuple[UUID, ...]:
    seen: set[UUID] = set()
    result: list[UUID] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


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




def _validate_relationship_graph_structure(
    relationships: tuple[dict[str, Any], ...],
    *,
    expected_root: UUID | None,
) -> UUID | None:
    enabled = tuple(item for item in relationships if item.get("enabled") is not False)
    if not enabled:
        return expected_root
    edges: list[tuple[UUID, UUID]] = []
    seen_edges: set[tuple[UUID, UUID]] = set()
    right_ids: set[UUID] = set()
    for relationship in enabled:
        left_id = UUID(str(relationship.get("left_dataset_id") or ""))
        right_id = UUID(str(relationship.get("right_dataset_id") or ""))
        edge = (left_id, right_id)
        if left_id == right_id:
            raise ValueError("Relationship cannot join a dataset to itself.")
        if edge in seen_edges:
            raise ValueError("Relationship plan contains a duplicate edge.")
        if right_id in right_ids:
            raise ValueError("Each joined dataset can have only one parent relationship.")
        seen_edges.add(edge)
        right_ids.add(right_id)
        edges.append(edge)

    roots = {left_id for left_id, _ in edges if left_id not in right_ids}
    if len(roots) != 1:
        raise ValueError("Relationship plan must contain exactly one acyclic root dataset.")
    root = next(iter(roots))
    if expected_root is not None and root != expected_root:
        raise ValueError("Relationship plan root must match the primary dataset.")

    connected = {root}
    remaining = list(edges)
    while remaining:
        progressed = False
        for edge in tuple(remaining):
            left_id, right_id = edge
            if left_id not in connected:
                continue
            if right_id in connected:
                raise ValueError("Relationship plan must not contain cycles or redundant paths.")
            connected.add(right_id)
            remaining.remove(edge)
            progressed = True
        if not progressed:
            raise ValueError("Every relationship must be reachable from the root dataset.")
    return root


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _recycle_times() -> tuple[str, str]:
    from app.core.settings import get_settings

    now = datetime.now(UTC)
    return now.isoformat(), (
        now + timedelta(days=get_settings().assistant_recycle_retention_days)
    ).isoformat()


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
