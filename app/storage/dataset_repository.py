from __future__ import annotations

import json
import logging
import shutil
from typing import Any
from uuid import UUID, uuid4

from app.storage.models import StoredDataset
from app.storage.repository_utils import dedupe_uuids as _dedupe_uuids
from app.storage.repository_utils import now_iso as _now_iso
from app.storage.repository_utils import recycle_times as _recycle_times
from app.storage.row_mappers import cleaning_run_from_row as _cleaning_run_from_row
from app.storage.row_mappers import column_metadata_from_row as _column_metadata_from_row
from app.storage.row_mappers import json_loads as _json_loads
from app.storage.row_mappers import optional_text as _optional_text
from app.storage.row_mappers import stored_dataset_from_row as _stored_dataset_from_row
from app.storage.row_mappers import (
    stored_dataset_group_from_row as _stored_dataset_group_from_row,
)

logger = logging.getLogger(__name__)


class DatasetRepositoryMixin:
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
        if not records:
            return 0

        return self.replace_raw_record_batches(dataset_id=dataset_id, batches=iter((records,)))

    def replace_raw_record_batches(
        self,
        *,
        dataset_id: UUID,
        batches: Any,
        preview: list[dict[str, Any]] | None = None,
        preview_limit: int = 50,
    ) -> int:
        """Replace raw records transactionally while retaining one bounded batch."""
        self._require_dataset_dir(dataset_id)
        inserted = 0
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM records WHERE dataset_id = ? AND source = ?",
                (str(dataset_id), "raw"),
            )
            for batch in batches:
                if not batch:
                    continue
                rows = []
                for record in batch:
                    if not isinstance(record, dict):
                        raise ValueError("Imported records must be JSON objects.")
                    inserted += 1
                    rows.append(
                        (
                            str(dataset_id),
                            "raw",
                            inserted,
                            json.dumps(record, ensure_ascii=False, default=str),
                        )
                    )
                    if preview is not None and len(preview) < preview_limit:
                        preview.append(record)
                connection.executemany(
                    """
                    INSERT INTO records (dataset_id, source, row_number, record)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
        if inserted:
            self._monitor_dataset_drift(dataset_id)
        return inserted

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
        self._monitor_dataset_drift(dataset_id)
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

    def analysis_column_cardinality(
        self,
        dataset_id: UUID,
        *,
        column_name: str,
    ) -> tuple[int, int]:
        """Return non-empty value and distinct counts for an active scalar column."""
        self.get_dataset(dataset_id)
        if not column_name:
            return 0, 0
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
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS value_count,
                       COUNT(DISTINCT lower(trim({expression}))) AS distinct_count
                FROM records
                WHERE dataset_id = ? AND source = ?
                  AND {expression} IS NOT NULL
                  AND trim({expression}) <> ''
                """,
                (
                    path_parameter,
                    str(dataset_id),
                    source,
                    path_parameter,
                    path_parameter,
                ),
            ).fetchone()
        if row is None:
            return 0, 0
        return int(row["value_count"] or 0), int(row["distinct_count"] or 0)

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
        self._monitor_dataset_drift(dataset_id)
        return self.get_cleaning_run(dataset_id, run_id)

    def _monitor_dataset_drift(self, dataset_id: UUID) -> None:
        try:
            from app.data_reliability import DataDriftService

            DataDriftService(self).scan_dataset(dataset_id)
        except Exception:
            logger.warning(
                "Dataset drift monitoring failed for %s.",
                dataset_id,
                exc_info=True,
            )

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
