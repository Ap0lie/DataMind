from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

from app.storage.repository_utils import now_iso
from app.storage.row_mappers import optional_text, report_from_row


class ReportRepositoryMixin:
    """Chart and report persistence for a user-scoped repository."""

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
                    now_iso(),
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
            now = now_iso()
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
                    optional_text(metadata.get("question")),
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
        columns = (
            "*"
            if include_content
            else (
                "id, dataset_id, user_id, created_at, updated_at, version, title, "
                "'' AS markdown, metadata, question"
            )
        )
        filters = ["user_id = ?", "deleted_at IS NULL"]
        parameters: list[object] = [self._user_id]
        if dataset_id is not None:
            self.get_dataset(dataset_id)
            filters.insert(0, "dataset_id = ?")
            parameters.insert(0, str(dataset_id))
        needle = str(query or "").strip().lower()
        if needle:
            pattern = f"%{needle}%"
            filters.append(
                """
                (
                    LOWER(COALESCE(title, '')) LIKE ?
                    OR LOWER(COALESCE(question, '')) LIKE ?
                    OR LOWER(COALESCE(markdown, '')) LIKE ?
                    OR LOWER(COALESCE(CAST(metadata AS TEXT), '')) LIKE ?
                )
                """
            )
            parameters.extend((pattern, pattern, pattern, pattern))
        sql_limit = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(max(limit, 0))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {columns} FROM reports "
                f"WHERE {' AND '.join(filters)} ORDER BY created_at DESC{sql_limit}",
                tuple(parameters),
            ).fetchall()
        reports = tuple(report_from_row(row) for row in rows)
        if include_content:
            return reports
        summary_keys = {
            "question",
            "route",
            "workflow",
            "nodes",
            "planner_source",
            "sql_source",
            "python_source",
            "report_source",
            "validation_issue_count",
        }
        return tuple(
            {
                **report,
                "metadata": {
                    key: value
                    for key, value in report["metadata"].items()
                    if key in summary_keys
                },
            }
            for report in reports
        )

    def get_report(self, report_id: UUID) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (str(report_id), self._user_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Report was not found: {report_id}")
        return report_from_row(row)

    def update_report(
        self, *, report_id: UUID, title: str | None = None
    ) -> dict[str, Any]:
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
                (next_title, now_iso(), str(report_id), self._user_id),
            )
        return self.get_report(report_id)

    def list_report_versions(self, dataset_id: UUID) -> tuple[dict[str, Any], ...]:
        self.get_dataset(dataset_id)
        return self.list_reports(dataset_id=dataset_id, include_content=False)

    def delete_report(self, report_id: UUID) -> None:
        from app.storage.repository_utils import recycle_times

        self.get_report(report_id)
        now, purge_after = recycle_times()
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
