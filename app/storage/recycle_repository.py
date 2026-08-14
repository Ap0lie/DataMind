from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from app.storage.repository_utils import now_iso, recycle_times


class AssetRecycleRepositoryMixin:
    """Soft deletion, restoration, and retention cleanup across stored assets."""

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
            now, purge_after = recycle_times()
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
        return tuple(
            sorted(assets, key=lambda item: item["deleted_at"], reverse=True)
        )

    def purge_expired_assets(self, *, now: str | None = None) -> int:
        cutoff = now or now_iso()
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
