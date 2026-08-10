from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.schemas.common import ApiModel

DriftStatus = Literal["baseline", "stable", "warning", "critical"]


class ColumnSnapshotResponse(ApiModel):
    name: str
    dtype: str
    missing_rate: float = Field(ge=0, le=1)
    unique_rate: float = Field(ge=0, le=1)
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    quantiles: tuple[float, ...] = ()
    value_signature: tuple[str, ...] = ()


class DatasetSnapshotResponse(ApiModel):
    snapshot_id: UUID
    dataset_id: UUID
    source: str
    row_count: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    fingerprint: str
    columns: tuple[ColumnSnapshotResponse, ...] = ()
    created_at: str


class DriftChangeResponse(ApiModel):
    change_type: str
    severity: Literal["info", "warning", "critical"]
    field: str | None = None
    previous_field: str | None = None
    current_field: str | None = None
    previous_value: Any = None
    current_value: Any = None
    score: float | None = None
    message: str


class DriftAffectedAssetResponse(ApiModel):
    asset_type: Literal["semantic_model", "relationship", "report"]
    asset_id: str
    status: Literal["stale"]
    reason: str


class DriftRecommendedActionResponse(ApiModel):
    action: Literal[
        "review_schema",
        "refresh_relationships",
        "run_cleaning",
        "rerun_analysis",
        "create_semantic_draft",
    ]
    label: str
    reason: str
    requires_authorization: bool = True


class DatasetDriftResponse(ApiModel):
    dataset_id: UUID
    status: DriftStatus
    snapshot: DatasetSnapshotResponse
    baseline_snapshot_id: UUID | None = None
    event_id: UUID | None = None
    changes: tuple[DriftChangeResponse, ...] = ()
    affected_assets: tuple[DriftAffectedAssetResponse, ...] = ()
    recommended_actions: tuple[DriftRecommendedActionResponse, ...] = ()
    scanned_at: str


class DatasetDriftHistoryResponse(ApiModel):
    events: tuple[DatasetDriftResponse, ...] = ()


class DatasetGroupDriftResponse(ApiModel):
    group_id: UUID
    status: DriftStatus
    datasets: tuple[DatasetDriftResponse, ...] = ()
    stale_relationship_count: int = Field(default=0, ge=0)
    scanned_at: str
