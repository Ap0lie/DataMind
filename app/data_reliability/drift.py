from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from statistics import fmean, pstdev
from typing import Any
from uuid import UUID, uuid4

from app.schemas.data_reliability import (
    ColumnSnapshotResponse,
    DatasetDriftHistoryResponse,
    DatasetDriftResponse,
    DatasetGroupDriftResponse,
    DatasetSnapshotResponse,
    DriftAffectedAssetResponse,
    DriftChangeResponse,
    DriftRecommendedActionResponse,
)

SAMPLE_LIMIT = 2_000
MISSING_RATE_THRESHOLD = 0.10
UNIQUE_RATE_THRESHOLD = 0.20
RELATIONSHIP_DROP_THRESHOLD = 0.20
RELATIONSHIP_MIN_MATCH_RATE = 0.50


class DataDriftService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def scan_dataset(self, dataset_id: UUID) -> DatasetDriftResponse:
        self.repository.get_dataset(dataset_id)
        snapshot_payload = _build_snapshot(self.repository, dataset_id)
        previous = self.repository.latest_data_snapshot(
            dataset_id,
            source=str(snapshot_payload["source"]),
        )
        if previous and previous["fingerprint"] == snapshot_payload["fingerprint"]:
            return _response_from_event(
                dataset_id=dataset_id,
                snapshot=previous,
                event=_latest_event_for_snapshot(
                    self.repository,
                    dataset_id=dataset_id,
                    snapshot=previous,
                ),
            )

        snapshot = self.repository.save_data_snapshot(snapshot_payload)
        if previous is None:
            return DatasetDriftResponse(
                dataset_id=dataset_id,
                status="baseline",
                snapshot=_snapshot_response(snapshot),
                scanned_at=str(snapshot["created_at"]),
            )

        changes = compare_snapshots(previous["profile"], snapshot["profile"])
        status = _status_for_changes(changes)
        group_ids = tuple(
            group.id
            for group in self.repository.list_dataset_groups()
            if dataset_id in group.dataset_ids
        )
        event_id = uuid4()
        affected = self._invalidate_assets(
            dataset_id=dataset_id,
            group_ids=group_ids,
            event_id=event_id,
            changes=changes,
            status=status,
        )
        affected.extend(
            self.refresh_relationships(
                dataset_id=dataset_id,
                event_id=event_id,
            )
        )
        recommendations = _recommended_actions(changes, affected)
        event = self.repository.save_data_drift_event(
            {
                "id": event_id,
                "dataset_id": dataset_id,
                "baseline_snapshot_id": previous["id"],
                "current_snapshot_id": snapshot["id"],
                "status": status,
                "changes": [item.model_dump(mode="json") for item in changes],
                "affected_assets": [
                    item.model_dump(mode="json") for item in affected
                ],
                "recommended_actions": [
                    item.model_dump(mode="json") for item in recommendations
                ],
            }
        )
        return _response_from_event(
            dataset_id=dataset_id,
            snapshot=snapshot,
            event=event,
        )

    def latest_dataset_status(self, dataset_id: UUID) -> DatasetDriftResponse:
        snapshot = self.repository.latest_data_snapshot(
            dataset_id,
            source=_active_source(self.repository, dataset_id),
        )
        if snapshot is None:
            return self.scan_dataset(dataset_id)
        return _response_from_event(
            dataset_id=dataset_id,
            snapshot=snapshot,
            event=_latest_event_for_snapshot(
                self.repository,
                dataset_id=dataset_id,
                snapshot=snapshot,
            ),
        )

    def dataset_history(
        self,
        dataset_id: UUID,
        *,
        limit: int = 50,
    ) -> DatasetDriftHistoryResponse:
        events = self.repository.list_data_drift_events(
            dataset_id=dataset_id,
            limit=limit,
        )
        return DatasetDriftHistoryResponse(
            events=tuple(
                _response_from_event(
                    dataset_id=dataset_id,
                    snapshot=self.repository.get_data_snapshot(
                        UUID(str(event["current_snapshot_id"]))
                    ),
                    event=event,
                )
                for event in events
            )
        )

    def scan_group(self, group_id: UUID) -> DatasetGroupDriftResponse:
        group = self.repository.get_dataset_group(group_id)
        datasets = tuple(self.scan_dataset(dataset_id) for dataset_id in group.dataset_ids)
        self.refresh_group_relationships(group_id)
        refreshed = self.repository.get_dataset_group(group_id)
        stale_count = sum(
            str(item.get("freshness_status") or "fresh") != "fresh"
            for item in refreshed.relationships
            if item.get("enabled", True)
        )
        status = _highest_status(
            tuple(item.status for item in datasets)
            + (("critical",) if stale_count else ())
        )
        return DatasetGroupDriftResponse(
            group_id=group_id,
            status=status,
            datasets=datasets,
            stale_relationship_count=stale_count,
            scanned_at=_now_iso(),
        )

    def latest_group_status(self, group_id: UUID) -> DatasetGroupDriftResponse:
        group = self.repository.get_dataset_group(group_id)
        datasets = tuple(
            self.latest_dataset_status(dataset_id) for dataset_id in group.dataset_ids
        )
        stale_count = sum(
            str(item.get("freshness_status") or "fresh") != "fresh"
            for item in group.relationships
            if item.get("enabled", True)
        )
        return DatasetGroupDriftResponse(
            group_id=group_id,
            status=_highest_status(
                tuple(item.status for item in datasets)
                + (("critical",) if stale_count else ())
            ),
            datasets=datasets,
            stale_relationship_count=stale_count,
            scanned_at=_now_iso(),
        )

    def refresh_group_relationships(self, group_id: UUID) -> tuple[dict[str, Any], ...]:
        group = self.repository.get_dataset_group(group_id)
        relationships = [
            self._relationship_status(dict(item))
            for item in group.relationships
        ]
        self.repository.replace_dataset_group_relationship_states(
            group_id=group_id,
            relationships=tuple(relationships),
        )
        return tuple(relationships)

    def refresh_relationships(
        self,
        *,
        dataset_id: UUID,
        event_id: UUID,
    ) -> list[DriftAffectedAssetResponse]:
        affected: list[DriftAffectedAssetResponse] = []
        for group in self.repository.list_dataset_groups():
            if dataset_id not in group.dataset_ids:
                continue
            relationships: list[dict[str, Any]] = []
            for source in group.relationships:
                relationship = self._relationship_status(dict(source))
                if relationship.get("freshness_status") == "stale":
                    relationship["drift_event_id"] = str(event_id)
                relationships.append(relationship)
            self.repository.replace_dataset_group_relationship_states(
                group_id=group.id,
                relationships=tuple(relationships),
            )
            for item in relationships:
                if item.get("freshness_status") != "stale":
                    continue
                affected.append(
                    DriftAffectedAssetResponse(
                        asset_type="relationship",
                        asset_id=str(item["relationship_id"]),
                        status="stale",
                        reason=str(item.get("stale_reason") or "检测到数据关系漂移。"),
                    )
                )
        return affected

    def _relationship_status(self, relationship: dict[str, Any]) -> dict[str, Any]:
        relationship["relationship_id"] = _relationship_id(relationship)
        left_id = UUID(str(relationship["left_dataset_id"]))
        right_id = UUID(str(relationship["right_dataset_id"]))
        left_column = str(relationship["left_column"])
        right_column = str(relationship["right_column"])
        left_columns = _dataset_columns(self.repository, left_id)
        right_columns = _dataset_columns(self.repository, right_id)
        if left_column not in left_columns or right_column not in right_columns:
            relationship.update(
                {
                    "freshness_status": "stale",
                    "stale_reason": "关联键已不存在于当前数据集字段中。",
                    "last_validated_at": _now_iso(),
                }
            )
            return relationship
        left_values = self.repository.sample_analysis_column_values(
            left_id,
            column_name=left_column,
            limit=500,
        )
        right_values = self.repository.sample_analysis_column_values(
            right_id,
            column_name=right_column,
            limit=500,
        )
        left_values = _expand_values(
            left_values,
            str(relationship.get("left_value_mode") or "scalar"),
            relationship.get("left_delimiter"),
        )
        right_values = _expand_values(
            right_values,
            str(relationship.get("right_value_mode") or "scalar"),
            relationship.get("right_delimiter"),
        )
        match_rate = (
            len(left_values & right_values) / len(left_values) if left_values else 0.0
        )
        baseline = relationship.get("baseline_match_rate")
        baseline_rate = float(baseline) if baseline is not None else match_rate
        drop = max(0.0, baseline_rate - match_rate)
        stale = (
            match_rate < RELATIONSHIP_MIN_MATCH_RATE
            or drop >= RELATIONSHIP_DROP_THRESHOLD
        )
        relationship.update(
            {
                "baseline_match_rate": round(baseline_rate, 4),
                "last_match_rate": round(match_rate, 4),
                "match_rate_drift": round(drop, 4),
                "freshness_status": "stale" if stale else "fresh",
                "stale_reason": (
                    f"关系样本匹配率从 {baseline_rate:.0%} 降至 {match_rate:.0%}。"
                    if stale
                    else ""
                ),
                "last_validated_at": _now_iso(),
            }
        )
        return relationship

    def _invalidate_assets(
        self,
        *,
        dataset_id: UUID,
        group_ids: tuple[UUID, ...],
        event_id: UUID,
        changes: tuple[DriftChangeResponse, ...],
        status: str,
    ) -> list[DriftAffectedAssetResponse]:
        if status == "stable":
            return []
        reason = "; ".join(item.message for item in changes[:4])
        affected: list[DriftAffectedAssetResponse] = []
        schema_changed = any(
            item.change_type
            in {"column_removed", "column_renamed", "type_changed"}
            for item in changes
        )
        if schema_changed:
            for model_id in self.repository.mark_semantic_models_stale(
                dataset_id=dataset_id,
                group_ids=group_ids,
                drift_event_id=event_id,
                reason=reason,
            ):
                affected.append(
                    DriftAffectedAssetResponse(
                        asset_type="semantic_model",
                        asset_id=str(model_id),
                        status="stale",
                        reason=reason,
                    )
                )
        for report_id in self.repository.mark_reports_stale(
            dataset_id=dataset_id,
            group_ids=group_ids,
            drift_event_id=event_id,
            reason=reason,
        ):
            affected.append(
                DriftAffectedAssetResponse(
                    asset_type="report",
                    asset_id=str(report_id),
                    status="stale",
                    reason=reason,
                )
            )
        return affected


def compare_snapshots(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> tuple[DriftChangeResponse, ...]:
    previous_columns = {
        str(item["name"]): item for item in previous.get("columns") or ()
    }
    current_columns = {
        str(item["name"]): item for item in current.get("columns") or ()
    }
    removed = set(previous_columns) - set(current_columns)
    added = set(current_columns) - set(previous_columns)
    renames = _rename_candidates(
        removed,
        added,
        previous_columns,
        current_columns,
    )
    changes: list[DriftChangeResponse] = []
    for old_name, new_name, score in renames:
        removed.discard(old_name)
        added.discard(new_name)
        changes.append(
            DriftChangeResponse(
                change_type="column_renamed",
                severity="critical",
                field=new_name,
                previous_field=old_name,
                current_field=new_name,
                score=score,
                message=f"字段 {old_name} 可能已重命名为 {new_name}。",
            )
        )
    changes.extend(
        DriftChangeResponse(
            change_type="column_removed",
            severity="critical",
            field=name,
            previous_field=name,
            message=f"字段 {name} 已被删除。",
        )
        for name in sorted(removed)
    )
    changes.extend(
        DriftChangeResponse(
            change_type="column_added",
            severity="warning",
            field=name,
            current_field=name,
            message=f"新增字段 {name}。",
        )
        for name in sorted(added)
    )
    for name in sorted(set(previous_columns) & set(current_columns)):
        changes.extend(
            _column_changes(name, previous_columns[name], current_columns[name])
        )
    previous_rows = int(previous.get("row_count") or 0)
    current_rows = int(current.get("row_count") or 0)
    row_delta = abs(current_rows - previous_rows) / max(previous_rows, 1)
    if row_delta >= 0.25:
        changes.append(
            DriftChangeResponse(
                change_type="row_count_drift",
                severity="critical" if row_delta >= 0.75 else "warning",
                previous_value=previous_rows,
                current_value=current_rows,
                score=round(row_delta, 4),
                message=f"数据行数从 {previous_rows} 变为 {current_rows}。",
            )
        )
    return tuple(changes)


def _build_snapshot(repository: Any, dataset_id: UUID) -> dict[str, Any]:
    records = repository.sample_analysis_records(dataset_id, limit=SAMPLE_LIMIT)
    row_count = repository.count_analysis_records(dataset_id)
    columns = tuple(
        _column_snapshot(name, records)
        for name in sorted({str(key) for row in records for key in row})
    )
    source = _active_source(repository, dataset_id)
    profile = {
        "source": source,
        "row_count": row_count,
        "sample_size": len(records),
        "columns": [item.model_dump(mode="json") for item in columns],
    }
    fingerprint = hashlib.sha256(
        json.dumps(profile, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "dataset_id": dataset_id,
        "source": source,
        "row_count": row_count,
        "sample_size": len(records),
        "fingerprint": fingerprint,
        "profile": profile,
    }


def _active_source(repository: Any, dataset_id: UUID) -> str:
    return "cleaned" if repository.preview_cleaned_records(dataset_id, limit=1) else "raw"


def _latest_event_for_snapshot(
    repository: Any,
    *,
    dataset_id: UUID,
    snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    event = repository.latest_data_drift_event(dataset_id)
    if event is None:
        return None
    if event.get("acknowledged_at"):
        return None
    if str(event.get("current_snapshot_id")) != str(snapshot.get("id")):
        return None
    # A raw -> cleaned transition normally establishes a new comparison baseline.
    # Older releases, however, also persisted real temporal precision-loss events
    # against the cleaned snapshot (for example, timestamp values truncated to a
    # date).  Those warnings remain actionable until the data is cleaned again and
    # must not disappear merely because their baseline source was ``raw``.
    try:
        baseline = repository.get_data_snapshot(
            UUID(str(event.get("baseline_snapshot_id")))
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    active_source = str(snapshot.get("source") or "")
    if str(baseline.get("source") or "") != active_source:
        temporal_changes = _temporal_precision_loss_changes(event)
        if not temporal_changes:
            return None
        filtered = dict(event)
        filtered["changes"] = temporal_changes
        filtered["status"] = _status_for_change_payloads(temporal_changes)
        # Assets can only be attributed to the complete historical event. Once
        # unrelated cross-source changes are removed, replaying those invalidations
        # would misleadingly attach them to the temporal warning alone.
        filtered["affected_assets"] = []
        filtered["recommended_actions"] = [
            item
            for item in event.get("recommended_actions") or ()
            if isinstance(item, dict) and item.get("action") == "run_cleaning"
        ]
        return filtered
    return event


def _is_temporal_precision_loss_event(event: dict[str, Any]) -> bool:
    """Recognize persisted raw-to-cleaned warnings caused by timestamp collapse."""

    return bool(_temporal_precision_loss_changes(event))


def _temporal_precision_loss_changes(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only actionable timestamp-collapse changes from a historical event."""

    changes: list[dict[str, Any]] = []
    for change in event.get("changes") or ():
        if not isinstance(change, dict):
            continue
        if str(change.get("change_type") or "") != "unique_rate_drift":
            continue
        if not _looks_temporal_field(str(change.get("field") or "")):
            continue
        previous = _finite_float(change.get("previous_value"))
        current = _finite_float(change.get("current_value"))
        if previous is None or current is None or previous <= 0:
            continue
        if previous - current >= UNIQUE_RATE_THRESHOLD and current < previous:
            changes.append(dict(change))
    return changes


def _status_for_change_payloads(changes: list[dict[str, Any]]) -> str:
    if any(str(item.get("severity") or "") == "critical" for item in changes):
        return "critical"
    return "warning" if changes else "stable"


def _looks_temporal_field(field: str) -> bool:
    lowered = field.lower()
    return any(token in lowered for token in ("date", "time", "日期", "时间"))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _column_snapshot(
    name: str,
    records: list[dict[str, Any]],
) -> ColumnSnapshotResponse:
    values = [row.get(name) for row in records]
    present = [value for value in values if value not in (None, "")]
    dtype = _infer_type(present)
    normalized = [_normalized_value(value) for value in present]
    numeric = [
        number
        for value in present
        if (number := _number(value)) is not None
    ]
    quantiles = _quantiles(numeric)
    signatures = tuple(
        hashlib.sha256(value.encode()).hexdigest()[:12]
        for value, _ in Counter(normalized).most_common(20)
    )
    return ColumnSnapshotResponse(
        name=name,
        dtype=dtype,
        missing_rate=(len(values) - len(present)) / max(len(values), 1),
        unique_rate=len(set(normalized)) / max(len(present), 1),
        mean=fmean(numeric) if numeric else None,
        std=pstdev(numeric) if len(numeric) > 1 else 0.0 if numeric else None,
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
        quantiles=quantiles,
        value_signature=signatures,
    )


def _column_changes(
    name: str,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[DriftChangeResponse]:
    changes: list[DriftChangeResponse] = []
    if previous.get("dtype") != current.get("dtype"):
        changes.append(
            DriftChangeResponse(
                change_type="type_changed",
                severity="critical",
                field=name,
                previous_value=previous.get("dtype"),
                current_value=current.get("dtype"),
                message=f"字段 {name} 的类型从 {previous.get('dtype')} 变为 {current.get('dtype')}。",
            )
        )
    missing_delta = abs(
        float(current.get("missing_rate") or 0)
        - float(previous.get("missing_rate") or 0)
    )
    if missing_delta >= MISSING_RATE_THRESHOLD:
        changes.append(
            DriftChangeResponse(
                change_type="missing_rate_drift",
                severity="critical" if missing_delta >= 0.30 else "warning",
                field=name,
                previous_value=previous.get("missing_rate"),
                current_value=current.get("missing_rate"),
                score=round(missing_delta, 4),
                message=f"字段 {name} 的缺失率变化了 {missing_delta:.0%}。",
            )
        )
    unique_delta = abs(
        float(current.get("unique_rate") or 0)
        - float(previous.get("unique_rate") or 0)
    )
    if unique_delta >= UNIQUE_RATE_THRESHOLD:
        changes.append(
            DriftChangeResponse(
                change_type="unique_rate_drift",
                severity="warning",
                field=name,
                previous_value=previous.get("unique_rate"),
                current_value=current.get("unique_rate"),
                score=round(unique_delta, 4),
                message=f"字段 {name} 的唯一率变化了 {unique_delta:.0%}。",
            )
        )
    previous_mean = previous.get("mean")
    current_mean = current.get("mean")
    if previous_mean is not None and current_mean is not None:
        scale = max(
            float(previous.get("std") or 0),
            float(current.get("std") or 0),
            1e-9,
        )
        shift = abs(float(current_mean) - float(previous_mean)) / scale
        if shift >= 1:
            changes.append(
                DriftChangeResponse(
                    change_type="distribution_drift",
                    severity="critical" if shift >= 2 else "warning",
                    field=name,
                    previous_value=previous_mean,
                    current_value=current_mean,
                    score=round(shift, 4),
                    message=f"字段 {name} 的均值偏移了 {shift:.2f} 个标准差。",
                )
            )
    return changes


def _rename_candidates(
    removed: set[str],
    added: set[str],
    previous: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> list[tuple[str, str, float]]:
    candidates: list[tuple[float, str, str]] = []
    for old_name in removed:
        for new_name in added:
            if previous[old_name].get("dtype") != current[new_name].get("dtype"):
                continue
            name_score = SequenceMatcher(
                None,
                _normalized_name(old_name),
                _normalized_name(new_name),
            ).ratio()
            previous_signature = set(previous[old_name].get("value_signature") or ())
            current_signature = set(current[new_name].get("value_signature") or ())
            signature_score = (
                len(previous_signature & current_signature)
                / max(len(previous_signature | current_signature), 1)
            )
            score = name_score * 0.65 + signature_score * 0.35
            if score >= 0.72:
                candidates.append((score, old_name, new_name))
    selected: list[tuple[str, str, float]] = []
    used_old: set[str] = set()
    used_new: set[str] = set()
    for score, old_name, new_name in sorted(candidates, reverse=True):
        if old_name in used_old or new_name in used_new:
            continue
        used_old.add(old_name)
        used_new.add(new_name)
        selected.append((old_name, new_name, round(score, 4)))
    return selected


def _recommended_actions(
    changes: tuple[DriftChangeResponse, ...],
    affected: list[DriftAffectedAssetResponse],
) -> tuple[DriftRecommendedActionResponse, ...]:
    types = {item.change_type for item in changes}
    asset_types = {item.asset_type for item in affected}
    actions: list[DriftRecommendedActionResponse] = []
    if types & {"column_removed", "column_renamed", "type_changed", "column_added"}:
        actions.extend(
            (
                DriftRecommendedActionResponse(
                    action="review_schema",
                    label="审查字段变化",
                    reason="字段绑定可能需要重新映射。",
                ),
                DriftRecommendedActionResponse(
                    action="create_semantic_draft",
                    label="创建新语义草稿",
                    reason="已发布语义版本保持不可变，新绑定应进入草稿。",
                ),
            )
        )
    if types & {
        "missing_rate_drift",
        "unique_rate_drift",
        "distribution_drift",
    }:
        actions.append(
            DriftRecommendedActionResponse(
                action="run_cleaning",
                label="重新清洗",
                reason="数据质量或指标分布已明显变化。",
            )
        )
    if "relationship" in asset_types:
        actions.append(
            DriftRecommendedActionResponse(
                action="refresh_relationships",
                label="重新识别关系",
                reason="关联键匹配率下降或字段已失效。",
            )
        )
    if "report" in asset_types:
        actions.append(
            DriftRecommendedActionResponse(
                action="rerun_analysis",
                label="重新分析",
                reason="历史报告引用了变化前的数据快照。",
            )
        )
    return tuple({item.action: item for item in actions}.values())


def _status_for_changes(changes: tuple[DriftChangeResponse, ...]) -> str:
    if any(item.severity == "critical" for item in changes):
        return "critical"
    if changes:
        return "warning"
    return "stable"


def _highest_status(statuses: tuple[str, ...]) -> str:
    order = {"baseline": 0, "stable": 1, "warning": 2, "critical": 3}
    return max(statuses or ("stable",), key=lambda item: order.get(item, 1))


def _response_from_event(
    *,
    dataset_id: UUID,
    snapshot: dict[str, Any],
    event: dict[str, Any] | None,
) -> DatasetDriftResponse:
    if event is None:
        return DatasetDriftResponse(
            dataset_id=dataset_id,
            status="baseline",
            snapshot=_snapshot_response(snapshot),
            scanned_at=str(snapshot["created_at"]),
        )
    return DatasetDriftResponse(
        dataset_id=dataset_id,
        status=str(event["status"]),
        snapshot=_snapshot_response(snapshot),
        baseline_snapshot_id=UUID(str(event["baseline_snapshot_id"])),
        event_id=UUID(str(event["id"])),
        changes=tuple(
            DriftChangeResponse.model_validate(item)
            for item in event.get("changes") or ()
        ),
        affected_assets=tuple(
            DriftAffectedAssetResponse.model_validate(item)
            for item in event.get("affected_assets") or ()
        ),
        recommended_actions=tuple(
            DriftRecommendedActionResponse.model_validate(item)
            for item in event.get("recommended_actions") or ()
        ),
        scanned_at=str(event["created_at"]),
    )


def _snapshot_response(snapshot: dict[str, Any]) -> DatasetSnapshotResponse:
    profile = snapshot.get("profile") or {}
    return DatasetSnapshotResponse(
        snapshot_id=UUID(str(snapshot["id"])),
        dataset_id=UUID(str(snapshot["dataset_id"])),
        source=str(snapshot["source"]),
        row_count=int(snapshot["row_count"]),
        sample_size=int(snapshot["sample_size"]),
        fingerprint=str(snapshot["fingerprint"]),
        columns=tuple(
            ColumnSnapshotResponse.model_validate(item)
            for item in profile.get("columns") or ()
        ),
        created_at=str(snapshot["created_at"]),
    )


def _dataset_columns(repository: Any, dataset_id: UUID) -> set[str]:
    return {
        str(key)
        for row in repository.sample_analysis_records(dataset_id, limit=100)
        for key in row
    }


def _expand_values(
    values: set[str],
    mode: str,
    delimiter: Any,
) -> set[str]:
    if mode != "delimited":
        return values
    separator = str(delimiter or "_")
    return {
        token.strip().lower()
        for value in values
        for token in value.split(separator)
        if token.strip()
    }


def _relationship_id(relationship: dict[str, Any]) -> str:
    existing = str(relationship.get("relationship_id") or "")
    if existing:
        return existing
    payload = ":".join(
        str(relationship.get(key) or "")
        for key in (
            "left_dataset_id",
            "left_column",
            "right_dataset_id",
            "right_column",
        )
    )
    return "rel_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _infer_type(values: list[Any]) -> str:
    if not values:
        return "unknown"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    numeric_count = sum(_number(value) is not None for value in values)
    if numeric_count / len(values) >= 0.9:
        return "number"
    date_count = sum(_looks_like_date(value) for value in values)
    if date_count / len(values) >= 0.9:
        return "date"
    return "text"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _looks_like_date(value: Any) -> bool:
    text = str(value).strip()
    if len(text) < 8 or not any(character in text for character in "-/T"):
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _quantiles(values: list[float]) -> tuple[float, ...]:
    if not values:
        return ()
    ordered = sorted(values)
    return tuple(
        ordered[round((len(ordered) - 1) * ratio)]
        for ratio in (0.25, 0.5, 0.75)
    )


def _normalized_value(value: Any) -> str:
    return str(value).strip().lower()


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
