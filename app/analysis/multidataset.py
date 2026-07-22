from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any
from uuid import UUID

import pandas as pd

from app.analysis.services import DatasetProfiler, _dataframe, apply_column_metadata_to_profile
from app.schemas.analysis import (
    DatasetJoinConfig,
    DatasetProfileResponse,
    DatasetReferenceResponse,
    JoinSuggestionCandidateResponse,
    JoinSuggestionResponse,
    MultiDatasetProfileResponse,
    ValidationIssueResponse,
)
from app.storage.dataset_store import DatasetStoreRepository, StoredDataset

MAX_JOIN_EXPANSION_RATIO = 10.0
MAX_DELIMITED_JOIN_EXPANSION_RATIO = 250.0
MAX_JOINED_ROWS = 1_000_000


@dataclass(frozen=True)
class PreparedMultiDatasetContext:
    dataframe: pd.DataFrame
    records: list[dict[str, Any]]
    profile: DatasetProfileResponse
    response: MultiDatasetProfileResponse | None
    validation_issues: tuple[ValidationIssueResponse, ...]


@dataclass(frozen=True)
class _JoinColumnStats:
    sample_values: set[str]
    distinct_ratio: float


def dataset_reference(
    repository: DatasetStoreRepository,
    dataset_id: UUID,
) -> tuple[DatasetReferenceResponse, DatasetProfileResponse]:
    dataset = repository.get_dataset(dataset_id)
    records = repository.read_analysis_records(dataset_id)
    profile = _profile(repository, dataset_id, records)
    return _reference(dataset, profile), profile


def suggest_dataset_joins(
    repository: DatasetStoreRepository,
    *,
    dataset_id: UUID,
    additional_dataset_ids: tuple[UUID, ...],
    records_by_dataset: Mapping[UUID, list[dict[str, Any]]] | None = None,
) -> JoinSuggestionResponse:
    primary_dataset = repository.get_dataset(dataset_id)
    left_records = _suggestion_records(repository, dataset_id, records_by_dataset)
    primary_profile = _profile(repository, dataset_id, left_records)
    primary_ref = _reference(primary_dataset, primary_profile)
    additional_refs: list[DatasetReferenceResponse] = []
    suggestions: list[JoinSuggestionCandidateResponse] = []
    issues: list[ValidationIssueResponse] = []
    left_metadata = _metadata_by_column(repository, dataset_id)
    left_columns = {column.name: column for column in primary_profile.columns}
    left_stats = _column_stats(left_records, left_columns.keys())
    stored_value_samples: dict[tuple[UUID, str], set[str]] = {}

    def comparable_values(target_dataset_id: UUID, column_name: str) -> set[str]:
        key = (target_dataset_id, column_name)
        if key not in stored_value_samples:
            stored_value_samples[key] = repository.sample_analysis_column_values(
                target_dataset_id,
                column_name=column_name,
            )
        return stored_value_samples[key]

    for right_dataset_id in _dedupe_dataset_ids(additional_dataset_ids, exclude=dataset_id):
        right_dataset = repository.get_dataset(right_dataset_id)
        right_records = _suggestion_records(repository, right_dataset_id, records_by_dataset)
        right_profile = _profile(repository, right_dataset_id, right_records)
        right_ref = _reference(right_dataset, right_profile)
        additional_refs.append(right_ref)
        right_metadata = _metadata_by_column(repository, right_dataset_id)
        right_columns = {column.name: column for column in right_profile.columns}
        right_stats = _column_stats(right_records, right_columns.keys())
        pair_candidates: list[JoinSuggestionCandidateResponse] = []
        left_candidate_names = _candidate_join_column_names(left_columns, left_metadata)
        right_candidate_names = _candidate_join_column_names(right_columns, right_metadata)
        for left_name in left_candidate_names:
            left_column = left_columns[left_name]
            for right_name in right_candidate_names:
                right_column = right_columns[right_name]
                left_column_stats = left_stats.get(left_name, _JoinColumnStats(set(), 0))
                right_column_stats = right_stats.get(right_name, _JoinColumnStats(set(), 0))
                left_values = left_column_stats.sample_values
                right_values = right_column_stats.sample_values
                if _needs_comparable_key_sample(
                    left_name,
                    right_name,
                    left_role=str(left_metadata.get(left_name, {}).get("role") or ""),
                    right_role=str(right_metadata.get(right_name, {}).get("role") or ""),
                ):
                    left_values = comparable_values(dataset_id, left_name)
                    right_values = comparable_values(right_dataset_id, right_name)
                candidate = _join_candidate(
                    left_dataset_id=dataset_id,
                    right_dataset_id=right_dataset_id,
                    left_name=left_name,
                    right_name=right_name,
                    left_type=left_column.dtype,
                    right_type=right_column.dtype,
                    left_role=str(left_metadata.get(left_name, {}).get("role") or ""),
                    right_role=str(right_metadata.get(right_name, {}).get("role") or ""),
                    left_values=left_values,
                    right_values=right_values,
                    left_distinct_ratio=left_column_stats.distinct_ratio,
                    right_distinct_ratio=right_column_stats.distinct_ratio,
                )
                if candidate.score >= 0.35:
                    pair_candidates.append(candidate)
        pair_candidates.sort(key=lambda item: item.score, reverse=True)
        suggestions.extend(pair_candidates[:5])
        if not pair_candidates:
            issues.append(
                ValidationIssueResponse(
                    severity="warning",
                    finding_ref="join_suggestions",
                    issue=f"未找到可推荐的关联字段：{primary_ref.name} -> {right_ref.name}",
                    suggestion="请手动选择 ID、日期或维度字段作为 join key。",
                )
            )

    return JoinSuggestionResponse(
        primary_dataset=primary_ref,
        additional_datasets=tuple(additional_refs),
        suggestions=tuple(suggestions),
        validation_issues=tuple(issues),
    )


def prepare_multi_dataset_context(
    repository: DatasetStoreRepository,
    *,
    dataset_id: UUID,
    additional_dataset_ids: tuple[UUID, ...],
    join_plan: tuple[DatasetJoinConfig, ...],
) -> PreparedMultiDatasetContext:
    primary_dataset = repository.get_dataset(dataset_id)
    primary_records = repository.read_analysis_records(dataset_id)
    if not primary_records:
        raise RuntimeError("Dataset has no raw records to analyze.")

    primary_frame = _dataframe(primary_records)
    primary_profile = _profile(repository, dataset_id, primary_records)
    primary_ref = _reference(primary_dataset, primary_profile)
    additional_ids = _dedupe_dataset_ids(additional_dataset_ids, exclude=dataset_id)
    if not additional_ids:
        return PreparedMultiDatasetContext(
            dataframe=primary_frame,
            records=primary_records,
            profile=primary_profile,
            response=None,
            validation_issues=(),
        )

    additional_refs: list[DatasetReferenceResponse] = []
    additional_frames: dict[UUID, pd.DataFrame] = {}
    datasets_by_id: dict[UUID, StoredDataset] = {dataset_id: primary_dataset}
    profiles_by_id: dict[UUID, DatasetProfileResponse] = {dataset_id: primary_profile}
    issues: list[ValidationIssueResponse] = []
    for additional_id in additional_ids:
        dataset = repository.get_dataset(additional_id)
        records = repository.read_analysis_records(additional_id)
        if not records:
            issues.append(
                ValidationIssueResponse(
                    severity="warning",
                    finding_ref="join_prepare",
                    issue=f"附加数据集没有可分析记录：{dataset.name}",
                    suggestion="请先导入或清洗该数据集。",
                )
            )
            continue
        profile = _profile(repository, additional_id, records)
        datasets_by_id[additional_id] = dataset
        profiles_by_id[additional_id] = profile
        additional_refs.append(_reference(dataset, profile))
        additional_frames[additional_id] = _dataframe(records)

    if not join_plan:
        issues.append(
            ValidationIssueResponse(
                severity="warning",
                finding_ref="join_prepare",
                issue="已选择多个数据集，但没有确认 join 配置；本次按主数据集分析。",
                suggestion="在分析页选择推荐 join key 或手动配置关联字段后重新运行。",
            )
        )
        response = _multi_dataset_response(
            primary_ref=primary_ref,
            additional_refs=tuple(additional_refs),
            join_plan=(),
            join_summary={"mode": "primary_dataset_fallback", "reason": "missing_join_plan"},
            joined_profile=primary_profile,
            column_source_map=dict.fromkeys(primary_frame.columns, primary_dataset.name),
            issues=tuple(issues),
        )
        return PreparedMultiDatasetContext(
            dataframe=primary_frame,
            records=primary_records,
            profile=primary_profile,
            response=response,
            validation_issues=tuple(issues),
        )

    joined = primary_frame.copy()
    column_source_map = {str(column): primary_dataset.name for column in joined.columns}
    column_lookup = {(dataset_id, str(column)): str(column) for column in primary_frame.columns}
    joined_dataset_ids = {dataset_id}
    join_summaries: list[dict[str, Any]] = []
    used_plan: list[DatasetJoinConfig] = []
    pending = list(join_plan)
    while pending:
        progressed = False
        for config in tuple(pending):
            if config.left_dataset_id not in joined_dataset_ids:
                continue
            pending.remove(config)
            progressed = True
            if config.right_dataset_id in joined_dataset_ids:
                issues.append(
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="join_prepare",
                        issue=f"关系形成重复路径或环，已跳过：{config.left_dataset_id} -> {config.right_dataset_id}",
                        suggestion="每张附表只应通过一条无环路径连接到主表。",
                    )
                )
                continue
            right_frame = additional_frames.get(config.right_dataset_id)
            right_dataset = datasets_by_id.get(config.right_dataset_id)
            if right_frame is None or right_dataset is None:
                issues.append(
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="join_prepare",
                        issue=f"附加数据集不可用或没有记录：{config.right_dataset_id}",
                        suggestion="确认该数据集属于当前用户且已经导入数据。",
                    )
                )
                continue
            left_key = column_lookup.get((config.left_dataset_id, config.left_column))
            if left_key is None or config.right_column not in right_frame.columns:
                issues.append(
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="join_prepare",
                        issue=f"join 字段不存在：{config.left_column} -> {config.right_column}",
                        suggestion="重新识别关系，或确认链路上游表已成功加入。",
                    )
                )
                continue

            left_value_mode = config.left_value_mode
            right_value_mode = config.right_value_mode
            left_delimiter = config.left_delimiter
            right_delimiter = config.right_delimiter
            before_rows = int(joined.shape[0])
            join_left = joined
            if left_value_mode == "scalar" and right_value_mode == "scalar":
                inferred = infer_join_value_match(
                    joined[[left_key]].rename(columns={left_key: config.left_column}).to_dict(orient="records"),
                    right_frame.to_dict(orient="records"),
                    config.left_column,
                    config.right_column,
                )
                if inferred[0] > _series_match_rate(joined[left_key], right_frame[config.right_column]):
                    _, left_value_mode, right_value_mode, left_delimiter, right_delimiter = inferred

            if left_value_mode == "delimited":
                join_left = _explode_delimited_column(join_left, left_key, left_delimiter or "_")
            if right_value_mode == "delimited":
                right_frame = _explode_delimited_column(
                    right_frame,
                    config.right_column,
                    right_delimiter or "_",
                )

            prefixed, right_column_lookup, source_updates = _prefix_right_frame(
                right_frame,
                right_dataset=right_dataset,
                reserved_columns={str(column) for column in join_left.columns},
            )
            right_key = right_column_lookup[config.right_column]
            estimated_rows = _estimate_join_rows(
                join_left[left_key],
                prefixed[right_key],
                join_type=config.join_type,
            )
            estimated_expansion = estimated_rows / max(before_rows, 1)
            expansion_limit = (
                MAX_DELIMITED_JOIN_EXPANSION_RATIO
                if "delimited" in {left_value_mode, right_value_mode}
                else MAX_JOIN_EXPANSION_RATIO
            )
            right_key_unique = bool(not prefixed[right_key].dropna().duplicated().any())
            if estimated_rows > MAX_JOINED_ROWS or estimated_expansion > expansion_limit:
                issues.append(
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="join_prepare",
                        issue=(
                            f"Join 预计产生 {estimated_rows} 行（约 {estimated_expansion:.1f} 倍），"
                            f"超过安全上限（{MAX_JOINED_ROWS} 行或 {expansion_limit:.0f} 倍），"
                            f"已跳过 {right_dataset.name}。"
                        ),
                        suggestion="改用唯一键维表，先聚合右表，或调整关系方向后重试。",
                    )
                )
                join_summaries.append(
                    {
                        "status": "skipped_row_expansion",
                        "left_dataset_id": str(config.left_dataset_id),
                        "right_dataset_id": str(config.right_dataset_id),
                        "right_dataset_name": right_dataset.name,
                        "left_column": config.left_column,
                        "right_column": config.right_column,
                        "join_type": config.join_type,
                        "before_rows": before_rows,
                        "estimated_rows": estimated_rows,
                        "estimated_expansion_ratio": round(estimated_expansion, 4),
                        "right_key_unique": right_key_unique,
                        "left_value_mode": left_value_mode,
                        "right_value_mode": right_value_mode,
                    }
                )
                continue

            indicator = f"__datamind_join_{len(join_summaries)}"
            merged = join_left.merge(
                prefixed,
                how=config.join_type,
                left_on=left_key,
                right_on=right_key,
                indicator=indicator,
            )
            unmatched_rows = int((merged[indicator] == "left_only").sum()) if config.join_type == "left" else 0
            joined = merged.drop(columns=[indicator])
            joined_dataset_ids.add(config.right_dataset_id)
            column_lookup.update(
                {
                    (config.right_dataset_id, original): prefixed_name
                    for original, prefixed_name in right_column_lookup.items()
                }
            )
            column_source_map.update(source_updates)
            actual_expansion = int(joined.shape[0]) / max(before_rows, 1)
            if actual_expansion > 1.05:
                issues.append(
                    ValidationIssueResponse(
                        severity="warning",
                        finding_ref="join_prepare",
                        issue=f"Join {right_dataset.name} 后行数扩大到 {actual_expansion:.2f} 倍，聚合指标可能重复。",
                        suggestion="复核右表关联键唯一性和金额、数量等指标的所属粒度。",
                    )
                )
            join_summaries.append(
                {
                    "status": "joined",
                    "left_dataset_id": str(config.left_dataset_id),
                    "right_dataset_id": str(config.right_dataset_id),
                    "right_dataset_name": right_dataset.name,
                    "left_column": config.left_column,
                    "right_column": config.right_column,
                    "left_join_column": left_key,
                    "right_join_column": right_key,
                    "join_type": config.join_type,
                    "before_rows": before_rows,
                    "estimated_rows": estimated_rows,
                    "after_rows": int(joined.shape[0]),
                    "row_expansion_ratio": round(actual_expansion, 4),
                    "right_key_unique": right_key_unique,
                    "left_value_mode": left_value_mode,
                    "right_value_mode": right_value_mode,
                    "unmatched_rows": unmatched_rows,
                }
            )
            used_plan.append(config)
        if not progressed:
            break

    issues.extend(
        ValidationIssueResponse(
            severity="warning",
            finding_ref="join_prepare",
            issue=f"关系无法从当前主表到达，已跳过：{config.left_dataset_id} -> {config.right_dataset_id}",
            suggestion="重新生成无环关系树，或将根表调整为该链路的起点。",
        )
        for config in pending
    )

    if joined.empty:
        issues.append(
            ValidationIssueResponse(
                severity="warning",
                finding_ref="join_prepare",
                issue="join 后结果为空；本次已 fallback 到主数据集分析。",
                suggestion="检查 join key 是否一致，或改用 left join。",
            )
        )
        joined = primary_frame.copy()
        column_source_map = {str(column): primary_dataset.name for column in joined.columns}

    joined_records = joined.where(pd.notna(joined), None).to_dict(orient="records")
    joined_profile = DatasetProfiler().profile(dataset_id=dataset_id, records=joined_records)
    response = _multi_dataset_response(
        primary_ref=primary_ref,
        additional_refs=tuple(additional_refs),
        join_plan=tuple(used_plan),
        join_summary={
            "mode": "joined" if used_plan else "primary_dataset_fallback",
            "dataset_count": 1 + len(additional_refs),
            "joined_dataset_count": len(joined_dataset_ids),
            "joined_row_count": int(joined.shape[0]),
            "joined_column_count": int(joined.shape[1]),
            "row_expansion_ratio": round(int(joined.shape[0]) / max(int(primary_frame.shape[0]), 1), 4),
            "skipped_join_count": len(join_plan) - len(used_plan),
            "joins": join_summaries,
        },
        joined_profile=joined_profile,
        column_source_map=column_source_map,
        issues=tuple(issues),
    )
    return PreparedMultiDatasetContext(
        dataframe=joined,
        records=joined_records,
        profile=joined_profile,
        response=response,
        validation_issues=tuple(issues),
    )


def _profile(
    repository: DatasetStoreRepository,
    dataset_id: UUID,
    records: list[dict[str, Any]],
) -> DatasetProfileResponse:
    return apply_column_metadata_to_profile(
        DatasetProfiler().profile(dataset_id=dataset_id, records=records),
        repository.list_column_metadata(dataset_id),
    )


def _reference(dataset: StoredDataset, profile: DatasetProfileResponse) -> DatasetReferenceResponse:
    return DatasetReferenceResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        status=dataset.status,
        row_count=profile.row_count,
        column_count=profile.column_count,
        columns=tuple(column.name for column in profile.columns),
    )


def _metadata_by_column(
    repository: DatasetStoreRepository,
    dataset_id: UUID,
) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("column_name")): item
        for item in repository.list_column_metadata(dataset_id)
        if item.get("column_name")
    }


def _dedupe_dataset_ids(
    dataset_ids: tuple[UUID, ...],
    *,
    exclude: UUID,
) -> tuple[UUID, ...]:
    seen: set[UUID] = {exclude}
    result: list[UUID] = []
    for dataset_id in dataset_ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        result.append(dataset_id)
    return tuple(result)


def _suggestion_records(
    repository: DatasetStoreRepository,
    dataset_id: UUID,
    records_by_dataset: Mapping[UUID, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if records_by_dataset is not None and dataset_id in records_by_dataset:
        return records_by_dataset[dataset_id]
    return repository.read_analysis_records(dataset_id)


def _join_candidate(
    *,
    left_dataset_id: UUID,
    right_dataset_id: UUID,
    left_name: str,
    right_name: str,
    left_type: str,
    right_type: str,
    left_role: str,
    right_role: str,
    left_values: set[str],
    right_values: set[str],
    left_distinct_ratio: float,
    right_distinct_ratio: float,
) -> JoinSuggestionCandidateResponse:
    left_norm = _normalize_column_name(left_name)
    right_norm = _normalize_column_name(right_name)
    score = 0.0
    reasons: list[str] = []
    if left_name.lower() == right_name.lower():
        score += 0.36
        reasons.append("字段同名")
    elif left_norm == right_norm:
        score += 0.3
        reasons.append("规范化字段名一致")
    elif left_norm and right_norm and (left_norm in right_norm or right_norm in left_norm):
        score += 0.14
        reasons.append("字段名相似")
    if left_role == "id" or right_role == "id":
        score += 0.22
        reasons.append("字段角色包含 id")
    elif left_role in {"dimension", "date"} or right_role in {"dimension", "date"}:
        score += 0.08
        reasons.append("字段角色适合作为维度/日期")
    if _type_family(left_type) == _type_family(right_type):
        score += 0.12
        reasons.append("字段类型兼容")
    match_rate, left_value_mode, right_value_mode, left_delimiter, right_delimiter = _join_value_match(
        left_name,
        right_name,
        left_values,
        right_values,
    )
    score += min(match_rate * 0.24, 0.24)
    if match_rate:
        if "delimited" in {left_value_mode, right_value_mode}:
            reasons.append(f"列表元素样本匹配率 {match_rate:.0%}")
        else:
            reasons.append(f"样本匹配率 {match_rate:.0%}")
    if right_distinct_ratio >= 0.9:
        score += 0.1
        reasons.append("右表键接近唯一")
    elif right_distinct_ratio < 0.5:
        score -= 0.18
        reasons.append("右表键重复率较高")
    if left_distinct_ratio < 0.5 and right_distinct_ratio < 0.5:
        score -= 0.2
        reasons.append("两侧键均高重复")
    score = max(0, min(score, 1))
    return JoinSuggestionCandidateResponse(
        left_dataset_id=left_dataset_id,
        right_dataset_id=right_dataset_id,
        left_column=left_name,
        right_column=right_name,
        join_type="left",
        score=score,
        reason="、".join(reasons) or "低置信度候选",
        left_type=left_type,
        right_type=right_type,
        left_role=left_role,
        right_role=right_role,
        estimated_match_rate=match_rate,
        left_value_mode=left_value_mode,
        right_value_mode=right_value_mode,
        left_delimiter=left_delimiter,
        right_delimiter=right_delimiter,
    )


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.replace("ID", "id").lower())


def _needs_comparable_key_sample(
    left_name: str,
    right_name: str,
    *,
    left_role: str,
    right_role: str,
) -> bool:
    left_normalized = _normalize_column_name(left_name)
    right_normalized = _normalize_column_name(right_name)
    names_match = bool(
        left_normalized
        and right_normalized
        and (
            left_normalized == right_normalized
            or left_normalized in right_normalized
            or right_normalized in left_normalized
        )
    )
    return names_match or (left_role == "id" and right_role == "id")


def _candidate_join_column_names(
    columns: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    if len(columns) <= 40:
        return tuple(columns.keys())
    scored: list[tuple[float, str]] = []
    for name in columns:
        lowered = name.lower()
        normalized = _normalize_column_name(name)
        role = str(metadata.get(name, {}).get("role") or "")
        score = 0.0
        if role == "id":
            score += 1.0
        elif role in {"date", "dimension"}:
            score += 0.35
        if any(token in lowered for token in ("_id", " id", "id_", "-id", "key", "code", "uuid", "ref")):
            score += 0.75
        if normalized.endswith(("id", "key", "code")):
            score += 0.45
        if any(token in lowered for token in ("date", "time", "name", "category", "type")):
            score += 0.25
        if score > 0:
            scored.append((score, name))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [name for _, name in scored[:40]]
    return tuple(selected or islice(columns, 40))


def _type_family(dtype: str) -> str:
    lowered = dtype.lower()
    if any(token in lowered for token in ("int", "float", "double", "number", "decimal")):
        return "number"
    if any(token in lowered for token in ("date", "time")):
        return "date"
    return "text"


def _column_stats(records: list[dict[str, Any]], columns: Iterable[str]) -> dict[str, _JoinColumnStats]:
    return {
        column: _JoinColumnStats(
            sample_values=_sample_values(records, column),
            distinct_ratio=_distinct_ratio(records, column),
        )
        for column in columns
    }


def _estimated_match_rate(left_values: set[str], right_values: set[str]) -> float:
    if not left_values or not right_values:
        return 0
    return len(left_values & right_values) / max(len(left_values), 1)


def infer_join_value_match(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    left_column: str,
    right_column: str,
) -> tuple[float, str, str, str | None, str | None]:
    """Return directional sample overlap and any safe list-value transforms."""
    return _join_value_match(
        left_column,
        right_column,
        _sample_values(left_records, left_column),
        _sample_values(right_records, right_column),
    )


def _join_value_match(
    left_name: str,
    right_name: str,
    left_values: set[str],
    right_values: set[str],
) -> tuple[float, str, str, str | None, str | None]:
    direct_rate = _estimated_match_rate(left_values, right_values)
    best = (direct_rate, "scalar", "scalar", None, None)
    left_tokens, left_delimiter = _collection_tokens(left_name, left_values)
    if left_delimiter:
        rate = _estimated_match_rate(left_tokens, right_values)
        if rate > best[0]:
            best = (rate, "delimited", "scalar", left_delimiter, None)
    right_tokens, right_delimiter = _collection_tokens(right_name, right_values)
    if right_delimiter:
        rate = _estimated_match_rate(left_values, right_tokens)
        if rate > best[0]:
            best = (rate, "scalar", "delimited", None, right_delimiter)
    return best


def _collection_tokens(column_name: str, values: set[str]) -> tuple[set[str], str | None]:
    normalized = _normalize_column_name(column_name)
    lowered = column_name.lower()
    collection_hint = (
        "list" in lowered
        or normalized.endswith(("ids", "keys", "codes"))
    )
    if not collection_hint:
        return set(), None
    for delimiter in ("_", ",", "|", ";"):
        split_values = [value.split(delimiter) for value in values if delimiter in value]
        if not split_values or max(len(parts) for parts in split_values) < 2:
            continue
        tokens = {
            token.strip().lower()
            for value in values
            for token in value.split(delimiter)
            if token.strip()
        }
        if tokens:
            return tokens, delimiter
    return set(), None


def _series_match_rate(left: pd.Series, right: pd.Series) -> float:
    left_values = {str(value).strip().lower() for value in left.dropna() if str(value).strip()}
    right_values = {str(value).strip().lower() for value in right.dropna() if str(value).strip()}
    return _estimated_match_rate(left_values, right_values)


def _explode_delimited_column(frame: pd.DataFrame, column: str, delimiter: str) -> pd.DataFrame:
    exploded = frame.copy()
    exploded[column] = exploded[column].map(lambda value: _split_delimited_value(value, delimiter))
    return exploded.explode(column, ignore_index=True)


def _split_delimited_value(value: Any, delimiter: str) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [None]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")] or [None]
    parts = [part.strip() for part in str(value).split(delimiter) if part.strip()]
    return parts or [None]


def _sample_values(records: list[dict[str, Any]], column: str) -> set[str]:
    values: set[str] = set()
    for record in records[:500]:
        value = record.get(column)
        if value is None or str(value).strip() == "":
            continue
        values.add(str(value).strip().lower())
    return values


def _distinct_ratio(records: list[dict[str, Any]], column: str) -> float:
    values = [str(record.get(column)).strip().lower() for record in records[:500] if record.get(column) not in (None, "")]
    if len(values) < 2:
        return 1.0 if values else 0.0
    return len(set(values)) / len(values)


def _prefix_right_frame(
    frame: pd.DataFrame,
    *,
    right_dataset: StoredDataset,
    reserved_columns: set[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    slug = _dataset_slug(right_dataset.name)
    if any(f"{slug}__{column}" in reserved_columns for column in frame.columns):
        slug = f"{slug}_{str(right_dataset.id)[:8]}"
    renamed: dict[str, str] = {}
    source_map: dict[str, str] = {}
    for column in frame.columns:
        column_name = str(column)
        next_name = f"{slug}__{column_name}"
        renamed[column_name] = next_name
        source_map[next_name] = right_dataset.name
    return frame.rename(columns=renamed), renamed, source_map


def _estimate_join_rows(
    left_series: pd.Series,
    right_series: pd.Series,
    *,
    join_type: str,
) -> int:
    left_counts = left_series.dropna().value_counts(dropna=True)
    right_counts = right_series.dropna().value_counts(dropna=True)
    right_multipliers = right_counts.reindex(left_counts.index, fill_value=0)
    if join_type == "left":
        right_multipliers = right_multipliers.clip(lower=1)
    estimated = int((left_counts * right_multipliers).sum())
    left_nulls = int(left_series.isna().sum())
    right_nulls = int(right_series.isna().sum())
    if left_nulls:
        estimated += left_nulls * max(right_nulls, 1 if join_type == "left" else 0)
    return estimated


def _dataset_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:32] or "dataset"


def _multi_dataset_response(
    *,
    primary_ref: DatasetReferenceResponse,
    additional_refs: tuple[DatasetReferenceResponse, ...],
    join_plan: tuple[DatasetJoinConfig, ...],
    join_summary: dict[str, Any],
    joined_profile: DatasetProfileResponse,
    column_source_map: dict[str, str],
    issues: tuple[ValidationIssueResponse, ...],
) -> MultiDatasetProfileResponse:
    return MultiDatasetProfileResponse(
        primary_dataset=primary_ref,
        additional_datasets=additional_refs,
        join_plan=join_plan,
        join_summary=join_summary,
        joined_profile=joined_profile,
        column_source_map=column_source_map,
        validation_issues=issues,
    )
