from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.analysis.model_router import AnalysisModelRouter, MCPAnalysisModelRouter
from app.analysis.multidataset import infer_join_value_match, suggest_dataset_joins
from app.analysis.prompt_utils import UNTRUSTED_INPUT_NOTICE, compact_prompt_records
from app.analysis.services import DatasetProfiler, apply_column_metadata_to_profile
from app.core.settings import get_settings
from app.schemas.dataset_store import DatasetRelationshipCandidate, DatasetRelationshipPlan
from app.semantic.embedding import cosine_similarity, get_semantic_embedding_provider
from app.storage.dataset_store import DatasetStoreRepository, StoredDatasetGroup

RELATIONSHIP_SAMPLE_LIMIT = 1000


@dataclass(frozen=True)
class DatasetGroupRelationshipSuggestions:
    candidates: tuple[DatasetRelationshipCandidate, ...]
    llm_used: bool
    compact_context: dict[str, Any]
    validation_issues: tuple[str, ...]


@dataclass(frozen=True)
class AutomaticDatasetRelationships:
    relationships: tuple[DatasetRelationshipPlan, ...]
    primary_dataset_id: UUID | None
    unresolved_dataset_ids: tuple[UUID, ...]


def suggest_dataset_group_relationships(
    repository: DatasetStoreRepository,
    *,
    group_id: UUID,
    router: AnalysisModelRouter | None = None,
) -> DatasetGroupRelationshipSuggestions:
    group = repository.get_dataset_group(group_id)
    records_by_dataset = {
        dataset_id: repository.sample_analysis_records(dataset_id, limit=RELATIONSHIP_SAMPLE_LIMIT)
        for dataset_id in group.dataset_ids
    }
    compact_context = _compact_group_context(repository, group, records_by_dataset=records_by_dataset)
    rule_candidates = _rule_candidates(repository, group, records_by_dataset=records_by_dataset)
    issues: list[str] = []
    needs_llm = len(rule_candidates) < max(len(group.dataset_ids) - 1, 1) or any(
        candidate.confidence < 0.65 for candidate in rule_candidates[: max(len(group.dataset_ids) - 1, 1)]
    )
    llm_candidates: tuple[DatasetRelationshipCandidate, ...] = ()
    llm_used = False
    if needs_llm:
        try:
            llm_candidates = _llm_candidates(
                group=group,
                compact_context=compact_context,
                rule_candidates=rule_candidates,
                records_by_dataset=records_by_dataset,
                router=router or MCPAnalysisModelRouter(),
            )
            llm_used = bool(llm_candidates)
        except Exception as exc:  # pragma: no cover - exercised by integration fallback paths.
            issues.append(f"LLM relationship suggestions unavailable; rule suggestions were used. {type(exc).__name__}: {exc}")

    candidates = _merge_candidates(rule_candidates, llm_candidates)
    if not candidates:
        issues.append("No relationship candidates were found. Please select join keys manually.")
    return DatasetGroupRelationshipSuggestions(
        candidates=candidates,
        llm_used=llm_used,
        compact_context=compact_context,
        validation_issues=tuple(issues),
    )


def select_automatic_dataset_relationships(
    group: StoredDatasetGroup,
    candidates: tuple[DatasetRelationshipCandidate, ...],
) -> AutomaticDatasetRelationships:
    """Choose a high-confidence, acyclic relationship tree with the safest root."""
    dataset_ids = tuple(group.dataset_ids)
    if len(dataset_ids) < 2:
        return AutomaticDatasetRelationships((), dataset_ids[0] if dataset_ids else None, ())

    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.confidence >= 0.55
        and (candidate.estimated_match_rate >= 0.1 or candidate.confidence >= 0.72)
        and candidate.relationship_type != "many_to_many"
    )
    best_by_pair: dict[frozenset[UUID], DatasetRelationshipCandidate] = {}
    for candidate in eligible:
        pair = frozenset((candidate.left_dataset_id, candidate.right_dataset_id))
        existing = best_by_pair.get(pair)
        if existing is None or candidate.confidence > existing.confidence:
            best_by_pair[pair] = candidate

    def tree_for_root(root_id: UUID) -> tuple[DatasetRelationshipPlan, ...]:
        connected = {root_id}
        plans: list[DatasetRelationshipPlan] = []
        while True:
            options: list[tuple[float, DatasetRelationshipPlan]] = []
            for candidate in best_by_pair.values():
                left_connected = candidate.left_dataset_id in connected
                right_connected = candidate.right_dataset_id in connected
                if left_connected == right_connected:
                    continue
                reverse = right_connected
                plan = _candidate_plan(candidate, reverse=reverse)
                safety_bonus = {
                    "many_to_one": 0.2,
                    "one_to_one": 0.15,
                    "unknown": -0.05,
                    "one_to_many": -0.35,
                }.get(plan.relationship_type, -0.5)
                options.append((plan.confidence + safety_bonus, plan))
            if not options:
                break
            _, selected_plan = max(options, key=lambda item: item[0])
            plans.append(selected_plan)
            connected.add(selected_plan.right_dataset_id)
        return tuple(plans)

    root_plans = {dataset_id: tree_for_root(dataset_id) for dataset_id in dataset_ids}

    def root_score(dataset_id: UUID) -> tuple[int, int, float, int]:
        plans = root_plans[dataset_id]
        safe_count = sum(plan.relationship_type in {"one_to_one", "many_to_one"} for plan in plans)
        unsafe_count = sum(plan.relationship_type == "one_to_many" for plan in plans)
        return len(plans), safe_count, sum(plan.confidence for plan in plans), -unsafe_count

    primary_dataset_id = max(dataset_ids, key=root_score)
    selected = root_plans[primary_dataset_id]
    connected_ids = {primary_dataset_id, *(plan.right_dataset_id for plan in selected)}
    unresolved = tuple(dataset_id for dataset_id in dataset_ids if dataset_id not in connected_ids)
    return AutomaticDatasetRelationships(selected, primary_dataset_id, unresolved)


def _candidate_plan(candidate: DatasetRelationshipCandidate, *, reverse: bool) -> DatasetRelationshipPlan:
    relationship_type = _reverse_relationship_type(candidate.relationship_type) if reverse else candidate.relationship_type
    return DatasetRelationshipPlan(
        left_dataset_id=candidate.right_dataset_id if reverse else candidate.left_dataset_id,
        right_dataset_id=candidate.left_dataset_id if reverse else candidate.right_dataset_id,
        left_column=candidate.right_column if reverse else candidate.left_column,
        right_column=candidate.left_column if reverse else candidate.right_column,
        join_type="left",
        left_value_mode=candidate.right_value_mode if reverse else candidate.left_value_mode,
        right_value_mode=candidate.left_value_mode if reverse else candidate.right_value_mode,
        left_delimiter=candidate.right_delimiter if reverse else candidate.left_delimiter,
        right_delimiter=candidate.left_delimiter if reverse else candidate.right_delimiter,
        enabled=True,
        confidence=candidate.confidence,
        source=candidate.source,
        reason=candidate.reason,
        relationship_type=relationship_type,
        risk_note=_risk_note(relationship_type),
    )


def _reverse_relationship_type(value: str) -> str:
    if value == "one_to_many":
        return "many_to_one"
    if value == "many_to_one":
        return "one_to_many"
    return value


def _rule_candidates(
    repository: DatasetStoreRepository,
    group: StoredDatasetGroup,
    *,
    records_by_dataset: dict[UUID, list[dict[str, Any]]],
) -> tuple[DatasetRelationshipCandidate, ...]:
    candidates: list[DatasetRelationshipCandidate] = []
    dataset_ids = tuple(group.dataset_ids)
    for index, left_id in enumerate(dataset_ids):
        for right_id in dataset_ids[index + 1 :]:
            left_entity = _entity_type_from_context(repository, left_id, records_by_dataset[left_id])
            right_entity = _entity_type_from_context(repository, right_id, records_by_dataset[right_id])
            oriented_left, oriented_right = _orient_pair(left_id, right_id, left_entity, right_entity)
            response = suggest_dataset_joins(
                repository,
                dataset_id=oriented_left,
                additional_dataset_ids=(oriented_right,),
                records_by_dataset=records_by_dataset,
            )
            for suggestion in response.suggestions[:3]:
                left_records = records_by_dataset[suggestion.left_dataset_id]
                right_records = records_by_dataset[suggestion.right_dataset_id]
                relationship_type = _relationship_type(
                    left_records,
                    right_records,
                    suggestion.left_column,
                    suggestion.right_column,
                    left_value_mode=suggestion.left_value_mode,
                    right_value_mode=suggestion.right_value_mode,
                )
                candidates.append(
                    DatasetRelationshipCandidate(
                        left_dataset_id=suggestion.left_dataset_id,
                        right_dataset_id=suggestion.right_dataset_id,
                        left_column=suggestion.left_column,
                        right_column=suggestion.right_column,
                        join_type=suggestion.join_type,
                        left_value_mode=suggestion.left_value_mode,
                        right_value_mode=suggestion.right_value_mode,
                        left_delimiter=suggestion.left_delimiter,
                        right_delimiter=suggestion.right_delimiter,
                        confidence=suggestion.score,
                        source="rules",
                        reason=suggestion.reason,
                        left_type=suggestion.left_type,
                        right_type=suggestion.right_type,
                        left_role=suggestion.left_role,
                        right_role=suggestion.right_role,
                        estimated_match_rate=suggestion.estimated_match_rate,
                        relationship_type=relationship_type,
                        risk_note=_risk_note(relationship_type),
                    )
                )
    provider = get_semantic_embedding_provider()
    column_names = list(dict.fromkeys([name for item in candidates for name in (item.left_column, item.right_column)]))
    vectors = provider.encode(column_names)
    vector_by_name = dict(zip(column_names, vectors, strict=True))
    enriched: list[DatasetRelationshipCandidate] = []
    for candidate in candidates:
        left_vector = vector_by_name.get(candidate.left_column, ())
        right_vector = vector_by_name.get(candidate.right_column, ())
        embedding_available = bool(left_vector and right_vector)
        embedding_score = cosine_similarity(left_vector, right_vector) if embedding_available else 0.0
        confidence = (
            min(1.0, candidate.confidence * 0.9 + embedding_score * 0.1)
            if candidate.estimated_match_rate > 0 and embedding_available
            else candidate.confidence
        )
        enriched.append(candidate.model_copy(update={"confidence": confidence, "embedding_score": embedding_score, "score_breakdown": {"rules_and_samples": candidate.confidence, "embedding": embedding_score}, "embedding_model_revision": provider.model_revision}))
    enriched.sort(key=lambda item: item.confidence, reverse=True)
    return tuple(enriched[:30])


def _llm_candidates(
    *,
    group: StoredDatasetGroup,
    compact_context: dict[str, Any],
    rule_candidates: tuple[DatasetRelationshipCandidate, ...],
    records_by_dataset: dict[UUID, list[dict[str, Any]]],
    router: AnalysisModelRouter,
) -> tuple[DatasetRelationshipCandidate, ...]:
    settings = get_settings()
    response = router.complete(
        provider=settings.planner_llm_provider,
        model=None,
        temperature=0.1,
        max_tokens=1200,
        metadata={"task": "dataset_group_relationship_suggestions", "group_id": str(group.id)},
        messages=[
            {
                "role": "system",
                "content": (
                    "You infer tabular dataset relationships from compressed schema summaries only. "
                    "Return strict JSON. Do not invent columns that are absent from the provided schema. "
                    f"{UNTRUSTED_INPUT_NOTICE}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "suggest_relationships",
                        "group": {"id": str(group.id), "name": group.name},
                        "compact_context": compact_context,
                        "rule_candidates": [candidate.model_dump(mode="json") for candidate in rule_candidates[:12]],
                        "output_contract": {
                            "relationships": [
                                {
                                    "left_dataset_id": "uuid",
                                    "right_dataset_id": "uuid",
                                    "left_column": "column",
                                    "right_column": "column",
                                    "relationship_type": "one_to_one|one_to_many|many_to_one|many_to_many|unknown",
                                    "confidence": 0.0,
                                    "reason": "short reason",
                                    "risk_note": "short risk",
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    payload = _extract_json_object(response.content)
    raw_relationships = payload.get("relationships") if isinstance(payload, dict) else None
    if not isinstance(raw_relationships, list):
        return ()
    dataset_ids = {str(dataset_id) for dataset_id in group.dataset_ids}
    validated: list[DatasetRelationshipCandidate] = []
    for item in raw_relationships[:12]:
        if not isinstance(item, dict):
            continue
        left_id = str(item.get("left_dataset_id") or "")
        right_id = str(item.get("right_dataset_id") or "")
        left_column = str(item.get("left_column") or "")
        right_column = str(item.get("right_column") or "")
        if left_id not in dataset_ids or right_id not in dataset_ids or left_id == right_id:
            continue
        if not _column_exists(compact_context, left_id, left_column) or not _column_exists(compact_context, right_id, right_column):
            continue
        left_uuid = UUID(left_id)
        right_uuid = UUID(right_id)
        left_records = records_by_dataset[left_uuid]
        right_records = records_by_dataset[right_uuid]
        match_rate, left_value_mode, right_value_mode, left_delimiter, right_delimiter = infer_join_value_match(
            left_records,
            right_records,
            left_column,
            right_column,
        )
        if match_rate <= 0 and float(item.get("confidence") or 0) < 0.75:
            continue
        relationship_type = _relationship_type(
            left_records,
            right_records,
            left_column,
            right_column,
            left_value_mode=left_value_mode,
            right_value_mode=right_value_mode,
        )
        confidence = max(0.0, min(float(item.get("confidence") or 0), 1.0))
        confidence = max(0.0, min(confidence * 0.75 + min(match_rate, 1) * 0.25, 1.0))
        validated.append(
            DatasetRelationshipCandidate(
                left_dataset_id=left_uuid,
                right_dataset_id=right_uuid,
                left_column=left_column,
                right_column=right_column,
                join_type="left",
                left_value_mode=left_value_mode,
                right_value_mode=right_value_mode,
                left_delimiter=left_delimiter,
                right_delimiter=right_delimiter,
                confidence=confidence,
                source="validated_llm",
                reason=str(item.get("reason") or "LLM semantic relationship candidate"),
                estimated_match_rate=match_rate,
                relationship_type=relationship_type,
                risk_note=str(item.get("risk_note") or _risk_note(relationship_type)),
            )
        )
    return tuple(validated)


def _compact_group_context(
    repository: DatasetStoreRepository,
    group: StoredDatasetGroup,
    *,
    records_by_dataset: dict[UUID, list[dict[str, Any]]],
) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    for dataset_id in group.dataset_ids:
        dataset = repository.get_dataset(dataset_id)
        records = records_by_dataset[dataset_id]
        profile = apply_column_metadata_to_profile(
            DatasetProfiler().profile(dataset_id=dataset_id, records=records),
            repository.list_column_metadata(dataset_id),
        )
        metadata = {
            str(item.get("column_name")): item
            for item in repository.list_column_metadata(dataset_id)
            if item.get("column_name")
        }
        tables.append(
            {
                "dataset_id": str(dataset_id),
                "name": dataset.name,
                "source_type": dataset.source_type,
                "entity_type": _entity_type(dataset.name, tuple(column.name for column in profile.columns)),
                "row_count": repository.count_analysis_records(dataset_id),
                "column_count": profile.column_count,
                "columns": [
                    {
                        "name": column.name,
                        "type": column.dtype,
                        "role": str(metadata.get(column.name, {}).get("role") or ""),
                        "missing_count": column.missing_count,
                        "distinct_count": column.distinct_count,
                        "sample_values": _sample_values(records, column.name, limit=5),
                    }
                    for column in profile.columns[:80]
                ],
                "preview_records": compact_prompt_records(records, max_rows=5, max_columns=60),
            }
        )
    return {"tables": tables, "max_preview_rows_per_table": 5, "contains_full_records": False}


def _merge_candidates(
    rule_candidates: tuple[DatasetRelationshipCandidate, ...],
    llm_candidates: tuple[DatasetRelationshipCandidate, ...],
) -> tuple[DatasetRelationshipCandidate, ...]:
    merged: dict[tuple[str, str, str, str], DatasetRelationshipCandidate] = {}
    for candidate in (*rule_candidates, *llm_candidates):
        key = (
            str(candidate.left_dataset_id),
            str(candidate.right_dataset_id),
            candidate.left_column,
            candidate.right_column,
        )
        existing = merged.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            merged[key] = candidate
    return tuple(sorted(merged.values(), key=lambda item: item.confidence, reverse=True)[:30])


def _entity_type_from_context(
    repository: DatasetStoreRepository,
    dataset_id: UUID,
    records: list[dict[str, Any]],
) -> str:
    dataset = repository.get_dataset(dataset_id)
    columns = tuple(records[0].keys()) if records else ()
    return _entity_type(dataset.name, columns)


def _entity_type(name: str, columns: tuple[str, ...]) -> str:
    lowered = name.lower()
    normalized_columns = {_normalize(column) for column in columns}
    if "all_data" in lowered or ("orderid" in normalized_columns and len(columns) > 18):
        return "wide"
    if any(token in lowered for token in ("item", "line", "detail", "payment", "review")):
        return "bridge" if any(token in lowered for token in ("item", "payment", "review")) else "fact"
    if any(token in lowered for token in ("order", "sale", "transaction", "invoice")):
        return "fact"
    if any(token in lowered for token in ("translation", "lookup", "category")):
        return "lookup"
    if any(token in lowered for token in ("customer", "product", "seller", "geo", "region", "user")):
        return "dimension"
    return "unknown"


def _orient_pair(left_id: UUID, right_id: UUID, left_entity: str, right_entity: str) -> tuple[UUID, UUID]:
    left_rank = _entity_rank(left_entity)
    right_rank = _entity_rank(right_entity)
    return (left_id, right_id) if left_rank >= right_rank else (right_id, left_id)


def _entity_rank(entity_type: str) -> int:
    return {"wide": 4, "fact": 3, "bridge": 2, "unknown": 1, "dimension": 0, "lookup": 0}.get(entity_type, 1)


def _relationship_type(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    left_column: str,
    right_column: str,
    *,
    left_value_mode: str = "scalar",
    right_value_mode: str = "scalar",
) -> str:
    left_ratio = _distinct_ratio(left_records, left_column)
    right_ratio = _distinct_ratio(right_records, right_column)
    if left_value_mode == "delimited" and right_value_mode == "scalar":
        return "many_to_one" if right_ratio >= 0.9 else "many_to_many"
    if right_value_mode == "delimited" and left_value_mode == "scalar":
        return "one_to_many" if left_ratio >= 0.9 else "many_to_many"
    if left_ratio >= 0.9 and right_ratio >= 0.9:
        return "one_to_one"
    if left_ratio < 0.9 <= right_ratio:
        return "many_to_one"
    if left_ratio >= 0.9 > right_ratio:
        return "one_to_many"
    if left_ratio > 0 and right_ratio > 0:
        return "many_to_many"
    return "unknown"


def _risk_note(relationship_type: str) -> str:
    if relationship_type in {"one_to_many", "many_to_many"}:
        return "Join may multiply rows; aggregations such as amount or quantity can be duplicated."
    if relationship_type == "many_to_one":
        return "Usually safe for adding dimension attributes, but verify unmatched rows."
    return ""


def _distinct_ratio(records: list[dict[str, Any]], column: str) -> float:
    values = [str(record.get(column)).strip().lower() for record in records[:1000] if record.get(column) not in (None, "")]
    if not values:
        return 0
    return len(set(values)) / len(values)


def _estimated_match_rate(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    left_column: str,
    right_column: str,
) -> float:
    left_values = {str(record.get(left_column)).strip().lower() for record in left_records[:500] if record.get(left_column) not in (None, "")}
    right_values = {str(record.get(right_column)).strip().lower() for record in right_records[:500] if record.get(right_column) not in (None, "")}
    if not left_values or not right_values:
        return 0
    return len(left_values & right_values) / max(len(left_values), 1)


def _sample_values(records: list[dict[str, Any]], column: str, *, limit: int) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records[:500]:
        value = record.get(column)
        if value is None or str(value).strip() == "":
            continue
        text = str(value).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text[:80])
        if len(values) >= limit:
            break
    return tuple(values)


def _column_exists(compact_context: dict[str, Any], dataset_id: str, column_name: str) -> bool:
    tables = compact_context.get("tables")
    if not isinstance(tables, list):
        return False
    for table in tables:
        if not isinstance(table, dict) or str(table.get("dataset_id")) != dataset_id:
            continue
        columns = table.get("columns")
        if not isinstance(columns, list):
            return False
        return any(isinstance(column, dict) and column.get("name") == column_name for column in columns)
    return False


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.replace("ID", "id").lower())


def _extract_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
