from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePath
from uuid import UUID

from app.analysis.query_intent import infer_query_intent
from app.analysis.services import DatasetProfiler
from app.schemas.analysis import DatasetJoinConfig
from app.storage.dataset_store import DatasetStoreRepository
from app.storage.models import StoredDataset

_DATASET_NOISE_TOKENS = {
    "csv",
    "data",
    "dataset",
    "datasets",
    "file",
    "olist",
    "table",
    "txt",
}
_STRICT_SCOPE_RE = re.compile(
    r"(?:仅|只)(?:使用|用|限于|限用)|only\s+(?:use|using)",
    re.IGNORECASE,
)
_NEGATED_DATASET_SCOPE_RE = re.compile(
    r"(?:不要|不得|请勿|禁止)\s*(?:使用|用|采用|读取|包括)\s*"
    r"[^,，;；。.!?！？\n]*"
    r"|(?:(?:do\s+not|don't|never)\s+(?:use|include|read)|without|excluding?)\s+"
    r"[^,，;；。.!?！？\n]*",
    re.IGNORECASE,
)
_GENERIC_DATASET_ALIAS_TOKENS = {
    "category",
    "name",
    "order",
    "orders",
    "product",
    "products",
}


@dataclass(frozen=True)
class AnalysisDatasetScope:
    dataset_id: UUID
    additional_dataset_ids: tuple[UUID, ...]
    join_plan: tuple[DatasetJoinConfig, ...]
    referenced_dataset_ids: tuple[UUID, ...] = ()
    allowlist_dataset_ids: tuple[UUID, ...] = ()
    denylist_dataset_ids: tuple[UUID, ...] = ()


def resolve_analysis_dataset_scope(
    repository: DatasetStoreRepository,
    *,
    question: str,
    dataset_id: UUID,
    additional_dataset_ids: tuple[UUID, ...],
    join_plan: tuple[DatasetJoinConfig, ...],
) -> AnalysisDatasetScope:
    """Reduce a relationship tree to the datasets needed by the question.

    Explicit dataset names are resolved from metadata first. Only that bounded
    set is sampled when the question names tables, so an explicit three-table
    request cannot pull records from unrelated members of a larger package.
    Field-source inference then adds required metric, filter and dimension
    owners. The resulting terminals are connected through the smallest subtree
    of the submitted relationship plan and re-rooted at the metric owner when
    one is known.
    """

    all_dataset_ids = _all_dataset_ids(
        dataset_id=dataset_id,
        additional_dataset_ids=additional_dataset_ids,
        join_plan=join_plan,
    )
    datasets = {item: repository.get_dataset(item) for item in all_dataset_ids}
    negated_scope_spans = _negated_dataset_scope_spans(question)
    strict_allowlist = bool(_STRICT_SCOPE_RE.search(question))
    if strict_allowlist:
        submitted_ids = set(all_dataset_ids)
        missing_declared_datasets = tuple(
            dataset
            for dataset in repository.list_datasets()
            if dataset.id not in submitted_ids
            and _dataset_reference_polarity(
                question,
                dataset.name,
                negated_scope_spans=negated_scope_spans,
            )[0]
        )
        if missing_declared_datasets:
            missing_names = ", ".join(
                sorted(dataset.name for dataset in missing_declared_datasets)
            )
            raise ValueError(
                "Strict dataset allowlist references datasets that were not submitted: "
                f"{missing_names}. Select the dataset package or submit its relationship plan."
            )
    allowlist_ids: list[UUID] = []
    denylist_ids: list[UUID] = []
    for item in all_dataset_ids:
        positive, negative = _dataset_reference_polarity(
            question,
            datasets[item].name,
            negated_scope_spans=negated_scope_spans,
        )
        if positive:
            allowlist_ids.append(item)
        if negative:
            denylist_ids.append(item)
    conflicting_ids = set(allowlist_ids) & set(denylist_ids)
    if conflicting_ids:
        conflicting_names = ", ".join(
            sorted(datasets[item].name for item in conflicting_ids)
        )
        raise ValueError(
            "Question both includes and excludes the same dataset(s): "
            f"{conflicting_names}."
        )
    explicit_ids = tuple(allowlist_ids)
    if strict_allowlist and not explicit_ids:
        raise ValueError(
            "Question declares an allowlist but no submitted dataset name could be resolved."
        )
    denied_ids = tuple(
        dict.fromkeys(
            (
                *denylist_ids,
                *(
                    item
                    for item in all_dataset_ids
                    if strict_allowlist and item not in set(explicit_ids)
                ),
            )
        )
    )
    available_dataset_ids = tuple(
        item for item in all_dataset_ids if item not in set(denied_ids)
    )
    if not available_dataset_ids:
        raise ValueError("Question excludes every submitted dataset.")

    inference_ids = explicit_ids or available_dataset_ids
    intent, column_sources = _infer_field_sources(
        repository,
        question=question,
        datasets=tuple(datasets[item] for item in inference_ids),
    )
    required_columns = tuple(
        dict.fromkeys(
            (
                *intent.required_dimensions,
                *((intent.required_metric,) if intent.required_metric else ()),
                *(item.column for item in intent.aggregations if item.column),
                *(item.column for item in intent.filters),
            )
        )
    )
    field_source_ids = tuple(
        dict.fromkeys(
            column_sources[column]
            for column in required_columns
            if column in column_sources
        )
    )
    referenced_ids = tuple(dict.fromkeys((*explicit_ids, *field_source_ids)))

    # A casual mention of one table is not enough evidence to discard the
    # submitted relationship context. Prune only for an explicit multi-table
    # scope, strict "only use" wording, or requirements spanning 2+ sources.
    scope_is_explicit = bool(denied_ids) or len(explicit_ids) >= 2 or strict_allowlist
    if not scope_is_explicit and len(field_source_ids) < 2:
        return AnalysisDatasetScope(
            dataset_id=dataset_id,
            additional_dataset_ids=_dedupe_ids(additional_dataset_ids, exclude=dataset_id),
            join_plan=join_plan,
            referenced_dataset_ids=referenced_ids,
            allowlist_dataset_ids=explicit_ids,
            denylist_dataset_ids=denied_ids,
        )
    if not referenced_ids:
        referenced_ids = explicit_ids or available_dataset_ids

    metric_source_id = (
        column_sources.get(intent.required_metric) if intent.required_metric else None
    )
    root_id = metric_source_id or (
        dataset_id if dataset_id in referenced_ids else referenced_ids[0]
    )
    if not join_plan:
        return AnalysisDatasetScope(
            dataset_id=root_id,
            additional_dataset_ids=tuple(
                item for item in referenced_ids if item != root_id
            ),
            join_plan=(),
            referenced_dataset_ids=referenced_ids,
            allowlist_dataset_ids=explicit_ids,
            denylist_dataset_ids=denied_ids,
        )
    denied_id_set = set(denied_ids)
    allowed_join_plan = tuple(
        item
        for item in join_plan
        if item.left_dataset_id not in denied_id_set
        and item.right_dataset_id not in denied_id_set
    )
    scoped_plan, ordered_ids = _minimal_oriented_subtree(
        allowed_join_plan,
        root_id=root_id,
        terminal_ids=referenced_ids,
    )
    if not denied_ids and set(ordered_ids) == set(all_dataset_ids):
        return AnalysisDatasetScope(
            dataset_id=dataset_id,
            additional_dataset_ids=_dedupe_ids(additional_dataset_ids, exclude=dataset_id),
            join_plan=join_plan,
            referenced_dataset_ids=referenced_ids,
            allowlist_dataset_ids=explicit_ids,
            denylist_dataset_ids=denied_ids,
        )

    return AnalysisDatasetScope(
        dataset_id=root_id,
        additional_dataset_ids=tuple(item for item in ordered_ids if item != root_id),
        join_plan=scoped_plan,
        referenced_dataset_ids=referenced_ids,
        allowlist_dataset_ids=explicit_ids,
        denylist_dataset_ids=denied_ids,
    )


def _infer_field_sources(
    repository: DatasetStoreRepository,
    *,
    question: str,
    datasets: tuple[StoredDataset, ...],
):
    qualified_records: list[dict[str, object]] = []
    column_sources: dict[str, UUID] = {}
    used_slugs: set[str] = set()
    for dataset in datasets:
        slug = _unique_dataset_slug(dataset, used_slugs)
        samples = repository.sample_analysis_records(dataset.id, limit=50)
        for record in samples:
            qualified: dict[str, object] = {}
            for column, value in record.items():
                name = f"{slug}__{column}"
                qualified[name] = value
                column_sources[name] = dataset.id
            qualified_records.append(qualified)
    if not qualified_records:
        profile = DatasetProfiler().profile(dataset_id=datasets[0].id, records=[])
    else:
        profile = DatasetProfiler().profile(
            dataset_id=datasets[0].id,
            records=qualified_records,
        )
    return infer_query_intent(question, profile), column_sources


def _minimal_oriented_subtree(
    join_plan: tuple[DatasetJoinConfig, ...],
    *,
    root_id: UUID,
    terminal_ids: tuple[UUID, ...],
) -> tuple[tuple[DatasetJoinConfig, ...], tuple[UUID, ...]]:
    if len(set(terminal_ids)) <= 1:
        return (), (root_id,)

    adjacency: dict[UUID, list[tuple[int, UUID]]] = {}
    for index, config in enumerate(join_plan):
        adjacency.setdefault(config.left_dataset_id, []).append(
            (index, config.right_dataset_id)
        )
        adjacency.setdefault(config.right_dataset_id, []).append(
            (index, config.left_dataset_id)
        )

    parents: dict[UUID, tuple[UUID, int] | None] = {root_id: None}
    queue: deque[UUID] = deque((root_id,))
    while queue:
        current = queue.popleft()
        for edge_index, neighbor in adjacency.get(current, ()):
            if neighbor in parents:
                continue
            parents[neighbor] = (current, edge_index)
            queue.append(neighbor)

    missing = tuple(item for item in terminal_ids if item not in parents)
    if missing:
        raise ValueError(
            "Question-referenced datasets are not connected by the submitted relationship plan."
        )

    selected_edges: set[int] = set()
    for terminal_id in terminal_ids:
        current = terminal_id
        while parents[current] is not None:
            parent, edge_index = parents[current]
            selected_edges.add(edge_index)
            current = parent

    scoped_adjacency: dict[UUID, list[tuple[int, UUID]]] = {}
    for edge_index in sorted(selected_edges):
        config = join_plan[edge_index]
        scoped_adjacency.setdefault(config.left_dataset_id, []).append(
            (edge_index, config.right_dataset_id)
        )
        scoped_adjacency.setdefault(config.right_dataset_id, []).append(
            (edge_index, config.left_dataset_id)
        )

    oriented: list[DatasetJoinConfig] = []
    ordered_ids: list[UUID] = [root_id]
    visited = {root_id}
    queue = deque((root_id,))
    while queue:
        current = queue.popleft()
        for edge_index, neighbor in scoped_adjacency.get(current, ()):
            if neighbor in visited:
                continue
            config = join_plan[edge_index]
            oriented.append(
                config
                if config.left_dataset_id == current
                else _reverse_join_config(config)
            )
            visited.add(neighbor)
            ordered_ids.append(neighbor)
            queue.append(neighbor)
    return tuple(oriented), tuple(ordered_ids)


def _reverse_join_config(config: DatasetJoinConfig) -> DatasetJoinConfig:
    return DatasetJoinConfig(
        left_dataset_id=config.right_dataset_id,
        right_dataset_id=config.left_dataset_id,
        left_column=config.right_column,
        right_column=config.left_column,
        join_type=config.join_type,
        left_value_mode=config.right_value_mode,
        right_value_mode=config.left_value_mode,
        left_delimiter=config.right_delimiter,
        right_delimiter=config.left_delimiter,
    )


def _all_dataset_ids(
    *,
    dataset_id: UUID,
    additional_dataset_ids: tuple[UUID, ...],
    join_plan: tuple[DatasetJoinConfig, ...],
) -> tuple[UUID, ...]:
    return tuple(
        dict.fromkeys(
            (
                dataset_id,
                *additional_dataset_ids,
                *(item for config in join_plan for item in (
                    config.left_dataset_id,
                    config.right_dataset_id,
                )),
            )
        )
    )


def _dataset_reference_polarity(
    question: str,
    dataset_name: str,
    *,
    negated_scope_spans: tuple[tuple[int, int], ...],
) -> tuple[bool, bool]:
    folded = question.casefold()
    positive = False
    negative = False
    for alias in _dataset_aliases(dataset_name):
        for matched in _alias_pattern(alias).finditer(folded):
            is_negative = any(
                start <= matched.start() and matched.end() <= end
                for start, end in negated_scope_spans
            )
            positive = positive or not is_negative
            negative = negative or is_negative
    return positive, negative


def _negated_dataset_scope_spans(question: str) -> tuple[tuple[int, int], ...]:
    return tuple(match.span() for match in _NEGATED_DATASET_SCOPE_RE.finditer(question))


def _dataset_aliases(dataset_name: str) -> tuple[str, ...]:
    stem = PurePath(dataset_name).stem.casefold()
    tokens = tuple(
        token
        for token in re.split(r"[^a-z0-9\u3400-\u9fff]+", stem)
        if token
    )
    meaningful = tuple(token for token in tokens if token not in _DATASET_NOISE_TOKENS)
    aliases: list[str] = []
    if meaningful:
        aliases.append("_".join(meaningful))
        aliases.extend(
            token
            for token in meaningful
            if len(token) >= 4 and token not in _GENERIC_DATASET_ALIAS_TOKENS
        )
    normalized_stem = "_".join(tokens)
    if normalized_stem and normalized_stem not in _DATASET_NOISE_TOKENS:
        aliases.append(normalized_stem)
    return tuple(
        item for item in dict.fromkeys(aliases) if len(item) >= 3 or _has_cjk(item)
    )


def _contains_alias(question: str, alias: str) -> bool:
    return _alias_pattern(alias).search(question) is not None


def _alias_pattern(alias: str) -> re.Pattern[str]:
    if _has_cjk(alias):
        return re.compile(re.escape(alias))
    parts = tuple(part for part in alias.split("_") if part)
    pattern = r"(?<![a-z0-9])" + r"[\s_.-]+".join(
        re.escape(part) for part in parts
    ) + r"(?![a-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def _has_cjk(value: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", value) is not None


def _unique_dataset_slug(dataset: StoredDataset, used_slugs: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "_", dataset.name.casefold()).strip("_")
    slug = slug[:40] or "dataset"
    if slug in used_slugs:
        slug = f"{slug}_{str(dataset.id)[:8]}"
    used_slugs.add(slug)
    return slug


def _dedupe_ids(items: Iterable[UUID], *, exclude: UUID) -> tuple[UUID, ...]:
    return tuple(dict.fromkeys(item for item in items if item != exclude))
