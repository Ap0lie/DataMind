from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any
from uuid import UUID

from app.analysis.intent_guard import validate_intent
from app.analysis.intent_prompts import compiler_messages, repair_messages
from app.analysis.model_router import AnalysisModelRouter
from app.analysis.query_intent import (
    infer_query_intent,
    infer_source_aggregations,
    negated_grouping_columns,
)
from app.analysis.services import DatasetProfiler
from app.core.settings import Settings, get_settings
from app.schemas.analysis_intent import (
    AnalysisIntentSpec,
    FieldBinding,
    IntentAggregation,
    IntentClause,
    IntentCompilationAttempt,
    IntentCompilationResult,
    IntentFilter,
    IntentGuardResult,
    IntentSourceSpan,
    RelationshipConstraint,
)
from app.storage.dataset_store import DatasetStoreRepository

_NEGATION_RE = re.compile(
    r"不要|不得|请勿|禁止|严禁|排除|忽略|无需|"
    r"\b(?:do\s+not|don't|never|without|excluding?)\b",
    re.IGNORECASE,
)
_STRICT_SCOPE_RE = re.compile(r"(?:仅|只)(?:使用|用|限于|限用)|only\s+(?:use|using)", re.I)
_RELATIONSHIP_RE = re.compile(
    r"[^,，;；。!?！？\n]*?(?:连接|关联|join)"
    r"[^,，;；。!?！？\n]*",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(r"[^,，;；。!?！？\n]+")
_GENERIC_ASSET_TOKENS = {
    "category",
    "name",
    "order",
    "orders",
    "product",
    "products",
}


@dataclass(frozen=True)
class IntentCompilationContext:
    assets: tuple[dict[str, Any], ...]
    profile: Any
    bindings: dict[str, FieldBinding]


class IntentCompilationHarness:
    def __init__(
        self,
        *,
        model_router: AnalysisModelRouter | None,
        settings: Settings | None = None,
    ) -> None:
        self._model_router = model_router
        self._settings = settings or get_settings()

    def compile(
        self,
        *,
        question: str,
        context: IntentCompilationContext,
        semantic_plan: dict[str, Any] | None = None,
    ) -> IntentCompilationResult:
        baseline = _deterministic_intent(question, context)
        baseline_guard = validate_intent(
            baseline,
            question=question,
            assets=context.assets,
        )
        attempts: list[IntentCompilationAttempt] = []
        model_intent: AnalysisIntentSpec | None = None
        model_guard: IntentGuardResult | None = None

        needs_model = (
            self._settings.intent_compiler_enabled
            and (
                _needs_model_compilation(question, baseline)
                or baseline_guard.status != "passed"
            )
        )
        should_compile = needs_model and self._model_router is not None
        if should_compile:
            messages = compiler_messages(
                question=question,
                assets=context.assets,
                semantic_plan=semantic_plan,
            )
            history: list[dict[str, Any]] = []
            for attempt_number in range(1, self._settings.intent_compiler_max_repairs + 2):
                provider: str | None = None
                model: str | None = None
                content = ""
                try:
                    response = self._model_router.complete(
                        messages=messages,
                        provider=self._settings.intent_compiler_provider,
                        temperature=0.0,
                        max_tokens=1800,
                        metadata={
                            "agent": "intent_compiler",
                            "optional_stage": self._settings.intent_compiler_mode == "shadow",
                            "timeout_seconds": self._settings.intent_compiler_timeout_seconds,
                            "structured_output": True,
                        },
                    )
                    provider = response.provider
                    model = response.model
                    content = str(response.content or "")
                    candidate = _parse_intent(content, question=question)
                    guard = validate_intent(
                        candidate,
                        question=question,
                        assets=context.assets,
                        baseline=baseline,
                    )
                    model_guard = guard
                    attempts.append(
                        IntentCompilationAttempt(
                            attempt=attempt_number,
                            status="succeeded" if guard.status == "passed" else "failed",
                            provider=provider,
                            model=model,
                            error=None if guard.status == "passed" else _guard_summary(guard),
                            guard=guard,
                        )
                    )
                    if guard.status == "passed":
                        model_intent = candidate.model_copy(update={"source": "llm"})
                        break
                    history.append(
                        {
                            "attempt": attempt_number,
                            "error": _guard_summary(guard),
                        }
                    )
                    messages = repair_messages(
                        question=question,
                        assets=context.assets,
                        invalid_content=content,
                        validation=guard,
                        prior_attempts=tuple(history),
                    )
                except Exception as exc:
                    error = str(exc)
                    attempts.append(
                        IntentCompilationAttempt(
                            attempt=attempt_number,
                            status="failed",
                            provider=provider,
                            model=model,
                            error=error,
                        )
                    )
                    history.append({"attempt": attempt_number, "error": error})
                    repair_guard = model_guard or IntentGuardResult(
                        status="repairable",
                        confidence=0,
                    )
                    messages = repair_messages(
                        question=question,
                        assets=context.assets,
                        invalid_content=content,
                        validation=repair_guard,
                        prior_attempts=tuple(history),
                    )

        selected = baseline
        selected_guard = baseline_guard
        if self._settings.intent_compiler_mode == "enforce" and (
            needs_model or baseline_guard.status != "passed"
        ):
            if model_intent is not None and model_guard is not None:
                selected = model_intent
                selected_guard = model_guard
            else:
                reasons = tuple(
                    dict.fromkeys(
                        attempt.error for attempt in attempts if attempt.error
                    )
                ) or ("Intent compiler could not produce a guarded interpretation.",)
                selected = baseline.model_copy(
                    update={
                        "requires_confirmation": True,
                        "confirmation_reasons": reasons,
                    }
                )
                selected_guard = IntentGuardResult(
                    status="confirmation_required",
                    issues=baseline_guard.issues,
                    confidence=min(baseline.confidence, 0.49),
                )

        return IntentCompilationResult(
            intent=selected,
            validation=selected_guard,
            attempts=tuple(attempts),
            mode=self._settings.intent_compiler_mode,
            model_intent=model_intent,
            metadata={
                "model_attempted": should_compile,
                "model_required": needs_model,
                "shadow_model_status": model_guard.status if model_guard else None,
                "baseline_status": baseline_guard.status,
            },
        )


def build_intent_compilation_context(
    repository: DatasetStoreRepository,
    *,
    dataset_ids: tuple[UUID, ...],
    authorized_dataset_ids: tuple[UUID, ...] = (),
) -> IntentCompilationContext:
    assets: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    bindings: dict[str, FieldBinding] = {}
    multiple = len(dataset_ids) > 1
    profile_dataset_ids = set(dataset_ids)
    for dataset_id in tuple(dict.fromkeys((*dataset_ids, *authorized_dataset_ids))):
        dataset = repository.get_dataset(dataset_id)
        include_schema = dataset_id in profile_dataset_ids
        records = (
            repository.sample_analysis_records(dataset_id, limit=30)
            if include_schema
            else ()
        )
        metadata = (
            {
                item["column_name"]: item
                for item in repository.list_column_metadata(dataset_id)
            }
            if include_schema
            else {}
        )
        columns = tuple(
            dict.fromkeys(
                (
                    *metadata,
                    *(str(column) for record in records for column in record),
                )
            )
        )
        slug = _dataset_slug(dataset.name)
        column_payload: list[dict[str, Any]] = []
        for column in columns:
            details = metadata.get(column, {})
            reference = f"{slug}__{column}" if multiple else column
            binding = FieldBinding(
                column=reference,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                dtype=str(details.get("override_type") or details.get("inferred_type") or "unknown"),
                role=str(details.get("role") or "") or None,
            )
            bindings[reference] = binding
            column_payload.append(
                {
                    "name": column,
                    "reference": reference,
                    "dtype": binding.dtype,
                    "role": binding.role,
                    "description": str(details.get("description") or "")[:240],
                }
            )
        for record in records:
            profile_records.append(
                {
                    (f"{slug}__{column}" if multiple else str(column)): value
                    for column, value in record.items()
                }
            )
        assets.append(
            {
                "dataset_id": dataset_id,
                "name": dataset.name,
                "sampled_row_count": len(records),
                "column_count": len(columns),
                "columns": tuple(column_payload),
            }
        )
    profile = DatasetProfiler().profile(
        dataset_id=dataset_ids[0],
        records=profile_records,
    )
    return IntentCompilationContext(
        assets=tuple(assets),
        profile=profile,
        bindings=bindings,
    )


def _deterministic_intent(
    question: str,
    context: IntentCompilationContext,
) -> AnalysisIntentSpec:
    semantic_question = _mask_relationship_asset_mentions(question, context.assets)
    inferred = infer_query_intent(semantic_question, context.profile)
    required_metric = _binding(context, inferred.required_metric)
    required_dimensions = _bindings(context, inferred.required_dimensions)
    candidate_metrics = _bindings(context, inferred.candidate_metrics)
    candidate_dimensions = _bindings(context, inferred.candidate_dimensions)
    clauses: list[IntentClause] = []

    if required_metric:
        clauses.append(
            _field_clause(question, "metric", "required", required_metric, "metric-1")
        )
    clauses.extend(
        _field_clause(question, "dimension", "required", binding, f"dimension-{index}")
        for index, binding in enumerate(required_dimensions, start=1)
    )
    forbidden_dimensions = _bindings(
        context,
        negated_grouping_columns(question, tuple(context.profile.categorical_columns)),
    )
    clauses.extend(
        _field_clause(question, "dimension", "forbidden", binding, f"forbidden-dimension-{index}")
        for index, binding in enumerate(forbidden_dimensions, start=1)
    )

    source_aggregations = infer_source_aggregations(
        question,
        tuple(
            (
                str(asset["name"]),
                tuple(str(item["name"]) for item in asset.get("columns", ())),
            )
            for asset in context.assets
        ),
    )
    aggregation_items = tuple(
        dict.fromkeys(
            (
                item.operation,
                item.column,
                item.alias,
            )
            for item in (*source_aggregations, *inferred.aggregations)
        )
    )
    aggregations = tuple(
        IntentAggregation(
            operation=operation,
            field=_binding(context, column),
            alias=alias,
        )
        for operation, column, alias in aggregation_items
    )
    filters = tuple(
        IntentFilter(
            field=_binding(context, item.column)
            or FieldBinding(column=item.column, confidence=0),
            operator=item.operator,
            value=item.value,
        )
        for item in inferred.filters
    )
    allowlist, denylist, dataset_clauses = _dataset_scope(question, context.assets)
    clauses.extend(dataset_clauses)
    relationships = _relationship_constraints(question, context.assets)
    clauses.extend(
        IntentClause(
            clause_id=f"relationship-{index}",
            kind="relationship",
            polarity=item.polarity,
            concept=item.operation,
            source_span=item.source_span,
            confidence=1,
        )
        for index, item in enumerate(relationships, start=1)
    )
    confidence = 1.0 if not any(binding.confidence < 1 for binding in _all_bindings(
        required_metric,
        required_dimensions,
        filters,
    )) else 0.7
    return AnalysisIntentSpec(
        question=question,
        clauses=tuple(clauses),
        required_metric=required_metric,
        candidate_metrics=candidate_metrics,
        required_dimensions=required_dimensions,
        candidate_dimensions=candidate_dimensions,
        aggregations=aggregations,
        filters=filters,
        derived_metrics=inferred.derived_metrics,
        dataset_allowlist=allowlist,
        dataset_denylist=denylist,
        strict_dataset_scope=bool(_STRICT_SCOPE_RE.search(question)),
        relationship_constraints=relationships,
        confidence=confidence,
    )


def _parse_intent(content: str, *, question: str) -> AnalysisIntentSpec:
    payload = _extract_json_object(content)
    if isinstance(payload.get("intent"), dict):
        payload = payload["intent"]
    payload.setdefault("question", question)
    payload["source"] = "llm"
    return AnalysisIntentSpec.model_validate(payload)


def _extract_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Intent compiler did not return a JSON object.")


def _binding(context: IntentCompilationContext, column: str | None) -> FieldBinding | None:
    if not column:
        return None
    if column in context.bindings:
        return context.bindings[column]
    leaf = column.rsplit("__", 1)[-1].casefold()
    matches = [
        binding
        for reference, binding in context.bindings.items()
        if reference.rsplit("__", 1)[-1].casefold() == leaf
    ]
    return matches[0] if len(matches) == 1 else FieldBinding(column=column, confidence=0)


def _bindings(
    context: IntentCompilationContext,
    columns: tuple[str, ...],
) -> tuple[FieldBinding, ...]:
    return tuple(binding for column in columns if (binding := _binding(context, column)))


def _field_clause(
    question: str,
    kind: str,
    polarity: str,
    binding: FieldBinding,
    clause_id: str,
) -> IntentClause:
    span = _best_span(question, binding.column, forbidden=polarity == "forbidden")
    return IntentClause(
        clause_id=clause_id,
        kind=kind,
        polarity=polarity,
        concept=binding.column.rsplit("__", 1)[-1],
        source_span=span,
        field=binding,
        confidence=binding.confidence,
    )


def _best_span(question: str, reference: str, *, forbidden: bool) -> IntentSourceSpan:
    leaf = reference.rsplit("__", 1)[-1].casefold()
    tokens = tuple(token for token in re.split(r"[^a-z0-9\u3400-\u9fff]+", leaf) if token)
    candidates = list(_CLAUSE_RE.finditer(question)) or [re.match(r".+", question)]
    selected = None
    for match in candidates:
        if match is None:
            continue
        text = match.group(0)
        is_forbidden = bool(_NEGATION_RE.search(text))
        if is_forbidden != forbidden:
            continue
        folded = text.casefold()
        if leaf in folded or any(len(token) >= 3 and token in folded for token in tokens):
            selected = match
            break
    if selected is None:
        selected = next(
            (
                match
                for match in candidates
                if match is not None and bool(_NEGATION_RE.search(match.group(0))) == forbidden
            ),
            None,
        )
    if selected is None:
        return IntentSourceSpan(text=question, start=0, end=len(question))
    start, end = selected.span()
    return IntentSourceSpan(text=question[start:end], start=start, end=end)


def _dataset_scope(
    question: str,
    assets: tuple[dict[str, Any], ...],
) -> tuple[tuple[UUID, ...], tuple[UUID, ...], list[IntentClause]]:
    allowlist: list[UUID] = []
    denylist: list[UUID] = []
    clauses: list[IntentClause] = []
    relationship_spans = tuple(match.span() for match in _RELATIONSHIP_RE.finditer(question))
    for asset in assets:
        dataset_id = UUID(str(asset["dataset_id"]))
        for match in _asset_matches(question, str(asset["name"])):
            if any(start <= match.start() and match.end() <= end for start, end in relationship_spans):
                continue
            containing = _containing_clause(question, match.start())
            forbidden = bool(_NEGATION_RE.search(containing.text))
            target = denylist if forbidden else allowlist
            target.append(dataset_id)
            clauses.append(
                IntentClause(
                    clause_id=f"dataset-{len(clauses) + 1}",
                    kind="dataset",
                    polarity="forbidden" if forbidden else "required",
                    concept=str(asset["name"]),
                    source_span=containing,
                    value=str(dataset_id),
                )
            )
            break
    return tuple(dict.fromkeys(allowlist)), tuple(dict.fromkeys(denylist)), clauses


def _relationship_constraints(
    question: str,
    assets: tuple[dict[str, Any], ...],
) -> tuple[RelationshipConstraint, ...]:
    output: list[RelationshipConstraint] = []
    for match in _RELATIONSHIP_RE.finditer(question):
        mentioned = sorted(
            (
                min(found.start() for found in matches),
                UUID(str(asset["dataset_id"])),
            )
            for asset in assets
            if (
                matches := tuple(
                    found
                    for alias in _asset_patterns(str(asset["name"]))
                    if (found := alias.search(match.group(0))) is not None
                )
            )
        )
        if len(mentioned) < 2:
            continue
        span = IntentSourceSpan(text=match.group(0), start=match.start(), end=match.end())
        output.append(
            RelationshipConstraint(
                left_dataset_id=mentioned[0][1],
                right_dataset_id=mentioned[1][1],
                polarity=(
                    "forbidden"
                    if _NEGATION_RE.search(match.group(0))
                    else "required"
                ),
                source_span=span,
            )
        )
    return tuple(output)


def _asset_matches(question: str, name: str) -> list[re.Match[str]]:
    return [match for pattern in _asset_patterns(name) for match in pattern.finditer(question)]


def _asset_patterns(name: str) -> tuple[re.Pattern[str], ...]:
    stem = PurePath(name).stem.casefold()
    tokens = tuple(
        token
        for token in re.split(r"[^a-z0-9\u3400-\u9fff]+", stem)
        if token not in {"csv", "data", "dataset", "file", "table", "txt", "olist"}
    )
    aliases = tuple(
        dict.fromkeys(
            (
                "_".join(tokens),
                *(
                    token
                    for token in tokens
                    if len(token) >= 4 and token not in _GENERIC_ASSET_TOKENS
                ),
                stem,
            )
        )
    )
    patterns: list[re.Pattern[str]] = []
    for alias in aliases:
        if len(alias) < 3:
            continue
        if re.search(r"[\u3400-\u9fff]", alias):
            patterns.append(re.compile(re.escape(alias), re.IGNORECASE))
        else:
            parts = tuple(part for part in re.split(r"[_\s.-]+", alias) if part)
            patterns.append(
                re.compile(
                    r"(?<![a-z0-9])" + r"[\s_.-]+".join(map(re.escape, parts)) + r"(?![a-z0-9])",
                    re.IGNORECASE,
                )
            )
    return tuple(patterns)


def _containing_clause(question: str, position: int) -> IntentSourceSpan:
    for match in _CLAUSE_RE.finditer(question):
        if match.start() <= position < match.end():
            return IntentSourceSpan(text=match.group(0), start=match.start(), end=match.end())
    return IntentSourceSpan(text=question, start=0, end=len(question))


def _mask_relationship_asset_mentions(
    question: str,
    assets: tuple[dict[str, Any], ...],
) -> str:
    masked = list(question)
    for relationship in _RELATIONSHIP_RE.finditer(question):
        text = relationship.group(0)
        for asset in assets:
            for pattern in _asset_patterns(str(asset["name"])):
                for match in pattern.finditer(text):
                    start = relationship.start() + match.start()
                    end = relationship.start() + match.end()
                    masked[start:end] = " " * (end - start)
    return "".join(masked)


def _dataset_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return slug[:32] or "dataset"


def _needs_model_compilation(question: str, intent: AnalysisIntentSpec) -> bool:
    return bool(
        _NEGATION_RE.search(question)
        or _STRICT_SCOPE_RE.search(question)
        or intent.relationship_constraints
        or len(intent.clauses) >= 4
        or any(token in question.casefold() for token in ("同时", "分别", "除非", "而不是", "unless", "instead of"))
    )


def _guard_summary(result: IntentGuardResult) -> str:
    return "; ".join(f"{item.code}: {item.message}" for item in result.issues) or result.status


def _all_bindings(
    metric: FieldBinding | None,
    dimensions: tuple[FieldBinding, ...],
    filters: tuple[IntentFilter, ...],
) -> tuple[FieldBinding, ...]:
    return tuple(
        item
        for item in (
            metric,
            *dimensions,
            *(filter_item.field for filter_item in filters),
        )
        if item is not None
    )
