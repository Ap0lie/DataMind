from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from app.schemas.analysis import AnalysisContractResponse, DatasetJoinConfig
from app.schemas.analysis_intent import (
    AnalysisIntentSpec,
    ContractGuardResult,
    FieldBinding,
    IntentClause,
    IntentGuardIssue,
    IntentGuardResult,
)

_NEGATION_RE = re.compile(
    r"不要|不得|请勿|禁止|严禁|排除|忽略|无需|"
    r"\b(?:do\s+not|don't|never|without|excluding?)\b",
    re.IGNORECASE,
)


def validate_intent(
    spec: AnalysisIntentSpec,
    *,
    question: str,
    assets: tuple[dict[str, Any], ...],
    baseline: AnalysisIntentSpec | None = None,
) -> IntentGuardResult:
    issues: list[IntentGuardIssue] = []
    asset_columns = _asset_columns(assets)
    asset_schema = _asset_schema(assets)
    known_ids = set(asset_columns)

    if spec.question.strip() != question.strip():
        issues.append(
            _issue(
                "question_mismatch",
                "Compiled intent does not preserve the current user question.",
                "Copy the current question without rewriting it.",
            )
        )

    for clause in spec.clauses:
        span = clause.source_span
        if span.end > len(question) or question[span.start : span.end] != span.text:
            issues.append(
                _issue(
                    "invalid_source_span",
                    f"Clause {clause.clause_id} does not cite an exact question span.",
                    "Use exact start/end offsets and verbatim text from the user question.",
                    clause.clause_id,
                )
            )
        has_negation = bool(_NEGATION_RE.search(span.text))
        if clause.polarity == "forbidden" and not has_negation:
            issues.append(
                _issue(
                    "unsupported_forbidden_polarity",
                    f"Clause {clause.clause_id} marks a positive span as forbidden.",
                    "Re-read the source span and preserve its polarity.",
                    clause.clause_id,
                )
            )
        if clause.polarity == "required" and has_negation:
            issues.append(
                _issue(
                    "negation_reversed",
                    f"Clause {clause.clause_id} converts a negated requirement into a required one.",
                    "Represent the constraint as forbidden instead of required.",
                    clause.clause_id,
                )
            )
        if (
            spec.source == "llm"
            and clause.polarity == "required"
            and clause.kind in {"metric", "dimension", "filter", "time", "grain"}
            and not _clause_concept_supported(clause)
        ):
            issues.append(
                _issue(
                    "unsupported_required_concept",
                    f"Clause {clause.clause_id} is not grounded in its cited source span.",
                    "Use the user's wording as concept, or remove the unsupported requirement.",
                    clause.clause_id,
                )
            )
        if clause.field is not None:
            issues.extend(_validate_field(clause.field, asset_columns, clause.clause_id))
            issues.extend(_validate_clause_semantics(clause, asset_schema))

    if spec.source == "llm":
        issues.extend(_required_clause_coverage_issues(spec))

    for binding in _all_bindings(spec):
        issues.extend(_validate_field(binding, asset_columns))

    unknown_ids = (
        set(spec.dataset_allowlist)
        | set(spec.dataset_denylist)
        | {
            item
            for relationship in spec.relationship_constraints
            for item in (relationship.left_dataset_id, relationship.right_dataset_id)
        }
    ) - known_ids
    if unknown_ids:
        issues.append(
            _issue(
                "unknown_dataset",
                "Intent references datasets outside the authorized request scope.",
                "Use only dataset IDs from authorized_assets.",
            )
        )
    for relationship in spec.relationship_constraints:
        span = relationship.source_span
        exact = span.end <= len(question) and question[span.start : span.end] == span.text
        if not exact:
            issues.append(
                _issue(
                    "invalid_relationship_span",
                    "Relationship constraint does not cite an exact question span.",
                    "Use the exact text that requires or forbids the relationship.",
                )
            )
        if relationship.polarity == "forbidden" and not _NEGATION_RE.search(span.text):
            issues.append(
                _issue(
                    "relationship_polarity_reversed",
                    "A relationship is marked forbidden without a negated source span.",
                    "Preserve the relationship polarity from the user wording.",
                )
            )
        if relationship.polarity == "required" and _NEGATION_RE.search(span.text):
            issues.append(
                _issue(
                    "relationship_polarity_reversed",
                    "A negated relationship is marked as required.",
                    "Preserve the forbidden relationship polarity from the user wording.",
                )
            )
    overlap = set(spec.dataset_allowlist) & set(spec.dataset_denylist)
    if overlap:
        issues.append(
            _issue(
                "dataset_scope_conflict",
                "The same dataset is both required and forbidden.",
                "Resolve the polarity conflict without widening scope.",
            )
        )
    if spec.strict_dataset_scope and not spec.dataset_allowlist:
        issues.append(
            _issue(
                "empty_strict_allowlist",
                "A strict dataset scope has no resolved dataset.",
                "Bind every explicitly allowed dataset or request user confirmation.",
            )
        )

    if baseline is not None:
        issues.extend(_baseline_preservation_issues(spec, baseline))

    conflicting_fields = _binding_keys(spec.required_dimensions) & {
        _binding_key(item.field)
        for item in spec.clauses
        if item.kind == "dimension" and item.polarity == "forbidden" and item.field
    }
    if conflicting_fields:
        issues.append(
            _issue(
                "field_polarity_conflict",
                "A dimension is simultaneously required and forbidden.",
                "Keep the interpretation supported by the exact source spans.",
            )
        )

    if issues:
        status = (
            "confirmation_required"
            if any(item.code in {"unknown_dataset", "empty_strict_allowlist"} for item in issues)
            else "repairable"
        )
        return IntentGuardResult(
            status=status,
            issues=tuple(_dedupe_issues(issues)),
            confidence=min(spec.confidence, 0.49 if status == "confirmation_required" else 0.69),
        )
    return IntentGuardResult(status="passed", confidence=spec.confidence)


def _baseline_preservation_issues(
    spec: AnalysisIntentSpec,
    baseline: AnalysisIntentSpec,
) -> list[IntentGuardIssue]:
    issues: list[IntentGuardIssue] = []
    if baseline.required_metric and (
        not spec.required_metric
        or not _same_column(baseline.required_metric.column, spec.required_metric.column)
    ):
        issues.append(
            _issue(
                "explicit_metric_omitted",
                "Compiled intent omitted the metric explicitly resolved from the question.",
                "Restore the explicit metric; candidate metrics cannot replace it.",
            )
        )
    missing_dimensions = [
        item.column
        for item in baseline.required_dimensions
        if not _matches_any(item.column, (value.column for value in spec.required_dimensions))
    ]
    if missing_dimensions:
        issues.append(
            _issue(
                "explicit_dimension_omitted",
                "Compiled intent omitted explicit dimensions: " + ", ".join(missing_dimensions),
                "Restore every explicitly requested dimension.",
            )
        )
    if not set(baseline.dataset_denylist).issubset(spec.dataset_denylist):
        issues.append(
            _issue(
                "dataset_denylist_weakened",
                "Compiled intent removed an explicit dataset exclusion.",
                "Restore every explicit dataset denylist entry.",
            )
        )
    baseline_forbidden = {
        frozenset((item.left_dataset_id, item.right_dataset_id))
        for item in baseline.relationship_constraints
        if item.polarity == "forbidden"
    }
    compiled_forbidden = {
        frozenset((item.left_dataset_id, item.right_dataset_id))
        for item in spec.relationship_constraints
        if item.polarity == "forbidden"
    }
    if not baseline_forbidden.issubset(compiled_forbidden):
        issues.append(
            _issue(
                "relationship_constraint_omitted",
                "Compiled intent removed an explicit forbidden relationship.",
                "Restore the relationship constraint without turning it into a dataset requirement.",
            )
        )
    return issues


def validate_analysis_contract(
    contract: AnalysisContractResponse,
    *,
    intent: AnalysisIntentSpec,
    join_plan: tuple[DatasetJoinConfig, ...] = (),
) -> ContractGuardResult:
    missing: list[str] = []
    preserved: list[str] = []
    issues: list[IntentGuardIssue] = []

    if intent.required_metric:
        required = intent.required_metric.column
        available_metrics = {
            value
            for value in (
                contract.metric,
                *(item.column for item in contract.aggregations),
            )
            if value
        }
        if _matches_any(required, available_metrics):
            preserved.append(f"metric:{required}")
        else:
            missing.append(f"metric:{required}")

    for binding in intent.required_dimensions:
        key = f"dimension:{binding.column}"
        if _matches_any(binding.column, contract.dimensions):
            preserved.append(key)
        else:
            missing.append(key)

    for required_filter in intent.filters:
        key = f"filter:{required_filter.field.column}{required_filter.operator}{required_filter.value}"
        matches = any(
            _same_column(required_filter.field.column, item.column)
            and required_filter.operator == item.operator
            and str(required_filter.value) == str(item.value)
            for item in contract.filters
        )
        (preserved if matches else missing).append(key)

    for aggregation in intent.aggregations:
        field = aggregation.field.column if aggregation.field else None
        key = f"aggregation:{aggregation.operation}:{field or '*'}"
        matches = any(
            item.operation == aggregation.operation
            and (
                field is None
                or (item.column is not None and _same_column(field, item.column))
            )
            for item in contract.aggregations
        )
        (preserved if matches else missing).append(key)

    if intent.time_field:
        key = f"time:{intent.time_field.column}"
        if contract.time_field and _same_column(intent.time_field.column, contract.time_field):
            preserved.append(key)
        else:
            missing.append(key)

    denied = set(intent.dataset_denylist) & set(contract.dataset_ids)
    if denied:
        issues.append(
            _issue(
                "contract_uses_forbidden_dataset",
                "Analysis contract includes a dataset forbidden by the approved intent.",
                "Re-resolve dataset scope before planning.",
            )
        )

    if intent.strict_dataset_scope:
        outside_allowlist = set(contract.dataset_ids) - set(intent.dataset_allowlist)
        if outside_allowlist:
            issues.append(
                _issue(
                    "contract_widens_dataset_scope",
                    "Analysis contract includes datasets outside the strict allowlist.",
                    "Re-resolve the contract using only explicitly allowed datasets.",
                )
            )

    forbidden_fields = {
        kind: tuple(
            clause.field.column
            for clause in intent.clauses
            if clause.kind == kind
            and clause.polarity == "forbidden"
            and clause.field is not None
        )
        for kind in ("metric", "dimension", "filter", "time")
    }
    forbidden_uses = [
        *(f"metric:{value}" for value in forbidden_fields["metric"] if _contract_uses_metric(contract, value)),
        *(f"dimension:{value}" for value in forbidden_fields["dimension"] if _matches_any(value, contract.dimensions)),
        *(f"filter:{value}" for value in forbidden_fields["filter"] if _matches_any(value, (item.column for item in contract.filters))),
        *(f"time:{value}" for value in forbidden_fields["time"] if contract.time_field and _same_column(value, contract.time_field)),
    ]
    if forbidden_uses:
        issues.append(
            _issue(
                "contract_uses_forbidden_field",
                "Analysis contract uses fields explicitly forbidden by the user: "
                + ", ".join(forbidden_uses),
                "Remove forbidden fields and rebuild the contract from approved clauses.",
            )
        )

    for constraint in intent.relationship_constraints:
        pair = {constraint.left_dataset_id, constraint.right_dataset_id}
        present = any(
            {item.left_dataset_id, item.right_dataset_id} == pair
            for item in join_plan
        )
        if constraint.polarity == "forbidden" and present:
            issues.append(
                _issue(
                    "forbidden_relationship_used",
                    "Analysis contract uses a relationship explicitly forbidden by the user.",
                    "Choose an allowed relationship path or ask the user to revise the constraint.",
                )
            )
        if constraint.polarity == "required" and not present:
            missing.append(
                "relationship:"
                f"{constraint.left_dataset_id}:{constraint.right_dataset_id}"
            )

    if missing:
        issues.append(
            _issue(
                "contract_requirement_missing",
                "Planner contract omitted explicit user requirements: " + ", ".join(missing),
                "Rebuild the contract from the approved intent before executing tools.",
            )
        )
    if issues:
        hard_failure = any(
            item.code
            in {
                "contract_uses_forbidden_dataset",
                "contract_uses_forbidden_field",
                "contract_widens_dataset_scope",
                "forbidden_relationship_used",
            }
            for item in issues
        )
        return ContractGuardResult(
            status="failed" if hard_failure else "repairable",
            issues=tuple(issues),
            preserved_requirements=tuple(preserved),
            missing_requirements=tuple(missing),
        )
    return ContractGuardResult(
        status="passed",
        preserved_requirements=tuple(preserved),
    )


def _required_clause_coverage_issues(
    spec: AnalysisIntentSpec,
) -> list[IntentGuardIssue]:
    requirements: list[tuple[str, FieldBinding]] = []
    if spec.required_metric:
        requirements.append(("metric", spec.required_metric))
    requirements.extend(("dimension", item) for item in spec.required_dimensions)
    if spec.time_field:
        requirements.append(("time", spec.time_field))
    requirements.extend(("filter", item.field) for item in spec.filters)
    requirements.extend(
        ("metric", item.field)
        for item in spec.aggregations
        if item.field is not None
    )
    issues: list[IntentGuardIssue] = []
    for kind, binding in requirements:
        if any(
            clause.kind == kind
            and clause.polarity == "required"
            and clause.field is not None
            and _same_binding(clause.field, binding)
            for clause in spec.clauses
        ):
            continue
        issues.append(
            _issue(
                "required_clause_missing_source",
                f"Required {kind} field has no matching source-backed clause: {binding.column}.",
                "Add one required clause with an exact source_span for this field.",
            )
        )
    dataset_clauses = tuple(
        clause
        for clause in spec.clauses
        if clause.kind == "dataset"
    )
    for dataset_id, polarity in (
        *((item, "required") for item in spec.dataset_allowlist),
        *((item, "forbidden") for item in spec.dataset_denylist),
    ):
        if any(
            clause.polarity == polarity
            and str(dataset_id) in {clause.concept, str(clause.value or "")}
            for clause in dataset_clauses
        ):
            continue
        issues.append(
            _issue(
                "dataset_clause_missing_source",
                f"Dataset scope entry has no matching source-backed clause: {dataset_id}.",
                "Add one dataset clause with the exact source span and dataset UUID as value.",
            )
        )
    return issues


def _asset_columns(assets: tuple[dict[str, Any], ...]) -> dict[UUID, set[str]]:
    output: dict[UUID, set[str]] = {}
    for asset in assets:
        try:
            dataset_id = UUID(str(asset["dataset_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        output[dataset_id] = {
            str(item.get("name") if isinstance(item, dict) else item)
            for item in asset.get("columns", ())
            if (item.get("name") if isinstance(item, dict) else item)
        }
    return output


def _asset_schema(
    assets: tuple[dict[str, Any], ...],
) -> dict[UUID, dict[str, dict[str, Any]]]:
    output: dict[UUID, dict[str, dict[str, Any]]] = {}
    for asset in assets:
        try:
            dataset_id = UUID(str(asset["dataset_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        columns: dict[str, dict[str, Any]] = {}
        for item in asset.get("columns", ()):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            columns[str(item["name"]).casefold()] = item
            if item.get("reference"):
                columns[str(item["reference"]).casefold()] = item
        output[dataset_id] = columns
    return output


def _validate_clause_semantics(
    clause: IntentClause,
    asset_schema: dict[UUID, dict[str, dict[str, Any]]],
) -> list[IntentGuardIssue]:
    binding = clause.field
    if binding is None or binding.dataset_id is None:
        return []
    metadata = asset_schema.get(binding.dataset_id, {}).get(binding.column.casefold())
    if metadata is None:
        metadata = asset_schema.get(binding.dataset_id, {}).get(
            binding.column.rsplit("__", 1)[-1].casefold()
        )
    if metadata is None:
        return []
    issues: list[IntentGuardIssue] = []
    actual_role = str(metadata.get("role") or "").casefold()
    actual_dtype = str(metadata.get("dtype") or "").casefold()
    if binding.role and actual_role and binding.role.casefold() != actual_role:
        issues.append(
            _issue(
                "field_role_mismatch",
                f"Field role does not match the authorized schema: {binding.column}.",
                "Copy the field role from the authorized schema.",
                clause.clause_id,
            )
        )
    if binding.dtype and actual_dtype and _dtype_family(binding.dtype) != _dtype_family(actual_dtype):
        issues.append(
            _issue(
                "field_type_mismatch",
                f"Field type does not match the authorized schema: {binding.column}.",
                "Copy the field type from the authorized schema.",
                clause.clause_id,
            )
        )
    if (
        clause.kind == "metric"
        and actual_role in {"id", "ignore"}
        and clause.aggregation not in {"count", "count_distinct"}
    ):
        issues.append(
            _issue(
                "unsafe_metric_role",
                f"Identifier or ignored field cannot be used as an additive metric: {binding.column}.",
                "Use count/count_distinct or bind a field with metric role.",
                clause.clause_id,
            )
        )
    if (
        clause.kind == "metric"
        and clause.aggregation in {"sum", "avg", "min", "max"}
        and _dtype_family(actual_dtype) not in {"integer", "number"}
    ):
        issues.append(
            _issue(
                "non_numeric_metric",
                f"Numeric aggregation is incompatible with field type: {binding.column}.",
                "Bind a numeric field or change to count/count_distinct.",
                clause.clause_id,
            )
        )
    return issues


def _dtype_family(value: str) -> str:
    folded = value.casefold()
    if any(token in folded for token in ("int", "uint")):
        return "integer"
    if any(token in folded for token in ("float", "double", "decimal", "number", "numeric")):
        return "number"
    if any(token in folded for token in ("date", "time")):
        return "date"
    if any(token in folded for token in ("bool",)):
        return "boolean"
    if not folded or folded == "unknown":
        return "unknown"
    return "text"


def _validate_field(
    binding: FieldBinding,
    asset_columns: dict[UUID, set[str]],
    clause_id: str | None = None,
) -> list[IntentGuardIssue]:
    if binding.dataset_id is not None:
        columns = asset_columns.get(binding.dataset_id)
        if columns is not None and _matches_any(binding.column, columns):
            return []
    elif sum(_matches_any(binding.column, columns) for columns in asset_columns.values()) == 1:
        return []
    return [
        _issue(
            "unknown_field",
            f"Field binding is not present in the authorized schema: {binding.column}.",
            "Bind the concept to one listed field and preserve its owning dataset.",
            clause_id,
        )
    ]


def _all_bindings(spec: AnalysisIntentSpec) -> Iterable[FieldBinding]:
    if spec.required_metric:
        yield spec.required_metric
    yield from spec.candidate_metrics
    yield from spec.required_dimensions
    yield from spec.candidate_dimensions
    if spec.time_field:
        yield spec.time_field
    for aggregation in spec.aggregations:
        if aggregation.field:
            yield aggregation.field
    for item in spec.filters:
        yield item.field


def _matches_any(reference: str, values: Iterable[str]) -> bool:
    return any(_same_column(reference, item) for item in values)


def _same_column(left: str, right: str) -> bool:
    left_folded = left.casefold()
    right_folded = right.casefold()
    return left_folded == right_folded or (
        left_folded.rsplit("__", 1)[-1] == right_folded.rsplit("__", 1)[-1]
    )


def _same_binding(left: FieldBinding, right: FieldBinding) -> bool:
    return (
        left.dataset_id in {None, right.dataset_id}
        or right.dataset_id is None
    ) and _same_column(left.column, right.column)


def _clause_concept_supported(clause: IntentClause) -> bool:
    span = _normalize_text(clause.source_span.text)
    candidates = {_normalize_text(clause.concept)}
    if clause.field is not None:
        candidates.add(_normalize_text(clause.field.column.rsplit("__", 1)[-1]))
    return any(len(candidate) >= 2 and candidate in span for candidate in candidates)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.casefold())


def _contract_uses_metric(contract: AnalysisContractResponse, field: str) -> bool:
    return bool(
        (contract.metric and _same_column(field, contract.metric))
        or _matches_any(
            field,
            (item.column for item in contract.aggregations if item.column),
        )
    )


def _binding_key(binding: FieldBinding) -> tuple[UUID | None, str]:
    return binding.dataset_id, binding.column.casefold()


def _binding_keys(bindings: Iterable[FieldBinding]) -> set[tuple[UUID | None, str]]:
    return {_binding_key(item) for item in bindings}


def _issue(
    code: str,
    message: str,
    suggestion: str,
    clause_id: str | None = None,
) -> IntentGuardIssue:
    return IntentGuardIssue(
        code=code,
        message=message,
        suggestion=suggestion,
        clause_id=clause_id,
    )


def _dedupe_issues(issues: Iterable[IntentGuardIssue]) -> list[IntentGuardIssue]:
    output: list[IntentGuardIssue] = []
    seen: set[tuple[str, str | None, str]] = set()
    for issue in issues:
        key = issue.code, issue.clause_id, issue.message
        if key not in seen:
            output.append(issue)
            seen.add(key)
    return output
