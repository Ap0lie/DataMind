from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
from sqlglot import exp, parse

from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisContractResponse,
    DatasetProfileResponse,
    InsightFindingResponse,
    MultiDatasetProfileResponse,
    PythonAnalysisResponse,
    SQLAnalysisResponse,
    StatisticalCheckResponse,
    StatisticalFindingVerdictResponse,
    StatisticalVerificationResponse,
    ValidationIssueResponse,
)

# Python's Unicode ``\w`` includes Chinese characters, so ``(?<![\w])`` drops
# ordinary claims such as "总额为999" or starts midway through "5,790". Only
# block an ASCII identifier prefix; CJK prose must remain a valid number boundary.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:[,.]\d+)*(?:\.\d+)?%?")
_EVIDENCE_RE = re.compile(r"evidence_id:([A-Za-z0-9_.:-]+)")
_COMPARISON_TERMS = (
    "高于",
    "低于",
    "更高",
    "更低",
    "增加",
    "增长",
    "下降",
    "差异",
    "相比",
    "领先",
    "落后",
    "最多",
    "最少",
    "排名",
    "占比",
    "higher",
    "lower",
    "increase",
    "decrease",
    "difference",
    "compared",
    "rank",
    "share",
)
_CAUSAL_REPLACEMENTS = {
    "导致": "伴随",
    "造成": "伴随",
    "驱动": "关联于",
    "引起": "伴随",
    "决定了": "与",
    "causes": "is associated with",
    "caused": "was associated with",
    "drives": "is associated with",
    "driven by": "associated with",
    "results in": "is associated with",
}
_CAUSAL_DISCLAIMERS = (
    "仅表示相关性",
    "不能证明因果",
    "不代表因果",
    "observational association",
    "does not establish causality",
)


def qualify_observational_findings(
    findings: tuple[InsightFindingResponse, ...],
    contract: AnalysisContractResponse,
) -> tuple[InsightFindingResponse, ...]:
    if contract.causal_claim_allowed:
        return findings
    qualified: list[InsightFindingResponse] = []
    for finding in findings:
        if not _contains_unqualified_causal_claim(finding.content):
            qualified.append(finding)
            continue
        content = finding.content
        for source, replacement in _CAUSAL_REPLACEMENTS.items():
            content = re.sub(re.escape(source), replacement, content, flags=re.IGNORECASE)
        qualified.append(
            finding.model_copy(
                update={
                    "content": f"观察性数据仅表示相关性，不能证明因果：{content}",
                    "confidence": "medium"
                    if finding.confidence == "high"
                    else finding.confidence,
                }
            )
        )
    return tuple(qualified)


def verify_statistical_analysis(
    *,
    contract: AnalysisContractResponse,
    profile: DatasetProfileResponse,
    dataframe: pd.DataFrame,
    findings: tuple[InsightFindingResponse, ...],
    evidence: tuple[dict[str, Any], ...] = (),
    sql_result: SQLAnalysisResponse | None = None,
    python_result: PythonAnalysisResponse | None = None,
    multi_dataset_context: MultiDatasetProfileResponse | None = None,
) -> StatisticalVerificationResponse:
    known_evidence = {
        str(item.get("evidence_id")): item
        for item in evidence
        if item.get("evidence_id")
    }
    comparison = _comparison_statistics(dataframe, contract)
    verdicts = tuple(
        _finding_verdict(
            index=index,
            finding=finding,
            known_evidence=known_evidence,
            contract=contract,
            sql_result=sql_result,
            python_result=python_result,
            comparison=comparison,
            multi_dataset_context=multi_dataset_context,
        )
        for index, finding in enumerate(findings, start=1)
    )
    numeric_verdicts = [
        verdict
        for finding, verdict in zip(findings, verdicts, strict=True)
        if _has_number(finding.content)
    ]
    numeric_supported = sum(verdict.status != "failed" for verdict in numeric_verdicts)
    coverage = (
        numeric_supported / len(numeric_verdicts) if numeric_verdicts else 1.0
    )

    checks = [
        _population_check(dataframe),
        _metric_check(contract, profile),
        _time_check(contract, profile),
        _request_coverage_check(
            contract=contract,
            sql_result=sql_result,
            python_result=python_result,
            evidence=evidence,
        ),
        _join_grain_check(
            contract=contract,
            context=multi_dataset_context,
            evidence=evidence,
        ),
        _numeric_evidence_check(numeric_verdicts, coverage),
        _comparison_check(findings, comparison),
        _causal_language_check(findings, contract),
    ]
    status = _overall_status(checks)
    failed = [check for check in checks if check.status == "failed"]
    warning = [check for check in checks if check.status == "warning"]
    summary = (
        f"统计审查{_status_label(status)}："
        f"{len(checks) - len(failed) - len(warning)} 项通过，"
        f"{len(warning)} 项警告，{len(failed)} 项失败。"
    )
    return StatisticalVerificationResponse(
        status=status,
        summary=summary,
        checks=tuple(checks),
        finding_verdicts=verdicts,
        requires_replan=bool(failed),
        numeric_evidence_coverage=coverage,
    )


def statistical_validation_issues(
    verification: StatisticalVerificationResponse,
) -> tuple[ValidationIssueResponse, ...]:
    return tuple(
        ValidationIssueResponse(
            severity="error" if check.status == "failed" else "warning",
            finding_ref=check.finding_ref,
            issue=check.message,
            suggestion=_suggestion(check.code),
        )
        for check in verification.checks
        if check.status in {"failed", "warning"}
    )


def reportable_findings(
    findings: tuple[InsightFindingResponse, ...],
    verification: StatisticalVerificationResponse | None,
) -> tuple[InsightFindingResponse, ...]:
    if verification is None:
        return findings
    failed_titles = {
        verdict.title
        for verdict in verification.finding_verdicts
        if verdict.status == "failed"
    }
    return tuple(finding for finding in findings if finding.title not in failed_titles)


def _finding_verdict(
    *,
    index: int,
    finding: InsightFindingResponse,
    known_evidence: dict[str, dict[str, Any]],
    contract: AnalysisContractResponse,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
    comparison: dict[str, Any] | None,
    multi_dataset_context: MultiDatasetProfileResponse | None,
) -> StatisticalFindingVerdictResponse:
    explicitly_referenced_ids = set(_EVIDENCE_RE.findall(finding.evidence))
    unknown_evidence_ids = explicitly_referenced_ids - set(known_evidence)
    evidence_ids = {
        value
        for value in explicitly_referenced_ids
        if value in known_evidence
    }
    source = finding.data_source.casefold()
    claim_kind = (
        "lineage"
        if "source_relationships_and_native_aggregates" in source
        else "analysis"
    )
    if (
        not explicitly_referenced_ids
        and not evidence_ids
        and "sql_result" in source
        and sql_result
        and sql_result.rows
    ):
        evidence_ids.add("sql_result")
    if (
        not explicitly_referenced_ids
        and not evidence_ids
        and "python" in source
        and python_result
        and (python_result.statistics or python_result.insights)
    ):
        evidence_ids.add("python_result")

    notes: list[str] = []
    failed = False
    warning = False
    numeric_claims = _numeric_claims(finding.content)
    if unknown_evidence_ids:
        failed = True
        notes.append(
            "结论引用了不存在的证据："
            + ", ".join(sorted(unknown_evidence_ids))
            + "。"
        )
    if numeric_claims:
        if not evidence_ids:
            failed = True
            notes.append("数值结论缺少可读取证据。")
        else:
            cited_evidence = []
            for evidence_id in sorted(evidence_ids):
                if evidence_id == "sql_result" and sql_result is not None:
                    cited_evidence.append(
                        (
                            evidence_id,
                            {
                                "status": "succeeded",
                                "result": sql_result.model_dump(mode="json"),
                            },
                        )
                    )
                elif evidence_id == "python_result" and python_result is not None:
                    cited_evidence.append(
                        (
                            evidence_id,
                            {
                                "status": "succeeded",
                                "result": {
                                    "python_result": python_result.model_dump(mode="json")
                                },
                            },
                        )
                    )
                elif evidence_id in known_evidence:
                    cited_evidence.append((evidence_id, known_evidence[evidence_id]))
            numeric_supported, numeric_notes = _numeric_claims_supported(
                claims=numeric_claims,
                claim_text=finding.content,
                cited_evidence=tuple(cited_evidence),
                contract=contract,
                claim_kind=claim_kind,
                require_native_grain=_finding_requires_native_grain(
                    contract=contract,
                    context=multi_dataset_context,
                ),
            )
            if not numeric_supported:
                failed = True
            notes.extend(numeric_notes)
            if not numeric_supported and finding.evidence:
                notes.append(f"结论证据说明：{finding.evidence}")
    if _is_comparison(finding.content) and not _is_descriptive_ranking(finding.content):
        if comparison is None:
            failed = True
            notes.append("比较结论缺少可计算的样本量、效应量或置信区间。")
        else:
            notes.append("比较支持统计已由确定性审查器计算。")
    if _contains_unqualified_causal_claim(finding.content):
        failed = True
        notes.append("观察性数据使用了未经限定的因果措辞。")
    elif any(disclaimer in finding.content.casefold() for disclaimer in _CAUSAL_DISCLAIMERS):
        warning = True
        notes.append("因果措辞已降级为观察性相关表述。")

    return StatisticalFindingVerdictResponse(
        finding_ref=f"finding_{index}",
        title=finding.title,
        status="failed" if failed else "warning" if warning else "passed",
        evidence_ids=tuple(sorted(evidence_ids)),
        sample_size=int(comparison["sample_size"]) if comparison else None,
        effect_size=comparison.get("effect_size") if comparison else None,
        confidence_interval=tuple(comparison["confidence_interval"])
        if comparison and comparison.get("confidence_interval")
        else None,
        notes=tuple(notes),
    )


def _numeric_claims(value: str) -> tuple[tuple[str, float, bool], ...]:
    claims: list[tuple[str, float, bool]] = []
    for match in _NUMBER_RE.finditer(value):
        token = match.group(0)
        normalized = token.rstrip("%").replace(",", "")
        try:
            number = float(normalized)
        except ValueError:
            continue
        if math.isfinite(number):
            claims.append((token, number, token.endswith("%")))
    return tuple(claims)


def _numeric_claims_supported(
    *,
    claims: tuple[tuple[str, float, bool], ...],
    claim_text: str,
    cited_evidence: tuple[tuple[str, dict[str, Any]], ...],
    contract: AnalysisContractResponse,
    claim_kind: str,
    require_native_grain: bool,
) -> tuple[bool, tuple[str, ...]]:
    notes: list[str] = []
    supported_claims: set[int] = set()
    native_claim_support = False
    failed = False
    for evidence_id, item in cited_evidence:
        candidates = _evidence_claim_candidates(
            item=item,
            contract=contract,
            claim_kind=claim_kind,
            claim_text=claim_text,
        )
        row_bound_values = _row_bound_claim_values(
            item=item,
            claim_text=claim_text,
            claims=claims,
            contract=contract,
        )
        contract_candidates = [
            candidate for candidate in candidates if candidate[1]
        ]
        query_candidates = [candidate for candidate in candidates if candidate[2]]
        if query_candidates and not contract_candidates:
            failed = True
            notes.append(
                f"证据 {evidence_id} 的查询未独立满足指标、过滤和分组合同。"
            )
            continue
        if not contract_candidates:
            failed = True
            notes.append(f"证据 {evidence_id} 没有可核对的确定性结果。")
            continue

        item_supported: set[int] = set()
        item_native_support = False
        item_values: set[float] = set()
        for values, _contract_ok, _query, native in contract_candidates:
            item_values.update(values)
            matches = {
                index
                for index, claim in enumerate(claims)
                if any(
                    _claim_matches_value(claim, value)
                    for value in (
                        values & row_bound_values[index]
                        if index in row_bound_values
                        else values
                    )
                )
            }
            item_supported.update(matches)
            if native and matches:
                item_native_support = True
        if not item_supported:
            failed = True
            nearest = sorted(
                item_values,
                key=lambda value: abs(value - claims[0][1]),
            )[:3]
            nearest_note = (
                "；最接近的证据值为 "
                + ", ".join(f"{value:,.6g}" for value in nearest)
                + (
                    f"（最大值 {max(item_values):,.6g}）"
                    if item_values
                    else ""
                )
                if nearest
                else ""
            )
            notes.append(
                f"证据 {evidence_id} 的结果不包含该结论中的任何数值{nearest_note}。"
            )
        supported_claims.update(item_supported)
        native_claim_support = native_claim_support or item_native_support

    missing = [
        token for index, (token, _number, _percent) in enumerate(claims)
        if index not in supported_claims
    ]
    if missing:
        failed = True
        notes.append(
            "结论数值与所引证据结果不一致：" + ", ".join(missing[:8]) + "。"
        )
    if require_native_grain and not native_claim_support:
        failed = True
        notes.append("Join 膨胀场景中的数值结论未引用同口径原生粒度证据。")
    if not failed:
        notes.append("结论数值已与其引用的同口径证据逐项核对。")
    return not failed, tuple(notes)


def _evidence_claim_candidates(
    *,
    item: dict[str, Any],
    contract: AnalysisContractResponse,
    claim_kind: str,
    claim_text: str,
) -> tuple[tuple[set[float], bool, bool, bool], ...]:
    if item.get("status", "succeeded") not in {"succeeded", "completed", None}:
        return ()
    result = _evidence_result(item)
    if not isinstance(result, dict):
        return ()

    annotated_coverage = item.get("contract_covered")
    candidates: list[tuple[set[float], bool, bool, bool]] = []
    if result.get("native_grain") is True:
        matches_contract = _native_grain_matches_contract(result, contract)
        lineage_population_match = (
            claim_kind == "lineage"
            and _native_grain_matches_population(result, contract)
            and _native_claim_identifies_source_metric(result, claim_text)
        )
        candidates.append(
            (
                _lineage_numeric_values(result)
                if claim_kind == "lineage"
                else _result_numeric_values(result),
                annotated_coverage is not False
                and (matches_contract or lineage_population_match),
                True,
                True,
            )
        )
        return tuple(candidates)

    nested_sql = result.get("sql_result")
    if isinstance(nested_sql, dict):
        candidates.extend(
            _sql_evidence_claim_candidates(
                nested_sql,
                contract=contract,
                annotated_coverage=annotated_coverage,
                sibling_claim_values=result.get("claim_values"),
            )
        )
    elif result.get("sql") and isinstance(result.get("rows"), (list, tuple)):
        candidates.extend(
            _sql_evidence_claim_candidates(
                result,
                contract=contract,
                annotated_coverage=annotated_coverage,
                sibling_claim_values=result.get("claim_values"),
            )
        )

    nested_python = result.get("python_result")
    if isinstance(nested_python, dict):
        execution_context = nested_python.get("execution_context")
        derived_from_evidence = bool(
            isinstance(execution_context, dict)
            and execution_context.get("input_evidence_id")
        )
        candidates.append(
            (
                _result_numeric_values(nested_python)
                | _result_numeric_values(result.get("claim_values")),
                # Artifact-backed Python evidence stores bounded claim values
                # beside the compact python_result contract payload.
                # Include both without borrowing values from another evidence.
                annotated_coverage is not False
                and (
                    _python_evidence_covers_contract(nested_python, contract)
                    or (annotated_coverage is True and derived_from_evidence)
                ),
                True,
                False,
            )
        )

    if candidates:
        return tuple(candidates)

    if (
        isinstance(result.get("rows"), (list, tuple))
        and result.get("metric")
        and result.get("aggregation")
    ):
        return (
            (
                _result_numeric_values(result),
                annotated_coverage is not False
                and _structured_aggregate_covers_contract(result, contract),
                True,
                False,
            ),
        )

    # Profile and relationship metadata is not analysis-result evidence. It may
    # support only the dedicated lineage finding that names those evidence kinds;
    # otherwise values such as match_rate must never validate a metric claim.
    if claim_kind == "lineage" and _is_relationship_evidence(item, result):
        return ((_lineage_numeric_values(result), True, False, False),)
    return ()


def _evidence_result(item: dict[str, Any]) -> dict[str, Any] | None:
    result = item.get("result")
    if isinstance(result, dict):
        return result
    claim_result = item.get("claim_result")
    if isinstance(claim_result, dict):
        return claim_result
    contract_result = item.get("contract_result")
    return contract_result if isinstance(contract_result, dict) else None


def _is_relationship_evidence(
    item: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    return bool(
        item.get("relationship_guard") is True
        or str(item.get("tool_name") or "") == "inspect_source_relationships"
    ) and isinstance(result.get("relationships"), (list, tuple))


def _sql_evidence_claim_candidates(
    result: dict[str, Any],
    *,
    contract: AnalysisContractResponse,
    annotated_coverage: Any,
    sibling_claim_values: Any = None,
) -> tuple[tuple[set[float], bool, bool, bool], ...]:
    sql = str(result.get("sql") or "")
    rows = list(result.get("rows") or ())
    try:
        statements = parse(sql, read="duckdb")
    except Exception:
        statements = []
    if len(statements) <= 1:
        values = _result_numeric_values(result)
        values.update(_result_numeric_values(sibling_claim_values))
        return (
            (
                values,
                annotated_coverage is not False
                and _sql_evidence_covers_contract(result, contract),
                True,
                False,
            ),
        )

    # Combined loop SQL must keep values and contract coverage in the same
    # statement partition. Otherwise a correct query can launder a number from
    # another query_index in the same SQLAnalysisResponse.
    if not rows or not all("query_index" in row for row in rows if isinstance(row, dict)):
        return tuple((set(), False, True, False) for _ in statements)
    candidates: list[tuple[set[float], bool, bool, bool]] = []
    for query_index, statement in enumerate(statements, start=1):
        partition = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("query_index")) == str(query_index)
        ]
        statement_result = {
            "sql": statement.sql(dialect="duckdb"),
            "rows": partition,
            "explanation": result.get("explanation") or "",
        }
        candidates.append(
            (
                _result_numeric_values(statement_result),
                annotated_coverage is not False
                and _sql_evidence_covers_contract(statement_result, contract),
                True,
                False,
            )
        )
    return tuple(candidates)


def _sql_evidence_covers_contract(
    result: dict[str, Any],
    contract: AnalysisContractResponse,
) -> bool:
    try:
        sql_result = SQLAnalysisResponse.model_validate(
            {
                "sql": result.get("sql") or "",
                "rows": result.get("rows") or (),
                "explanation": result.get("explanation") or "",
            }
        )
    except (TypeError, ValueError):
        return False
    return analysis_contract_covered(
        contract=contract,
        sql_result=sql_result,
        python_result=None,
    )


def _python_evidence_covers_contract(
    result: dict[str, Any],
    contract: AnalysisContractResponse,
) -> bool:
    try:
        python_result = PythonAnalysisResponse.model_validate(result)
    except (TypeError, ValueError):
        return False
    return analysis_contract_covered(
        contract=contract,
        sql_result=None,
        python_result=python_result,
    )


def _structured_aggregate_covers_contract(
    result: dict[str, Any],
    contract: AnalysisContractResponse,
) -> bool:
    metric = str(result.get("metric") or "")
    if contract.metric and not _column_equivalent(metric, contract.metric):
        return False
    aggregation = str(result.get("aggregation") or "")
    expected_aggregations = {
        item.operation
        for item in contract.aggregations
        if item.column is None or _column_equivalent(str(item.column), metric)
    }
    if expected_aggregations and aggregation not in expected_aggregations:
        return False
    if len(expected_aggregations) > 1:
        return False

    result_filters = result.get("filters") or ()
    if not isinstance(result_filters, (list, tuple)):
        return False
    if len(result_filters) != len(contract.filters):
        return False
    for expected in contract.filters:
        if not any(
            isinstance(actual, dict)
            and _column_equivalent(str(actual.get("column") or ""), expected.column)
            and str(actual.get("operator") or "=") == expected.operator
            and _value_equivalent(actual.get("value"), expected.value)
            for actual in result_filters
        ):
            return False

    expected_grain = _contract_grain_columns(contract)
    actual_grain = result.get("grain")
    if actual_grain is None:
        actual_grain = [result["group_by"]] if result.get("group_by") else ["dataset"]
    if not isinstance(actual_grain, (list, tuple)):
        return False
    actual_columns = tuple(
        str(item) for item in actual_grain if str(item).casefold() != "dataset"
    )
    return len(actual_columns) == len(expected_grain) and all(
        _contains_equivalent_column(actual_columns, item) for item in expected_grain
    )


def _result_numeric_values(value: Any, *, key: str = "") -> set[float]:
    ignored_keys = {
        "sql",
        "code",
        "explanation",
        "evidence_id",
        "source_dataset_id",
        "dataset_id",
        "action_hash",
        "result_hash",
        "filters",
        "arguments",
        "row_count",
        "source_row_count",
        "filtered_row_count",
        "dataset_count",
        "column_count",
        "joined_row_count",
        "joined_column_count",
        "distinct_count",
        "missing_count",
        "match_rate",
        "left_non_null_count",
        "right_non_null_count",
        "left_distinct_count",
        "right_distinct_count",
        "left_duplicate_count",
        "right_duplicate_count",
    }
    if key in ignored_keys or value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        number = float(value)
        return {number} if math.isfinite(number) else set()
    if isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", stripped):
            return set()
        try:
            number = float(stripped.rstrip("%").replace(",", ""))
        except ValueError:
            return set()
        values = {number}
        if stripped.endswith("%"):
            values.add(number / 100)
        return values
    if isinstance(value, dict):
        values: set[float] = set()
        for child_key, child in value.items():
            values.update(_result_numeric_values(child, key=str(child_key)))
            if child_key == "rows" and isinstance(child, (list, tuple)):
                values.update(_tabular_derived_numeric_values(child))
        return values
    if isinstance(value, (list, tuple)):
        values = {float(len(value))} if key == "rows" else set()
        for child in value:
            values.update(_result_numeric_values(child))
        return values
    return set()


def _lineage_numeric_values(value: Any) -> set[float]:
    """Extract numbers only for an explicitly typed relationship/lineage claim."""

    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        number = float(value)
        return {number} if math.isfinite(number) else set()
    if isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
            stripped,
        ):
            return set()
        try:
            number = float(stripped.rstrip("%").replace(",", ""))
        except ValueError:
            return set()
        return {number, number / 100} if stripped.endswith("%") else {number}
    if isinstance(value, dict):
        values: set[float] = set()
        for child in value.values():
            values.update(_lineage_numeric_values(child))
        return values
    if isinstance(value, (list, tuple)):
        values: set[float] = set()
        for child in value:
            values.update(_lineage_numeric_values(child))
        return values
    return set()


def _row_bound_claim_values(
    *,
    item: dict[str, Any],
    claim_text: str,
    claims: tuple[tuple[str, float, bool], ...],
    contract: AnalysisContractResponse,
) -> dict[int, set[float]]:
    """Bind a labeled numeric claim to values from that same result row."""

    result = _evidence_result(item)
    if not isinstance(result, dict):
        return {}
    row_groups: list[list[dict[str, Any]]] = []
    nested_sql = result.get("sql_result")
    if isinstance(nested_sql, dict) and isinstance(nested_sql.get("rows"), (list, tuple)):
        row_groups.append([row for row in nested_sql["rows"] if isinstance(row, dict)])
    if isinstance(result.get("rows"), (list, tuple)):
        row_groups.append([row for row in result["rows"] if isinstance(row, dict)])
    if isinstance(result.get("claim_rows"), (list, tuple)):
        row_groups.append([row for row in result["claim_rows"] if isinstance(row, dict)])
    rows = [row for group in row_groups for row in group]
    if not rows:
        return {}

    expected_dimensions = _contract_grain_columns(contract)
    dimension_columns = {
        str(column)
        for row in rows
        for column in row
        if any(_column_equivalent(str(column), expected) for expected in expected_dimensions)
    }
    if not dimension_columns:
        dimension_columns = {
            str(column)
            for row in rows
            for column, value in row.items()
            if isinstance(value, str)
            and column not in {"evidence_id", "query_index"}
        }

    label_values: dict[str, set[float]] = {}
    label_text: dict[str, str] = {}
    for row in rows:
        for column in dimension_columns:
            value = row.get(column)
            if value is None:
                continue
            label = str(value).strip()
            if not label:
                continue
            folded = label.casefold()
            label_text[folded] = label
            label_values.setdefault(folded, set()).update(_result_numeric_values(row))
    occurrences: list[tuple[int, str]] = []
    folded_claim = claim_text.casefold()
    for folded, label in label_text.items():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        occurrences.extend((match.start(), folded) for match in pattern.finditer(claim_text))
    if not occurrences:
        return {}
    occurrences.sort()

    bound: dict[int, set[float]] = {}
    cursor = 0
    for index, (token, _number, _percent) in enumerate(claims):
        token_start = folded_claim.find(token.casefold(), cursor)
        if token_start < 0:
            token_start = folded_claim.find(token.casefold())
        if token_start < 0:
            continue
        cursor = token_start + len(token)
        preceding = [
            (position, label)
            for position, label in occurrences
            if 0 <= token_start - position <= 120
        ]
        if not preceding:
            continue
        _position, label = max(preceding, key=lambda item: item[0])
        bound[index] = label_values[label]
    return bound


def _tabular_derived_numeric_values(rows: list[Any] | tuple[Any, ...]) -> set[float]:
    """Expose deterministic column totals produced by the report summarizer.

    A finding may summarize grouped SQL rows into an overall total. Keeping the
    derivation column-local allows that valid path without permitting arithmetic
    across unrelated evidence items.
    """

    columns: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for column, value in row.items():
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                columns.setdefault(str(column), []).append(number)
    values = {
        float(sum(numbers))
        for numbers in columns.values()
        if len(numbers) > 1
    }
    folded = {name.casefold(): numbers for name, numbers in columns.items()}
    if folded.keys() & {"aov", "average_order_value", "avg_order_value"}:
        numerator = next(
            (
                numbers
                for name, numbers in folded.items()
                if name in {"gmv", "revenue", "sales", "total_price", "total_amount"}
            ),
            None,
        )
        denominator = next(
            (numbers for name, numbers in folded.items() if name == "order_count"),
            None,
        )
        if numerator and denominator and sum(denominator):
            values.add(float(sum(numerator) / sum(denominator)))
    return values


def _claim_matches_value(
    claim: tuple[str, float, bool],
    evidence_value: float,
) -> bool:
    _token, number, percent = claim
    expected_values = (number, number / 100) if percent else (number,)
    return any(
        math.isclose(expected, evidence_value, rel_tol=1e-6, abs_tol=1e-9)
        for expected in expected_values
    )


def _finding_requires_native_grain(
    *,
    contract: AnalysisContractResponse,
    context: MultiDatasetProfileResponse | None,
) -> bool:
    if context is None or not contract.metric:
        return False
    expansion = float(context.join_summary.get("row_expansion_ratio") or 1)
    return expansion >= 1.05


def _population_check(dataframe: pd.DataFrame) -> StatisticalCheckResponse:
    row_count = int(dataframe.shape[0])
    passed = row_count > 0
    return StatisticalCheckResponse(
        code="population_non_empty",
        status="passed" if passed else "failed",
        severity="info" if passed else "error",
        message=(
            f"分析总体包含 {row_count} 行记录。"
            if passed
            else "分析总体为空，无法形成有效结论。"
        ),
        details={"row_count": row_count},
    )


def _metric_check(
    contract: AnalysisContractResponse,
    profile: DatasetProfileResponse,
) -> StatisticalCheckResponse:
    if not contract.metric:
        return StatisticalCheckResponse(
            code="metric_type",
            status="not_applicable",
            severity="info",
            message="本次分析未指定数值指标，使用计数或文本分析路径。",
        )
    column = next(
        (item for item in profile.columns if item.name == contract.metric),
        None,
    )
    passed = bool(column and column.is_numeric)
    return StatisticalCheckResponse(
        code="metric_type",
        status="passed" if passed else "failed",
        severity="info" if passed else "error",
        message=(
            f"指标 {contract.metric} 已验证为数值字段。"
            if passed
            else f"指标 {contract.metric} 不存在或不是可聚合数值字段。"
        ),
        details={"metric": contract.metric},
    )


def _time_check(
    contract: AnalysisContractResponse,
    profile: DatasetProfileResponse,
) -> StatisticalCheckResponse:
    if contract.analysis_type != "trend":
        return StatisticalCheckResponse(
            code="time_coverage",
            status="not_applicable",
            severity="info",
            message="本次分析不要求时间趋势。",
        )
    valid = bool(
        contract.time_field
        and any(column.name == contract.time_field for column in profile.columns)
    )
    return StatisticalCheckResponse(
        code="time_coverage",
        status="passed" if valid else "warning",
        severity="info" if valid else "warning",
        message=(
            f"趋势分析使用时间字段 {contract.time_field}。"
            if valid
            else "用户请求了趋势分析，但没有可靠时间字段；不得推断真实时间变化。"
        ),
        details={"time_field": contract.time_field},
    )


def _request_coverage_check(
    *,
    contract: AnalysisContractResponse,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
    evidence: tuple[dict[str, Any], ...] = (),
) -> StatisticalCheckResponse:
    has_execution_result = sql_result is not None or python_result is not None
    undeclared_requirements = _undeclared_contract_requirements(contract)
    if undeclared_requirements:
        return StatisticalCheckResponse(
            code="request_coverage",
            status="failed",
            severity="error",
            message=(
                "分析契约未结构化用户明确声明的要求："
                + "、".join(undeclared_requirements)
                + "；不得用空契约将执行结果标记为已覆盖。"
            ),
            finding_ref="analysis_contract",
            details={
                "dimensions": (
                    ["analysis_contract.dimensions"]
                    if "维度" in undeclared_requirements
                    else []
                ),
                "filters": (
                    ["analysis_contract.filters"]
                    if "过滤条件" in undeclared_requirements
                    else []
                ),
                "aggregations": (
                    ["analysis_contract.aggregations"]
                    if "聚合指标" in undeclared_requirements
                    else []
                ),
                "contract_incomplete": undeclared_requirements,
            },
        )
    if (
        not contract.aggregations
        and not contract.filters
        and (not contract.dimensions or not has_execution_result)
    ):
        return StatisticalCheckResponse(
            code="request_coverage",
            status="not_applicable",
            severity="info",
            message="用户问题未声明必须覆盖的维度、过滤或聚合。",
        )

    statements: list[exp.Expression] = []
    if sql_result is not None:
        try:
            statements = parse(sql_result.sql, read="duckdb")
        except Exception:
            statements = []

    candidates = [
        (
            f"sql_statement_{index}",
            _statement_contract_gaps(
                statement=statement,
                contract=contract,
                result_rows=sql_result.rows if sql_result is not None else (),
            ),
        )
        for index, statement in enumerate(statements, start=1)
    ]
    if python_result is not None:
        candidates.append(
            (
                "python_result",
                _python_contract_gaps(
                    python_result=python_result,
                    contract=contract,
                ),
            )
        )
    native_gaps = _native_evidence_contract_gaps(contract, evidence)
    if native_gaps is not None:
        candidates.append(("native_grain_evidence", native_gaps))
    if not candidates:
        candidates.append(("no_execution_result", _empty_contract_gaps(contract)))

    # A contract is covered only when one execution result satisfies the whole
    # request. Selecting the closest candidate for diagnostics prevents separate
    # SQL statements from being spliced into a result that never existed.
    covered_by, missing = min(candidates, key=lambda item: _gap_score(item[1]))
    passed = not any(missing.values())
    missing_dimensions = list(missing["dimensions"])
    missing_filters = list(missing["filters"])
    missing_aggregations = list(missing["aggregations"])
    unexpected_dimensions = list(missing.get("unexpected_dimensions", ()))
    unexpected_filters = list(missing.get("unexpected_filters", ()))
    return StatisticalCheckResponse(
        code="request_coverage",
        status="passed" if passed else "failed",
        severity="info" if passed else "error",
        message=(
            "执行结果覆盖了用户明确要求的维度、过滤条件和聚合指标。"
            if passed
            else "执行结果未完整回答用户问题："
            + "；".join(
                f"{label}缺少 {', '.join(values)}"
                for label, values in (
                    ("维度", missing_dimensions),
                    ("未授权分组维度", unexpected_dimensions),
                    ("过滤", missing_filters),
                    ("未授权过滤", unexpected_filters),
                    ("聚合", missing_aggregations),
                )
                if values
            )
        ),
        finding_ref="analysis_contract",
        details=(
            {
                "required_dimensions": list(_contract_grain_columns(contract)),
                "required_filters": [
                    f"{item.column}{item.operator}{item.value}"
                    for item in contract.filters
                ],
                "required_aggregations": [
                    f"{item.operation}({item.column or '*'})"
                    for item in contract.aggregations
                ],
                "covered_by": covered_by,
            }
            if passed
            else missing
        ),
    )


def _undeclared_contract_requirements(
    contract: AnalysisContractResponse,
) -> list[str]:
    """Detect an empty contract that contradicts explicit wording in its objective."""

    objective = contract.objective.casefold()
    missing: list[str] = []
    declares_dimension = bool(
        re.search(
            r"(?:按|依据|根据|group\s+by)\s*[^,，;；。.!?！？\n]+?"
            r"(?:统计|汇总|聚合|分组|分析|查看|compare|aggregate)",
            objective,
            flags=re.IGNORECASE,
        )
    )
    declares_filter = bool(
        re.search(r"(?:过滤|筛选|只看|限定|where\b)", objective)
        or re.search(r"(?<![<>!=])(?:==|=|!=|<>|>=|<=)(?![=])", objective)
    )
    declares_aggregation = any(
        token in objective
        for token in (
            "总额",
            "总计",
            "合计",
            "平均",
            "均值",
            "数量",
            "订单数",
            "sum(",
            "avg(",
            "count(",
            " total",
            " average",
        )
    )
    if declares_dimension and not (contract.dimensions or contract.grain != ("dataset",)):
        missing.append("维度")
    if declares_filter and not contract.filters:
        missing.append("过滤条件")
    if declares_aggregation and not contract.aggregations:
        missing.append("聚合指标")
    return missing


def analysis_contract_covered(
    *,
    contract: AnalysisContractResponse,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
    evidence: tuple[dict[str, Any], ...] = (),
) -> bool:
    """Return whether deterministic outputs cover the contract's explicit request."""

    return _request_coverage_check(
        contract=contract,
        sql_result=sql_result,
        python_result=python_result,
        evidence=evidence,
    ).status in {"passed", "not_applicable"}


def analysis_contract_gaps(
    *,
    contract: AnalysisContractResponse,
    sql_result: SQLAnalysisResponse | None,
    python_result: PythonAnalysisResponse | None,
    evidence: tuple[dict[str, Any], ...] = (),
) -> dict[str, tuple[str, ...]]:
    check = _request_coverage_check(
        contract=contract,
        sql_result=sql_result,
        python_result=python_result,
        evidence=evidence,
    )
    details = check.details if isinstance(check.details, dict) else {}
    return {
        key: tuple(str(item) for item in details.get(key, ()) if str(item))
        for key in (
            "dimensions",
            "unexpected_dimensions",
            "filters",
            "unexpected_filters",
            "aggregations",
        )
    }


def _statement_contract_gaps(
    *,
    statement: exp.Expression,
    contract: AnalysisContractResponse,
    result_rows: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> dict[str, list[str]]:
    group_columns = _statement_group_columns(statement)
    required_dimensions = _contract_grain_columns(contract)
    actual_filters, unexpected_filters = _statement_filters(statement)
    unexpected_filters.extend(
        _unexpected_population_operations(statement, result_rows=result_rows)
    )
    unmatched_actual = list(actual_filters)
    missing_filters: list[str] = []
    for expected in contract.filters:
        match_index = next(
            (
                index
                for index, actual in enumerate(unmatched_actual)
                if _column_equivalent(actual[0], expected.column)
                and actual[1] == expected.operator
                and _value_equivalent(actual[2], expected.value)
            ),
            None,
        )
        if match_index is None:
            missing_filters.append(
                f"{expected.column}{expected.operator}{expected.value}"
            )
        else:
            unmatched_actual.pop(match_index)
    unexpected_filters.extend(
        f"{column}{operator}{value}"
        for column, operator, value in unmatched_actual
    )
    missing: dict[str, list[str]] = {
        "dimensions": [
            dimension
            for dimension in required_dimensions
            if not _contains_equivalent_column(group_columns, dimension)
        ],
        "filters": missing_filters,
        "aggregations": [
            f"{item.operation}({item.column or '*'})"
            for item in contract.aggregations
            if not _aggregation_present(
                statement=statement,
                operation=item.operation,
                column=item.column,
            )
        ],
    }
    if contract.grain or contract.dimensions or contract.time_field:
        unexpected = [
            column
            for column in sorted(group_columns)
            if not _contains_equivalent_column(required_dimensions, column)
        ]
        if unexpected:
            missing["unexpected_dimensions"] = unexpected
    if unexpected_filters:
        missing["unexpected_filters"] = unexpected_filters
    return missing


def _python_contract_gaps(
    *,
    python_result: PythonAnalysisResponse,
    contract: AnalysisContractResponse,
) -> dict[str, list[str]]:
    missing = _empty_contract_gaps(contract)
    context = python_result.execution_context
    if context is not None:
        missing["dimensions"] = [
            dimension
            for dimension in _contract_grain_columns(contract)
            if not _contains_equivalent_column(
                context.referenced_columns,
                dimension,
            )
        ]
        unmatched_filters = list(context.applied_filters)
        missing_filters: list[str] = []
        for expected in contract.filters:
            match_index = next(
                (
                    index
                    for index, actual in enumerate(unmatched_filters)
                    if _column_equivalent(actual.column, expected.column)
                    and actual.operator == expected.operator
                    and _value_equivalent(actual.value, expected.value)
                ),
                None,
            )
            if match_index is None:
                missing_filters.append(
                    f"{expected.column}{expected.operator}{expected.value}"
                )
            else:
                unmatched_filters.pop(match_index)
        missing["filters"] = missing_filters
    missing["aggregations"] = [
        f"{item.operation}({item.column or '*'})"
        for item in contract.aggregations
        if item.alias not in python_result.statistics
    ]
    return missing


def _native_evidence_contract_gaps(
    contract: AnalysisContractResponse,
    evidence: tuple[dict[str, Any], ...],
) -> dict[str, list[str]] | None:
    native_results = [
        result
        for item in evidence
        if item.get("status", "succeeded") == "succeeded"
        and str(item.get("tool_name") or "") == "aggregate_source_dataset"
        and isinstance((result := _evidence_result(item)), dict)
        and result.get("native_grain") is True
    ]
    if not native_results:
        return None
    return {
        "dimensions": [],
        "filters": [],
        "aggregations": [
            f"{aggregation.operation}({aggregation.column or '*'})"
            for aggregation in contract.aggregations
            if not any(
                _native_grain_matches_aggregation(result, contract, aggregation)
                for result in native_results
            )
        ],
    }


def _empty_contract_gaps(
    contract: AnalysisContractResponse,
) -> dict[str, list[str]]:
    return {
        "dimensions": list(_contract_grain_columns(contract)),
        "filters": [
            f"{item.column}{item.operator}{item.value}" for item in contract.filters
        ],
        "aggregations": [
            f"{item.operation}({item.column or '*'})"
            for item in contract.aggregations
        ],
    }


def _gap_score(gaps: dict[str, list[str]]) -> tuple[int, int, int, int]:
    return (
        sum(len(values) for values in gaps.values()),
        len(gaps.get("unexpected_dimensions", ()))
        + len(gaps.get("unexpected_filters", ())),
        len(gaps.get("dimensions", ())),
        len(gaps.get("filters", ())),
    )


def _contract_grain_columns(contract: AnalysisContractResponse) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value
            for value in (
                *contract.dimensions,
                *(item for item in contract.grain if item.casefold() != "dataset"),
                contract.time_field,
            )
            if value
        )
    )


def _statement_group_columns(statement: exp.Expression) -> set[str]:
    group = statement.args.get("group")
    if group is None:
        return set()
    aliases = {
        select.alias.casefold(): next(select.this.find_all(exp.Column)).name.casefold()
        for select in statement.selects
        if isinstance(select, exp.Alias)
        and select.alias
        and any(True for _ in select.this.find_all(exp.Column))
    }
    return {
        aliases.get(column.name.casefold(), column.name.casefold())
        for column in group.find_all(exp.Column)
    }


def _filter_present(
    statement: exp.Expression,
    column: str,
    operator: str,
    value: object,
) -> bool:
    where = statement.args.get("where")
    if where is None:
        return False
    comparison_type = {
        "=": exp.EQ,
        "!=": exp.NEQ,
        ">": exp.GT,
        ">=": exp.GTE,
        "<": exp.LT,
        "<=": exp.LTE,
    }[operator]
    for comparison in where.find_all(comparison_type):
        left_columns = list(comparison.this.find_all(exp.Column))
        right_value = _sql_value(comparison.expression)
        if (
            any(_column_equivalent(item.name, column) for item in left_columns)
            and right_value is not None
            and _value_equivalent(right_value, value)
        ):
            return True
    return False


def _statement_filters(
    statement: exp.Expression,
) -> tuple[list[tuple[str, str, object]], list[str]]:
    """Return exact conjunctive predicates and reject broader WHERE semantics."""

    where = statement.args.get("where")
    if where is None:
        return [], []
    root = where.this
    if any(True for _ in root.find_all(exp.Or)) or isinstance(root, exp.Or):
        return [], [root.sql(dialect="duckdb")]
    if any(True for _ in root.find_all(exp.Not)) or isinstance(root, exp.Not):
        return [], [root.sql(dialect="duckdb")]

    predicates: list[exp.Expression] = []

    def collect(node: exp.Expression) -> None:
        if isinstance(node, exp.Paren):
            collect(node.this)
        elif isinstance(node, exp.And):
            collect(node.this)
            collect(node.expression)
        else:
            predicates.append(node)

    collect(root)
    actual: list[tuple[str, str, object]] = []
    unexpected: list[str] = []
    operator_by_type: dict[type[exp.Expression], str] = {
        exp.EQ: "=",
        exp.NEQ: "!=",
        exp.GT: ">",
        exp.GTE: ">=",
        exp.LT: "<",
        exp.LTE: "<=",
    }
    reverse_operator = {"=": "=", "!=": "!=", ">": "<", ">=": "<=", "<": ">", "<=": ">="}
    for predicate in predicates:
        operator = operator_by_type.get(type(predicate))
        if operator is None:
            unexpected.append(predicate.sql(dialect="duckdb"))
            continue
        left = predicate.this
        right = predicate.expression
        if isinstance(left, exp.Column) and (value := _sql_value(right)) is not None:
            actual.append((left.name, operator, value))
            continue
        if isinstance(right, exp.Column) and (value := _sql_value(left)) is not None:
            actual.append((right.name, reverse_operator[operator], value))
            continue
        unexpected.append(predicate.sql(dialect="duckdb"))
    return actual, unexpected


def _unexpected_population_operations(
    statement: exp.Expression,
    *,
    result_rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[str]:
    unexpected: list[str] = []
    for key in ("having", "qualify", "offset"):
        node = statement.args.get(key)
        if node is not None:
            unexpected.append(f"{key.upper()} {node.sql(dialect='duckdb')}")

    nested_selects = [
        select
        for select in statement.find_all(exp.Select)
        if select is not statement
    ]
    for select in nested_selects:
        for key in ("where", "having", "qualify", "limit", "offset"):
            node = select.args.get(key)
            if node is not None:
                unexpected.append(
                    f"INNER {key.upper()} {node.sql(dialect='duckdb')}"
                )

    limit = statement.args.get("limit")
    if limit is not None:
        raw_limit = _sql_value(limit.expression)
        try:
            limit_value = int(str(raw_limit)) if raw_limit is not None else None
        except ValueError:
            limit_value = None
        if limit_value is None or not result_rows or len(result_rows) >= limit_value:
            unexpected.append(f"LIMIT {limit.sql(dialect='duckdb')}")
    return unexpected


def _aggregation_present(
    *,
    statement: exp.Expression,
    operation: str,
    column: str | None,
) -> bool:
    node_type = {
        "sum": exp.Sum,
        "avg": exp.Avg,
        "min": exp.Min,
        "max": exp.Max,
        "count": exp.Count,
        "count_distinct": exp.Count,
    }[operation]
    # Only the final SELECT projection can establish the returned metric.
    # Scanning the entire AST lets an unused CTE lend its SUM/AVG to an outer
    # query that actually returns COUNT(*) or another incompatible result.
    for projection in statement.selects:
        nodes = [projection] if isinstance(projection, node_type) else []
        nodes.extend(projection.find_all(node_type))
        for node in nodes:
            node_columns = [item.name for item in node.find_all(exp.Column)]
            if column and not any(
                _column_equivalent(candidate, column) for candidate in node_columns
            ):
                continue
            is_distinct = any(True for _ in node.find_all(exp.Distinct))
            if operation == "count_distinct" and not is_distinct:
                continue
            if operation == "count" and is_distinct:
                continue
            return True
    return False


def _sql_value(node: exp.Expression) -> object | None:
    if isinstance(node, exp.Literal):
        return node.this
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    return None


def _value_equivalent(left: object, right: object) -> bool:
    if isinstance(right, bool):
        return str(left).casefold() == str(right).casefold()
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return False
    return str(left).casefold() == str(right).casefold()


def _column_equivalent(left: str, right: str) -> bool:
    left_folded = left.casefold()
    right_folded = right.casefold()
    return left_folded == right_folded or left_folded.rsplit("__", 1)[-1] == right_folded.rsplit("__", 1)[-1]


def _contains_equivalent_column(columns: object, expected: str) -> bool:
    return any(_column_equivalent(str(column), expected) for column in columns)


def _join_grain_check(
    *,
    contract: AnalysisContractResponse,
    context: MultiDatasetProfileResponse | None,
    evidence: tuple[dict[str, Any], ...],
) -> StatisticalCheckResponse:
    if context is None:
        return StatisticalCheckResponse(
            code="join_grain",
            status="not_applicable",
            severity="info",
            message="单数据集分析不需要 Join 粒度审查。",
        )
    summary = context.join_summary
    expansion = float(summary.get("row_expansion_ratio") or 1)
    native_candidates = [
        (item, result)
        for item in evidence
        if isinstance((result := _evidence_result(item)), dict)
        and result.get("native_grain") is True
        and item.get("status", "succeeded") == "succeeded"
    ]
    skipped = int(summary.get("skipped_join_count") or 0)
    required_aggregations = tuple(
        aggregation
        for aggregation in contract.aggregations
        if aggregation.column and aggregation.operation in {"sum", "avg", "min", "max"}
    )
    if not required_aggregations and contract.metric:
        required_aggregations = (
            AnalysisAggregationResponse(
                operation="sum",
                column=contract.metric,
                alias="native_metric",
            ),
        )
    native_matches = [
        (item, aggregation)
        for item, result in native_candidates
        for aggregation in required_aggregations
        if _native_grain_matches_aggregation(result, contract, aggregation)
    ]
    native_grain = bool(required_aggregations) and all(
        any(match is aggregation for _, match in native_matches)
        for aggregation in required_aggregations
    )
    if (expansion >= 1.05 or skipped) and required_aggregations and not native_grain:
        metrics = ", ".join(
            aggregation.column or "*" for aggregation in required_aggregations
        )
        return StatisticalCheckResponse(
            code="join_grain",
            status="failed",
            severity="error",
            message=(
                f"Join 后行数扩大到 {expansion:.2f} 倍或跳过了不安全连接，且指标 "
                f"{metrics} 未全部获得源表原生粒度证据。"
            ),
            finding_ref="join_prepare",
            details={
                "row_expansion_ratio": expansion,
                "native_grain_evidence": native_grain,
                "native_grain_candidate_count": len(native_candidates),
                "native_grain_match_count": len(native_matches),
                "skipped_join_count": skipped,
            },
        )
    status = "warning" if skipped else "passed"
    return StatisticalCheckResponse(
        code="join_grain",
        status=status,
        severity="warning" if skipped else "info",
        message=(
            f"已检测并跳过 {skipped} 个不安全 Join；其结果不得进入指标汇总。"
            if skipped
            else f"Join 粒度审查通过，行膨胀比例为 {expansion:.2f}。"
        ),
        finding_ref="join_prepare",
        details={
            "row_expansion_ratio": expansion,
            "native_grain_evidence": native_grain,
            "native_grain_candidate_count": len(native_candidates),
            "native_grain_match_count": len(native_matches),
            "skipped_join_count": skipped,
        },
    )


def _native_grain_matches_contract(
    result: dict[str, Any],
    contract: AnalysisContractResponse,
) -> bool:
    return any(
        _native_grain_matches_aggregation(result, contract, aggregation)
        for aggregation in contract.aggregations
    )


def _native_grain_matches_aggregation(
    result: dict[str, Any],
    contract: AnalysisContractResponse,
    aggregation: AnalysisAggregationResponse,
) -> bool:
    metric = str(result.get("metric") or "")
    if (
        not metric
        or not aggregation.column
        or not _column_equivalent(metric, aggregation.column)
        or str(result.get("aggregation") or "") != aggregation.operation
    ):
        return False
    if "__" in aggregation.column:
        expected_source = _normalized_source_reference(
            aggregation.column.rsplit("__", 1)[0]
        )
        actual_source = _normalized_source_reference(
            str(result.get("source_dataset") or "")
        )
        if not actual_source or actual_source != expected_source:
            return False
    source_dataset_id = str(result.get("source_dataset_id") or "")
    if not source_dataset_id or source_dataset_id not in {
        str(dataset_id) for dataset_id in contract.dataset_ids
    }:
        return False

    result_filters = result.get("filters") or ()
    if not isinstance(result_filters, (list, tuple)):
        return False
    for expected in contract.filters:
        if not any(
            isinstance(actual, dict)
            and _column_equivalent(str(actual.get("column") or ""), expected.column)
            and str(actual.get("operator") or "=") == expected.operator
            and _value_equivalent(actual.get("value"), expected.value)
            for actual in result_filters
        ):
            return False
    if len(result_filters) != len(contract.filters):
        return False

    expected_grain = _contract_grain_columns(contract)
    actual_grain = result.get("grain")
    if actual_grain is None:
        actual_grain = [result["group_by"]] if result.get("group_by") else ["dataset"]
    if not isinstance(actual_grain, (list, tuple)):
        return False
    actual_columns = tuple(
        str(item) for item in actual_grain if str(item).casefold() != "dataset"
    )
    return (
        len(actual_columns) == len(expected_grain)
        and all(_contains_equivalent_column(actual_columns, item) for item in expected_grain)
    )


def _normalized_source_reference(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return re.sub(r"(?:dataset)?(?:csv)?$", "", normalized)


def _native_grain_matches_population(
    result: dict[str, Any],
    contract: AnalysisContractResponse,
) -> bool:
    """Validate a separately requested native metric at the contract population/grain."""

    source_dataset_id = str(result.get("source_dataset_id") or "")
    if not source_dataset_id or source_dataset_id not in {
        str(dataset_id) for dataset_id in contract.dataset_ids
    }:
        return False
    aggregation = str(result.get("aggregation") or "")
    allowed_operations = {item.operation for item in contract.aggregations} or {"sum"}
    if aggregation not in allowed_operations:
        return False
    result_filters = result.get("filters") or ()
    if not isinstance(result_filters, (list, tuple)):
        return False
    if len(result_filters) != len(contract.filters):
        return False
    for expected in contract.filters:
        if not any(
            isinstance(actual, dict)
            and _column_equivalent(str(actual.get("column") or ""), expected.column)
            and str(actual.get("operator") or "=") == expected.operator
            and _value_equivalent(actual.get("value"), expected.value)
            for actual in result_filters
        ):
            return False
    expected_grain = _contract_grain_columns(contract)
    actual_grain = result.get("grain")
    if actual_grain is None:
        actual_grain = [result["group_by"]] if result.get("group_by") else ["dataset"]
    if not isinstance(actual_grain, (list, tuple)):
        return False
    actual_columns = tuple(
        str(item) for item in actual_grain if str(item).casefold() != "dataset"
    )
    return len(actual_columns) == len(expected_grain) and all(
        _contains_equivalent_column(actual_columns, item) for item in expected_grain
    )


def _native_claim_identifies_source_metric(
    result: dict[str, Any],
    claim_text: str,
) -> bool:
    source = str(result.get("source_dataset") or "").strip()
    metric = str(result.get("metric") or "").strip()
    folded = claim_text.casefold()
    return bool(
        source
        and metric
        and source.casefold() in folded
        and metric.casefold() in folded
    )


def _numeric_evidence_check(
    verdicts: list[StatisticalFindingVerdictResponse],
    coverage: float,
) -> StatisticalCheckResponse:
    missing = [verdict.title for verdict in verdicts if verdict.status == "failed"]
    return StatisticalCheckResponse(
        code="numeric_evidence",
        status="passed" if coverage == 1 else "failed",
        severity="info" if coverage == 1 else "error",
        message=(
            f"数值结论证据覆盖率为 {coverage:.0%}。"
            if coverage == 1
            else f"数值结论证据覆盖率仅 {coverage:.0%}，缺失：{', '.join(missing[:5])}。"
        ),
        details={"coverage": coverage, "unsupported_findings": missing},
    )


def _comparison_check(
    findings: tuple[InsightFindingResponse, ...],
    comparison: dict[str, Any] | None,
) -> StatisticalCheckResponse:
    count = sum(
        _is_comparison(finding.content) and not _is_descriptive_ranking(finding.content)
        for finding in findings
    )
    if not count:
        return StatisticalCheckResponse(
            code="comparison_support",
            status="not_applicable",
            severity="info",
            message="本次结论不包含比较性陈述。",
        )
    if comparison is None:
        return StatisticalCheckResponse(
            code="comparison_support",
            status="failed",
            severity="error",
            message="比较性结论缺少足够样本或可计算的数值指标。",
        )
    return StatisticalCheckResponse(
        code="comparison_support",
        status="passed",
        severity="info",
        message=(
            f"比较性结论已披露样本量 n={comparison['sample_size']}，"
            "并计算效应量或 95% 置信区间。"
        ),
        details=comparison,
    )


def _is_descriptive_ranking(value: str) -> bool:
    folded = value.casefold()
    return any(
        token in folded
        for token in (
            "排名",
            "前三",
            "前五",
            "最高",
            "最低",
            "最多",
            "最少",
            "top ",
            "highest",
            "lowest",
            "largest",
            "smallest",
        )
    )


def _causal_language_check(
    findings: tuple[InsightFindingResponse, ...],
    contract: AnalysisContractResponse,
) -> StatisticalCheckResponse:
    if contract.causal_claim_allowed:
        return StatisticalCheckResponse(
            code="causal_language",
            status="passed",
            severity="info",
            message="分析契约允许基于已声明实验设计进行因果解释。",
        )
    unsupported = [
        finding.title
        for finding in findings
        if _contains_unqualified_causal_claim(finding.content)
    ]
    return StatisticalCheckResponse(
        code="causal_language",
        status="failed" if unsupported else "passed",
        severity="error" if unsupported else "info",
        message=(
            f"观察性结论仍包含未经限定的因果措辞：{', '.join(unsupported[:5])}。"
            if unsupported
            else "观察性结论未使用未经限定的因果措辞。"
        ),
        details={"unsupported_findings": unsupported},
    )


def _comparison_statistics(
    dataframe: pd.DataFrame,
    contract: AnalysisContractResponse,
) -> dict[str, Any] | None:
    metric = contract.metric
    dimension = next(
        (item for item in contract.dimensions if item in dataframe.columns),
        None,
    )
    if not metric and dimension:
        counts = dataframe[dimension].dropna().value_counts()
        if len(counts) < 2:
            return None
        total = int(counts.sum())
        if total < 2:
            return None
        left_count, right_count = int(counts.iloc[0]), int(counts.iloc[1])
        left_share, right_share = left_count / total, right_count / total
        difference = left_share - right_share
        standard_error = math.sqrt(
            (left_share * (1 - left_share) + right_share * (1 - right_share))
            / total
        )
        effect = 2 * (
            math.asin(math.sqrt(left_share))
            - math.asin(math.sqrt(right_share))
        )
        return {
            "sample_size": total,
            "metric": "count",
            "dimension": dimension,
            "groups": [str(counts.index[0]), str(counts.index[1])],
            "mean_difference": difference,
            "effect_size": effect,
            "confidence_interval": [
                difference - 1.96 * standard_error,
                difference + 1.96 * standard_error,
            ],
        }
    if not metric or metric not in dataframe.columns:
        return None
    values = pd.to_numeric(dataframe[metric], errors="coerce")
    valid = values.dropna()
    if len(valid) < 2:
        return None
    if dimension:
        frame = pd.DataFrame({"group": dataframe[dimension], "value": values}).dropna()
        group_names = list(frame["group"].value_counts().head(2).index)
        if len(group_names) >= 2:
            left = frame.loc[frame["group"] == group_names[0], "value"]
            right = frame.loc[frame["group"] == group_names[1], "value"]
            if len(left) >= 2 and len(right) >= 2:
                difference = float(left.mean() - right.mean())
                standard_error = math.sqrt(
                    float(left.var(ddof=1)) / len(left)
                    + float(right.var(ddof=1)) / len(right)
                )
                pooled_denominator = len(left) + len(right) - 2
                pooled_variance = (
                    ((len(left) - 1) * float(left.var(ddof=1)))
                    + ((len(right) - 1) * float(right.var(ddof=1)))
                ) / pooled_denominator
                effect = (
                    difference / math.sqrt(pooled_variance)
                    if pooled_variance > 0
                    else 0.0
                )
                return {
                    "sample_size": int(len(left) + len(right)),
                    "metric": metric,
                    "dimension": dimension,
                    "groups": [str(group_names[0]), str(group_names[1])],
                    "mean_difference": difference,
                    "effect_size": effect,
                    "confidence_interval": [
                        difference - 1.96 * standard_error,
                        difference + 1.96 * standard_error,
                    ],
                }
    standard_error = float(valid.std(ddof=1)) / math.sqrt(len(valid))
    mean = float(valid.mean())
    return {
        "sample_size": len(valid),
        "metric": metric,
        "mean": mean,
        "effect_size": None,
        "confidence_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
    }


def _overall_status(checks: list[StatisticalCheckResponse]) -> str:
    if any(check.status == "failed" for check in checks):
        return "failed"
    if any(check.status == "warning" for check in checks):
        return "warning"
    return "passed"


def _has_number(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def _is_comparison(text: str) -> bool:
    folded = text.casefold()
    return any(term in folded for term in _COMPARISON_TERMS)


def _contains_unqualified_causal_claim(text: str) -> bool:
    folded = text.casefold()
    if any(disclaimer in folded for disclaimer in _CAUSAL_DISCLAIMERS):
        return False
    return any(term in folded for term in _CAUSAL_REPLACEMENTS)


def _status_label(status: str) -> str:
    return {"passed": "通过", "warning": "有警告", "failed": "失败"}[status]


def _suggestion(code: str) -> str:
    return {
        "population_non_empty": "补充可分析记录后重新运行。",
        "metric_type": "恢复字段类型自动推断，或选择已确认的数值指标。",
        "time_coverage": "提供真实日期字段，或将问题改为非时间序列分析。",
        "request_coverage": "按分析契约重新生成查询，补齐缺失的维度、过滤和聚合。",
        "join_grain": "在来源事实表粒度先聚合，再按安全关系连接。",
        "numeric_evidence": "重新执行 SQL/Python 工具并让结论引用 evidence_id。",
        "comparison_support": "补充样本量、效应量或 95% 置信区间。",
        "causal_language": "改用相关性措辞，或提供受控实验设计和因果识别假设。",
    }.get(code, "根据审查信息重新规划分析。")
