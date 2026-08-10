from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

import duckdb
import pandas as pd
from sqlglot import exp, parse

from app.analysis.python_execution import PythonAnalysisExecutor
from app.analysis.query_intent import SOURCE_METRIC_ALIASES
from app.analysis.services import (
    PlannedAnalysis,
    _apply_plan_filters,
    _run_python,
    _run_sql,
)
from app.schemas.analysis import DatasetProfileResponse, PythonExecutionContextResponse
from app.semantic.service import SemanticLayerService
from app.storage.dataset_store import DatasetStoreRepository


class LoopErrorType(StrEnum):
    TRANSIENT = "transient"
    INVALID_ARGUMENTS = "invalid_arguments"
    SQL_ERROR = "sql_error"
    PYTHON_ERROR = "python_error"
    CHART_ERROR = "chart_error"
    VALIDATION_ERROR = "validation_error"
    POLICY_ERROR = "policy_error"
    DATA_INSUFFICIENT = "data_insufficient"
    PROVIDER_ERROR = "provider_error"
    FATAL_STATE_ERROR = "fatal_state_error"


@dataclass(frozen=True)
class ToolExecution:
    tool_name: str
    arguments: dict[str, Any]
    action_hash: str
    result: dict[str, Any] | None = None
    error_type: LoopErrorType | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    _tool("inspect_analysis_context", "Inspect the fixed question, schema and semantic scope.", {}),
    _tool(
        "inspect_source_datasets",
        "Inspect every allowed source dataset before aggregating joined multi-table data.",
        {},
    ),
    _tool("profile_dataset", "Profile the scoped analysis dataframe.", {}),
    _tool(
        "aggregate_dataset",
        "Aggregate a metric or record count by an optional dimension.",
        {"group_by": {"type": "string"}, "metric": {"type": "string"}, "aggregation": {"type": "string", "enum": ["sum", "avg", "min", "max", "count", "count_distinct"]}},
    ),
    _tool(
        "aggregate_source_dataset",
        (
            "Aggregate one original allowed source dataset at its native grain. "
            "Use this instead of the joined dataset for fact-table amounts that could be duplicated."
        ),
        {
            "dataset": {"type": "string"},
            "group_by": {"type": "string"},
            "metric": {"type": "string"},
            "aggregation": {
                "type": "string",
                "enum": ["sum", "avg", "min", "max", "count", "count_distinct"],
            },
        },
        required=("dataset", "metric", "aggregation"),
    ),
    _tool(
        "detect_anomalies",
        "Find numeric outliers for one metric using a deterministic z-score rule.",
        {"metric": {"type": "string"}, "threshold": {"type": "number"}},
        required=("metric",),
    ),
    _tool(
        "analyze_text",
        "Summarize one text column without sending rows outside the job.",
        {"field": {"type": "string"}},
        required=("field",),
    ),
    _tool("execute_semantic_query", "Execute the persisted published semantic plan deterministically.", {}),
    _tool(
        "execute_safe_sql",
        "Execute one read-only SELECT against the scoped table named dataset.",
        {"sql": {"type": "string"}},
        required=("sql",),
    ),
    _tool(
        "execute_python_analysis",
        (
            "Execute one sandboxed Python analysis attempt. Code must define analyze(df). "
            "Optionally use rows from earlier SQL evidence as the input dataframe."
        ),
        {
            "code": {"type": "string"},
            "evidence_id": {"type": "string"},
        },
        required=("code",),
    ),
    _tool(
        "generate_chart",
        "Create a bounded chart from an earlier evidence item.",
        {"evidence_id": {"type": "string"}, "chart_type": {"type": "string", "enum": ["bar", "line", "pie"]}, "x": {"type": "string"}, "y": {"type": "string"}, "title": {"type": "string"}},
        required=("evidence_id", "chart_type", "x", "y"),
    ),
    _tool(
        "validate_evidence",
        "Run deterministic checks on one earlier evidence item.",
        {"evidence_id": {"type": "string"}},
        required=("evidence_id",),
    ),
)
TOOL_NAMES = frozenset(item["function"]["name"] for item in TOOL_DEFINITIONS)


def canonical_action_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    normalized = dict(arguments)
    if tool_name == "execute_safe_sql" and isinstance(normalized.get("sql"), str):
        normalized["sql"] = _canonical_sql(str(normalized["sql"]))
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()


def canonical_result_hash(result: dict[str, Any] | None) -> str | None:
    if not result:
        return None
    canonical = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _canonical_sql(sql: str) -> str:
    try:
        statements = parse(sql, read="duckdb")
    except Exception:
        return " ".join(sql.split())

    canonical: list[str] = []
    for statement in statements:
        normalized = statement.transform(_strip_sql_alias)
        normalized = normalized.transform(_strip_sql_qualifier)
        canonical.append(
            normalized.sql(
                dialect="duckdb",
                normalize=True,
                pretty=False,
                comments=False,
            )
        )
    return ";".join(canonical)


def _strip_sql_alias(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Alias):
        return node.this.copy()
    return node


def _strip_sql_qualifier(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Table):
        normalized = node.copy()
        normalized.set("alias", None)
        return normalized
    if isinstance(node, exp.Column) and node.table:
        normalized = node.copy()
        normalized.set("table", None)
        return normalized
    return node


def error_fingerprint(error_type: LoopErrorType, message: str) -> str:
    normalized = " ".join(message.lower().split())[:500]
    return hashlib.sha256(f"{error_type}:{normalized}".encode()).hexdigest()


def classify_tool_error(tool_name: str, exc: Exception) -> LoopErrorType:
    text = str(exc).lower()
    if isinstance(exc, (KeyError, TypeError)) or "required" in text or "unknown field" in text:
        return LoopErrorType.INVALID_ARGUMENTS
    if "forbidden" in text or "only select" in text or "policy" in text:
        return LoopErrorType.POLICY_ERROR
    if "timeout" in text or "temporar" in text:
        return LoopErrorType.TRANSIENT
    if tool_name == "execute_safe_sql":
        return LoopErrorType.SQL_ERROR
    if tool_name == "execute_python_analysis":
        return LoopErrorType.PYTHON_ERROR
    if tool_name == "generate_chart":
        return LoopErrorType.CHART_ERROR
    if tool_name == "validate_evidence":
        return LoopErrorType.VALIDATION_ERROR
    if "no data" in text or "empty" in text:
        return LoopErrorType.DATA_INSUFFICIENT
    return LoopErrorType.FATAL_STATE_ERROR


def load_evidence_result(
    repository: DatasetStoreRepository,
    dataset_id: UUID,
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    """Load one evidence payload regardless of its storage representation."""

    result = evidence.get("result")
    if isinstance(result, dict):
        return result
    artifact_id = evidence.get("artifact_id")
    if not artifact_id:
        return None
    artifact = repository.get_artifact(dataset_id, UUID(str(artifact_id)))
    content = artifact.get("content")
    if not isinstance(content, dict):
        return None
    nested = content.get("result")
    return nested if isinstance(nested, dict) else content


class AgentToolRuntime:
    """Read-only, job-scoped tool view. Identity and dataset scope are server supplied."""

    def __init__(
        self,
        *,
        repository: DatasetStoreRepository,
        job_id: UUID,
        dataset_id: UUID,
        allowed_dataset_ids: tuple[UUID, ...],
        dataframe: pd.DataFrame,
        question: str,
        profile: DatasetProfileResponse,
        plan: PlannedAnalysis,
        planner_decision: dict[str, Any] | None,
        python_executor: PythonAnalysisExecutor,
        evidence: tuple[dict[str, Any], ...] = (),
    ) -> None:
        if dataset_id not in allowed_dataset_ids:
            raise ValueError("Primary dataset is outside the job scope.")
        for allowed_id in allowed_dataset_ids:
            repository.get_dataset(allowed_id)
        self.repository = repository
        self.job_id = job_id
        self.dataset_id = dataset_id
        self.allowed_dataset_ids = allowed_dataset_ids
        self.source_dataframe = dataframe
        self.source_row_count = len(dataframe)
        self.dataframe = _apply_plan_filters(dataframe, plan.filters)
        self.question = question
        self.profile = profile
        self.plan = plan
        self.planner_decision = planner_decision
        self.python_executor = python_executor
        self._source_frames: dict[UUID, pd.DataFrame] = {}
        self.evidence: dict[str, dict[str, Any]] = {}
        for item in evidence:
            hydrated = dict(item)
            hydrated["result"] = load_evidence_result(
                repository,
                dataset_id,
                hydrated,
            )
            self.evidence[str(hydrated.get("evidence_id"))] = hydrated

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecution:
        action_hash = canonical_action_hash(tool_name, arguments)
        if tool_name not in TOOL_NAMES:
            return ToolExecution(tool_name, arguments, action_hash, error_type=LoopErrorType.POLICY_ERROR, error="Tool is not in the job allowlist.")
        try:
            result = getattr(self, f"_tool_{tool_name}")(arguments)
            return ToolExecution(tool_name, arguments, action_hash, result=_compact(result))
        except Exception as exc:
            return ToolExecution(tool_name, arguments, action_hash, error_type=classify_tool_error(tool_name, exc), error=str(exc)[:1200])

    def _tool_inspect_analysis_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": self.question,
            "dataset_id": str(self.dataset_id),
            "allowed_dataset_ids": [str(item) for item in self.allowed_dataset_ids],
            "columns": [str(item) for item in self.dataframe.columns],
            "row_count": len(self.dataframe),
            "planned_route": self.plan.route,
            "metric": self.plan.metric_column,
            "dimension": self.plan.category_column,
            "time": self.plan.time_column,
            "semantic_model_id": self.planner_decision.get("semantic_model_id") if self.planner_decision else None,
        }

    def _tool_inspect_source_datasets(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        datasets: list[dict[str, Any]] = []
        for dataset_id in self.allowed_dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            frame = self._source_dataframe(dataset_id)
            datasets.append(
                {
                    "dataset_id": str(dataset_id),
                    "name": dataset.name,
                    "row_count": len(frame),
                    "columns": [str(column) for column in frame.columns[:80]],
                    "native_grain": True,
                }
            )
        return {
            "datasets": datasets,
            "instruction": (
                "Aggregate monetary facts on their original source dataset before joining "
                "multiple one-to-many tables."
            ),
        }

    def _tool_profile_dataset(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.profile.model_dump(mode="json")

    def _tool_aggregate_dataset(self, arguments: dict[str, Any]) -> dict[str, Any]:
        frame = self.dataframe
        group_by = _optional_column(frame, arguments.get("group_by"))
        metric = _optional_column(frame, arguments.get("metric"))
        aggregation = str(arguments.get("aggregation") or ("sum" if metric else "count"))
        if aggregation not in {"sum", "avg", "min", "max", "count", "count_distinct"}:
            raise ValueError("Unsupported aggregation.")
        if metric is None:
            if group_by:
                result = frame.groupby(group_by, dropna=False).size().reset_index(name="row_count")
            else:
                result = pd.DataFrame([{"row_count": len(frame)}])
        else:
            values = pd.to_numeric(frame[metric], errors="coerce")
            work = frame.assign(__metric=values).dropna(subset=["__metric"])
            function = {"sum": "sum", "avg": "mean", "min": "min", "max": "max", "count": "count", "count_distinct": "nunique"}[aggregation]
            if group_by:
                result = work.groupby(group_by, dropna=False)["__metric"].agg(function).reset_index(name=f"{aggregation}_{metric}")
            else:
                value = getattr(work["__metric"], function)()
                result = pd.DataFrame([{f"{aggregation}_{metric}": value}])
        rows = _frame_rows(result.head(200))
        return {"rows": rows, "row_count": len(rows), "group_by": group_by, "metric": metric, "aggregation": aggregation}

    def _tool_aggregate_source_dataset(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        dataset_id = self._resolve_source_dataset(arguments.get("dataset"))
        dataset = self.repository.get_dataset(dataset_id)
        frame = self._source_dataframe(dataset_id)
        scoped = self._source_contract_scope(frame)
        if scoped["missing_filters"] or scoped["missing_dimensions"]:
            missing = (*scoped["missing_filters"], *scoped["missing_dimensions"])
            raise ValueError(
                "Source dataset cannot satisfy the contract population/grain: "
                + ", ".join(missing)
            )
        original_row_count = len(frame)
        frame = _apply_plan_filters(frame, scoped["filters"])
        required_group_by = next(iter(scoped["dimensions"]), None)
        group_by = _optional_column(
            frame, arguments.get("group_by") or required_group_by
        )
        if len(scoped["dimensions"]) > 1 or (
            scoped["dimensions"] and group_by != required_group_by
        ):
            raise ValueError("Source aggregation does not preserve the contract grain.")
        metric = _required_column(frame, arguments.get("metric"))
        aggregation = str(arguments.get("aggregation") or "sum")
        result = _aggregate_frame(
            frame,
            group_by=group_by,
            metric=metric,
            aggregation=aggregation,
        )
        return _source_aggregate_result(
            dataset_id=dataset_id,
            dataset_name=dataset.name,
            source_row_count=original_row_count,
            filtered_row_count=len(frame),
            group_by=group_by,
            metric=metric,
            aggregation=aggregation,
            filters=scoped["filters"],
            result=result,
        )

    def _tool_detect_anomalies(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metric = _required_column(self.dataframe, arguments.get("metric"))
        threshold = max(1.0, min(float(arguments.get("threshold") or 3.0), 10.0))
        values = pd.to_numeric(self.dataframe[metric], errors="coerce")
        deviation = float(values.std(ddof=0) or 0)
        if not deviation:
            return {"metric": metric, "threshold": threshold, "count": 0, "rows": []}
        scores = (values - float(values.mean())) / deviation
        matches = self.dataframe.loc[scores.abs() >= threshold].copy().head(100)
        matches["z_score"] = scores.loc[matches.index]
        return {"metric": metric, "threshold": threshold, "count": int((scores.abs() >= threshold).sum()), "rows": _frame_rows(matches)}

    def _tool_analyze_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        field = _required_column(self.dataframe, arguments.get("field"))
        values = self.dataframe[field].dropna().astype(str)
        if values.empty:
            raise ValueError("Text field has no data.")
        top = values.value_counts().head(20)
        return {"field": field, "non_empty": len(values), "average_length": round(float(values.str.len().mean()), 2), "top_values": [{"value": str(key)[:200], "count": int(value)} for key, value in top.items()]}

    def _tool_execute_semantic_query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.planner_decision:
            raise ValueError("No published semantic decision is attached to this job.")
        return SemanticLayerService(self.repository).execute_semantic_plan(self.planner_decision)

    def _tool_execute_safe_sql(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sql = str(arguments.get("sql") or "").strip()
        _validate_safe_dataset_sql(sql)
        connection = duckdb.connect(":memory:")
        try:
            connection.register("dataset", self.dataframe)
            result = connection.execute(sql).fetchdf().head(1000)
        finally:
            connection.close()
        return {"sql": sql, "rows": _frame_rows(result), "explanation": "Loop safe SQL over the job-scoped analysis dataframe."}

    def _tool_execute_python_analysis(self, arguments: dict[str, Any]) -> dict[str, Any]:
        code = str(arguments.get("code") or "").strip()
        if not code:
            raise ValueError("Python code is required.")
        evidence_id = str(arguments.get("evidence_id") or "").strip() or None
        frame = self.dataframe
        source_row_count = self.source_row_count
        applied_filters = self.plan.filters
        if evidence_id:
            source = self._required_evidence(evidence_id)
            source_result = source.get("result") if isinstance(source.get("result"), dict) else {}
            rows = _evidence_rows(source_result)
            if not rows:
                raise ValueError("Referenced evidence has no tabular rows for Python analysis.")
            frame = pd.DataFrame.from_records(rows)
            source_row_count = len(frame)
            applied_filters = ()
        python_result = self.python_executor(code, frame).model_copy(
            update={
                "execution_context": PythonExecutionContextResponse(
                    source_row_count=source_row_count,
                    input_row_count=len(frame),
                    input_evidence_id=evidence_id,
                    applied_filters=applied_filters,
                    referenced_columns=_python_referenced_columns(code, frame.columns),
                )
            }
        )
        return {
            "code": code,
            "python_result": python_result.model_dump(mode="json"),
        }

    def _tool_generate_chart(self, arguments: dict[str, Any]) -> dict[str, Any]:
        evidence = self._required_evidence(arguments.get("evidence_id"))
        result = evidence.get("result") if isinstance(evidence.get("result"), dict) else {}
        rows = _evidence_rows(result)
        if not rows:
            charts = (result.get("python_result") or {}).get("charts") or []
            rows = charts[0].get("data") if charts and isinstance(charts[0], dict) else []
        if not isinstance(rows, list) or not rows:
            raise ValueError("Referenced evidence has no chartable rows.")
        x, y = str(arguments.get("x") or ""), str(arguments.get("y") or "")
        if x not in rows[0] or y not in rows[0]:
            raise KeyError("Chart x/y field is not present in evidence rows.")
        return {"chart": {"title": str(arguments.get("title") or "自主分析图表"), "chart_type": str(arguments.get("chart_type") or "bar"), "spec": {"x": x, "y": y}, "data": rows[:500], "explanation": "Generated from verified loop evidence.", "related_finding_ids": []}}

    def _tool_validate_evidence(self, arguments: dict[str, Any]) -> dict[str, Any]:
        evidence = self._required_evidence(arguments.get("evidence_id"))
        result = evidence.get("result")
        valid = isinstance(result, dict) and bool(result)
        return {"valid": valid, "evidence_id": str(evidence.get("evidence_id")), "checks": {"non_empty": valid, "scoped": True, "bounded": len(json.dumps(result, default=str)) <= 250_000}}

    def _required_evidence(self, value: Any) -> dict[str, Any]:
        evidence = self.evidence.get(str(value or ""))
        if evidence is None:
            raise KeyError("Unknown evidence_id.")
        return evidence

    def required_source_aggregates(
        self,
        excluded: set[tuple[str, str, str]] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Compute source-grain totals explicitly requested by a multi-table question."""
        if len(self.allowed_dataset_ids) < 2:
            return ()
        question = _normalized_source_name(self.question)
        requested: list[dict[str, Any]] = []
        excluded = excluded or set()
        for dataset_id in self.allowed_dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            if _normalized_source_name(dataset.name) not in question:
                continue
            frame = self._source_dataframe(dataset_id)
            scoped = self._source_contract_scope(frame)
            if (
                scoped["missing_filters"]
                or scoped["missing_dimensions"]
                or len(scoped["dimensions"]) > 1
            ):
                continue
            group_by = next(iter(scoped["dimensions"]), None)
            metrics = [
                metric
                for metric, metric_aliases in SOURCE_METRIC_ALIASES.items()
                if metric in frame.columns
                and any(
                    _normalized_source_name(alias) in question
                    for alias in metric_aliases
                )
                and (str(dataset_id), metric, "sum") not in excluded
            ]
            if not metrics:
                continue
            filtered_frame = _apply_plan_filters(frame, scoped["filters"])
            for metric in metrics:
                result = _aggregate_frame(
                    filtered_frame,
                    group_by=group_by,
                    metric=metric,
                    aggregation="sum",
                )
                requested.append(
                    _source_aggregate_result(
                        dataset_id=dataset_id,
                        dataset_name=dataset.name,
                        source_row_count=len(frame),
                        filtered_row_count=len(filtered_frame),
                        group_by=group_by,
                        metric=metric,
                        aggregation="sum",
                        filters=scoped["filters"],
                        result=result,
                    )
                )
        return tuple(requested)

    def _source_contract_scope(self, frame: pd.DataFrame) -> dict[str, Any]:
        filters = []
        missing_filters: list[str] = []
        for condition in self.plan.filters:
            column = _source_contract_column(frame, condition.column)
            if column is None:
                missing_filters.append(condition.column)
                continue
            filters.append(condition.model_copy(update={"column": column}))

        dimensions = []
        missing_dimensions: list[str] = []
        contract_dimensions = tuple(
            dict.fromkeys(
                item
                for item in (*self.plan.requested_dimensions, self.plan.time_column)
                if item
            )
        )
        for dimension in contract_dimensions:
            column = _source_contract_column(frame, dimension)
            if column is None:
                missing_dimensions.append(dimension)
                continue
            dimensions.append(column)
        return {
            "filters": tuple(filters),
            "dimensions": tuple(dimensions),
            "missing_filters": tuple(missing_filters),
            "missing_dimensions": tuple(missing_dimensions),
        }

    def source_relationships(self) -> dict[str, Any]:
        """Profile shared identifier keys across the original scoped datasets."""
        if len(self.allowed_dataset_ids) < 2:
            return {"relationships": [], "risk_count": 0}
        sources: list[tuple[UUID, str, pd.DataFrame]] = []
        for dataset_id in self.allowed_dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            sources.append((dataset_id, dataset.name, self._source_dataframe(dataset_id)))
        relationships: list[dict[str, Any]] = []
        for left_index, (left_id, left_name, left_frame) in enumerate(sources):
            for right_id, right_name, right_frame in sources[left_index + 1 :]:
                shared_keys = sorted(
                    column
                    for column in set(map(str, left_frame.columns)).intersection(
                        map(str, right_frame.columns)
                    )
                    if _relationship_key_name(column)
                )
                for key in shared_keys:
                    relationship = _source_key_relationship(
                        left_id=left_id,
                        left_name=left_name,
                        left_values=left_frame[key],
                        right_id=right_id,
                        right_name=right_name,
                        right_values=right_frame[key],
                        key=key,
                    )
                    if relationship is not None:
                        relationships.append(relationship)
        relationships.sort(
            key=lambda item: (
                float(item.get("overlap_rate") or 0),
                -int(item.get("left_duplicate_count") or 0)
                - int(item.get("right_duplicate_count") or 0),
            ),
            reverse=True,
        )
        bounded = relationships[:24]
        return {
            "relationships": bounded,
            "risk_count": sum(
                1
                for item in bounded
                if item.get("relationship_type") in {"one_to_many", "many_to_one", "many_to_many"}
            ),
            "method": "source_key_uniqueness_and_overlap",
        }

    def _source_dataframe(self, dataset_id: UUID) -> pd.DataFrame:
        if dataset_id not in self.allowed_dataset_ids:
            raise ValueError("Source dataset is outside the job scope.")
        cached = self._source_frames.get(dataset_id)
        if cached is not None:
            return cached
        records = self.repository.read_analysis_records(dataset_id)
        if not records:
            raise ValueError("Source dataset has no analysis records.")
        frame = pd.DataFrame.from_records(records)
        self._source_frames[dataset_id] = frame
        return frame

    def _resolve_source_dataset(self, value: Any) -> UUID:
        requested = str(value or "").strip()
        normalized = _normalized_source_name(requested)
        for dataset_id in self.allowed_dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            if requested == str(dataset_id) or normalized == _normalized_source_name(
                dataset.name
            ):
                return dataset_id
        raise KeyError(f"Unknown or disallowed source dataset: {requested}")

    def legacy_fallback(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        sql_result = None
        if self.plan.route in {"sql", "hybrid"}:
            sql_result = _run_sql(self.source_dataframe, self.plan)
            output["sql_result"] = sql_result.model_dump(mode="json")
        if self.plan.route in {"python", "hybrid"}:
            referenced_columns = tuple(
                dict.fromkeys(
                    column
                    for column in (
                        *self.plan.requested_dimensions,
                        self.plan.time_column,
                        self.plan.metric_column,
                    )
                    if column and column in self.dataframe.columns
                )
            )
            python_result = _run_python(
                self.source_dataframe,
                self.plan,
                sql_result,
                self.question,
            ).model_copy(
                update={
                    "execution_context": PythonExecutionContextResponse(
                        source_row_count=self.source_row_count,
                        input_row_count=len(self.dataframe),
                        applied_filters=self.plan.filters,
                        referenced_columns=referenced_columns,
                    )
                }
            )
            output["python_result"] = python_result.model_dump(mode="json")
        return output


def _evidence_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("rows")
    if not isinstance(rows, list) and isinstance(result.get("sql_result"), dict):
        rows = result["sql_result"].get("rows")
    if not isinstance(rows, (list, tuple)):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _python_referenced_columns(code: str, columns: Any) -> tuple[str, ...]:
    available = tuple(str(column) for column in columns)
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return ()
    literal_names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    attribute_names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "df"
    }
    referenced = literal_names | attribute_names
    return tuple(column for column in available if column in referenced)


def _validate_safe_dataset_sql(sql: str) -> None:
    if not sql:
        raise ValueError("SQL is required.")
    statements = parse(sql, read="duckdb")
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise ValueError("Only one SELECT statement is allowed.")
    root = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command, exp.Copy, exp.Attach, exp.Pragma)
    if any(root.find(kind) is not None for kind in forbidden):
        raise ValueError("SQL contains a forbidden statement.")
    forbidden_functions = {"read_csv", "read_csv_auto", "read_parquet", "sqlite_scan", "postgres_scan", "httpfs", "glob"}
    if any(str(function.sql_name()).lower() in forbidden_functions for function in root.find_all(exp.Func)):
        raise ValueError("External table functions are forbidden.")
    tables = {str(table.name).lower() for table in root.find_all(exp.Table)}
    ctes = {str(cte.alias_or_name).lower() for cte in root.find_all(exp.CTE)}
    if any(table != "dataset" and table not in ctes for table in tables):
        raise ValueError("SQL may only read the job-scoped dataset table.")
    if any(str(join.args.get("kind") or "").upper() in {"CROSS", "NATURAL"} for join in root.find_all(exp.Join)):
        raise ValueError("CROSS and NATURAL JOIN are forbidden.")


def _required_column(frame: pd.DataFrame, value: Any) -> str:
    column = str(value or "")
    if column not in frame.columns:
        raise KeyError(f"Unknown field: {column}")
    return column


def _optional_column(frame: pd.DataFrame, value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _required_column(frame, value)


def _source_contract_column(frame: pd.DataFrame, value: str) -> str | None:
    if value in frame.columns:
        return value
    suffix = value.rsplit("__", 1)[-1]
    matches = [str(column) for column in frame.columns if str(column) == suffix]
    return matches[0] if len(matches) == 1 else None


def _aggregate_frame(
    frame: pd.DataFrame,
    *,
    group_by: str | None,
    metric: str,
    aggregation: str,
) -> pd.DataFrame:
    if aggregation not in {"sum", "avg", "min", "max", "count", "count_distinct"}:
        raise ValueError("Unsupported aggregation.")
    values = pd.to_numeric(frame[metric], errors="coerce")
    work = frame.assign(__metric=values).dropna(subset=["__metric"])
    function = {
        "sum": "sum",
        "avg": "mean",
        "min": "min",
        "max": "max",
        "count": "count",
        "count_distinct": "nunique",
    }[aggregation]
    output_name = f"{aggregation}_{metric}"
    if group_by:
        return (
            work.groupby(group_by, dropna=False)["__metric"]
            .agg(function)
            .reset_index(name=output_name)
        )
    return pd.DataFrame([{output_name: getattr(work["__metric"], function)()}])


def _source_aggregate_result(
    *,
    dataset_id: UUID,
    dataset_name: str,
    source_row_count: int,
    filtered_row_count: int,
    group_by: str | None,
    metric: str,
    aggregation: str,
    filters: tuple[Any, ...],
    result: pd.DataFrame,
) -> dict[str, Any]:
    quoted_metric = metric.replace('"', '""')
    alias = f"{aggregation}_{metric}".replace('"', '""')
    group_sql = ""
    if group_by:
        quoted_group = group_by.replace('"', '""')
        group_sql = f'\"{quoted_group}\", '
    source_name = dataset_name.replace('"', '""')
    sql = (
        f'SELECT {group_sql}{aggregation.upper()}(CAST(\"{quoted_metric}\" AS DOUBLE)) '
        f'AS \"{alias}\" FROM \"{source_name}\"'
    )
    where_parts = [
        f'\"{item.column.replace(chr(34), chr(34) * 2)}\" {item.operator} '
        + _source_sql_literal(item.value)
        for item in filters
    ]
    if where_parts:
        sql += f" WHERE {' AND '.join(where_parts)}"
    if group_by:
        sql += f' GROUP BY \"{group_by.replace(chr(34), chr(34) * 2)}\"'
    rows = _frame_rows(result.head(1000))
    return {
        "sql": sql,
        "rows": rows,
        "explanation": (
            f"Source-grain {aggregation} of {metric} from {dataset_name}; joined rows were not used."
        ),
        "source_dataset_id": str(dataset_id),
        "source_dataset": dataset_name,
        "source_row_count": source_row_count,
        "filtered_row_count": filtered_row_count,
        "group_by": group_by,
        "grain": [group_by] if group_by else ["dataset"],
        "metric": metric,
        "aggregation": aggregation,
        "filters": [item.model_dump(mode="json") for item in filters],
        "native_grain": True,
    }


def _source_sql_literal(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _relationship_key_name(column: str) -> bool:
    normalized = _normalized_source_name(column)
    return normalized == "id" or normalized.endswith(("id", "key"))


def _source_key_relationship(
    *,
    left_id: UUID,
    left_name: str,
    left_values: pd.Series,
    right_id: UUID,
    right_name: str,
    right_values: pd.Series,
    key: str,
) -> dict[str, Any] | None:
    left = left_values.dropna().astype(str).str.strip()
    right = right_values.dropna().astype(str).str.strip()
    left = left[left != ""]
    right = right[right != ""]
    if left.empty or right.empty:
        return None
    left_distinct = set(left.unique())
    right_distinct = set(right.unique())
    overlap = left_distinct.intersection(right_distinct)
    overlap_rate = len(overlap) / max(min(len(left_distinct), len(right_distinct)), 1)
    if overlap_rate < 0.5:
        return None
    left_unique = len(left) == len(left_distinct)
    right_unique = len(right) == len(right_distinct)
    relationship_type = {
        (True, True): "one_to_one",
        (True, False): "one_to_many",
        (False, True): "many_to_one",
        (False, False): "many_to_many",
    }[(left_unique, right_unique)]
    risk_note = ""
    if relationship_type == "many_to_many":
        risk_note = (
            "Both source keys repeat; a direct row-level join can multiply records and duplicate measures."
        )
    elif relationship_type in {"one_to_many", "many_to_one"}:
        risk_note = (
            "The child-side key repeats; parent-grain measures can be duplicated after the join."
        )
    return {
        "left_dataset_id": str(left_id),
        "left_dataset": left_name,
        "left_column": key,
        "left_non_null_count": len(left),
        "left_distinct_count": len(left_distinct),
        "left_duplicate_count": len(left) - len(left_distinct),
        "left_key_unique": left_unique,
        "right_dataset_id": str(right_id),
        "right_dataset": right_name,
        "right_column": key,
        "right_non_null_count": len(right),
        "right_distinct_count": len(right_distinct),
        "right_duplicate_count": len(right) - len(right_distinct),
        "right_key_unique": right_unique,
        "overlap_count": len(overlap),
        "overlap_rate": round(overlap_rate, 4),
        "relationship_type": relationship_type,
        "risk_note": risk_note,
    }


def _normalized_source_name(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _frame_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.where(frame.notna(), None).to_json(orient="records", force_ascii=False, date_format="iso"))


def _compact(value: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= 250_000:
        return value
    compact = dict(value)
    if isinstance(compact.get("rows"), list):
        compact["rows"] = compact["rows"][:200]
        compact["truncated"] = True
    return compact
