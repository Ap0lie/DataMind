from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean
from typing import Protocol

from app.core.enums import McpCapability
from app.mcp.models import MCPServer, MCPTool
from app.mcp.tool_schemas import (
    AggregationRequest,
    AggregationResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    AnomalyRecord,
    ColumnProfile,
    DataCell,
    DatasetProfileResponse,
    DatasetRequest,
)


class DataAnalysisBackend(Protocol):
    async def profile_dataset(self, request: DatasetRequest) -> DatasetProfileResponse:
        """Return dataset shape, column types, missingness, and numeric summaries."""

    async def aggregate_dataset(self, request: AggregationRequest) -> AggregationResponse:
        """Aggregate tabular records with optional group-by columns."""

    async def detect_anomalies(
        self,
        request: AnomalyDetectionRequest,
    ) -> AnomalyDetectionResponse:
        """Detect simple numeric outliers."""


class DataAnalysisMCPServer:
    def __init__(self, backend: DataAnalysisBackend, name: str = "data-analysis-mcp") -> None:
        self._backend = backend
        self._server = MCPServer(
            name=name,
            description="Data Analysis MCP server for tabular profiling, aggregation, and QA.",
            tools=(
                _analysis_tool(
                    "profile_dataset",
                    "Profile an inline tabular dataset.",
                    DatasetRequest.model_json_schema(),
                    DatasetProfileResponse.model_json_schema(),
                ),
                _analysis_tool(
                    "aggregate_dataset",
                    "Aggregate an inline tabular dataset.",
                    AggregationRequest.model_json_schema(),
                    AggregationResponse.model_json_schema(),
                ),
                _analysis_tool(
                    "detect_anomalies",
                    "Detect z-score anomalies in numeric columns.",
                    AnomalyDetectionRequest.model_json_schema(),
                    AnomalyDetectionResponse.model_json_schema(),
                ),
            ),
        )

    @property
    def server(self) -> MCPServer:
        return self._server

    async def list_tools(self) -> tuple[MCPTool, ...]:
        return self._server.tools

    async def invoke(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        match tool_name:
            case "profile_dataset":
                profile_response = await self._backend.profile_dataset(
                    DatasetRequest.model_validate(arguments)
                )
                return profile_response.model_dump(mode="json")
            case "aggregate_dataset":
                aggregation_response = await self._backend.aggregate_dataset(
                    AggregationRequest.model_validate(arguments)
                )
                return aggregation_response.model_dump(mode="json")
            case "detect_anomalies":
                anomaly_response = await self._backend.detect_anomalies(
                    AnomalyDetectionRequest.model_validate(arguments)
                )
                return anomaly_response.model_dump(mode="json")
            case _:
                raise LookupError(f"Unknown Data Analysis MCP tool: {tool_name}")

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        return await self.invoke(tool_name, arguments)


class InMemoryDataAnalysisBackend:
    async def profile_dataset(self, request: DatasetRequest) -> DatasetProfileResponse:
        columns = tuple(sorted({key for record in request.records for key in record}))
        profiles = tuple(
            _profile_column(column, tuple(record.get(column) for record in request.records))
            for column in columns
        )
        return DatasetProfileResponse(
            dataset_id=request.dataset_id,
            row_count=len(request.records),
            column_count=len(columns),
            columns=profiles,
        )

    async def aggregate_dataset(self, request: AggregationRequest) -> AggregationResponse:
        groups: dict[tuple[DataCell, ...], list[dict[str, DataCell]]] = defaultdict(list)
        for record in request.records:
            key = tuple(record.get(column) for column in request.group_by)
            groups[key].append(record)

        rows: list[dict[str, DataCell]] = []
        for group_key, records in groups.items():
            row: dict[str, DataCell] = dict(
                zip(request.group_by, group_key, strict=True)
            )
            for metric in request.metrics:
                alias = metric.alias or f"{metric.operation}_{metric.column}"
                row[alias] = _aggregate(records, metric.column, metric.operation)
            rows.append(row)
        return AggregationResponse(dataset_id=request.dataset_id, rows=tuple(rows))

    async def detect_anomalies(
        self,
        request: AnomalyDetectionRequest,
    ) -> AnomalyDetectionResponse:
        anomalies: list[AnomalyRecord] = []
        for column in request.columns:
            series = tuple(_numeric(record.get(column)) for record in request.records)
            values = tuple(value for value in series if value is not None)
            if len(values) < 2:
                continue
            center = mean(values)
            variance = sum((value - center) ** 2 for value in values) / len(values)
            stddev = sqrt(variance)
            if stddev == 0:
                continue
            for index, value in enumerate(series):
                if value is None:
                    continue
                zscore = (value - center) / stddev
                if abs(zscore) >= request.zscore_threshold:
                    anomalies.append(
                        AnomalyRecord(
                            row_index=index,
                            column=column,
                            value=value,
                            zscore=zscore,
                        )
                    )
        return AnomalyDetectionResponse(dataset_id=request.dataset_id, anomalies=tuple(anomalies))


def _analysis_tool(
    name: str,
    description: str,
    input_schema: dict[str, object],
    output_schema: dict[str, object],
) -> MCPTool:
    return MCPTool(
        name=name,
        capability=McpCapability.DATA_ANALYSIS,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=15.0,
        max_retries=1,
        retry_backoff_seconds=0.05,
    )


def _profile_column(column: str, values: tuple[DataCell, ...]) -> ColumnProfile:
    non_null = tuple(value for value in values if value is not None)
    numeric_values = tuple(_numeric(value) for value in non_null)
    numeric = tuple(value for value in numeric_values if value is not None)
    inferred_type = _infer_type(non_null, numeric)
    return ColumnProfile(
        name=column,
        inferred_type=inferred_type,
        missing_count=len(values) - len(non_null),
        distinct_count=len(set(non_null)),
        min_value=min(numeric) if numeric else None,
        max_value=max(numeric) if numeric else None,
        mean=mean(numeric) if numeric else None,
    )


def _infer_type(values: tuple[DataCell, ...], numeric: tuple[float, ...]) -> str:
    if not values:
        return "unknown"
    if len(numeric) == len(values):
        return "numeric"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    return "categorical"


def _aggregate(records: list[dict[str, DataCell]], column: str, operation: str) -> DataCell:
    if operation == "count":
        return len([record for record in records if record.get(column) is not None])
    values = tuple(
        value
        for value in (_numeric(record.get(column)) for record in records)
        if value is not None
    )
    if not values:
        return None
    match operation:
        case "sum":
            return sum(values)
        case "avg":
            return mean(values)
        case "min":
            return min(values)
        case "max":
            return max(values)
        case _:
            raise ValueError(f"Unsupported aggregation operation: {operation}")


def _numeric(value: DataCell) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return None
