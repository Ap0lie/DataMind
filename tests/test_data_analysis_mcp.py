from __future__ import annotations

import pytest

from app.core.enums import McpCapability
from app.mcp.bootstrap import build_mock_mcp_runtime
from app.mcp.data_analysis_server import DataAnalysisMCPServer, InMemoryDataAnalysisBackend
from app.mcp.runtime import InMemoryMCPRuntime


@pytest.mark.asyncio
async def test_data_analysis_mcp_registers_tools() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(DataAnalysisMCPServer(InMemoryDataAnalysisBackend()))

    registry = await runtime.discover(McpCapability.DATA_ANALYSIS)
    tool_names = {tool.name for tool in registry.tools}

    assert tool_names == {"profile_dataset", "aggregate_dataset", "detect_anomalies"}


@pytest.mark.asyncio
async def test_profile_dataset_returns_column_statistics() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(DataAnalysisMCPServer(InMemoryDataAnalysisBackend()))

    result = await runtime.invoke(
        "profile_dataset",
        {
            "dataset_id": "sales-demo",
            "records": (
                {"region": "apac", "revenue": 100.0},
                {"region": "emea", "revenue": 150.0},
                {"region": "apac", "revenue": None},
            ),
        },
    )

    assert result.ok
    assert result.data["row_count"] == 3
    revenue = next(column for column in result.data["columns"] if column["name"] == "revenue")
    assert revenue["inferred_type"] == "numeric"
    assert revenue["missing_count"] == 1
    assert revenue["mean"] == 125.0


@pytest.mark.asyncio
async def test_aggregate_dataset_groups_records() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(DataAnalysisMCPServer(InMemoryDataAnalysisBackend()))

    result = await runtime.invoke(
        "aggregate_dataset",
        {
            "records": (
                {"region": "apac", "revenue": 100.0},
                {"region": "apac", "revenue": 150.0},
                {"region": "emea", "revenue": 80.0},
            ),
            "group_by": ("region",),
            "metrics": ({"column": "revenue", "operation": "sum", "alias": "total_revenue"},),
        },
    )

    assert result.ok
    rows = {row["region"]: row["total_revenue"] for row in result.data["rows"]}
    assert rows == {"apac": 250.0, "emea": 80.0}


@pytest.mark.asyncio
async def test_detect_anomalies_returns_outliers() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(DataAnalysisMCPServer(InMemoryDataAnalysisBackend()))

    result = await runtime.invoke(
        "detect_anomalies",
        {
            "records": (
                {"day": "d1", "revenue": 100.0},
                {"day": "d2", "revenue": 102.0},
                {"day": "d3", "revenue": 500.0},
            ),
            "columns": ("revenue",),
            "zscore_threshold": 1.0,
        },
    )

    assert result.ok
    assert result.data["anomalies"]
    assert result.data["anomalies"][0]["row_index"] == 2


@pytest.mark.asyncio
async def test_bootstrap_registers_data_analysis_nlp_and_model_router() -> None:
    runtime = await build_mock_mcp_runtime()

    filesystem_registry = await runtime.discover(McpCapability.FILESYSTEM)
    data_analysis_registry = await runtime.discover(McpCapability.DATA_ANALYSIS)
    nlp_registry = await runtime.discover(McpCapability.NLP)
    model_router_registry = await runtime.discover(McpCapability.MODEL_ROUTER)

    assert filesystem_registry.tools
    assert data_analysis_registry.tools
    assert nlp_registry.tools
    assert model_router_registry.tools
