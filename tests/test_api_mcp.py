from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio
async def test_mcp_tools_endpoint_lists_registered_tools() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/mcp/tools")

    assert response.status_code == 200
    payload = response.json()
    tool_names = {tool["name"] for tool in payload["tools"]}
    assert "filesystem_upload_dataset" in tool_names
    assert "filesystem_read_preview" in tool_names
    assert "profile_dataset" in tool_names
    assert "aggregate_dataset" in tool_names
    assert "detect_anomalies" in tool_names
    assert "language_detection" in tool_names
    assert "model_completion" in tool_names


@pytest.mark.asyncio
async def test_mcp_tools_endpoint_filters_by_capability() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/mcp/tools", params={"capability": "data_analysis"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tools"]
    assert {tool["capability"] for tool in payload["tools"]} == {"data_analysis"}


@pytest.mark.asyncio
async def test_mcp_invoke_endpoint_calls_tool() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/mcp/invoke",
            json={
                "tool_name": "language_detection",
                "arguments": {
                    "document_id": "test-doc",
                    "text": "hello",
                    "backend": "rule_based",
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"]
    assert payload["status"] == "success"
    assert payload["data"]["language"] == "en"


@pytest.mark.asyncio
async def test_mcp_invoke_endpoint_returns_invalid_arguments() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/mcp/invoke",
            json={"tool_name": "language_detection", "arguments": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert not payload["ok"]
    assert payload["status"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_mcp_invoke_endpoint_profiles_dataset() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/mcp/invoke",
            json={
                "tool_name": "profile_dataset",
                "arguments": {
                    "dataset_id": "sales-demo",
                    "records": (
                        {"region": "apac", "revenue": 100.0},
                        {"region": "emea", "revenue": 150.0},
                    ),
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"]
    assert payload["data"]["row_count"] == 2
