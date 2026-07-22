from __future__ import annotations

import pytest

from app.core.enums import McpCapability
from app.mcp.mock_server import MockMCPServer
from app.mcp.models import MCPInvocationStatus, MCPServer, MCPTool
from app.mcp.runtime import InMemoryMCPRuntime


@pytest.mark.asyncio
async def test_registry_discovers_nlp_tools() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.nlp())

    registry = await runtime.discover(McpCapability.NLP)

    assert "mock-nlp" in registry.servers
    assert {tool.name for tool in registry.tools} == {"language_detection", "sentiment_analysis"}


@pytest.mark.asyncio
async def test_runtime_invokes_language_detection_tool() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.nlp())

    result = await runtime.invoke("language_detection", {"text": "你好, DataMind"})

    assert result.ok
    assert result.status == MCPInvocationStatus.SUCCESS
    assert result.data["language"] == "zh"


@pytest.mark.asyncio
async def test_runtime_invokes_sentiment_analysis_tool() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.nlp())

    result = await runtime.invoke("sentiment_analysis", {"text": "DataMind is excellent"})

    assert result.ok
    assert result.data["sentiment"] == "positive"


@pytest.mark.asyncio
async def test_runtime_returns_tool_not_found_error() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.nlp())

    result = await runtime.invoke("missing_tool", {})

    assert not result.ok
    assert result.status == MCPInvocationStatus.TOOL_NOT_FOUND
    assert result.error is not None


@pytest.mark.asyncio
async def test_runtime_returns_invalid_arguments_error() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.nlp())

    result = await runtime.invoke("language_detection", {})

    assert not result.ok
    assert result.status == MCPInvocationStatus.INVALID_ARGUMENTS
    assert result.error is not None
    assert "Missing required MCP argument" in result.error.message


@pytest.mark.asyncio
async def test_runtime_returns_server_not_found_error() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.nlp())

    result = await runtime.invoke("language_detection", {"text": "hello"}, server_name="missing")

    assert not result.ok
    assert result.status == MCPInvocationStatus.SERVER_NOT_FOUND


@pytest.mark.asyncio
async def test_runtime_returns_timeout_error() -> None:
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(MockMCPServer.slow(delay_seconds=0.1))

    result = await runtime.invoke("slow_tool", {}, timeout_seconds=0.01)

    assert not result.ok
    assert result.status == MCPInvocationStatus.TIMEOUT


@pytest.mark.asyncio
async def test_runtime_retries_failed_invocation() -> None:
    calls = {"count": 0}

    def flaky_handler(_: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return {"ok": True}

    server = MockMCPServer(
        server=MCPServer(
            name="mock-flaky",
            description="Mock flaky server.",
            tools=(
                MCPTool(
                    name="flaky_tool",
                    capability=McpCapability.NLP,
                    description="Fails once then succeeds.",
                    max_retries=1,
                ),
            ),
        ),
        handlers={"flaky_tool": flaky_handler},
    )
    runtime = InMemoryMCPRuntime()
    await runtime.register_server(server)

    result = await runtime.invoke("flaky_tool", {})

    assert result.ok
    assert result.attempts == 2
    assert calls["count"] == 2
