from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.enums import McpCapability
from app.mcp.models import MCPServer, MCPTool

MCPHandler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class MockMCPServer:
    def __init__(self, server: MCPServer, handlers: dict[str, MCPHandler]) -> None:
        self._server = server
        self._handlers = handlers

    @property
    def server(self) -> MCPServer:
        return self._server

    async def list_tools(self) -> tuple[MCPTool, ...]:
        return self._server.tools

    async def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise LookupError(f"Mock MCP tool is not registered: {tool_name}")
        result = handler(arguments)
        if hasattr(result, "__await__"):
            return await result
        return result

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.invoke(tool_name, arguments)

    @classmethod
    def nlp(cls) -> MockMCPServer:
        return cls(
            server=MCPServer(
                name="mock-nlp",
                description="Mock NLP MCP server for unit tests.",
                tools=(
                    MCPTool(
                        name="language_detection",
                        capability=McpCapability.NLP,
                        description="Detect text language.",
                        input_schema={"type": "object", "required": ["text"]},
                        output_schema={"type": "object", "required": ["language", "confidence"]},
                    ),
                    MCPTool(
                        name="sentiment_analysis",
                        capability=McpCapability.NLP,
                        description="Analyze text sentiment.",
                        input_schema={"type": "object", "required": ["text"]},
                        output_schema={"type": "object", "required": ["sentiment", "confidence"]},
                    ),
                ),
            ),
            handlers={
                "language_detection": _detect_language,
                "sentiment_analysis": _analyze_sentiment,
            },
        )

    @classmethod
    def slow(cls, delay_seconds: float) -> MockMCPServer:
        async def slow_handler(_: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(delay_seconds)
            return {"ok": True}

        return cls(
            server=MCPServer(
                name="mock-slow",
                description="Mock slow MCP server for timeout tests.",
                tools=(
                    MCPTool(
                        name="slow_tool",
                        capability=McpCapability.NLP,
                        description="Slow tool for timeout tests.",
                        timeout_seconds=delay_seconds + 1,
                    ),
                ),
            ),
            handlers={"slow_tool": slow_handler},
        )


def _detect_language(arguments: dict[str, Any]) -> dict[str, Any]:
    text = str(arguments.get("text", ""))
    language = "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"
    return {"language": language, "confidence": 0.95}


def _analyze_sentiment(arguments: dict[str, Any]) -> dict[str, Any]:
    text = str(arguments.get("text", "")).lower()
    positive_markers = ("good", "great", "excellent", "love", "喜欢", "优秀")
    negative_markers = ("bad", "poor", "hate", "broken", "糟糕", "讨厌")
    positive_score = sum(marker in text for marker in positive_markers)
    negative_score = sum(marker in text for marker in negative_markers)
    if positive_score > negative_score:
        return {"sentiment": "positive", "confidence": 0.8}
    if negative_score > positive_score:
        return {"sentiment": "negative", "confidence": 0.8}
    return {"sentiment": "neutral", "confidence": 0.6}
