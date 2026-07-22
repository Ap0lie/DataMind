from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from app.core.enums import McpCapability
from app.mcp.contracts import MCPServerClient
from app.mcp.models import (
    MCPInvocationError,
    MCPInvocationResult,
    MCPInvocationStatus,
    MCPRegistry,
    MCPServer,
    MCPTool,
    MCPToolCatalog,
    MCPToolCatalogItem,
)


class InMemoryMCPRuntime:
    """Async-first MCP runtime for tests and local development."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPServerClient] = {}
        self._servers: dict[str, MCPServer] = {}

    async def register_server(self, client: MCPServerClient) -> None:
        server = client.server
        self._clients[server.name] = client
        self._servers[server.name] = server.model_copy(update={"tools": await client.list_tools()})

    async def discover(self, capability: McpCapability | None = None) -> MCPRegistry:
        if capability is None:
            return MCPRegistry(servers=self._servers.copy())

        filtered_servers: dict[str, MCPServer] = {}
        for server_name, server in self._servers.items():
            tools = tuple(
                tool for tool in server.tools if tool.capability == capability and tool.enabled
            )
            if tools:
                filtered_servers[server_name] = server.model_copy(update={"tools": tools})
        return MCPRegistry(servers=filtered_servers)

    async def catalog(self, capability: McpCapability | None = None) -> MCPToolCatalog:
        registry = await self.discover(capability)
        return MCPToolCatalog(
            tools=tuple(
                MCPToolCatalogItem(
                    server_name=server_name,
                    name=tool.name,
                    capability=tool.capability,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                    timeout_seconds=tool.timeout_seconds,
                    max_retries=tool.max_retries,
                    enabled=tool.enabled,
                )
                for server_name, server in registry.servers.items()
                for tool in server.tools
            )
        )

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        server_name: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> MCPInvocationResult:
        started_at = datetime.now(UTC)
        if server_name is not None and server_name not in self._servers:
            return self._error_result(
                started_at=started_at,
                server_name=server_name,
                tool_name=tool_name,
                status=MCPInvocationStatus.SERVER_NOT_FOUND,
                message="MCP server was not found.",
            )

        candidates = self._find_candidates(tool_name=tool_name, server_name=server_name)
        if not candidates:
            return self._error_result(
                started_at=started_at,
                server_name=server_name or "unknown",
                tool_name=tool_name,
                status=MCPInvocationStatus.TOOL_NOT_FOUND,
                message="MCP tool was not found.",
            )

        enabled_candidates = [
            (candidate_server_name, candidate_client, candidate_tool)
            for candidate_server_name, candidate_client, candidate_tool in candidates
            if candidate_tool.enabled
        ]
        if not enabled_candidates:
            selected_server_name, _, selected_tool = candidates[0]
            return self._error_result(
                started_at=started_at,
                server_name=selected_server_name,
                tool_name=tool_name,
                status=MCPInvocationStatus.TOOL_DISABLED,
                message="MCP tool is disabled.",
                capability=selected_tool.capability,
            )

        selected_server_name, selected_client, selected_tool = enabled_candidates[0]
        validation_error = self._validate_arguments(selected_tool, arguments)
        if validation_error is not None:
            return self._error_result(
                started_at=started_at,
                server_name=selected_server_name,
                tool_name=tool_name,
                status=MCPInvocationStatus.INVALID_ARGUMENTS,
                message=validation_error,
                capability=selected_tool.capability,
            )

        effective_timeout = timeout_seconds or selected_tool.timeout_seconds
        effective_retries = selected_tool.max_retries if max_retries is None else max_retries
        attempts = 0

        while attempts <= effective_retries:
            attempts += 1
            try:
                data = await asyncio.wait_for(
                    selected_client.invoke(tool_name, arguments),
                    timeout=effective_timeout,
                )
                finished_at = datetime.now(UTC)
                return MCPInvocationResult(
                    server_name=selected_server_name,
                    tool_name=tool_name,
                    capability=selected_tool.capability,
                    status=MCPInvocationStatus.SUCCESS,
                    data=data,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=self._duration_ms(started_at, finished_at),
                    attempts=attempts,
                )
            except TimeoutError:
                if attempts > effective_retries:
                    return self._error_result(
                        started_at=started_at,
                        server_name=selected_server_name,
                        tool_name=tool_name,
                        status=MCPInvocationStatus.TIMEOUT,
                        message=f"MCP invocation timed out after {effective_timeout} seconds.",
                        capability=selected_tool.capability,
                        attempts=attempts,
                    )
            except Exception as exc:
                if attempts > effective_retries:
                    return self._error_result(
                        started_at=started_at,
                        server_name=selected_server_name,
                        tool_name=tool_name,
                        status=MCPInvocationStatus.FAILED,
                        message=str(exc),
                        capability=selected_tool.capability,
                        details={"exception_type": exc.__class__.__name__},
                        attempts=attempts,
                    )

            if selected_tool.retry_backoff_seconds > 0:
                await asyncio.sleep(selected_tool.retry_backoff_seconds)

        return self._error_result(
            started_at=started_at,
            server_name=selected_server_name,
            tool_name=tool_name,
            status=MCPInvocationStatus.FAILED,
            message="MCP invocation failed after retry loop.",
            capability=selected_tool.capability,
            attempts=max(attempts, 1),
        )

    def _find_candidates(
        self,
        *,
        tool_name: str,
        server_name: str | None,
    ) -> list[tuple[str, MCPServerClient, MCPTool]]:
        server_names: Iterable[str] = (server_name,) if server_name else self._servers.keys()
        candidates: list[tuple[str, MCPServerClient, MCPTool]] = []
        for name in server_names:
            server = self._servers.get(name)
            client = self._clients.get(name)
            if server is None or client is None or not server.enabled:
                continue
            candidates.extend(
                (name, client, tool)
                for tool in server.tools
                if tool.name == tool_name
            )
        return candidates

    def _error_result(
        self,
        *,
        started_at: datetime,
        server_name: str,
        tool_name: str,
        status: MCPInvocationStatus,
        message: str,
        capability: McpCapability | None = None,
        details: dict[str, Any] | None = None,
        attempts: int = 1,
    ) -> MCPInvocationResult:
        finished_at = datetime.now(UTC)
        return MCPInvocationResult(
            server_name=server_name,
            tool_name=tool_name,
            capability=capability,
            status=status,
            error=MCPInvocationError(code=status, message=message, details=details or {}),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=self._duration_ms(started_at, finished_at),
            attempts=attempts,
        )

    @staticmethod
    def _duration_ms(started_at: datetime, finished_at: datetime) -> float:
        return (finished_at - started_at).total_seconds() * 1000

    @staticmethod
    def _validate_arguments(tool: MCPTool, arguments: dict[str, Any]) -> str | None:
        schema = tool.input_schema
        if not schema:
            return None
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [
                field for field in required if isinstance(field, str) and field not in arguments
            ]
            if missing:
                return f"Missing required MCP argument(s): {', '.join(missing)}."
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, definition in properties.items():
                if name not in arguments or not isinstance(definition, dict):
                    continue
                expected_type = definition.get("type")
                if isinstance(expected_type, str) and not _matches_json_type(
                    arguments[name],
                    expected_type,
                ):
                    return f"Invalid MCP argument '{name}': expected {expected_type}."
        return None


def _matches_json_type(value: Any, expected_type: str) -> bool:
    match expected_type:
        case "string":
            return isinstance(value, str)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case "boolean":
            return isinstance(value, bool)
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, (list, tuple))
        case _:
            return True
