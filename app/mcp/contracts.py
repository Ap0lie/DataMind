from __future__ import annotations

from typing import Any, Protocol

from app.core.enums import McpCapability
from app.mcp.models import MCPInvocationResult, MCPRegistry, MCPServer, MCPTool, MCPToolCatalog


class MCPServerClient(Protocol):
    @property
    def server(self) -> MCPServer:
        """Return static MCP server metadata."""

    async def list_tools(self) -> tuple[MCPTool, ...]:
        """Return tools exposed by one MCP server."""

    async def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool and return raw structured data."""


class MCPRuntime(Protocol):
    async def register_server(self, client: MCPServerClient) -> None:
        """Register one MCP server client."""

    async def discover(self, capability: McpCapability | None = None) -> MCPRegistry:
        """Discover available MCP servers and tools."""

    async def catalog(self, capability: McpCapability | None = None) -> MCPToolCatalog:
        """Return a flattened MCP tool catalog."""

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        server_name: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> MCPInvocationResult:
        """Dynamically invoke an MCP tool."""


class McpClient(MCPServerClient, Protocol):
    async def list_tools(self) -> tuple[MCPTool, ...]:
        """Return tools exposed by one MCP server."""

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool."""


class McpRegistry(Protocol):
    async def register(self, server_name: str, client: McpClient) -> None:
        """Register one MCP server client."""

    async def find_tools(self, capability: McpCapability) -> tuple[MCPTool, ...]:
        """Find tool descriptors by capability."""


McpToolDescriptor = MCPTool
