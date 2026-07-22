from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import McpCapability


class MCPModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MCPInvocationStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TOOL_NOT_FOUND = "tool_not_found"
    SERVER_NOT_FOUND = "server_not_found"
    TOOL_DISABLED = "tool_disabled"
    INVALID_ARGUMENTS = "invalid_arguments"


class MCPTool(MCPModel):
    name: str = Field(min_length=1)
    capability: McpCapability
    description: str = Field(min_length=1)
    input_schema: dict[str, object] = Field(default_factory=dict)
    output_schema: dict[str, object] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=0, ge=0)
    retry_backoff_seconds: float = Field(default=0.0, ge=0.0)
    enabled: bool = True


class MCPServer(MCPModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tools: tuple[MCPTool, ...] = Field(default_factory=tuple)
    enabled: bool = True

    @property
    def capabilities(self) -> tuple[McpCapability, ...]:
        return tuple(sorted({tool.capability for tool in self.tools}, key=str))


class MCPRegistry(MCPModel):
    servers: dict[str, MCPServer] = Field(default_factory=dict)

    @property
    def tools(self) -> tuple[MCPTool, ...]:
        return tuple(tool for server in self.servers.values() for tool in server.tools)


class MCPToolCatalogItem(MCPModel):
    server_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    capability: McpCapability
    description: str = Field(min_length=1)
    input_schema: dict[str, object] = Field(default_factory=dict)
    output_schema: dict[str, object] = Field(default_factory=dict)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    enabled: bool


class MCPToolCatalog(MCPModel):
    tools: tuple[MCPToolCatalogItem, ...] = Field(default_factory=tuple)


class MCPInvocationError(MCPModel):
    code: MCPInvocationStatus
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class MCPInvocationResult(MCPModel):
    invocation_id: UUID = Field(default_factory=uuid4)
    server_name: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    capability: McpCapability | None = None
    status: MCPInvocationStatus
    data: dict[str, Any] = Field(default_factory=dict)
    error: MCPInvocationError | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=1)

    @property
    def ok(self) -> bool:
        return self.status == MCPInvocationStatus.SUCCESS
