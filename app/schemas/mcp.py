from __future__ import annotations

from typing import Any

from pydantic import Field

from app.core.enums import McpCapability
from app.mcp.models import MCPInvocationStatus
from app.schemas.common import ApiModel


class MCPToolCatalogItemResponse(ApiModel):
    server_name: str
    name: str
    capability: McpCapability
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    timeout_seconds: float
    max_retries: int
    enabled: bool


class MCPToolCatalogResponse(ApiModel):
    tools: tuple[MCPToolCatalogItemResponse, ...]


class MCPInvokeRequest(ApiModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    server_name: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0)


class MCPInvocationErrorResponse(ApiModel):
    code: MCPInvocationStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class MCPInvokeResponse(ApiModel):
    server_name: str
    tool_name: str
    capability: McpCapability | None
    status: MCPInvocationStatus
    ok: bool
    data: dict[str, Any]
    error: MCPInvocationErrorResponse | None
    duration_ms: float
    attempts: int
