from __future__ import annotations

import json
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.v1.deps import current_user_id
from app.core.enums import McpCapability
from app.core.settings import get_settings
from app.mcp.bootstrap import build_mcp_runtime
from app.mcp.models import MCPInvocationResult
from app.mcp.tool_schemas import ModelRouterRequest
from app.schemas.mcp import (
    MCPInvocationErrorResponse,
    MCPInvokeRequest,
    MCPInvokeResponse,
    MCPToolCatalogItemResponse,
    MCPToolCatalogResponse,
)

router = APIRouter()


@router.get("/tools", response_model=MCPToolCatalogResponse)
async def list_mcp_tools(
    capability: Annotated[McpCapability | None, Query()] = None,
    _user_id: str = Depends(current_user_id),
) -> MCPToolCatalogResponse:
    runtime = await build_mcp_runtime()
    catalog = await runtime.catalog(capability)
    return MCPToolCatalogResponse(
        tools=tuple(
            MCPToolCatalogItemResponse(
                server_name=tool.server_name,
                name=tool.name,
                capability=tool.capability,
                description=tool.description,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                timeout_seconds=tool.timeout_seconds,
                max_retries=tool.max_retries,
                enabled=tool.enabled,
            )
            for tool in catalog.tools
        )
    )


@router.post("/invoke", response_model=MCPInvokeResponse)
async def invoke_mcp_tool(
    request: MCPInvokeRequest,
    _user_id: str = Depends(current_user_id),
) -> MCPInvokeResponse:
    settings = get_settings()
    if settings.environment.lower() == "production" and not settings.allow_mcp_invoke:
        raise HTTPException(status_code=403, detail="Direct MCP invocation is disabled.")
    runtime = await build_mcp_runtime()
    result = await runtime.invoke(
        request.tool_name,
        request.arguments,
        server_name=request.server_name,
        timeout_seconds=request.timeout_seconds,
        max_retries=request.max_retries,
    )
    return _invoke_response(result)


@router.post("/model-stream")
def stream_model_completion(
    request: MCPInvokeRequest,
    _user_id: str = Depends(current_user_id),
) -> StreamingResponse:
    settings = get_settings()
    if settings.environment.lower() == "production" and not settings.allow_mcp_invoke:
        raise HTTPException(status_code=403, detail="Direct MCP invocation is disabled.")
    if request.tool_name != "model_completion":
        return StreamingResponse(
            iter(("Only model_completion supports streaming.",)),
            media_type="text/plain",
        )
    model_request = ModelRouterRequest.model_validate(request.arguments)
    return StreamingResponse(
        _stream_deepseek_completion(model_request),
        media_type="text/plain; charset=utf-8",
    )


def _invoke_response(result: MCPInvocationResult) -> MCPInvokeResponse:
    return MCPInvokeResponse(
        server_name=result.server_name,
        tool_name=result.tool_name,
        capability=result.capability,
        status=result.status,
        ok=result.ok,
        data=result.data,
        error=(
            MCPInvocationErrorResponse(
                code=result.error.code,
                message=result.error.message,
                details=result.error.details,
            )
            if result.error
            else None
        ),
        duration_ms=result.duration_ms,
        attempts=result.attempts,
    )


def _stream_deepseek_completion(model_request: ModelRouterRequest):
    settings = get_settings()
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    if not api_key:
        yield "DeepSeek API key is not configured."
        return

    model = model_request.model or settings.llm_model or "deepseek-chat"
    base_url = (settings.llm_base_url or "https://api.deepseek.com").rstrip("/")
    payload = {
        "model": model,
        "messages": tuple(
            {"role": message.role, "content": message.content}
            for message in model_request.messages
        ),
        "temperature": model_request.temperature,
        "stream": True,
    }
    if model_request.max_tokens is not None:
        payload["max_tokens"] = model_request.max_tokens

    http_request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    try:
        with urlopen(http_request, timeout=settings.llm_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                delta = event.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield str(content)
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        yield f"DeepSeek API error {exc.code}: {message}"
    except URLError as exc:
        yield f"DeepSeek API connection failed: {exc}"
