from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from app.analysis.prompt_utils import enforce_prompt_budget
from app.core.settings import Settings, get_settings
from app.mcp.bootstrap import build_mcp_runtime
from app.mcp.models import MCPInvocationStatus
from app.mcp.runtime import InMemoryMCPRuntime
from app.mcp.tool_schemas import ModelRouterResponse
from app.security.rate_limit import enforce_rate_limit


class AnalysisModelRouter(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ModelRouterResponse:
        """Call the DeepSeek model completion endpoint through MCP."""


class MCPAnalysisModelRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        runtime: InMemoryMCPRuntime | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._runtime = runtime

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ModelRouterResponse:
        prompt_chars = enforce_prompt_budget(
            messages,
            max_chars=self._settings.llm_prompt_max_chars,
        )
        request_metadata = {**(metadata or {}), "prompt_chars": prompt_chars}
        return _run_async(
            self._complete_async(
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=request_metadata,
                tools=tools or [],
                tool_choice=tool_choice,
            )
        )

    async def _complete_async(
        self,
        *,
        messages: list[dict[str, Any]],
        provider: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        metadata: dict[str, object],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
    ) -> ModelRouterResponse:
        enforce_rate_limit(
            f"llm:{metadata.get('user_id') or 'shared'}:{metadata.get('agent') or 'unknown'}",
            limit=self._settings.llm_rate_limit,
            window_seconds=60,
        )
        runtime = self._runtime or await build_mcp_runtime(self._settings)
        provider_timeout = _provider_timeout_seconds(self._settings, metadata)
        result = await runtime.invoke(
            "model_completion",
            {
                "messages": messages,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens or self._settings.llm_max_tokens,
                "metadata": metadata,
                "tools": tools,
                "tool_choice": tool_choice,
            },
            server_name="model-router-mcp",
            # Keep the MCP envelope slightly wider than the provider request so
            # provider timeouts are reported accurately instead of as MCP timeouts.
            timeout_seconds=provider_timeout + 5.0,
            max_retries=0,
        )
        if result.status != MCPInvocationStatus.SUCCESS:
            message = result.error.message if result.error else "Model Router MCP failed."
            raise RuntimeError(message)
        return ModelRouterResponse.model_validate(result.data)


def _provider_timeout_seconds(settings: Settings, metadata: dict[str, object]) -> float:
    if str(metadata.get("agent") or "").strip().lower() == "assistant":
        return min(settings.assistant_llm_timeout_seconds, settings.assistant_timeout_seconds)
    return settings.llm_timeout_seconds


def _run_async[T](awaitable: Awaitable[T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, awaitable)
        return future.result()
