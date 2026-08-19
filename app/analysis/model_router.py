from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from app.analysis.prompt_utils import enforce_prompt_budget
from app.core.settings import Settings, get_settings
from app.harness.context import (
    ContextBudgetExceeded,
    ContextBudgetManager,
    PromptEnvelope,
    resolve_context_profile,
)
from app.harness.node import (
    current_node_name,
    emit_context_event,
    remaining_node_timeout,
)
from app.mcp.bootstrap import build_mcp_runtime
from app.mcp.models import MCPInvocationStatus
from app.mcp.runtime import InMemoryMCPRuntime
from app.mcp.tool_schemas import ModelRouterRequest, ModelRouterResponse
from app.security.rate_limit import enforce_rate_limit

logger = logging.getLogger("datamind.context_budget")


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
        context: PromptEnvelope | None = None,
    ) -> ModelRouterResponse:
        """Call the DeepSeek model completion endpoint through MCP."""

    def stream_complete(
        self,
        *,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], None],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
        context: PromptEnvelope | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> ModelRouterResponse:
        """Stream visible answer tokens and return the final provider metadata."""


class MCPAnalysisModelRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        runtime: InMemoryMCPRuntime | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._runtime = runtime
        self._context_budget = ContextBudgetManager.from_settings(self._settings)

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
        context: PromptEnvelope | None = None,
    ) -> ModelRouterResponse:
        resolved_max_tokens = max_tokens or self._settings.llm_max_tokens
        messages, budget_metadata = self._prepare_messages(
            messages=messages,
            context=context,
            metadata=metadata,
            max_tokens=resolved_max_tokens,
            streaming=False,
        )
        prompt_chars = enforce_prompt_budget(
            messages,
            max_chars=self._settings.llm_prompt_max_chars,
        )
        request_metadata = {
            **(metadata or {}),
            "prompt_chars": prompt_chars,
            "context_budget": budget_metadata,
        }
        return _run_async(
            self._complete_async(
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=resolved_max_tokens,
                metadata=request_metadata,
                tools=tools or [],
                tool_choice=tool_choice,
            )
        )

    def stream_complete(
        self,
        *,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], None],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
        context: PromptEnvelope | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> ModelRouterResponse:
        resolved_max_tokens = max_tokens or self._settings.llm_max_tokens
        messages, budget_metadata = self._prepare_messages(
            messages=messages,
            context=context,
            metadata=metadata,
            max_tokens=resolved_max_tokens,
            streaming=True,
        )
        prompt_chars = enforce_prompt_budget(
            messages,
            max_chars=self._settings.llm_prompt_max_chars,
        )
        request_metadata = {
            **(metadata or {}),
            "prompt_chars": prompt_chars,
            "context_budget": budget_metadata,
        }
        return _run_async(
            self._stream_complete_async(
                messages=messages,
                on_delta=on_delta,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=resolved_max_tokens,
                metadata=request_metadata,
                cancel_check=cancel_check,
            )
        )

    def _prepare_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        context: PromptEnvelope | None,
        metadata: dict[str, object] | None,
        max_tokens: int,
        streaming: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        agent = str((metadata or {}).get("agent") or "")
        profile = resolve_context_profile(
            agent=agent,
            node_name=current_node_name(),
            streaming=streaming,
        )
        try:
            prepared = self._context_budget.prepare(
                context or PromptEnvelope.from_messages(messages),
                profile=profile,
                output_tokens=max_tokens,
            )
        except ContextBudgetExceeded as exc:
            payload = {"profile": profile, "error": str(exc)}
            logger.error("context.budget_exceeded %s", json.dumps(payload, ensure_ascii=False))
            emit_context_event(
                "context.budget_exceeded",
                status="failed",
                message="Required model context exceeds its configured budget.",
                payload=payload,
            )
            raise
        report = prepared.report.as_metadata()
        logger.info("context.budget_evaluated %s", json.dumps(report, ensure_ascii=False))
        emit_context_event(
            "context.budget_evaluated",
            status="completed" if report["fits"] else "warning",
            message="Model context budget evaluated.",
            payload=report,
        )
        if report["compressed"]:
            emit_context_event(
                "context.compressed",
                status="completed",
                message="Model context was deterministically compressed.",
                payload=report,
            )
        if report["suppressed_sections"]:
            emit_context_event(
                "context.section_suppressed",
                status="completed",
                message="Optional model context sections were suppressed.",
                payload={
                    "profile": profile,
                    "sections": report["suppressed_sections"],
                    "duration_ms": report["duration_ms"],
                },
            )
        if not report["fits"]:
            emit_context_event(
                "context.budget_exceeded",
                status="warning",
                message="Shadow context proposal could not fit the configured budget.",
                payload=report,
            )
        return prepared.messages, report

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
        if provider_timeout <= 0:
            raise TimeoutError("The current workflow node deadline has expired.")
        shutdown_margin = min(0.5, provider_timeout * 0.1)
        provider_metadata = {
            **metadata,
            "timeout_seconds": max(0.05, provider_timeout - shutdown_margin),
        }
        result = await runtime.invoke(
            "model_completion",
            {
                "messages": messages,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens or self._settings.llm_max_tokens,
                "metadata": provider_metadata,
                "tools": tools,
                "tool_choice": tool_choice,
            },
            server_name="model-router-mcp",
            # The provider receives a small shutdown margin inside this strict
            # envelope so the workflow deadline remains a real wall-clock cap.
            timeout_seconds=provider_timeout,
            max_retries=0,
        )
        if result.status != MCPInvocationStatus.SUCCESS:
            message = result.error.message if result.error else "Model Router MCP failed."
            raise RuntimeError(message)
        return ModelRouterResponse.model_validate(result.data)

    async def _stream_complete_async(
        self,
        *,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], None],
        provider: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        metadata: dict[str, object],
        cancel_check: Callable[[], None] | None,
    ) -> ModelRouterResponse:
        enforce_rate_limit(
            f"llm:{metadata.get('user_id') or 'shared'}:{metadata.get('agent') or 'unknown'}",
            limit=self._settings.llm_rate_limit,
            window_seconds=60,
        )
        # Incremental provider bytes cannot be represented by the current
        # request/response MCP envelope. The same configured backend is used
        # directly for this final-answer channel; tool decisions still use MCP.
        from app.mcp.model_router_server import ConfiguredModelRouterBackend

        request = ModelRouterRequest.model_validate(
            {
                "messages": messages,
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens or self._settings.llm_max_tokens,
                "metadata": metadata,
            }
        )
        return await ConfiguredModelRouterBackend(self._settings).stream_complete(
            request,
            on_delta,
            cancel_check=cancel_check,
        )


def _provider_timeout_seconds(settings: Settings, metadata: dict[str, object]) -> float:
    configured: float
    agent = str(metadata.get("agent") or "").strip().lower()
    if agent == "assistant" or agent.startswith("assistant_"):
        configured = min(
            settings.assistant_llm_timeout_seconds, settings.assistant_timeout_seconds
        )
    elif metadata.get("optional_stage") is True:
        configured = min(settings.llm_timeout_seconds, settings.llm_optional_timeout_seconds)
    else:
        configured = settings.llm_timeout_seconds
    requested = metadata.get("timeout_seconds")
    if isinstance(requested, (int, float)) and not isinstance(requested, bool) and requested > 0:
        configured = min(configured, float(requested))
    return remaining_node_timeout(configured) or 0.0


def _run_async[T](awaitable: Awaitable[T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, awaitable)
        return future.result()
