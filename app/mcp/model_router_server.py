from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.enums import McpCapability
from app.core.settings import Settings
from app.mcp.models import MCPServer, MCPTool
from app.mcp.tool_schemas import ModelRouterRequest, ModelRouterResponse


class ModelRouterBackend(Protocol):
    async def complete(self, request: ModelRouterRequest) -> ModelRouterResponse:
        """Complete a model request through the configured chat completion backend."""


class ModelRouterMCPServer:
    def __init__(self, backend: ModelRouterBackend, name: str = "model-router-mcp") -> None:
        self._backend = backend
        self._server = MCPServer(
            name=name,
            description="Model Router MCP server for DataMind LLM calls.",
            tools=(
                MCPTool(
                    name="model_completion",
                    capability=McpCapability.MODEL_ROUTER,
                    description="Generate text with the configured DataMind model provider.",
                    input_schema=ModelRouterRequest.model_json_schema(),
                    output_schema=ModelRouterResponse.model_json_schema(),
                    timeout_seconds=60.0,
                    max_retries=1,
                    retry_backoff_seconds=0.2,
                ),
            ),
        )

    @property
    def server(self) -> MCPServer:
        return self._server

    async def list_tools(self) -> tuple[MCPTool, ...]:
        return self._server.tools

    async def invoke(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        if tool_name != "model_completion":
            raise LookupError(f"Unknown Model Router MCP tool: {tool_name}")
        request = ModelRouterRequest.model_validate(arguments)
        response = await self._backend.complete(request)
        return response.model_dump(mode="json")

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        return await self.invoke(tool_name, arguments)


class MockModelRouterBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def complete(self, request: ModelRouterRequest) -> ModelRouterResponse:
        model = request.model or self._settings.llm_model
        prompt_text = "\n".join(_content_as_text(message.content) for message in request.messages)
        scripted = request.metadata.get("mock_tool_calls")
        if request.tools and isinstance(scripted, list) and scripted:
            selected = scripted[0]
            tool_calls = (selected,) if isinstance(selected, dict) else ()
            return ModelRouterResponse(
                provider="mock",
                model=model,
                content=None,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        content = f"[mock:{model}] {prompt_text[:240]}"
        prompt_tokens = max(1, len(prompt_text.split()))
        completion_tokens = max(1, len(content.split()))
        return ModelRouterResponse(
            provider="mock",
            model=model,
            content=content,
            finish_reason="stop",
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )


class DeepSeekModelRouterBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def complete(self, request: ModelRouterRequest) -> ModelRouterResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: ModelRouterRequest) -> ModelRouterResponse:
        api_key = _secret_value(self._settings.deepseek_api_key) or _secret_value(
            self._settings.llm_api_key
        )
        if not api_key:
            raise RuntimeError("DeepSeek API key is not configured.")

        model = request.model or self._settings.deepseek_model or "deepseek-chat"
        base_url = (
            self._settings.deepseek_base_url
            or self._settings.llm_base_url
            or "https://api.deepseek.com"
        ).rstrip("/")
        payload: dict[str, Any] = {
            "model": model,
            "messages": tuple(
                _provider_message(message, text_only=True) for message in request.messages
            ),
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice or "auto"

        http_request = Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(
                http_request,
                timeout=_provider_timeout_seconds(self._settings, request),
            ) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API error {exc.code}: {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"DeepSeek API connection failed: {exc}") from exc

        return _model_router_response("deepseek", model, response_data)


class KimiModelRouterBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def complete(self, request: ModelRouterRequest) -> ModelRouterResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: ModelRouterRequest) -> ModelRouterResponse:
        api_key = _secret_value(self._settings.kimi_api_key) or _secret_value(
            self._settings.llm_api_key
        )
        if not api_key:
            raise RuntimeError("Kimi API key is not configured.")

        model = request.model or self._settings.kimi_model or "moonshot-v1-32k"
        base_url = (self._settings.kimi_base_url or "https://api.moonshot.cn/v1").rstrip("/")
        temperature = request.temperature
        if model.strip().lower() == "kimi-k2.6":
            # Moonshot currently requires the provider-defined temperature for K2.6.
            # Normalize here because core analysis stages intentionally request
            # different temperatures and should not need provider-specific logic.
            temperature = 1.0
        payload: dict[str, Any] = {
            "model": model,
            "messages": _kimi_provider_messages(request.messages),
            "temperature": temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice or "auto"

        http_request = Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_data = _kimi_request_with_transient_retry(
            http_request,
            timeout_seconds=_provider_timeout_seconds(self._settings, request),
            max_retries=self._settings.llm_transient_retries,
            backoff_seconds=self._settings.llm_retry_backoff_seconds,
        )

        return _model_router_response("kimi", model, response_data)


def _model_router_response(
    provider: str,
    fallback_model: str,
    response_data: dict[str, Any],
) -> ModelRouterResponse:
    choice = response_data["choices"][0]
    message = choice["message"]
    content_value = message.get("content")
    usage = response_data.get("usage") or {}
    return ModelRouterResponse(
        provider=provider,
        model=str(response_data.get("model") or fallback_model),
        content=str(content_value) if content_value is not None else None,
        tool_calls=tuple(
            item for item in message.get("tool_calls") or () if isinstance(item, dict)
        ),
        finish_reason=str(choice.get("finish_reason") or "") or None,
        token_usage={
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    )


_TRANSIENT_PROVIDER_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_PROVIDER_BACKOFF_SECONDS = 30.0


def _kimi_request_with_transient_retry(
    http_request: Request,
    *,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    """Call Kimi while retrying only errors that are safe to repeat.

    Authentication, permission, and request-validation failures deliberately fail
    on the first attempt. This keeps invalid evaluation configuration visible and
    prevents a paid request from being repeated when retrying cannot help.
    """

    attempt = 0
    while True:
        attempt += 1
        try:
            with urlopen(http_request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code not in _TRANSIENT_PROVIDER_STATUS_CODES or attempt > max_retries:
                suffix = f" after {attempt} attempts" if attempt > 1 else ""
                raise RuntimeError(
                    f"Kimi API error {exc.code}{suffix}: {message}"
                ) from exc
        except (URLError, TimeoutError) as exc:
            if attempt > max_retries:
                suffix = f" after {attempt} attempts" if attempt > 1 else ""
                raise RuntimeError(f"Kimi API connection failed{suffix}: {exc}") from exc

        if backoff_seconds > 0:
            time.sleep(
                min(
                    backoff_seconds * (2 ** (attempt - 1)),
                    _MAX_PROVIDER_BACKOFF_SECONDS,
                )
            )


def _provider_timeout_seconds(settings: Settings, request: ModelRouterRequest) -> float:
    if str(request.metadata.get("agent") or "").strip().lower() == "assistant":
        return min(settings.assistant_llm_timeout_seconds, settings.assistant_timeout_seconds)
    return settings.llm_timeout_seconds


class ConfiguredModelRouterBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._deepseek = DeepSeekModelRouterBackend(settings)
        self._kimi = KimiModelRouterBackend(settings)
        self._mock = MockModelRouterBackend(settings)

    async def complete(self, request: ModelRouterRequest) -> ModelRouterResponse:
        provider = _provider_for_request(request, self._settings)
        if provider == "deepseek":
            return await self._deepseek.complete(request)
        if provider == "kimi":
            try:
                return await self._kimi.complete(request)
            except RuntimeError:
                allow_fallback = request.metadata.get("allow_provider_fallback")
                if allow_fallback is None:
                    allow_fallback = self._settings.llm_allow_provider_fallback
                if allow_fallback is False:
                    raise
                # A model id is provider-specific. Reusing a Kimi model name on
                # DeepSeek hides the original failure behind a second 400.
                fallback_request = request.model_copy(
                    update={"provider": "deepseek", "model": None}
                )
                return await self._deepseek.complete(fallback_request)
        if provider == "mock":
            return await self._mock.complete(request)
        raise RuntimeError(f"Unsupported LLM provider: {provider}")


def _provider_for_request(request: ModelRouterRequest, settings: Settings) -> str:
    explicit = request.provider or request.metadata.get("provider")
    if explicit:
        return str(explicit).strip().lower()
    if settings.llm_provider:
        return settings.llm_provider.lower()
    agent = str(request.metadata.get("agent") or "").strip().lower()
    match agent:
        case "planner" | "design_framework" | "round_plan":
            provider = settings.planner_llm_provider
        case "sql":
            provider = settings.sql_llm_provider
        case "python" | "python_agent" | "round_python":
            provider = settings.python_llm_provider
        case "reflect" | "reflection":
            provider = settings.reflection_llm_provider
        case "review":
            provider = settings.review_llm_provider
        case "multimodal" | "vision":
            provider = settings.multimodal_llm_provider
        case "report" | "integrate" | "chart_refine":
            provider = settings.report_llm_provider
        case _:
            provider = settings.default_llm_provider
    return provider.lower()


def _secret_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())
    return str(value)


def _content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url")
                detail = image_url.get("detail") if isinstance(image_url, dict) else ""
                parts.append(f"[image_url omitted for text-only provider; detail={detail}]")
            elif item.get("type") == "file":
                parts.append("[file content omitted for text-only provider]")
        return "\n".join(part for part in parts if part).strip() or str(content)
    return str(content)


def _provider_message(message: Any, *, text_only: bool = False) -> dict[str, Any]:
    content = _content_as_text(message.content) if text_only else message.content
    payload: dict[str, Any] = {"role": message.role, "content": content}
    if message.tool_calls:
        payload["tool_calls"] = list(message.tool_calls)
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _kimi_provider_messages(messages: Any) -> tuple[dict[str, Any], ...]:
    payload = []
    for message in messages:
        item = _provider_message(message)
        content = item.get("content")
        has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
        if not has_content and not item.get("tool_calls"):
            continue
        payload.append(item)
    return tuple(payload)
