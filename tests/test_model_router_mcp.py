from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from app.analysis.model_router import (
    _provider_timeout_seconds as mcp_provider_timeout_seconds,
)
from app.core.enums import McpCapability
from app.core.settings import Settings
from app.mcp.model_router_server import (
    ConfiguredModelRouterBackend,
    KimiModelRouterBackend,
    MockModelRouterBackend,
    ModelRouterMCPServer,
    _kimi_provider_messages,
)
from app.mcp.runtime import InMemoryMCPRuntime
from app.mcp.tool_schemas import ModelRouterRequest


def test_kimi_provider_omits_empty_non_tool_messages() -> None:
    request = ModelRouterRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "第一次分析"},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "请重新分析"},
            ]
        }
    )

    messages = _kimi_provider_messages(request.messages)

    assert messages == (
        {"role": "user", "content": "第一次分析"},
        {"role": "user", "content": "请重新分析"},
    )


@pytest.mark.asyncio
async def test_model_router_mcp_registers_and_invokes_completion() -> None:
    runtime = InMemoryMCPRuntime()
    settings = Settings(llm_model="mock-model")
    await runtime.register_server(ModelRouterMCPServer(MockModelRouterBackend(settings)))

    registry = await runtime.discover(McpCapability.MODEL_ROUTER)
    result = await runtime.invoke(
        "model_completion",
        {"messages": [{"role": "user", "content": "Summarize DataMind"}]},
    )

    assert {tool.name for tool in registry.tools} == {"model_completion"}
    assert result.ok
    assert result.data["provider"] == "mock"
    assert result.data["model"] == "mock-model"
    assert result.data["token_usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_configured_model_router_allows_global_mock_override() -> None:
    backend = ConfiguredModelRouterBackend(Settings(llm_provider="mock", llm_model="mock-model"))

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Write report"}],
                "metadata": {"agent": "report"},
            }
        )
    )

    assert response.provider == "mock"
    assert response.model == "mock-model"


@pytest.mark.asyncio
async def test_mock_model_router_accepts_multimodal_content_parts() -> None:
    backend = ConfiguredModelRouterBackend(Settings(llm_provider="mock", llm_model="mock-model"))

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Explain chart"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgo=",
                                    "detail": "auto",
                                },
                            },
                        ],
                    }
                ],
                "metadata": {"agent": "report"},
            }
        )
    )

    assert response.provider == "mock"
    assert "Explain chart" in response.content
    assert "image_url omitted" in response.content


@pytest.mark.asyncio
async def test_mock_model_router_supports_scripted_single_tool_call() -> None:
    backend = MockModelRouterBackend(Settings(llm_model="mock-model"))
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "profile_dataset", "arguments": "{}"},
    }

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Inspect the data"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "profile_dataset",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
                "metadata": {"mock_tool_calls": [tool_call]},
            }
        )
    )

    assert response.content is None
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == (tool_call,)


@pytest.mark.asyncio
async def test_configured_model_router_routes_python_agent_to_deepseek(monkeypatch) -> None:
    async def fake_deepseek_complete(self: object, request: ModelRouterRequest) -> object:
        return type(
            "Response",
            (),
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "content": "def analyze(df): return {}",
                "token_usage": {"total_tokens": 1},
            },
        )()

    monkeypatch.setattr(
        "app.mcp.model_router_server.DeepSeekModelRouterBackend.complete",
        fake_deepseek_complete,
    )
    backend = ConfiguredModelRouterBackend(
        Settings(llm_provider=None, python_llm_provider="deepseek")
    )

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Analyze with Python"}],
                "metadata": {"agent": "python"},
            }
        )
    )

    assert response.provider == "deepseek"
    assert response.model == "deepseek-chat"


@pytest.mark.asyncio
async def test_explicit_kimi_request_can_disable_cross_provider_fallback(monkeypatch) -> None:
    deepseek_called = False

    async def fail_kimi(self: object, request: ModelRouterRequest) -> object:
        raise RuntimeError("Kimi API error 400: tools unsupported")

    async def fake_deepseek(self: object, request: ModelRouterRequest) -> object:
        nonlocal deepseek_called
        deepseek_called = True
        return object()

    monkeypatch.setattr(
        "app.mcp.model_router_server.KimiModelRouterBackend.complete", fail_kimi
    )
    monkeypatch.setattr(
        "app.mcp.model_router_server.DeepSeekModelRouterBackend.complete", fake_deepseek
    )
    backend = ConfiguredModelRouterBackend(
        Settings(llm_allow_provider_fallback=False)
    )
    request = ModelRouterRequest.model_validate(
        {
            "provider": "kimi",
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "Analyze"}],
        }
    )

    with pytest.raises(RuntimeError, match="Kimi API error 400"):
        await backend.complete(request)
    assert deepseek_called is False


@pytest.mark.asyncio
async def test_kimi_fallback_clears_provider_specific_model_id(monkeypatch) -> None:
    captured_model: str | None = "not-called"

    async def fail_kimi(self: object, request: ModelRouterRequest) -> object:
        raise RuntimeError("Kimi temporarily unavailable")

    async def fake_deepseek(self: object, request: ModelRouterRequest) -> object:
        nonlocal captured_model
        captured_model = request.model
        return type("Response", (), {"provider": "deepseek"})()

    monkeypatch.setattr(
        "app.mcp.model_router_server.KimiModelRouterBackend.complete", fail_kimi
    )
    monkeypatch.setattr(
        "app.mcp.model_router_server.DeepSeekModelRouterBackend.complete", fake_deepseek
    )
    backend = ConfiguredModelRouterBackend(Settings(llm_allow_provider_fallback=True))

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "provider": "kimi",
                "model": "kimi-k2.6",
                "messages": [{"role": "user", "content": "Analyze"}],
            }
        )
    )

    assert response.provider == "deepseek"
    assert captured_model is None


@pytest.mark.asyncio
async def test_kimi_backend_uses_moonshot_chat_completions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "moonshot-v1-32k",
                    "choices": [{"message": {"content": "Kimi report"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    monkeypatch.setattr("app.mcp.model_router_server.urlopen", fake_urlopen)
    backend = KimiModelRouterBackend(Settings(kimi_api_key="test-key", llm_timeout_seconds=12))

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "生成报告"}],
                "temperature": 0.2,
            }
        )
    )

    assert captured["url"] == "https://api.moonshot.cn/v1/chat/completions"
    assert captured["payload"]["model"] == "moonshot-v1-32k"
    assert captured["payload"]["messages"][0]["content"] == "生成报告"
    assert captured["timeout"] == 12
    assert response.provider == "kimi"
    assert response.content == "Kimi report"


@pytest.mark.asyncio
async def test_kimi_assistant_uses_dedicated_longer_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "kimi-k2.6",
                    "choices": [{"message": {"content": "Assistant answer"}}],
                    "usage": {},
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeHTTPResponse:
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    monkeypatch.setattr("app.mcp.model_router_server.urlopen", fake_urlopen)
    settings = Settings(
        kimi_api_key="test-key",
        llm_timeout_seconds=12,
        assistant_llm_timeout_seconds=90,
        assistant_timeout_seconds=300,
    )

    response = await KimiModelRouterBackend(settings).complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "读取报告后回答"}],
                "metadata": {"agent": "assistant"},
            }
        )
    )

    assert captured["timeout"] == 90
    assert response.content == "Assistant answer"


@pytest.mark.asyncio
async def test_kimi_k26_normalizes_temperature_to_provider_required_value(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "kimi-k2.6",
                    "choices": [{"message": {"content": "K2.6 answer"}}],
                    "usage": {},
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeHTTPResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse()

    monkeypatch.setattr("app.mcp.model_router_server.urlopen", fake_urlopen)
    settings = Settings(kimi_api_key="test-key", kimi_model="kimi-k2.6")

    await KimiModelRouterBackend(settings).complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "分析数据"}],
                "temperature": 0.0,
            }
        )
    )

    assert captured["payload"]["temperature"] == 1.0


@pytest.mark.asyncio
async def test_kimi_backend_retries_transient_overload_then_succeeds(monkeypatch) -> None:
    calls = 0
    delays: list[float] = []

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "kimi-k2.6",
                    "choices": [{"message": {"content": "Recovered"}}],
                    "usage": {},
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeHTTPResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {},
                BytesIO(b'{"error":{"type":"engine_overloaded_error"}}'),
            )
        return FakeHTTPResponse()

    monkeypatch.setattr("app.mcp.model_router_server.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "app.mcp.model_router_server.time.sleep", lambda delay: delays.append(delay)
    )
    backend = KimiModelRouterBackend(
        Settings(
            kimi_api_key="test-key",
            kimi_model="kimi-k2.6",
            llm_transient_retries=4,
            llm_retry_backoff_seconds=2,
        )
    )

    response = await backend.complete(
        ModelRouterRequest.model_validate(
            {"messages": [{"role": "user", "content": "分析数据"}]}
        )
    )

    assert calls == 3
    assert delays == [2, 4]
    assert response.content == "Recovered"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401])
async def test_kimi_backend_does_not_retry_non_transient_http_errors(
    monkeypatch, status_code: int
) -> None:
    calls = 0

    def fake_urlopen(request: object, timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            status_code,
            "Invalid request",
            {},
            BytesIO(b'{"error":{"message":"invalid"}}'),
        )

    monkeypatch.setattr("app.mcp.model_router_server.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "app.mcp.model_router_server.time.sleep",
        lambda _delay: pytest.fail("non-transient errors must not sleep or retry"),
    )
    backend = KimiModelRouterBackend(
        Settings(kimi_api_key="test-key", llm_transient_retries=4)
    )

    with pytest.raises(RuntimeError, match=rf"Kimi API error {status_code}:"):
        await backend.complete(
            ModelRouterRequest.model_validate(
                {"messages": [{"role": "user", "content": "分析数据"}]}
            )
        )

    assert calls == 1


def test_mcp_model_router_timeout_distinguishes_assistant_calls() -> None:
    settings = Settings(
        llm_timeout_seconds=30,
        assistant_llm_timeout_seconds=120,
        assistant_timeout_seconds=100,
    )

    assert mcp_provider_timeout_seconds(settings, {"agent": "report"}) == 30
    assert mcp_provider_timeout_seconds(settings, {"agent": "assistant"}) == 100


@pytest.mark.asyncio
async def test_kimi_backend_preserves_multimodal_content_parts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "moonshot-v1-32k",
                    "choices": [{"message": {"content": "Kimi vision report"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeHTTPResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse()

    monkeypatch.setattr("app.mcp.model_router_server.urlopen", fake_urlopen)
    backend = KimiModelRouterBackend(Settings(kimi_api_key="test-key"))

    await backend.complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "生成报告"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgo=",
                                    "detail": "auto",
                                },
                            },
                        ],
                    }
                ],
                "metadata": {"agent": "report"},
            }
        )
    )

    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
