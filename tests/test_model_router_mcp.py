from __future__ import annotations

import asyncio
import json

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
    _provider_for_request,
    _provider_retry_count,
    _provider_timeout_seconds,
    _ProviderHTTPError,
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


def test_provider_routing_keeps_analysis_fast_and_report_specialized() -> None:
    settings = Settings(
        default_llm_provider="deepseek",
        planner_llm_provider="deepseek",
        reflection_llm_provider="deepseek",
        report_llm_provider="kimi",
        review_llm_provider="kimi",
    )

    def provider(agent: str) -> str:
        return _provider_for_request(
            ModelRouterRequest.model_validate(
                {
                    "messages": [{"role": "user", "content": "test"}],
                    "metadata": {"agent": agent},
                }
            ),
            settings,
        )

    assert provider("planner") == "deepseek"
    assert provider("integrate") == "deepseek"
    assert provider("chart_refine") == "deepseek"
    assert provider("review") == "kimi"
    assert provider("report_execute") == "kimi"


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

    async def fake_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        captured.update(url=url, payload=payload, timeout=timeout_seconds)
        return {
            "model": "moonshot-v1-32k",
            "choices": [{"message": {"content": "Kimi report"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    monkeypatch.setattr("app.mcp.model_router_server._post_json", fake_post_json)
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
    assert captured["timeout"] == pytest.approx(12, abs=0.1)
    assert response.provider == "kimi"
    assert response.content == "Kimi report"


@pytest.mark.asyncio
async def test_kimi_backend_streams_provider_sse_deltas(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_stream_json_events(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ):
        captured.update(payload=payload, timeout=timeout_seconds)
        events = (
            {
                "model": "kimi-k2.6",
                "choices": [{"delta": {"content": "第一段"}, "finish_reason": None}],
            },
            {
                "model": "kimi-k2.6",
                "choices": [{"delta": {"content": "第二段"}, "finish_reason": None}],
            },
            {
                "model": "kimi-k2.6",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10,
                },
            },
        )
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}"
        yield "data: [DONE]"

    monkeypatch.setattr(
        "app.mcp.model_router_server._stream_json_events",
        fake_stream_json_events,
    )
    deltas: list[str] = []
    response = await KimiModelRouterBackend(
        Settings(
            kimi_api_key="test-key",
            kimi_model="kimi-k2.6",
            assistant_llm_timeout_seconds=90,
        )
    ).stream_complete(
        ModelRouterRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "流式回答"}],
                "metadata": {"agent": "assistant"},
            }
        ),
        deltas.append,
    )

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert deltas == ["第一段", "第二段"]
    assert response.content == "第一段第二段"
    assert response.token_usage["total_tokens"] == 10


@pytest.mark.asyncio
async def test_kimi_assistant_uses_dedicated_longer_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        captured["timeout"] = timeout_seconds
        return {
            "model": "kimi-k2.6",
            "choices": [{"message": {"content": "Assistant answer"}}],
            "usage": {},
        }

    monkeypatch.setattr("app.mcp.model_router_server._post_json", fake_post_json)
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

    assert captured["timeout"] == pytest.approx(90, abs=0.1)
    assert response.content == "Assistant answer"


@pytest.mark.asyncio
async def test_kimi_k26_normalizes_temperature_to_provider_required_value(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        captured["payload"] = payload
        return {
            "model": "kimi-k2.6",
            "choices": [{"message": {"content": "K2.6 answer"}}],
            "usage": {},
        }

    monkeypatch.setattr("app.mcp.model_router_server._post_json", fake_post_json)
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

    async def fake_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _ProviderHTTPError(429, '{"error":{"type":"engine_overloaded_error"}}')
        return {
            "model": "kimi-k2.6",
            "choices": [{"message": {"content": "Recovered"}}],
            "usage": {},
        }

    async def fake_sleep(delay: float, *, deadline: float, provider: str) -> None:
        delays.append(delay)

    monkeypatch.setattr("app.mcp.model_router_server._post_json", fake_post_json)
    monkeypatch.setattr("app.mcp.model_router_server._sleep_before_retry", fake_sleep)
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

    async def fake_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        nonlocal calls
        calls += 1
        raise _ProviderHTTPError(status_code, '{"error":{"message":"invalid"}}')

    async def fail_sleep(delay: float, *, deadline: float, provider: str) -> None:
        pytest.fail("non-transient errors must not sleep or retry")

    monkeypatch.setattr("app.mcp.model_router_server._post_json", fake_post_json)
    monkeypatch.setattr("app.mcp.model_router_server._sleep_before_retry", fail_sleep)
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
        llm_optional_timeout_seconds=8,
        assistant_llm_timeout_seconds=120,
        assistant_timeout_seconds=100,
    )

    assert mcp_provider_timeout_seconds(settings, {"agent": "report"}) == 30
    assert (
        mcp_provider_timeout_seconds(
            settings, {"agent": "integrate", "optional_stage": True}
        )
        == 8
    )
    assert mcp_provider_timeout_seconds(settings, {"agent": "assistant"}) == 100
    assert (
        mcp_provider_timeout_seconds(
            settings, {"agent": "assistant_continuation"}
        )
        == 100
    )
    assert mcp_provider_timeout_seconds(
        settings,
        {"agent": "report", "timeout_seconds": 3.5},
    ) == 3.5
    optional_request = ModelRouterRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "polish"}],
            "metadata": {"agent": "integrate", "optional_stage": True},
        }
    )
    assert _provider_timeout_seconds(settings, optional_request) == 8
    assert _provider_retry_count(settings, optional_request) == 0
    deadline_request = ModelRouterRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "report"}],
            "metadata": {"agent": "report", "timeout_seconds": 2.5},
        }
    )
    assert _provider_timeout_seconds(settings, deadline_request) == 2.5
    continuation_request = ModelRouterRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "continue"}],
            "metadata": {"agent": "assistant_continuation"},
        }
    )
    assert _provider_timeout_seconds(settings, continuation_request) == 100


@pytest.mark.asyncio
async def test_kimi_provider_call_is_cancellable(monkeypatch) -> None:
    started = asyncio.Event()
    canceled = asyncio.Event()

    async def slow_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            canceled.set()
            raise
        return {}

    monkeypatch.setattr("app.mcp.model_router_server._post_json", slow_post_json)
    task = asyncio.create_task(
        KimiModelRouterBackend(Settings(kimi_api_key="test-key")).complete(
            ModelRouterRequest.model_validate(
                {"messages": [{"role": "user", "content": "分析"}]}
            )
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert canceled.is_set()


@pytest.mark.asyncio
async def test_kimi_backend_preserves_multimodal_content_parts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(
        url: str, *, payload: dict, headers: dict, timeout_seconds: float
    ) -> dict:
        captured["payload"] = payload
        return {
            "model": "moonshot-v1-32k",
            "choices": [{"message": {"content": "Kimi vision report"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    monkeypatch.setattr("app.mcp.model_router_server._post_json", fake_post_json)
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
