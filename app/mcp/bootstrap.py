from __future__ import annotations

from threading import Lock

from app.core.settings import Settings, get_settings
from app.mcp.contracts import MCPRuntime
from app.mcp.data_analysis_server import DataAnalysisMCPServer, InMemoryDataAnalysisBackend
from app.mcp.filesystem_server import FilesystemMCPServer
from app.mcp.model_router_server import ConfiguredModelRouterBackend, ModelRouterMCPServer
from app.mcp.nlp_server import MockNLPBackend, NLPMCPServer
from app.mcp.runtime import InMemoryMCPRuntime
from app.storage.dataset_store import DatasetStoreRepository

_runtime: InMemoryMCPRuntime | None = None
_runtime_settings_key: tuple[object, ...] | None = None
_runtime_lock = Lock()


async def register_mock_mcp_servers(
    runtime: MCPRuntime,
    settings: Settings | None = None,
) -> None:
    resolved_settings = settings or get_settings()
    await runtime.register_server(
        FilesystemMCPServer(DatasetStoreRepository(resolved_settings.dataset_store_path))
    )
    await runtime.register_server(DataAnalysisMCPServer(InMemoryDataAnalysisBackend()))
    await runtime.register_server(NLPMCPServer(MockNLPBackend()))
    await runtime.register_server(
        ModelRouterMCPServer(ConfiguredModelRouterBackend(resolved_settings))
    )


async def build_mcp_runtime(
    settings: Settings | None = None,
    *,
    reuse: bool = True,
) -> InMemoryMCPRuntime:
    global _runtime, _runtime_settings_key
    resolved_settings = settings or get_settings()
    settings_key = _settings_key(resolved_settings)
    if reuse and _runtime is not None and _runtime_settings_key == settings_key:
        return _runtime

    # Registration is fast and happens once per API/worker process. A regular
    # lock works across the sync/async entry points used by the local runtime.
    with _runtime_lock:
        if reuse and _runtime is not None and _runtime_settings_key == settings_key:
            return _runtime
        runtime = InMemoryMCPRuntime()
        await register_mock_mcp_servers(runtime, resolved_settings)
        if reuse:
            _runtime = runtime
            _runtime_settings_key = settings_key
        return runtime


async def build_mock_mcp_runtime() -> InMemoryMCPRuntime:
    """Backward-compatible test helper that returns an isolated runtime."""
    runtime = InMemoryMCPRuntime()
    await register_mock_mcp_servers(runtime)
    return runtime


def reset_mcp_runtime() -> None:
    """Reset the process runtime for tests that replace settings/backends."""
    global _runtime, _runtime_settings_key
    with _runtime_lock:
        _runtime = None
        _runtime_settings_key = None


def _settings_key(settings: Settings) -> tuple[object, ...]:
    return (
        settings.llm_provider,
        settings.default_llm_provider,
        settings.cleaning_llm_provider,
        settings.planner_llm_provider,
        settings.sql_llm_provider,
        settings.python_llm_provider,
        settings.reflection_llm_provider,
        settings.report_llm_provider,
        settings.review_llm_provider,
        settings.deepseek_model,
        settings.kimi_model,
        settings.deepseek_base_url,
        settings.kimi_base_url,
        settings.llm_base_url,
        settings.llm_timeout_seconds,
        settings.llm_max_tokens,
        settings.llm_prompt_max_chars,
        settings.llm_allow_provider_fallback,
        settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else None,
        settings.kimi_api_key.get_secret_value() if settings.kimi_api_key else None,
        settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
    )
