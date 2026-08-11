from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATAMIND_", env_file=".env", extra="ignore")

    app_name: str = "DataMind API"
    app_version: str = "0.1.0"
    build_sha: str = "local"
    api_prefix: str = "/api/v1"
    debug: bool = Field(default=False)
    dataset_store_path: str = "data/datasets"
    database_url: str | None = None
    redis_url: str = "redis://127.0.0.1:6379/0"
    execution_backend: str = "local"
    checkpoint_backend: str = "sqlite"
    auth_mode: str = "legacy"
    session_cookie_name: str = "datamind_session"
    session_ttl_seconds: int = Field(default=43200, ge=300)
    session_absolute_ttl_seconds: int = Field(default=604800, ge=3600)
    session_cookie_secure: bool = False
    csrf_header_name: str = "X-CSRF-Token"
    allow_mcp_invoke: bool = False
    worker_lease_seconds: int = Field(default=120, ge=30)
    python_runner_url: str | None = None
    python_runner_shared_secret: SecretStr | None = None
    python_runner_timeout_seconds: float = Field(default=35.0, gt=0)
    python_runner_container_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    python_sandbox_image: str = "datamind-python-sandbox:latest"
    python_runner_temp_path: str = "/var/lib/datamind-runner"
    python_runner_volume_name: str = "datamind-runner-temp"
    otel_exporter_otlp_endpoint: str | None = None
    environment: str = "development"
    display_timezone: str = "Asia/Singapore"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    rate_limits_enabled: bool = False
    login_rate_limit: int = Field(default=10, ge=1)
    job_rate_limit: int = Field(default=30, ge=1)
    llm_rate_limit: int = Field(default=120, ge=1)
    llm_provider: str | None = None
    default_llm_provider: str = "deepseek"
    cleaning_llm_provider: str = "deepseek"
    planner_llm_provider: str = "deepseek"
    sql_llm_provider: str = "deepseek"
    python_llm_provider: str = "deepseek"
    reflection_llm_provider: str = "deepseek"
    report_llm_provider: str = "kimi"
    review_llm_provider: str = "kimi"
    multimodal_llm_provider: str = "kimi"
    deepseek_model: str = "deepseek-chat"
    kimi_model: str = "moonshot-v1-32k"
    deepseek_api_key: SecretStr | None = None
    kimi_api_key: SecretStr | None = None
    deepseek_base_url: str | None = None
    kimi_base_url: str | None = None
    llm_model: str = "deepseek-chat"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_optional_timeout_seconds: float = Field(default=8.0, gt=0, le=30)
    llm_transient_retries: int = Field(default=4, ge=0, le=10)
    llm_retry_backoff_seconds: float = Field(default=2.0, ge=0.0, le=30.0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_prompt_max_chars: int = Field(default=120_000, ge=10_000)
    context_budget_enabled: bool = True
    context_budget_mode: str = Field(default="shadow", pattern="^(shadow|enforce)$")
    llm_context_window_tokens: int = Field(default=65_536, ge=8_192, le=2_000_000)
    context_safety_ratio: float = Field(default=0.15, ge=0.05, le=0.40)
    llm_allow_provider_fallback: bool = True
    semantic_embedding_enabled: bool = False
    semantic_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    semantic_embedding_model_path: str = "data/models/bge-small-zh-v1.5"
    semantic_embedding_revision: str = "4e17e244a0fb63bfb78fca8fcf95079fcc664f5c"
    semantic_embedding_device: str = "cpu"
    semantic_embedding_local_files_only: bool = True
    semantic_embedding_batch_size: int = Field(default=32, ge=1, le=256)
    semantic_embedding_cache_size: int = Field(default=4096, ge=128)
    semantic_embedding_required: bool = False
    agent_loop_enabled: bool = True
    agent_loop_default_mode: str = Field(default="loop", pattern="^(legacy|loop)$")
    agent_loop_allow_request_override: bool = True
    agent_loop_provider: str = "deepseek"
    agent_loop_model: str | None = None
    agent_loop_max_tool_calls: int = Field(default=12, ge=1, le=50)
    agent_loop_max_decisions: int = Field(default=16, ge=1, le=64)
    agent_loop_max_tool_attempts: int = Field(default=3, ge=1, le=8)
    agent_loop_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    agent_loop_max_tokens: int = Field(default=50_000, ge=1000)
    analysis_fast_path_enabled: bool = True
    analysis_fast_path_max_rows: int = Field(default=1000, ge=1, le=1_000_000)
    cleaning_loop_enabled: bool = True
    cleaning_loop_max_decisions: int = Field(default=8, ge=1, le=32)
    cleaning_loop_max_tool_calls: int = Field(default=5, ge=1, le=20)
    cleaning_loop_max_strategy_attempts: int = Field(default=2, ge=1, le=5)
    cleaning_loop_timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    cleaning_loop_max_tokens: int = Field(default=12_000, ge=1000)
    report_loop_enabled: bool = True
    report_loop_max_decisions: int = Field(default=6, ge=1, le=20)
    report_loop_max_revisions: int = Field(default=2, ge=0, le=5)
    report_loop_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    report_loop_max_tokens: int = Field(default=8_000, ge=1000)
    dataset_upload_max_bytes: int = Field(
        default=209_715_200, ge=1024, le=1_073_741_824
    )
    assistant_enabled: bool = True
    assistant_llm_provider: str = "kimi"
    assistant_llm_model: str = "kimi-k2.6"
    assistant_max_tool_calls: int = Field(default=8, ge=1, le=20)
    assistant_max_context_chars: int = Field(default=60_000, ge=10_000, le=240_000)
    assistant_fast_path_enabled: bool = True
    # Final-answer output is sized dynamically from the question and evidence.
    # The ask/execute values are per-call ceilings, not fixed allocations.
    assistant_completion_min_tokens: int = Field(default=1_536, ge=256, le=8_192)
    assistant_ask_max_tokens: int = Field(default=4_096, ge=512, le=16_384)
    assistant_execute_max_tokens: int = Field(default=8_192, ge=512, le=32_768)
    assistant_completion_total_max_tokens: int = Field(
        default=24_576, ge=1_024, le=65_536
    )
    assistant_max_continuations: int = Field(default=5, ge=0, le=12)
    assistant_retrieval_slow_ms: int = Field(default=3_000, ge=100, le=300_000)
    assistant_first_token_slow_ms: int = Field(default=15_000, ge=100, le=600_000)
    assistant_total_slow_ms: int = Field(default=60_000, ge=100, le=1_800_000)
    assistant_llm_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    assistant_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    assistant_image_max_bytes: int = Field(default=5_242_880, ge=1024, le=20_971_520)
    assistant_data_file_max_bytes: int = Field(default=209_715_200, ge=1024, le=1_073_741_824)
    assistant_data_file_max_count: int = Field(default=20, ge=1, le=100)
    assistant_data_batch_max_bytes: int = Field(default=1_073_741_824, ge=1024)
    assistant_recycle_retention_days: int = Field(default=30, ge=1, le=365)
    assistant_rate_limit: int = Field(default=30, ge=1)
    assistant_memory_enabled: bool = True
    assistant_memory_summary_messages: int = Field(default=12, ge=4, le=100)
    assistant_memory_summary_chars: int = Field(default=24_000, ge=1_000, le=240_000)
    assistant_memory_summary_max_chars: int = Field(default=3_000, ge=500, le=20_000)
    assistant_memory_retrieval_limit: int = Field(default=8, ge=1, le=50)
    assistant_memory_context_chars: int = Field(default=4_000, ge=500, le=40_000)
    assistant_memory_ttl_days: int = Field(default=180, ge=1, le=3_650)
    assistant_memory_recycle_days: int = Field(default=30, ge=1, le=365)
    assistant_memory_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    assistant_memory_relevance_threshold: float = Field(default=0.32, ge=0.0, le=1.0)
    assistant_memory_prefilter_limit: int = Field(default=100, ge=8, le=500)
    assistant_memory_mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    assistant_memory_experience_enabled: bool = True
    assistant_memory_model_extraction_enabled: bool = True
    assistant_memory_auto_dormancy_enabled: bool = False
    assistant_memory_dormancy_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    assistant_memory_dormancy_min_feedback: int = Field(default=3, ge=1, le=100)
    assistant_memory_wrong_feedback_limit: int = Field(default=2, ge=1, le=20)

    @model_validator(mode="after")
    def validate_production_profile(self) -> Settings:
        if self.assistant_completion_total_max_tokens < self.assistant_completion_min_tokens:
            raise ValueError(
                "Assistant total completion budget must be at least its per-call minimum."
            )
        environment = self.environment.lower()
        runner_secret = (
            self.python_runner_shared_secret.get_secret_value().strip()
            if self.python_runner_shared_secret
            else ""
        )
        if environment == "runner":
            if not runner_secret:
                raise ValueError(
                    "Python Runner requires DATAMIND_PYTHON_RUNNER_SHARED_SECRET."
                )
            return self
        if environment != "production":
            return self
        if self.execution_backend.lower() != "celery":
            raise ValueError("Production requires DATAMIND_EXECUTION_BACKEND=celery.")
        if self.auth_mode.lower() != "session":
            raise ValueError("Production requires DATAMIND_AUTH_MODE=session.")
        if not self.database_url:
            raise ValueError("Production requires DATAMIND_DATABASE_URL.")
        if not self.session_cookie_secure:
            raise ValueError("Production requires secure session cookies.")
        if not self.python_runner_url or not self.python_runner_url.strip():
            raise ValueError("Production requires DATAMIND_PYTHON_RUNNER_URL.")
        if not runner_secret:
            raise ValueError(
                "Production requires DATAMIND_PYTHON_RUNNER_SHARED_SECRET."
            )
        if self.semantic_embedding_enabled and not self.semantic_embedding_local_files_only:
            raise ValueError("Production semantic embedding must use local files only.")
        if self.semantic_embedding_required and not self.semantic_embedding_enabled:
            raise ValueError("Required semantic embedding must be enabled.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
