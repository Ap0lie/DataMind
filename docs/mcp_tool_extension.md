# MCP Tool Extension Guide

DataMind v1 keeps MCP deliberately small. External capabilities should
enter through MCP, but only two server categories belong in the MVP:

- Filesystem MCP for uploaded datasets and local file reads.
- Model Router MCP for provider-routed DeepSeek, Kimi, and mock LLM calls.

The current codebase still contains a Data Analysis MCP helper for local
profiling while the PRD migration is in progress. Do not add crawler, scheduler,
knowledge graph, vector database, RAG, or notification MCP servers for v1.

## Add A V1 Tool

1. Define Pydantic input/output schemas in `app/mcp/tool_schemas.py`.
2. Add an `MCPTool` descriptor with `name`, `capability`, schemas, timeout, and retry settings.
3. Implement an async server client compatible with `MCPServerClient`.
4. Register it with `MCPRuntime.register_server`.
5. Add unit tests for discovery, invocation, validation errors, timeout, and output shape.

## Runtime API

Registered tools can be inspected and invoked through FastAPI:

```text
GET  /api/v1/mcp/tools
POST /api/v1/mcp/invoke
```

Example model invocation:

```json
{
  "server_name": "model-router-mcp",
  "tool_name": "model_completion",
  "arguments": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "Explain this dataset."}
    ]
  }
}
```

The response uses normalized MCP invocation statuses:

```text
success, failed, timeout, tool_not_found, server_not_found, tool_disabled, invalid_arguments
```

## Model Router Settings

LLM settings live in `app/core/settings.py` and are read from `DATAMIND_`
environment variables:

```text
DATAMIND_DEFAULT_LLM_PROVIDER=deepseek
DATAMIND_PLANNER_LLM_PROVIDER=deepseek
DATAMIND_SQL_LLM_PROVIDER=deepseek
DATAMIND_REPORT_LLM_PROVIDER=kimi
DATAMIND_DEEPSEEK_MODEL=deepseek-chat
DATAMIND_KIMI_MODEL=moonshot-v1-32k
DATAMIND_DEEPSEEK_BASE_URL=https://api.deepseek.com
DATAMIND_KIMI_BASE_URL=https://api.moonshot.cn/v1
DATAMIND_LLM_API_KEY=
DATAMIND_DEEPSEEK_API_KEY=
DATAMIND_KIMI_API_KEY=
DATAMIND_LLM_TIMEOUT_SECONDS=30
DATAMIND_LLM_MAX_TOKENS=2048
```

DeepSeek handles deterministic planning and SQL tasks by default. Kimi handles
high-value report, review, and future multimodal tasks by default. The mock
router remains available for tests and local fallback.
