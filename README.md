# DataMind

DataMind is an AI data analysis copilot based on LangGraph and MCP.
It focuses on the v1 PRD scope: upload structured datasets, profile data,
ask questions in natural language, generate SQL or Python analysis, create
visualizations, and produce structured web reports with Markdown export fallback.

The project is a local-first agent engineering application with an optional
production profile. It is not a full BI platform or enterprise semantic layer.

## V1 Scope

- CSV and Excel dataset upload
- DuckDB-oriented analysis workflow
- Dataset preview, schema, type summary, missing values, duplicates, and basic statistics
- Planner Agent for routing analysis tasks
- SQL Agent for text-to-SQL, execution, and explanation
- Python Agent for EDA, statistics, and chart generation
- Structured web report generation with Markdown compatibility export
- MCP boundary for Filesystem and Model Router capabilities
- LangGraph checkpoint persistence and node execution harness
- Redis/Celery durable analysis workers in the production profile
- SQLite local persistence and PostgreSQL production persistence
- Cookie sessions, CSRF protection, rate limits, and user-scoped data
- Optional one-shot Docker sandbox for generated Python code
- React + Vite + Tailwind CSS UI
- Docker Compose deployment

Out of scope for the current version: enterprise RBAC/SSO, arbitrary multi-table
SQL, automatic semantic-layer management, AutoML, and ML training.

## Stack

- Python 3.12
- FastAPI
- LangGraph
- Pydantic v2
- DuckDB
- SQLite / PostgreSQL
- Redis / Celery
- Alembic / SQLAlchemy Core
- OpenTelemetry
- Pandas
- Plotly
- React
- Vite
- Tailwind CSS
- Docker Compose

## Directory Structure

```text
app/
  api/          FastAPI adapters
  agents/       Agent contracts and analysis helpers
  core/         domain entities, enums, settings, ports
  mcp/          MCP Runtime and Model Router/Data Analysis tools
  harness/      LangGraph node reliability, validation, and trace boundary
  python_runner/ controlled Docker sandbox runner
  schemas/      HTTP schemas
  workflows/    LangGraph workflow adapter
frontend/
  react/        React + Vite + Tailwind user interface
docs/
tests/
```

Some legacy architecture modules are still present while the codebase is being
trimmed toward the new PRD. They should not be expanded for v1 unless they
directly support the dataset analysis flow.

## Local Development

```bash
conda create -y -n datamind-py312 python=3.12
conda activate datamind-py312
pip install -e ".[dev]"
uvicorn app.main:create_app --factory --reload --host 127.0.0.1 --port 8010
```

Health checks:

```bash
curl http://127.0.0.1:8010/api/v1/health
curl http://127.0.0.1:8010/api/v1/health/live
curl http://127.0.0.1:8010/api/v1/health/ready
```

Swagger UI:

```text
http://127.0.0.1:8010/docs
```

Run the React UI:

```bash
cd frontend/react
npm install
npm run dev
```

React dev server:

```text
http://127.0.0.1:5173
```

The product frontend is the React app. React pages load datasets, cleaned
records, analysis summaries, job events, and reports from FastAPI. Local mode
uses SQLite and a local executor; production uses PostgreSQL, Redis, and Celery.

## LLM Configuration

LLM access is exposed through Model Router MCP. DataMind currently supports
layered provider routing; agents should not call provider APIs directly.

Create `.env` from `.env.example`:

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
```

Set `DATAMIND_LLM_PROVIDER=mock` to globally force mock model routing in tests or local no-key runs.

### Kimi data assistant

The sidebar Kimi workspace keeps user-scoped conversation history, accepts JPEG/PNG/WebP images and CSV/XLSX/JSON/TXT data packages, and retrieves existing DataMind analysis results and reports. Ask mode exposes only read and plan-preview tools. Execute mode can run cleaning, relationship, analysis, report, and semantic-model operations only for explicitly granted assets; every mutation is scope checked, idempotent, and audited. Kimi may attach bounded task preferences to the cleaning, Planner, SQL, Python, visualization, review, and report stages. These preferences are persisted for retry/recovery and are always treated as untrusted user instructions below immutable safety prompts. Soft deletion always requires a separate confirmation and keeps assets recoverable for 30 days.

```env
DATAMIND_ASSISTANT_ENABLED=true
DATAMIND_ASSISTANT_LLM_PROVIDER=kimi
DATAMIND_ASSISTANT_LLM_MODEL=kimi-k2.6
DATAMIND_ASSISTANT_MAX_TOOL_CALLS=8
DATAMIND_ASSISTANT_MAX_CONTEXT_CHARS=60000
DATAMIND_ASSISTANT_TIMEOUT_SECONDS=300
DATAMIND_ASSISTANT_IMAGE_MAX_BYTES=5242880
DATAMIND_ASSISTANT_DATA_FILE_MAX_BYTES=209715200
DATAMIND_ASSISTANT_DATA_FILE_MAX_COUNT=20
DATAMIND_ASSISTANT_DATA_BATCH_MAX_BYTES=1073741824
DATAMIND_ASSISTANT_RECYCLE_RETENTION_DAYS=30
DATAMIND_ASSISTANT_RATE_LIMIT=30
```

Assistant attachments are streamed into the shared protected `assistant-attachments` directory and are served only through authenticated API routes. Multi-file imports parse one file at a time, create one dataset group, start `cleaning_strategy=auto`, save validated relationship suggestions, and grant Kimi management access to the new package. Local API lifespan and production Celery Beat purge expired recycle-bin assets. `DATAMIND_KIMI_API_KEY` is required when the assistant provider is Kimi; readiness reports `assistant_model=not_configured` otherwise.

### Bounded autonomous analysis loop

Loop Engineering is the default analysis path. The legacy workflow remains available as an explicit compatibility mode:

```env
DATAMIND_AGENT_LOOP_ENABLED=true
DATAMIND_AGENT_LOOP_DEFAULT_MODE=loop
DATAMIND_AGENT_LOOP_ALLOW_REQUEST_OVERRIDE=true
DATAMIND_AGENT_LOOP_PROVIDER=deepseek
DATAMIND_AGENT_LOOP_MODEL=deepseek-chat
```

The analysis request accepts `agent_mode: auto | legacy | loop`. `auto` resolves to Loop by default. Loop mode exposes only job-scoped, read-only analysis tools; each model decision can select one tool. SQL/Python/chart errors are classified and returned to the controller for bounded repair. Tool, decision, token and time budgets force a deterministic fallback instead of an unbounded retry cycle. Deployments can explicitly disable Loop or select `legacy` when compatibility is required.

Cleaning uploads now use an asynchronous bounded Loop (`cleaning_strategy: auto | rules | llm | hybrid`). In `auto`, the model selects a strategy from aggregate schema/quality evidence; generated cleaning code runs only in the one-shot Python sandbox. Every candidate must pass row/column retention, missingness and duplicate quality gates before one cleaning version is committed and activated. Cancellation or failure preserves the previous active version.

When analysis runs with `agent_mode=loop`, report generation also uses a bounded sub-loop: strategy selection, draft generation, deterministic evidence validation, at most two revisions, and at most one request back to the read-only analysis Loop. Only `report_commit` persists the report by job id. Configure these paths with `DATAMIND_CLEANING_LOOP_*` and `DATAMIND_REPORT_LOOP_*`; production Compose enables the cleaning, analysis, and report Loops by default. Compatibility modes remain available through explicit environment overrides.

## Tests And Quality

```bash
# Fast unit suite (default, no Python subprocess)
python -m pytest

# Real LangGraph with mock model/Python execution boundaries
python -m pytest -o addopts="" -m workflow

# FastAPI + temporary SQLite + DuckDB + mocked services
python -m pytest -o addopts="" -m integration

# Explicit local subprocess/timeout/isolation checks
python -m pytest -o addopts="" -m sandbox

# Explicit project benchmark tests
python -m pytest -o addopts="" -m benchmark

# Deterministic PR release gate and optional benchmark tracks
python -m app.evaluation.cli run --suite release
python -m app.evaluation.cli run --suite provider --repeats 3
python -m app.evaluation.cli run --suite performance --backend compose
python -m app.evaluation.cli run --suite resilience --backend compose
python -m app.evaluation.cli history --database data/datamind.db
python -m app.evaluation.cli calibrate --runs run1.json run2.json run3.json run4.json run5.json --output baseline.json

npm --prefix frontend/react run test:e2e
ruff check .
mypy app tests
```

Every backend test belongs to exactly one of `unit`, `workflow`, `integration`,
`sandbox`, or `benchmark`. The Push/PR CI workflow runs unit, workflow, integration,
the deterministic release benchmark, frontend build, and Playwright. The separate `production-smoke.yml` workflow
runs manually or weekly against real PostgreSQL, Redis, Celery, Cookie auth,
and the controlled Python Runner; local sandbox checks remain explicit.

The scheduled `benchmarks.yml` workflow keeps real-provider and production-stack
measurements outside PR CI. Benchmark artifacts include per-case JSONL, a Markdown
summary, JUnit, environment/model identity, corpus checksum, latency and token
availability. Missing telemetry is reported as `metric_unavailable`, never as zero.
External public corpora are opt-in through `BENCHMARK_DATA_ROOT` and a SHA-256 manifest;
DataMind never copies user database rows into benchmark artifacts.

## Docker Compose

Production Compose requires Docker Desktop/Engine and TLS termination for secure
session cookies:

```bash
cp .env.production.example .env.production
docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production build --pull
docker compose --env-file .env.production up -d
```

The production stack serves React and `/api/v1` from the same HTTPS domain via
Caddy and Nginx. Configure DNS, secrets and model keys in `.env.production`.
See [docs/deployment.md](docs/deployment.md) for first deployment, upgrades,
health checks, backups and SQLite migration.

After the stack is ready, run the same end-to-end acceptance used by the
production Smoke workflow:

```bash
python scripts/production_smoke.py --base-url http://127.0.0.1:8010/api/v1
```

This verifies login/CSRF, dataset persistence, asynchronous autonomous cleaning,
Celery analysis, Loop result retrieval, and persisted report delivery.

Services:

- FastAPI: `http://localhost:8010`
- PostgreSQL: internal service `postgres:5432`
- Redis: internal service `redis:6379`
- Celery analysis worker
- Controlled Python Runner and one-shot no-network sandbox containers

Run `alembic upgrade head` before a manual production deployment. Existing
SQLite data can be copied with:

```bash
python -m app.storage.migrate_sqlite_to_postgres --source data/datamind.db --target POSTGRES_URL
```

## Documentation

- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)
- [MCP Tool Extension Guide](docs/mcp_tool_extension.md)
