# Development Guide

## Environment

Use Python 3.12.

```bash
conda create -y -n datamind-py312 python=3.12
conda activate datamind-py312
pip install -e ".[dev]"
```

## Run Locally

```bash
uvicorn app.main:create_app --factory --reload --host 127.0.0.1 --port 8010
cd frontend/react
npm install
npm run dev
```

FastAPI Swagger:

```text
http://127.0.0.1:8010/docs
```

## Quality Gate

Run this before finishing a change:

```bash
python -m pytest
npm --prefix frontend/react run test:e2e
ruff check .
mypy app tests
```

## V1 Architecture Rules

- Prioritize the dataset analysis flow from the PRD.
- LangGraph is the only workflow scheduler.
- Agent nodes execute through `NodeExecutionHarness` for validation, transient retries, and trace events.
- Model calls use the shared context budget manager. Keep `DATAMIND_CONTEXT_BUDGET_MODE=shadow`
  while calibrating a provider/model pair, run `python -m app.evaluation.cli run --suite context`,
  and switch to `enforce` only after the required-contract and latency gates pass.
- LLM calls must enter through Model Router MCP.
- The internal MCP Runtime is reused once per API or Worker process.
- Dataset/file access should enter through Filesystem MCP or a local service boundary.
- LangGraph belongs in workflow adapters, not in core domain models.
- Local development defaults to SQLite, legacy headers, and the local executor.
- Production settings must use PostgreSQL, Cookie sessions, Redis/Celery, and secure cookies.

## Test Layout

```text
tests/
  test_*                 unit and architecture tests
  integration/           cross-module runtime flows
```

Unit tests should use in-memory or local adapters. External systems belong in
explicit integration tests.

## README Diagrams

The localized architecture and end-to-end workflow diagrams are generated from
`scripts/render_readme_architecture.mjs`. After installing the frontend
dependencies, regenerate both SVG and PNG assets whenever system boundaries or
workflow stages change:

```bash
node scripts/render_readme_architecture.mjs
```

Commit the generated files in `docs/assets/` together with the source change and
inspect both language variants before opening a pull request.
