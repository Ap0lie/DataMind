# Contributing To DataMind

DataMind is currently an actively developed, all-rights-reserved project. Discuss
substantial changes with the repository owner before opening a pull request.

## Development Setup

Use Python 3.12 and Node.js 24. Follow the local setup in `README.md`, copy
`.env.example` to `.env`, and use mock providers for deterministic development.
Never commit credentials or user data.

## Change Guidelines

- Keep LangGraph as the workflow orchestrator and route node execution through
  the existing Harness boundaries.
- Preserve user isolation, evidence validation, idempotency, and sandbox policy.
- Prefer the existing domain modules over adding behavior to large entry files.
- Keep HTTP contracts backward compatible unless the change includes a migration
  and coordinated frontend update.
- Update `prd.md`, README files, schemas, and migrations when behavior changes.

## Verification

Run the layers affected by the change:

```bash
python -m pytest
python -m pytest -o addopts="" -m workflow
python -m pytest -o addopts="" -m integration
python -m app.evaluation.cli run --suite release
npm --prefix frontend/react run build
npm --prefix frontend/react run test:e2e
ruff check .
mypy app tests
```

Real-provider, sandbox, performance, and resilience suites are explicit because
they require credentials or production-like infrastructure.

## Pull Requests

Describe the user-visible behavior, migration or compatibility impact, tests run,
and remaining risks. Keep generated files reproducible; regenerate README
architecture assets with `node scripts/render_readme_architecture.mjs` when the
system boundary changes.
