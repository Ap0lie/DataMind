# DataMind Server Deployment

## Stack

The production Compose project runs:

- Caddy as the public HTTPS gateway.
- Nginx serving the React production bundle and proxying `/api/v1`.
- FastAPI, Celery Worker and Celery Beat.
- PostgreSQL and password-protected Redis.
- The controlled Python Runner and one-shot, network-disabled sandbox containers.

Only ports 80 and 443 are public. FastAPI port 8010 binds to `127.0.0.1` by
default for server-side diagnostics. PostgreSQL, Redis and the Python Runner are
not published to the host.

## Requirements

- A Linux server with Docker Engine and Docker Compose v2.
- At least 8 GB RAM and 20 GB free disk; 16 GB RAM is recommended when semantic
  embeddings and concurrent analysis are enabled.
- A domain whose A/AAAA record points to the server.
- Inbound TCP 80/443 and UDP 443 allowed by the firewall.
- Outbound HTTPS access for model providers and the first image/model build.

The Python Runner controls one-shot sandbox containers through the host Docker
socket. Deploy this stack on a dedicated trusted host; do not expose the Runner
port or Docker socket through a reverse proxy.

## First Deployment

1. Prepare production configuration:

```bash
cp .env.production.example .env.production
openssl rand -hex 32
```

Place separate generated values in:

```text
DATAMIND_POSTGRES_PASSWORD
DATAMIND_REDIS_PASSWORD
DATAMIND_PYTHON_RUNNER_SHARED_SECRET
```

Set `DATAMIND_DOMAIN`, `DATAMIND_PUBLIC_ORIGIN`, and the DeepSeek/Kimi keys.
Secrets should contain URL-safe characters because the database and Redis
passwords are embedded in internal connection URLs. Never commit
`.env.production`.

2. Validate and build:

```bash
docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production build --pull
```

The application image downloads the pinned `BAAI/bge-small-zh-v1.5` revision
during the build. The API and Worker reuse this image layer; the Python sandbox
does not contain the embedding model.

Assistant structured summaries, versioned semantic memory, maintenance leases, usage
audit, and validated analysis experience use the existing application database and
BGE cache. API, Worker, and Beat must share the same PostgreSQL database; no extra
vector database or memory service is required. Memory maintenance runs after answer
commit, outside the first-token path, and is recovered from its database lease after
Worker restarts. Daily Beat cleanup recycles stale or superseded unpinned memory and
permanently removes only expired recycle items.

`DATAMIND_ASSISTANT_MEMORY_ENABLED` is the default for new users; each user can turn
long-term reads and writes off without deleting existing memory. Tune
`DATAMIND_ASSISTANT_MEMORY_RELEVANCE_THRESHOLD`,
`DATAMIND_ASSISTANT_MEMORY_PREFILTER_LIMIT`, and
`DATAMIND_ASSISTANT_MEMORY_MMR_LAMBDA` only after running the deterministic `memory`
benchmark. `DATAMIND_ASSISTANT_MEMORY_EXPERIENCE_ENABLED` controls read-only Planner
experience reuse independently.

Memory v3 model extraction runs only in the post-answer maintenance job. Keep
`DATAMIND_ASSISTANT_MEMORY_MODEL_EXTRACTION_ENABLED=true` when the Kimi provider is
configured; deterministic extraction remains the fallback. Relevance and utility are
scored separately. Start with
`DATAMIND_ASSISTANT_MEMORY_AUTO_DORMANCY_ENABLED=false`, retain at least five valid
Memory benchmark batches, then enable it only if harmful-memory adoption remains
below the release threshold. Dormancy is reversible and never applies to pinned
memory. The default policy requires three feedback signals or two explicit `wrong`
ratings and a utility below `0.25`.

If the server cannot reach the official Python package index, set
`DATAMIND_PYPI_INDEX_URL` to an approved internal or regional mirror before
building. This value is used only while building images.

3. Start the stack:

```bash
docker compose --env-file .env.production up -d
docker compose --env-file .env.production ps
```

Caddy obtains and renews the TLS certificate automatically after DNS and ports
are correct. Open `https://YOUR_DOMAIN`. The API is available through the same
origin at `/api/v1`; no browser-facing API host configuration is required.

4. Verify readiness and logs:

```bash
curl -fsS https://YOUR_DOMAIN/api/v1/health/live
curl -fsS https://YOUR_DOMAIN/api/v1/health/ready
docker compose --env-file .env.production logs --tail=200 api worker frontend gateway
```

## Upgrade

```bash
git pull
docker compose --env-file .env.production build --pull
docker compose --env-file .env.production run --rm migrate
docker compose --env-file .env.production up -d --remove-orphans
```

Jobs and reports remain in PostgreSQL and the `datamind-data` volume. Running
jobs use database leases and checkpoints; Worker shutdown has a 90-second grace
period before Docker stops the container.

## Backup And Restore

Create regular PostgreSQL and persistent-volume backups:

```bash
docker compose --env-file .env.production exec -T postgres \
  pg_dump -U datamind -d datamind -Fc > datamind-postgres.dump
docker run --rm -v datamind_datamind-data:/source:ro -v "$PWD":/backup \
  alpine tar -czf /backup/datamind-data.tar.gz -C /source .
```

The Caddy volume contains certificate state and may also be backed up. Redis is
not the business source of truth, but its AOF volume preserves queued delivery
state during routine restarts.

## Existing SQLite Data

Run Alembic, then import into an empty PostgreSQL database:

```bash
docker compose --env-file .env.production run --rm migrate
docker compose --env-file .env.production run --rm \
  -v "$PWD/data:/import:ro" api \
  python -m app.storage.migrate_sqlite_to_postgres \
  --source /import/datamind.db \
  --target "postgresql+psycopg://datamind:PASSWORD@postgres:5432/datamind"
```

Use the URL-encoded PostgreSQL password in `--target`. The importer preserves
dataset, report, cleaning-run and job UUIDs and skips rows already present.

## Operations

Useful commands:

```bash
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs -f api worker
docker compose --env-file .env.production restart worker
docker compose --env-file .env.production exec api python -m app.python_runner.smoke
docker compose --env-file .env.production down
```

Do not use `down -v` in production unless permanent deletion of PostgreSQL,
uploads, Redis state and TLS state is intentional.

Set `DATAMIND_OTEL_EXPORTER_OTLP_ENDPOINT` to export API, Worker and Workflow
spans. User-visible ordered events remain in PostgreSQL when no collector is
configured.
