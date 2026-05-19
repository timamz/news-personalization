# News Personalization Service

Personalized news digest and event-notification service powered by LLM agents,
RSS/Atom feeds, public Telegram channels, and Reddit subreddits.

The backend owns the news domain and exposes a generic REST API. Frontends, such
as the Telegram bot, call that API and receive digest/event deliveries through
webhooks.

## Repository Structure

```text
.
|-- .github/workflows/ci.yml        # GitHub Actions lint, tests, Docker builds
|-- .env.example                    # root-level Compose variables
|-- AGENTS.md                       # development guide for humans and agents
|-- CLAUDE.md                       # pointer to AGENTS.md
|-- docker-compose.yml              # local/devbox service orchestration
|-- backend/
|   |-- Dockerfile
|   |-- README.md
|   |-- alembic/                    # database migrations
|   |-- backups/                    # rolling pg_dump target, gitignored
|   |-- src/news_service/
|   |   |-- agents/                 # conversational, discovery, digest, event agents
|   |   |-- api/                    # FastAPI routes
|   |   |-- core/                   # settings, logging, LLM, Redis, guardrails
|   |   |-- db/                     # SQLAlchemy session setup
|   |   |-- models/                 # persistence models
|   |   |-- schemas/                # API/data schemas
|   |   |-- services/               # source fetching, search, delivery helpers
|   |   `-- tasks/                  # Celery app, beat tasks, workers
|   `-- tests/
|       |-- integration/
|       `-- unit/
|-- benchmark/
|   |-- README.md
|   |-- data/corpus/                # committed scenario corpora
|   |-- economics/                  # baseline/economics scripts and prompts
|   |-- scripts/                    # corpus and diploma-metric helpers
|   |-- src/news_benchmark/         # fake clock, fake services, DB harness
|   `-- tests/integration/          # end-to-end agent scenario tests
|-- infra/grafana/
|   |-- dashboards/                 # business and technical dashboards
|   `-- provisioning/               # Grafana datasource/dashboard config
`-- tgbot/
    |-- Dockerfile
    |-- README.md
    |-- src/tgbot/                  # aiogram handlers, backend client, webhook server
    `-- tests/unit/
```

Each Python service has its own `pyproject.toml`, `uv.lock`, source tree, tests,
and Dockerfile where applicable.

## Services

`docker-compose.yml` starts:

- `postgres` - PostgreSQL 16 with pgvector.
- `redis` - broker/cache used by Celery and conversation state.
- `app` - FastAPI backend on <http://localhost:8000>.
- `worker` - Celery worker for polling, discovery, digest, and delivery tasks.
- `beat` - Celery Beat scheduler.
- `tgbot` - Telegram frontend and webhook server on port `8001`.
- `db-backup` - hourly `pg_dump` checkpoint to `backend/backups/checkpoint.sql`.
- `grafana` - dashboards on <http://localhost:3000>.

API docs are available at <http://localhost:8000/docs> when the backend is
running.

## Source Support

The system ingests:

- RSS/Atom feeds.
- Public Telegram channels.
- Reddit subreddits.

Direct user-specified source attachment supports Telegram channels and Reddit
subreddits. Source discovery can find RSS/Atom feeds, Telegram channels, and
Reddit subreddits.

## Quick Start

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp tgbot/.env.example tgbot/.env
```

Fill in the required provider keys and bot token in the copied env files. Then
create the external Compose volumes used by this repo:

```bash
docker volume create news-personalization_pgdata
docker volume create news-personalization_tgbot_home
docker volume create news-personalization_grafana_data
```

Start the stack:

```bash
docker compose up --build -d
```

Rebuild a single service or service group:

```bash
docker compose up --build -d app worker beat
docker compose up --build -d tgbot
```

## Development

Install dev dependencies inside the service you are changing:

```bash
cd backend
uv sync --extra dev
```

Use the same pattern in `tgbot/` and `benchmark/`.

Common checks:

```bash
cd backend
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q

cd ../tgbot
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q

cd ../benchmark
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/integration -q
```

Benchmark integration tests create throwaway databases on the configured
devbox Postgres instance and use the benchmark fake-world harness. See
`benchmark/README.md` for details.

## Database Backups

The `db-backup` service runs alongside Postgres and writes a plain-text
`pg_dump` of the `news` database to `backend/backups/checkpoint.sql` every hour.
The file is overwritten on each run, so it is a single rolling checkpoint rather
than a retention system.

`backend/backups/` is gitignored except for `.gitkeep`. Remove the service from
`docker-compose.yml` if you do not want the dump running.

## Service Docs

- `backend/README.md`
- `benchmark/README.md`
- `tgbot/README.md`
