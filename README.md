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

Fill in the required provider keys and bot token in the copied env files.

## Environment Files

### `.env`

Root Compose variables. No provider tokens are required here.

- `GRAFANA_ADMIN_PASSWORD` - Grafana admin password. Defaults to `admin` if
  unset.

### `backend/.env`

Backend, Celery worker, and Celery Beat settings.

Required:

- `OPENAI_API_KEY` - OpenAI or OpenAI-compatible provider key used by LiteLLM.
  With the default config, this key is used for chat, judge, and embeddings.
- `YANDEX_SEARCH_API_KEY` - Yandex Cloud Search API key for a service account
  with the `search-api.executor` role.
- `LLM_MODEL_PRICING_USD_PER_1M` - JSON price table. It must contain every model
  configured by `LITELLM_MODEL`, `LITELLM_JUDGE_MODEL`, and
  `LITELLM_EMBEDDING_MODEL`; the backend refuses to start when an entry is
  missing.

Usually edited:

- `LITELLM_MODEL` - main chat model in LiteLLM format, for example
  `openai/gpt-5.4-nano`.
- `LITELLM_JUDGE_MODEL` - model used by judge calls.
- `LITELLM_EMBEDDING_MODEL` - embedding model. The default is
  `openai/text-embedding-3-small`.
- `YANDEX_SEARCH_TYPE` - Yandex search index suffix. Default is `COM`; supported
  values in the example are `COM`, `RU`, `KK`, `TR`, `BY`, and `UZ`.
- `OPENAI_API_BASE` - optional OpenAI-compatible proxy base URL.
- `DEEPSEEK_API_KEY` and `DEEPSEEK_API_BASE` - required only when a configured
  chat or judge model uses a `deepseek/...` LiteLLM model string. Keep an
  embedding provider configured separately because DeepSeek does not provide
  embeddings.
- `PROXY_URL` - optional SOCKS5 proxy for outbound HTTP.
- `ADMIN_ALERT_WEBHOOK_URL` - optional operator alert webhook, usually a tgbot
  `/deliver/{token}/{chat_id}` URL for the admin chat.

Normally left as-is under Docker Compose:

- `DATABASE_URL` - Compose overrides this inside backend containers to point at
  the `postgres` service.
- `REDIS_URL` - Compose overrides this inside backend containers to point at the
  `redis` service.

### `tgbot/.env`

Telegram frontend settings.

Required:

- `BOT_TOKEN` - Telegram Bot API token from BotFather.

Usually edited:

- `WEBHOOK_PUBLIC_HOST` - host the backend should use when posting deliveries to
  the bot. The Docker Compose default is `tgbot`.
- `BACKEND_URL` - backend API URL. The Docker Compose default is
  `http://app:8000`.
- `PROXY_URL` - optional SOCKS5 proxy for Telegram API calls.

Normally left as-is under Docker Compose:

- `WEBHOOK_HOST` - bind host for the bot webhook server.
- `WEBHOOK_PORT` - bind port for the bot webhook server.
- `BOT_STORAGE_PATH` - SQLite file used for Telegram ID to backend API key
  mapping.

### `benchmark/.env`

Only needed when running `benchmark/` directly. The benchmark harness loads
`benchmark/.env` first and then `backend/.env`, so it can inherit the backend's
`OPENAI_API_KEY`, `YANDEX_SEARCH_API_KEY`, model settings, and pricing table.

Fill this file when you need to override:

- `BENCHMARK_PG_HOST`, `BENCHMARK_PG_PORT`, `BENCHMARK_PG_ADMIN_USER`,
  `BENCHMARK_PG_ADMIN_PASSWORD`, `BENCHMARK_PG_ADMIN_DB` - devbox Postgres used
  for per-run throwaway databases.
- `BENCHMARK_REDIS_URL` - Redis used by benchmark runs.
- `LITELLM_MODEL`, `LITELLM_JUDGE_MODEL`, `LITELLM_EMBEDDING_MODEL` - benchmark
  model overrides.

After env files are configured, create the external Compose volumes used by this
repo:

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
