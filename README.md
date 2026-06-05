# News Personalization Service

LLM-assisted news subscriptions with a Telegram frontend.

Try the bot now: [@scopeon_bot](https://t.me/scopeon_bot).

## What It Does

This repository contains a news service that lets a user describe what they want
to follow, keeps a set of relevant sources for that topic, and delivers either
scheduled digests or event-style notifications.

Supported source families are:

- RSS/Atom feeds.
- Public Telegram channels.
- Reddit subreddits.

The backend owns users, subscriptions, source polling, stored news items,
embeddings, LLM agents, scheduling, and webhook delivery. The Telegram bot is a
frontend: it talks to the backend API, stores the Telegram-user-to-backend-key
mapping locally, and receives backend deliveries through its webhook server.

## Repository Map

```text
.
|-- docker-compose.yml        # local/devbox stack entry point
|-- backend/                  # FastAPI app, Celery tasks, agents, persistence
|-- tgbot/                    # Telegram bot frontend and delivery webhook
|-- benchmark/                # end-to-end scenario harness
|-- infra/grafana/            # Grafana dashboards and provisioning
|-- AGENTS.md                 # development guide
`-- README.md                 # this overview
```

Useful deeper docs:

- [backend/README.md](backend/README.md)
- [tgbot/README.md](tgbot/README.md)
- [benchmark/README.md](benchmark/README.md)

## Runtime Stack

`docker compose up --build -d` starts:

- `postgres` - PostgreSQL 16 with pgvector.
- `redis` - broker/cache for Celery and conversation state.
- `app` - FastAPI backend on <http://localhost:8000>.
- `worker` - Celery worker for polling, discovery, digest, and delivery tasks.
- `beat` - Celery Beat scheduler.
- `tgbot` - Telegram frontend and webhook server on port `8001`.
- `db-backup` - hourly rolling dump to `backend/backups/checkpoint.sql`.
- `grafana` - dashboards on <http://localhost:3000>.

API docs are available at <http://localhost:8000/docs> when the backend is
running.

## Quick Start

Create local env files:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp tgbot/.env.example tgbot/.env
```

Fill the required values:

- `backend/.env`: `OPENAI_API_KEY`, `YANDEX_SEARCH_API_KEY`,
  `LLM_MODEL_PRICING_USD_PER_1M`.
- `tgbot/.env`: `BOT_TOKEN`.
- `.env`: optional `GRAFANA_ADMIN_PASSWORD`.

Create the external Docker volumes used by the stack:

```bash
docker volume create news-personalization_pgdata
docker volume create news-personalization_tgbot_home
docker volume create news-personalization_grafana_data
```

Start everything:

```bash
docker compose up --build -d
```

Rebuild selected services:

```bash
docker compose up --build -d app worker beat
docker compose up --build -d tgbot
```

## Development Checks

Run checks inside the service you are changing.

Backend:

```bash
cd backend
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q
```

Telegram bot:

```bash
cd tgbot
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q
```

Benchmark:

```bash
cd benchmark
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/integration -q
```
