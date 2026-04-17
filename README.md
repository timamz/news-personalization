# News Personalization Service

Personalized news digest service powered by LLM agents, RSS feeds, public Telegram channels,
Reddit subreddits, and public X/Twitter accounts.

## Repository Structure

```
backend/    — FastAPI backend, Celery workers, LLM agents, PostgreSQL + pgvector
tgbot/      — Telegram bot frontend (aiogram)
```

## Quick Start

```bash
cp backend/.env.example backend/.env  # configure OpenAI key and other backend secrets
cp tgbot/.env.example tgbot/.env  # configure Telegram bot token
docker compose up --build -d      # start all services
```

API docs: http://localhost:8000/docs

## Rebuild a single service

```bash
docker compose up --build -d tgbot   # rebuild only the Telegram bot
docker compose up --build -d app     # rebuild only the backend API
```

Service-specific development and testing instructions:
- [`backend/README.md`](backend/README.md)
- [`tgbot/README.md`](tgbot/README.md)

## Database backups

The `db-backup` service runs alongside Postgres and writes a plain-text
`pg_dump` of the `news` database to `backend/backups/checkpoint.sql` every hour.
The file is overwritten on each run (single rolling checkpoint, no retention).
It is intended as a lightweight safety net for local and devbox deployments so
a broken migration or corrupt data directory can be recovered to the state of
the last hour; it is not a substitute for offsite backups in real production.

The `backend/backups/` directory is gitignored (except for `.gitkeep`). Remove
the service from `docker-compose.yml` if you don't want the dump running.
