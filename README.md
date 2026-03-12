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
