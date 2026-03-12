# News Backend

FastAPI backend for the news personalization service. It exposes the API, runs Celery tasks,
and owns PostgreSQL/pgvector data. Source ingestion supports RSS feeds, public Telegram
channels, and Reddit subreddits.

## Quick Start

Configure backend environment:

```bash
cp .env.example .env
```

Start backend stack from repository root:

```bash
cd ..
docker compose up --build -d app worker beat
```

## Development Setup

```bash
uv sync --extra dev
```

Note: dev tools are configured as the `dev` extra, so use `uv sync --extra dev`
instead of `uv sync --dev`.

## Lint and Format

```bash
uv run ruff check .
uv run ruff format .
```

## Testing

```bash
PYTHONPATH=src uv run python -m pytest tests/unit -q
PYTHONPATH=src uv run python -m pytest tests/integration -q
```
