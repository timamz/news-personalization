# News Telegram Bot

Telegram interface for the news personalization service. Runs as a Docker container alongside the backend via the shared `docker-compose.yml`.

## Quick Start

Configure the bot token in `.env`:

```bash
cp .env.example .env   # set BOT_TOKEN
```

Then start everything from the repository root:

```bash
cd ..
docker compose up --build -d
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
uv run python -m pytest tests/unit -q
```
