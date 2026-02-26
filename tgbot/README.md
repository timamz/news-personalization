# News Telegram Bot

Telegram interface for the news personalization service. Runs as a Docker container alongside the backend via the shared `docker-compose.yml`.

## Quick Start

Configure the bot token in `.env`:

```bash
cp .env.example .env   # set BOT_TOKEN
```

Then start everything from the backend directory:

```bash
cd ../news-backend
docker compose up --build -d
```

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format .
uv run pytest
```
