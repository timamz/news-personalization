# News Service

Personalized news digest service powered by LLM agents and RSS.

## Quick Start

```bash
cp .env.example .env        # configure your keys
docker-compose up            # start all services
```

## Development

```bash
uv sync --all-extras         # install all dependencies
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest                # run tests
```
