# news-benchmark

Integration and economics harness for the `news-service` backend.

The benchmark package drives backend agents end-to-end under controlled
conditions. It creates throwaway Postgres databases, patches time, swaps real
external integrations for fakes where a test needs determinism, and records LLM
usage for cost analysis.

## Structure

```text
benchmark/
|-- .env.example
|-- data/corpus/                 # committed digest/event/search corpora
|-- economics/
|   |-- constants.py
|   |-- prompts.py
|   `-- run_baseline.py          # live API baseline runner
|-- scripts/                     # corpus generation and diploma metric helpers
|-- src/news_benchmark/
|   |-- clock.py                 # FakeClock + datetime patching
|   |-- config.py                # devbox Postgres/Redis/model settings
|   |-- cost_ledger.py           # LiteLLM token/cost capture
|   |-- db.py                    # throwaway DB create/drop + Alembic install
|   |-- fakes/                   # adapters, delivery, search, Celery shim, world
|   |-- scheduler.py             # virtual async scheduler
|   `-- simulator.py
`-- tests/integration/           # S-conv, S-digest, S-discovery, S-event, etc.
```

## Test Harness

`tests/conftest.py` performs the load-bearing setup before backend modules are
imported:

1. Loads benchmark/backend `.env` values.
2. Installs the fake clock.
3. Wraps LiteLLM calls with the cost ledger.
4. Creates a per-run throwaway Postgres database on the configured devbox.
5. Runs Alembic migrations and points `news_service` at the throwaway database.

The `world` fixture installs deterministic fakes for search, delivery, Celery,
article fetching, and source adapters. Tests can still exercise the real backend
agent code and persistence layer.

## Running Tests

Tests require access to the configured Postgres and Redis instances. By default
those point at the devbox addresses in `src/news_benchmark/config.py`; override
them in `benchmark/.env` when needed.

```bash
cd benchmark
uv sync --extra dev
uv run pytest tests/integration -q
```

## Economics Baseline

`economics/run_baseline.py` sends the prompt set in `economics/prompts.py` to a
live backend API, reads resulting database state, and writes JSON/log output
under `benchmark/economics/results/`.

```bash
cd benchmark
uv run python economics/run_baseline.py \
  --api-url http://100.73.138.67:8000 \
  --db-url postgresql://news:news@100.73.138.67:5432/news
```

`benchmark/economics/results/` is gitignored.
