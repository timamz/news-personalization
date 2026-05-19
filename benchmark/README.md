# news-benchmark

Integration-test harness for the `news-service` backend.

## What's here

Only the infrastructure needed to drive the backend agents end-to-end
under controlled conditions. All heavyweight scenario-data-generation,
audits, LLM-as-judge rubrics, and the matrix runner have been removed.
What remains is a small library of primitives that the forthcoming
pytest-based integration tests compose per-test.

### Primitives

- `clock.py` -- FakeClock + `install_clock_patch()` that monkey-patches
  `datetime.datetime.utcnow` / `now(tz)` so the backend sees virtual
  time.
- `cost_ledger.py` -- wraps `litellm.acompletion` / `litellm.aembedding`
  to record per-call cost + token counts, tagged by current agent.
- `config.py` -- devbox Postgres/Redis + model strings.
- `db.py` -- `create_bench_db()` / `drop_bench_db()` / schema install.
- `scheduler.py` -- `VirtualScheduler`: heap-based async scheduler that
  advances the FakeClock between events, for cron/poll-driven tests.
- `fakes/adapters.py` -- `FakeAdapter` and `make_scenario_poll_adapter()`
  for impersonating RSS/Telegram/Reddit ingest.
- `fakes/article_fetch.py` -- URL->body cache.
- `fakes/celery_shim.py` -- inline async dispatch of `send_task` /
  `.delay`.
- `fakes/delivery.py` -- webhook capture (no real HTTP POST).
- `fakes/search.py` -- deterministic fake `search_web`.
- `fakes/world.py` -- wires every fake into `news_service.*` module
  globals via monkey-patch (covers canonical + `from X import Y`
  aliases + Celery).

### What's NOT here

- No scenarios, no data generators, no audits.
- No matrix runner, no CLI entry point, no report writer.
- No LLM-as-judge rubrics.

Those lived in prior commits (snapshot: `535dab2`). They'll be replaced
by much smaller per-test fixtures under `tests/` once we start writing
the integration suite.

## Running tests

Tests require access to the devbox Postgres + Redis (Tailscale) and a
valid LiteLLM provider key. See `.env.example`.

```bash
cd benchmark
uv sync --extra dev
uv run pytest
```
