# news-benchmark

LLM-as-judge benchmark harness for the `news-service` backend.

Drives the real backend under a virtual clock with a hand-labeled, LLM-fabricated
world. Measures pass/fail of deterministic assertions, classification precision /
recall on event notifications, LLM-judged quality rubrics on digests and
conversations, and per-agent USD cost — for any LiteLLM model combination.

## Layout

```
benchmark/
  src/news_benchmark/
    run.py                    CLI entry + boot order
    config.py                 BenchmarkConfig (pydantic-settings)
    clock.py                  FakeClock + datetime monkey-patch
    scheduler.py              Virtual event-loop scheduler
    db.py                     Throwaway bench DB lifecycle on devbox
    redis_ns.py               Prefixed Redis wrapper
    cost_ledger.py            litellm wrappers + per-agent tagging
    tagging.py                agent_tag ContextVar stack
    fakes/                    Search, delivery, article-fetch, adapters
    scenarios/                Scenario skeletons + loaders (data in ../data/scenarios/)
    simulator/                LLM-persona user driver
    judge/                    Rubric judges + deterministic assertions
    report/                   JSON / markdown / transcript writers
    data_gen/                 Offline pipeline that turns skeletons into full fixtures
  data/
    scenarios/                Committed generated fixtures (bodies, corpus, labels)
  results/                    Gitignored run artifacts
```

## Data-creation report

See [../reports/benchmark_data_generation.md](../../reports/benchmark_data_generation.md)
for the full record of how every piece of scenario data was produced, what
model wrote it, what audits it passed, and what to regenerate if a headline
changes.

## Run

```
cd benchmark
uv sync --extra dev
uv run python -m news_benchmark.run \
    --scenarios s01,s02,s03,s04,s05 \
    --models default \
    --seed 42 \
    --repeat 1 \
    --out-dir results
```

`--models default` uses whatever `LITELLM_MODEL` / `LITELLM_JUDGE_MODEL` are
set to in the environment (inherited from `../backend/.env` if not set here).
