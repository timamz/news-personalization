# Reproducing the monthly unit-economics numbers

This folder hosts two benchmarks:

| Benchmark | File | What it measures |
|---|---|---|
| **Baseline** (per-scenario cost of a single pipeline call) | `run_baseline.py` | Dollar cost of one writer/judge/assessor/verifier invocation on a fixed synthetic corpus. Used as a sanity floor. |
| **One-month steady state** (authoritative) | `../tests/integration/test_s_e2e_month.py` | Dollar cost of a full simulated calendar month: two subscriptions, onboarding, polling every 30 min, daily digests, event alerts, weekly verifier, plus four force-invoked maintenance paths. Produces the number quoted in the thesis. |

Both hit the **real LLM provider** and the real Yandex Search API, so each run costs actual money. Typical 30-day run: **\$2-3**. Budget guardrail aborts at \$30.

---

## Prerequisites

1. Python 3.12+, `uv` installed.
2. `benchmark/.env` populated with:
   - `OPENAI_API_KEY` (or whichever LiteLLM provider you are pricing)
   - `LITELLM_MODEL`, `LITELLM_EMBEDDING_MODEL`, `LITELLM_JUDGE_MODEL`
   - `YANDEX_SEARCH_API_KEY`
3. A Postgres + pgvector instance the benchmark can talk to (the test uses `async_session_factory` from `news_service.db.session`). The canonical run points at devbox (`100.73.138.67:5432`, `news`/`news`/`news`).
4. `cd benchmark && uv sync --extra dev`

## The canonical 30-day run

```bash
cd benchmark
RUN_E2E_MONTH=1 E2E_ITEMS_MODE=avg \
  uv run pytest tests/integration/test_s_e2e_month.py -s -x \
  | tee /tmp/e2e_30d.log
```

- Wall-clock: ~1h45m (real LLM round-trips dominate; Yandex is synchronous).
- Writes `economics/results/e2e_month_v2_<run_id>.json` with the full ledger.
- Last canonical run at time of writing: `eaca1303` -> `cost_total_usd=2.301`.

### Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `RUN_E2E_MONTH` | unset | Must be `1`. Test is skipped otherwise. |
| `E2E_ITEMS_MODE` | `avg` | `avg` = `AVG_ITEMS_PER_SOURCE_PER_DAY` (3). `max` reruns at the capacity-planning worst case (`MAX_ITEMS_PER_SOURCE_PER_DAY` in `constants.py`). |
| `E2E_SIM_DAYS` | `30` | Shorten to iterate on wiring without paying the full month. Force-invocations at days 15/18/21/24 will day-clamp onto the last day in very short runs. |
| `E2E_POLL_MINUTES` | `30` | Polling cadence in simulated minutes. Lower = more polls = more embedding cost. |

### What the result JSON carries

```json
{
  "run_id": "eaca1303",
  "simulated_days": 30,
  "digest_sub_id": "...",
  "event_sub_id": "...",
  "digest_deliveries": 14,
  "event_deliveries": 5,
  "cost_digest_usd": 0.261,
  "cost_event_usd": 0.157,
  "cost_unattributed_usd": 0.378,
  "cost_yandex_usd": 1.505,
  "cost_total_usd": 2.301,
  "cost_summary": { "by_agent": { ... }, "by_call_type": { ... } },
  "ledger_rows": [ ... every LLM + embedding + yandex call ... ]
}
```

- `cost_digest_usd` / `cost_event_usd` are attributed via the production `current_subscription_id` ContextVar.
- `cost_unattributed_usd` is mostly `_poll_all_feeds` embedding work that runs outside a `subscription_tag(...)` block. Expect ~15-20% of the total here.
- `cost_yandex_usd` counts Finder + Digest-Writer + Verifier web-search calls at the flat Yandex per-query rate (`$0.005`).

## Force-invoked maintenance paths

Four paths that *must* be priced but whose organic triggers are flaky to engineer are force-invoked at:

| Day | Path | Why forced |
|---|---|---|
| 15 | `run_reflector(...)` with canonical staleness reason | Organic reflector needs a real staleness DB state and the threshold env capture-at-import is fragile. |
| 18 | `_deliver_digest(digest_sub_id)` with `judge_digest` monkey-patched to REVISE once | Forces one Writer -> Judge -> Writer revision round. |
| 21 | `deliver_event_notifications_batch(...)` with `judge_batch_events` patched to REVISE once | Forces one Assessor + Event Judge revision round. |
| 24 | `run_event_verifier(...)` on the event sub | Guarantees a verifier tick even if organic schedule has not lined up. |

Each force-invocation is a real production call. The only thing bypassed is the trigger condition; the LLM dollar cost is identical to the organic path.

## The baseline benchmark

```bash
cd benchmark
uv run python economics/run_baseline.py
```

Writes one JSON per scenario under `economics/results/<run_id>.json`. Much cheaper (~\$0.05 per scenario). Useful when iterating on prompts or a single agent in isolation -- not a substitute for the month-long run.

## Common pitfalls

- **`pytest-asyncio` "another operation is in progress"** when running multiple integration tests in one invocation. Not a code bug: asyncpg connection reuse across event loops. Run the month test alone (`-k test_one_month_steady_state_cost`) or use `-x`.
- **Forgetting `RUN_E2E_MONTH=1`**: the test silently skips.
- **Empty `cost_digest_usd` / `cost_event_usd`**: the backend `current_subscription_id` ContextVar is not set. Fix by rebuilding the image on devbox if the benchmark is running against stale code.
- **Yandex `UNAUTHENTICATED`**: the API key lost the `search-api.executor` role or expired. Mint a new key and update `benchmark/.env`.

## Where the authoritative number lives

Whatever run JSON has the highest `run_id` timestamp under `economics/results/e2e_month_v2_*.json` and `cost_total_usd` at the canonical `items_mode=avg`, `simulated_days=30` settings. At the time of writing: **`e2e_month_v2_eaca1303.json`**, total **\$2.301**.
