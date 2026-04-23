# AGENTS.md

Development guide for humans and AI agents working on this codebase.

---

## Project Philosophy

**Keep it simple.** Add complexity only when there is no simpler alternative that meets the requirement. Every abstraction, service, and dependency must justify its existence. If in doubt, do less.

**Scalable and reliable without being over-engineered.** The architecture should handle growth, but not be pre-optimised for problems that don't exist yet.

**The backend is agnostic to any frontend.** The backend exposes a generic REST API and delivers digests via webhooks to arbitrary URLs. It has zero knowledge of Telegram, web, mobile, or any other interface. Frontend services are responsible for registering webhook URLs and translating backend payloads into their native format. This separation is a core architectural principle — never introduce frontend-specific logic into the backend.

**Provider-agnostic LLM stack.** All LLM calls go through LiteLLM, all web search goes through the Yandex Cloud Search API (REST), and agents are built on Google ADK. The entire system can run on OpenAI, Gemini, Anthropic, self-hosted models, or any LiteLLM-supported provider by changing one config string.

---

## Monorepo Structure

```
/
  docker-compose.yml   — orchestrates all services (single entry point)
  CLAUDE.md            — this file
  README.md            — project overview and quick start
  backend/             — FastAPI backend, Celery workers, LLM agents
    .env.example       — backend environment variables template
  tgbot/               — Telegram bot frontend (aiogram)
    .env.example       — Telegram bot environment variables template
```

Each service has its own `Dockerfile`, `pyproject.toml`, `uv.lock`, source code, and tests. New frontends are added as sibling directories (e.g. `webapp/`, `mobile/`).

All services run in Docker. `docker compose up --build -d` starts everything. Individual services can be rebuilt independently: `docker compose up --build -d tgbot`.

---

## Multi-Agent Architecture (backend)

### Core Agents

| Agent | Location | Kind | Trigger | Tools / Outputs |
|---|---|---|---|---|
| **Conversational Agent** | `agents/conversational/agent.py` (tools in `agents/conversational/tools.py`) | ADK tool-use loop | Every user message | `create_subscription`, `update_subscription`, `get_subscriptions`, `remember`, `add_source`, `remove_source`, `set_user_language`, `set_user_timezone`, `trigger_digest_now`, `trigger_source_discovery`, `delete_subscription`, `close_scenario` |
| **Discovery Agent** | `agents/source_discovery/pipeline.py` | ADK looped agent (no hard round cap) | Subscription creation / reflector trigger | `spawn_finder(strategy)` (launches one Finder per call, ADK invokes in parallel), `inspect_source(url)`, `submit_selection(urls)`, `abort(reason)` |
| **Source Finder** | `agents/source_discovery/finder.py` | ADK ReAct, one per strategy | Spawned by Discovery Agent via `spawn_finder` | `search_existing_sources` (pgvector), `tool_search_web` (Yandex Search API), `validate_and_score_source` (fetch posts, embed, cosine) |
| **Digest Writer** | `agents/digest/writer.py` | ADK agent (plans + optionally researches + composes in one loop) | Scheduled digest delivery | `search_web` (Yandex Search API), `submit_digest(digest_text, used_item_ids)` -- items already arrive with full article body from ingest time, so no fetch tool is needed |
| **Quality Judge** | `agents/digest/judge.py` | Single structured-output LLM call | After Writer produces draft | Returns `QualityScores` (relevance/format/conciseness 1-5, `PASS` or `REVISE`+feedback) |
| **Pipeline Reflector** | `agents/digest/reflector.py` | ADK agent | After delivery, only when a health trigger fires | `fetch_source_items(source_id, since_days_ago, limit)`, `remove_source(url, reason)` (auto-discovered only), `trigger_source_discovery(reason)`, `emit_status` |
| **Batch Event Assessor** | `agents/event/batch_assessor.py` | Single structured-output LLM call | New items from polling cycle | Returns `BatchAssessmentResult` (per-item `is_relevant` + `notification_body`); accepts optional `critic_feedback_per_item` for judge revision turns |
| **Event Judge** | `agents/event/judge.py` | Single structured-output LLM call | After Batch Assessor when any item is relevant | Returns `BatchJudgeResult` with per-item `PASS`/`REVISE`+feedback; `overall` is REVISE iff any item is |
| **Event Verifier** | `agents/event/verifier.py` | ADK agent | Weekly per active event sub (beat daily, self-throttled via `last_reflected_at`) | `web_search`, `fetch_source_items(source_id, since_days_ago, limit)`, `trigger_source_discovery(reason)`, `emit_missed_event(title, summary, source_url, happened_at)`, `emit_status` |
| **Source Poller** | `tasks/poll_feeds.py` | Celery Beat task (no LLM) | Every 30 min | Fetches RSS / Telegram / Reddit, enriches each item with the full article body at ingest time, embeds new items |

### Pipeline Flows

**Source Discovery Pipeline** (triggered by Conversational Agent or Reflector):
```
discovery_agent.loop:
  plan strategies -> spawn_finder(s1), spawn_finder(s2), ...  (ADK runs in parallel)
                  -> review deduped scored pool
                  -> optionally inspect_source(url) or spawn more finders
                  -> submit_selection(urls) or abort(reason)
```
A single looped Discovery Agent. The prompt suggests starting with 3 diverse strategies emitted as parallel `spawn_finder` calls, each of which runs one Source Finder (ReAct loop with Yandex Search API + pgvector + validation/scoring). The agent then reviews the deduped pool, may inspect candidates or spawn more finders, and finalizes via `submit_selection` (or `abort`). Whatever the agent submits is accepted verbatim -- there is no post-hoc backfill or score gate. The prompt instructs the agent to always submit at least one candidate when the pool is non-empty and to reserve `abort` for the case where every strategy returned nothing. When the pipeline does finish with zero selected sources, `run_and_persist_discovery` returns `status="no_sources_found"`, which the Conversational Agent surfaces to the user as a clear error so they can broaden or refine the topic rather than ending up with an empty subscription.

**Digest Pipeline** (triggered by Celery Beat schedule, `agents/digest/pipeline.py`):
```
fetch_candidates(DB, no LLM)
  -> [write_digest() <-> judge_digest()]   (max 3 revisions)
  -> validate_used_item_ids + validate_digest_text
  -> [run_reflector()]                     (only when health triggers fire)
  -> mark_as_sent() + update contribution streaks
```
Candidates are fetched via cosine similarity + recency (no LLM). The Digest Writer is one ADK agent that handles planning, optional research (web search), and composition in a single loop; candidate items already carry the full article body from ingest-time enrichment. The Judge is a separate structured-output call (Generator/Critic pattern). The Reflector runs only when `_compute_reflect_reasons` returns a non-empty list, using four triggers: REVISE verdict after max revisions, per-source drift (aggregate `source_description_embedding` cosine below `reflector_drift_similarity_threshold`), per-source staleness (no new item for `reflector_source_staleness_days`), or per-source contribution streak (`digests_since_last_contribution >= reflector_contribution_streak_threshold`). The Reflector is non-blocking; it may remove auto-discovered sources, queue discovery, or inspect via `fetch_source_items` before deciding.

**Event Pipeline** (triggered by polling):
```
poll_feeds() -> deliver_event_notifications_batch(item_ids):
  for each subscription:
    assess_batch_events(items, history)
    if any relevant:
        [judge_batch_events() <-> assess_batch_events(only REVISE items, critic_feedback_per_item)]
          (max `event_judge_max_revisions`, default 2; items still REVISE afterwards are dropped)
    deliver() + insert SentItem  for every PASS relevant item
```
All new items from a polling cycle are batched. One LLM call per subscription evaluates all items together (enabling cross-item deduplication and notification-history checks), reducing N*M calls to M calls. When at least one item is relevant, an Event Judge (`agents/event/judge.py`) critiques the assessor's output per item. On REVISE, the assessor re-runs for only those items with the critic feedback injected; items still REVISE after the bounded loop are dropped from delivery rather than forced through. The Judge is fail-open (tier 2): on exception the loop falls through with the unreviewed assessor output.

**Event Verification Pipeline** (weekly outcome-check):
```
Celery Beat (daily) -> reflect_event_subscriptions():
  select event subs where last_reflected_at < now - event_reflector_interval_days
  for each:
    run_event_verifier(user_spec, notification_history, source_contexts, lookback)
    for each missed_event the agent emitted:
        insert synthetic NewsItem (source=sentinel "_verifier") + SentItem
        deliver catch-up via webhook
    for each discovery_reason: celery_app.send_task(DISCOVER_SOURCES_TASK, (sub_id, reason))
    stamp last_reflected_at
```
The Event Verifier is the event-subscription analog of the Digest Pipeline Reflector — an autonomous ADK agent that decides, per confirmed miss, whether to deliver a catch-up and whether to queue source discovery, using `fetch_source_items` to distinguish "source did not cover it" from "assessor missed it." A single confirmed miss caused by a coverage gap is sufficient to trigger discovery; no quota threshold. Non-blocking (tier 3): per-sub failures are logged and swallowed.

### Key Design Decisions

**`user_spec` + `retrieval_query` pair.** Each subscription has a freeform `user_spec` markdown document authored by the Conversational Agent and read verbatim by every downstream LLM. No fixed schema — whatever the agent finds useful: topic, presentation, exclusions, tone, language quirks. Separately, the agent supplies a short `retrieval_query` (topic + entities only, no formatting) that is embedded into `topic_embedding` for cosine retrieval. Splitting the two keeps presentation noise out of the vector while letting the spec evolve freely.

**One persistent conversation per user.** The backend exposes a single streaming endpoint (`POST /subscriptions/conversations/stream`, user-keyed, no `conversation_id`). Redis stores one `ConversationState` per user under `conv:user:{user_id}` with a long dormancy TTL (`conversation_ttl_seconds`, 30 days default). The agent compacts its own transcript via the `close_scenario` tool when a logical task (onboarding, create/edit subscription, add/remove sources, delete, trigger digest, Q&A, cancellation) finishes; closed scenarios move from hot `messages` into a one-line `compacted_log` entry rendered into the next turn's prompt. A deterministic byte-size guardrail (`conversation_hot_max_bytes`) trims the oldest messages if the agent forgets.

**LiteLLM for all LLM calls.** `core/llm.py` wraps `litellm.acompletion()` and `litellm.aembedding()`. Model configured via `LITELLM_MODEL=openai/gpt-5.4-nano` (or any LiteLLM-supported string). Retry logic in `core/llm_retry.py` catches `litellm` exception types.

**Yandex Cloud Search API for web search.** `services/search.py` POSTs to `https://searchapi.api.cloud.yandex.net/v2/web/search` with an `Api-Key` header, decodes the base64-encoded XML in the response's `rawData` field, and returns up to ten formatted results. Configured via `YANDEX_SEARCH_API_KEY` (issued in the Yandex Cloud console for a service account with the `search-api.executor` role) and `YANDEX_SEARCH_TYPE` (default `COM` -- Yandex.com international index; other valid values: `RU`, `KK`, `TR`, `BY`, `UZ`). No folder ID is required: the REST endpoint accepts the API key on its own because the key already carries the folder binding server-side.

**Google ADK for agentic agents.** The Conversational Agent, Discovery Agent, Source Finders, Digest Writer, Pipeline Reflector, and Event Verifier are all ADK `Agent` instances with `LiteLlm` models and tool functions. ADK manages the tool-use / ReAct loop (LLM call -> tool execution -> result feedback -> repeat). `agents/adk_runner.py` wraps ADK's `Runner` with `run_agent` (streaming events) and `run_agent_text` (final text). Single-shot structured-output calls (Digest Judge, Event Judge, Batch Event Assessor) bypass ADK and call `core/llm.chat_completion` directly with a Pydantic `response_format`.

**Content guardrails.** `orchestration/guardrails.py` wraps external content in `<untrusted-content>` boundary tags, scans for injection patterns, validates LLM outputs (phantom item IDs, cron expressions, notification body length).

---

## Multi-Service Architecture

The system is composed of independent services that communicate over HTTP:

- **Backend** (`backend/`) — the core service. Manages users, subscriptions, source ingestion (RSS + public Telegram channels + Reddit subreddits), news items, embeddings, digest generation, and event notifications. Delivers digests and event alerts by POSTing to webhook URLs.
- **Telegram Bot** (`tgbot/`) — a frontend. Translates Telegram commands into backend API calls and receives digest webhooks to forward to users.
- **Future frontends** — web app, mobile app, email service, etc. Each is a sibling directory with the same pattern: call the backend API, expose a webhook endpoint for deliveries.

The backend never imports from or depends on any frontend. Frontends depend only on the backend's public REST API.

---

## API Design

All endpoints that involve LLM processing use **NDJSON streaming** (`application/x-ndjson`). There are no non-streaming variants — callers that don't need progress updates consume the stream and use the final `"done"` event.

---

## Development Principles

### General
- Python 3.12+. Use modern syntax: `type X = ...`, `match`, `X | Y` unions.
- Prefer the standard library. Add a dependency only if it saves substantial code.
- No clever one-liners that sacrifice readability. Code is read far more than it is written.
- Functions do one thing. If a function needs a comment to explain what it does, rename it or split it.
- No dead code, no commented-out code, no TODO comments left in commits.

### Typing
- All functions must have type annotations. No `Any` unless truly unavoidable.
- Use `Pydantic v2` for all data validation and serialization.

### Error Handling

Three tiers for pipeline stages:

1. **Critical (must succeed)**: Planner, Composer, Batch Assessor.
   - Raise `DigestPipelineError` / `EventPipelineError` (from `core/exceptions.py`) after retries are exhausted.
   - Let the Celery task handle retry/abort decisions.
   - Never silently return None for a critical failure.

2. **Quality gate (best-effort)**: Judge.
   - If the judge fails, log a warning and use the unreviewed draft.
   - Never block the pipeline on a quality gate failure.

3. **Non-blocking (nice-to-have)**: Reflector.
   - Log and swallow. Never let these failures affect the user.

General rules:
- Fail loudly and early. Raise specific exceptions, not bare `Exception`.
- Never silently swallow errors. Log then re-raise or handle explicitly.
- External calls (LLM, Yandex Search API, RSS/Telegram sources, DB) must have explicit timeout and retry logic.
- Use `@with_llm_retry()` for LLM calls; the decorator handles transient errors.
- After retries are exhausted, raise a typed exception (never bare `Exception`).
- Broad `except Exception` is acceptable only at task boundaries (Celery tasks, API route handlers) where you must prevent an unhandled crash.

---

## Testing (TDD)

Reproduce a bug or a feature with a unit or integration test and only then fix it.

Be verbose and direct in README.md and code documentation.
Don't use inline code comments, only codeblocks on top of classes and methods.
Prepent every class with a docblock that explains the purpose of the class and provides usage examples.
Use English only to write doclbocks, using only ASCII.

Respect the DDD paradigm.
Respect the principles of testing in the "Angry Tests" book of Yegor Bugayenko.
Favor "fail fast" paradigm over "fail safe": throw exception earlier.

Include as much context as possible in exception messages.

Cover every change with a unit test to guarantee repeatability.
One logical assertion per test: multiple `assert` statements are fine when they verify the same behavioral claim (e.g. checking several fields of one response). Split into separate tests only when the assertions test independent behaviors.
Assert at least once in every test.
Keep test cases as short as possible.
Verify only one specific behavioral pattern per test.
Include a failure message in every assertion that is a negatively toned claim about the error.

Map each test file one-to-one with the feature file it tests.
Name tests as full English sentences, stating what the object under test does.
Spell "cannot" and "dont" without apostrophes in test method names.

Don't share object attributes between tests.
Don't use setUp() or tearDown() idioms in tests.
Don't use static literals or other shared constants in tests.
Prepare a clean state at the start of tests instead of cleaning up after themselves.
Don't rely on default configurations of the objects under test, provide custom arguments.

Don't test functionality irrelevant to the test's stated purpose.
Don't provide functionality in objects used only by tests.
Don't assert on side effects such as logging output in tests.
Don't check the behavior of setters, getters, or constructors in tests.
Don't assert on error messages or codes in tests.
Favor fake objects and stubs over mocks in tests.
Use Hamcrest matchers in tests if available.

Use irregular inputs in tests, such as non-ASCII strings.
Use random values as inputs in tests.
Inline small fixtures in tests instead of loading them from files.
Create large fixtures at runtime rather than store them in files.
Create supplementary fixture objects to avoid code duplication in tests.

Close resources in tests, such as file handlers, sockets, and database connections.
Store temporary files in temporary directories, not in the codebase directory.
Don't mock the file system, sockets, or memory managers in tests.
Don't print any log messages in tests.
Configure the testing framework to disable logging from the objects under test.
Never wait indefinitely for any event in tests; always stop waiting on a timeout.
Verify object behavior in multi-threaded, concurrent environments in tests.
Retry potentially flaky code blocks in tests.
Assume the absence of an Internet connection in tests.
Use ephemeral TCP ports in tests, generated using appropriate library functions.

---

## Code Quality — Ruff

Ruff is the single tool for both linting and formatting. No other linters or formatters.

```toml
# pyproject.toml (in each service)
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["B008"]  # Depends() in FastAPI defaults is idiomatic
```

- Run `ruff check .` and `ruff format .` before every commit.
- CI will reject any PR that fails either check.
- Do not disable ruff rules inline unless absolutely necessary, and always add a comment explaining why.

---

## CI/CD — GitHub Actions

Every push to any branch runs:
1. `ruff check . && ruff format --check .` (per service)
2. `pytest tests/unit` (per service)
3. `docker build` smoke test (per service)

Merging to `main` additionally runs:
4. `pytest tests/integration` (requires Docker services)
5. Multi-platform image build (`linux/amd64`, `linux/arm64`)

**PRs cannot be merged if CI fails.**

Branch strategy: `main` is always deployable. Feature work happens on short-lived branches merged via PR.

### Deployment

All deployment targets the **`devbox`** Docker context (`ssh://timamz@100.73.138.67`). Never run `docker compose up` locally with production bot tokens -- it will conflict with the remote instance.

**There is no local stack. Do not start one.** All testing, smoke tests, and manual verification must hit the devbox containers that are already running. The API is reachable at `http://100.73.138.67:8000` (Tailscale). Postgres at `100.73.138.67:5432` (`news`/`news`/`news`). If a code change needs to be exercised, rebuild the affected service on devbox (`docker --context devbox compose up --build -d <service>`) -- never `docker compose up` locally. The token in `tgbot/.env` is the production bot token and will conflict with the devbox bot if run locally.

```bash
docker --context devbox compose up --build -d          # deploy full stack
docker --context devbox compose up --build -d tgbot     # deploy single service
docker --context devbox logs -f tgbot                   # check logs
```

#### Deploy-time facts worth knowing

- **Compose project name is `news-monorepo`** (derived from the directory name of the compose file on the client machine). Containers are named `news-monorepo-<service>-1`.
- **pgdata volume is external and legacy-named** `news-personalization_pgdata` (pre-dating the rename to `news-monorepo`), declared `external: true` in `docker-compose.yml`. It must be pre-created before `up`:
  ```bash
  docker --context devbox volume create news-personalization_pgdata
  ```
  If it does not exist, the stack fails to start with `external volume "news-personalization_pgdata" not found`. The same applies to `news-personalization_tgbot_home`.
- **Bind mounts resolve on the daemon host, not the client.** When you deploy via `docker --context devbox compose up`, any `- ./foo:/bar` mount looks for `./foo` on the *devbox* filesystem, not your Mac. The repo is not checked out on devbox, so file-based bind mounts silently turn into empty-directory mounts on the container side (Docker auto-creates a directory at the destination when the source is missing). Do not rely on bind mounts for config files -- bake config into the image with a service-local `Dockerfile` instead. Directory bind mounts used as scratch space (e.g. `db-backup`'s `/backups`) are fine because an empty directory is the intended state.
- **Postgres is on pgvector image** (`pgvector/pgvector:pg16`). The pgvector extension must exist in the `news` database. The baseline Alembic migration (`0001_baseline`) runs `CREATE EXTENSION IF NOT EXISTS vector` before creating tables.
- **App container runs `alembic upgrade head` on startup** (see the compose `command:` for the `app` service). On a fresh DB this applies `0001_baseline` in one pass. On an existing DB at baseline, it is a no-op. Any future schema changes ship as new revision files on top of the baseline.

#### Wiping the DB (zero-state redeploy)

When you want to drop all data and start clean:

```bash
docker --context devbox compose down
docker --context devbox volume rm news-personalization_pgdata
docker --context devbox volume create news-personalization_pgdata
docker --context devbox compose up --build -d
```

Verify afterward:

```bash
docker --context devbox exec news-monorepo-postgres-1 \
  psql -U news -d news -c 'SELECT version_num FROM alembic_version;' -c '\dt'
```

`version_num` should be `0001_baseline`; `\dt` should list `alembic_version` plus the 8 model tables (`users`, `subscriptions`, `subscription_sources`, `sources`, `news_items`, `sent_items`, `source_removal_log`, `failed_tasks`). Confirm all are empty with a row-count roll-up if you want to be sure.

#### Troubleshooting

- **`external volume ... not found`**: run the `volume create` command above.
- **Yandex Search API returning `UNAUTHENTICATED`**: the API key is missing the `search-api.executor` role on the folder it is bound to, or the key has been revoked. Mint a new key in the Yandex Cloud console and update `YANDEX_SEARCH_API_KEY` in `backend/.env`.
- **SSH connection reset mid-build**: retry. The `docker --context devbox compose up --build` command is idempotent; partial image layers are cached.
- **Containers from an unrelated project on devbox**: devbox is a shared host running several projects (`bjj_bot`, `link-to-audio-bot`, `news-monorepo`). Always scope commands with `--filter name=news-monorepo` or `docker compose -p news-monorepo`.

---

## Dependency Management — uv

Each service manages its own dependencies independently.

```bash
cd backend  # or cd tgbot
uv add <package>          # add runtime dependency
uv add --optional dev <package>   # add dev/test dependency to the dev extra
uv sync --extra dev       # install with dev tools
uv run pytest             # run in managed environment
```

- `uv.lock` is committed to the repository. Always run `uv sync` after pulling.
- Pin the Python version in `.python-version` (per service).

Key backend dependencies:
- `litellm` — provider-agnostic LLM access (chat completions + embeddings)
- `google-adk` — agent framework with tool-calling loop
- `croniter` — cron expression validation
- `feedparser`, `beautifulsoup4`, `selenium` — source content parsing
- `pgvector` — vector similarity search in PostgreSQL

---

## Docker

- Multi-stage `Dockerfile` per service: `builder` stage installs deps, `runtime` stage is minimal.
- Run as a non-root user in the final image.
- All services defined in the root `docker-compose.yml`. `docker compose up` must bring up the full stack.
- Use `pgvector/pgvector:pg16` for Postgres (supports ARM64 + AMD64).
- Web search is a cloud dependency (Yandex Cloud Search API); no local search container is part of the stack.
- No secrets in the image. All config via environment variables read from service-local `.env` files.

---

## Configuration

- Backend config lives in `backend/src/news_service/core/config.py` using `pydantic-settings`.
- Bot config lives in `tgbot/src/tgbot/core/config.py` using `pydantic-settings`.
- Required secrets raise an error at startup if missing.
- Each service's `.env.example` documents required variables. `.env` files are gitignored.

Key backend settings:
- `LITELLM_MODEL` — LLM model string in LiteLLM format (e.g. `openai/gpt-5.4-nano`)
- `LITELLM_EMBEDDING_MODEL` — embedding model (e.g. `openai/text-embedding-3-small`)
- `LITELLM_JUDGE_MODEL` — separate model for quality judge
- `YANDEX_SEARCH_API_KEY` — Yandex Cloud Search API key (service account with `search-api.executor`)
- `YANDEX_SEARCH_TYPE` — Yandex search index suffix: `COM` (default), `RU`, `KK`, `TR`, `BY`, `UZ`
- `OPENAI_API_KEY` — read by LiteLLM from environment (not in Settings class)

---

## Database

- Async SQLAlchemy with `asyncpg`. No synchronous DB calls in async context.
- All schema changes via Alembic migrations. Never modify the DB schema manually.
- `pgvector` for embeddings (1536 dimensions by default, configurable).
- The `sources` table (model: `Source`) stores all source types (RSS, Telegram, Reddit). The `subscription_sources` join table links subscriptions to their fixed sources; digest/event retrieval must use only those sources.

---

## Logging

- Structured JSON logs via the standard `logging` module configured in `core/logging.py`.
- Log levels: `DEBUG` locally, `INFO` in production.
- Always include context: `subscription_id`, `source_id`, `user_id` where relevant.
- Never log secrets or PII.

---

When the content of this file becomes outdated (like architecture description), update this file also.
