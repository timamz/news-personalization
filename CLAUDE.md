# AGENTS.md

Development guide for humans and AI agents working on this codebase.

---

## Project Philosophy

**Keep it simple.** Add complexity only when there is no simpler alternative that meets the requirement. Every abstraction, service, and dependency must justify its existence. If in doubt, do less.

**Scalable and reliable without being over-engineered.** The architecture should handle growth, but not be pre-optimised for problems that don't exist yet.

**The backend is agnostic to any frontend.** The backend exposes a generic REST API and delivers digests via webhooks to arbitrary URLs. It has zero knowledge of Telegram, web, mobile, or any other interface. Frontend services are responsible for registering webhook URLs and translating backend payloads into their native format. This separation is a core architectural principle — never introduce frontend-specific logic into the backend.

**Provider-agnostic LLM stack.** All LLM calls go through LiteLLM, all web search goes through SearXNG, and agents are built on Google ADK. The entire system can run on OpenAI, Gemini, Anthropic, self-hosted models, or any LiteLLM-supported provider by changing one config string.

---

## Monorepo Structure

```
/
  docker-compose.yml   — orchestrates all services (single entry point)
  CLAUDE.md            — this file
  README.md            — project overview and quick start
  searxng/             — SearXNG metasearch engine config
    settings.yml       — engine configuration (google, bing, duckduckgo, brave)
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
| **Conversational Agent** | `agents/conversational.py` | ADK tool-use loop | Every user message | `save_subscription`, `get_subscriptions`, `remember`, `add_source`, `remove_source`, `set_user_language`, `set_user_timezone`, `trigger_digest_now`, `delete_subscription`, `close_scenario` |
| **Discovery Agent** | `agents/source_discovery/pipeline.py` | ADK looped agent (max 2 rounds) | Subscription creation / reflector trigger | `run_parallel_search(strategies)` (spawns N parallel finders), `submit_results()` |
| **Source Finder** | `agents/source_discovery/finder.py` | ADK ReAct, one per strategy | Spawned by Discovery Agent | `search_existing_sources` (pgvector), `tool_search_web` (SearXNG), `validate_and_score_source` (fetch posts, embed, cosine) |
| **Digest Writer** | `agents/digest/writer.py` | ADK agent (plans + researches + composes in one loop) | Scheduled digest delivery | `fetch_article` (httpx + BeautifulSoup, budgeted), `search_web` (SearXNG, budgeted), `submit_digest(digest_text, used_item_ids)` |
| **Quality Judge** | `agents/digest/judge.py` | Single structured-output LLM call | After Writer produces draft | Returns `QualityScores` (relevance/format/conciseness 1-5, `PASS` or `REVISE`+feedback) |
| **Pipeline Reflector** | `agents/digest/reflector.py` | ADK agent | After delivery, when health triggers fire | `remove_source(url, reason)` (auto-discovered only), `trigger_source_discovery(reason)`, `emit_status` |
| **Batch Event Assessor** | `agents/event/batch_assessor.py` | Single structured-output LLM call | New items from polling cycle | Returns `BatchAssessmentResult` (per-item `is_relevant` + `notification_body`) |
| **Event Preview** | `agents/event/preview.py` | Single structured-output LLM call | Subscription creation (preview recent events) | Returns relevant items from recent feed history |
| **Source Poller** | `tasks/poll_feeds.py` | Celery Beat task (no LLM) | Every 30 min | Fetches RSS / Telegram / Reddit / Twitter, embeds new items |

### Pipeline Flows

**Source Discovery Pipeline** (triggered by Conversational Agent or Reflector):
```
discovery_agent.loop:
  plan strategies -> run_parallel_search(N strategies) -> review results
                  -> [optional second round]
                  -> submit_results()
```
A single looped Discovery Agent decides on 2-5 initial strategies, fans out to N parallel Source Finders (each running its own ReAct loop with SearXNG + pgvector + validation/scoring), reviews the deduped pool, optionally refines with another round, and finalizes. Capped at 2 rounds; results are deduped by normalized URL and ranked by cosine relevance score.

**Digest Pipeline** (triggered by Celery Beat schedule, `agents/digest/pipeline.py`):
```
fetch_candidates(DB, no LLM)
  -> [write_digest() <-> judge_digest()]   (max 2 revisions)
  -> validate_used_item_ids + validate_digest_text
  -> [run_reflector()]                     (only when health triggers fire)
  -> mark_as_sent()
```
Candidates are fetched via cosine similarity + recency (no LLM). The Digest Writer is one ADK agent that handles planning, optional research (article fetch + web search), and composition in a single loop. The Judge is a separate structured-output call (Generator/Critic pattern). The Reflector runs only when `_should_reflect` fires (judge failed, verdict ≠ PASS, low source coverage, low avg score, or stale `last_reflected_at`); on its own it is non-blocking and may remove dead auto-discovered sources and queue a discovery task.

**Event Pipeline** (triggered by polling):
```
poll_feeds() -> deliver_event_notifications_batch(item_ids) -> assess_batch_events() per subscription
```
All new items from a polling cycle are batched. One LLM call per subscription evaluates all items together (enabling cross-item deduplication and notification-history checks), reducing N*M calls to M calls.

### Key Design Decisions

**`user_spec` as source of truth.** Each subscription has a `user_spec` text field — a markdown document the Conversational Agent writes and pipelines read. Contains: topic, preferences, exclusions, format instructions, source reflections. Replaces the old `canonical_prompt`, `prompt_summary`, `short_label` fields.

**One persistent conversation per user.** The backend exposes a single streaming endpoint (`POST /subscriptions/conversations/stream`, user-keyed, no `conversation_id`). Redis stores one `ConversationState` per user under `conv:user:{user_id}` with a long dormancy TTL (`conversation_ttl_seconds`, 30 days default). The agent compacts its own transcript via the `close_scenario` tool when a logical task (onboarding, create/edit subscription, add/remove sources, delete, trigger digest, Q&A, cancellation) finishes; closed scenarios move from hot `messages` into a one-line `compacted_log` entry rendered into the next turn's prompt. A deterministic byte-size guardrail (`conversation_hot_max_bytes`) trims the oldest messages if the agent forgets.

**LiteLLM for all LLM calls.** `core/llm.py` wraps `litellm.acompletion()` and `litellm.aembedding()`. Model configured via `LITELLM_MODEL=openai/gpt-5.4-nano` (or any LiteLLM-supported string). Retry logic in `core/llm_retry.py` catches `litellm` exception types.

**SearXNG for web search.** `services/search.py` calls a self-hosted SearXNG instance. No external API keys needed for search. Configurable via `SEARXNG_URL` and `WEB_SEARCH_PROVIDER` settings.

**Google ADK for agentic agents.** The Conversational Agent, Discovery Agent, Source Finders, Digest Writer, and Pipeline Reflector are all ADK `Agent` instances with `LiteLlm` models and tool functions. ADK manages the tool-use / ReAct loop (LLM call -> tool execution -> result feedback -> repeat). `agents/adk_runner.py` wraps ADK's `Runner` with `run_agent` (streaming events) and `run_agent_text` (final text). Single-shot structured-output calls (Quality Judge, Batch Event Assessor, Event Preview) bypass ADK and call `core/llm.chat_completion` directly with a Pydantic `response_format`.

**Pipeline observability.** `orchestration/tracing.py` records `PipelineEvent` rows with trace_id, timing, token usage, and input/output summaries. `EvaluationResult` rows store quality scores from the judge for trend analysis.

**Content guardrails.** `orchestration/guardrails.py` wraps external content in `<untrusted-content>` boundary tags, scans for injection patterns, validates LLM outputs (phantom item IDs, cron expressions, notification body length).

---

## Multi-Service Architecture

The system is composed of independent services that communicate over HTTP:

- **Backend** (`backend/`) — the core service. Manages users, subscriptions, source ingestion (RSS + public Telegram channels + Reddit subreddits + public X/Twitter accounts), news items, embeddings, digest generation, and event notifications. Delivers digests and event alerts by POSTing to webhook URLs.
- **Telegram Bot** (`tgbot/`) — a frontend. Translates Telegram commands into backend API calls and receives digest webhooks to forward to users.
- **SearXNG** (`searxng/`) — self-hosted metasearch engine for provider-independent web search.
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

3. **Non-blocking (nice-to-have)**: Reflector, tracing, observability.
   - Log and swallow. Never let these failures affect the user.

General rules:
- Fail loudly and early. Raise specific exceptions, not bare `Exception`.
- Never silently swallow errors. Log then re-raise or handle explicitly.
- External calls (LLM, SearXNG, RSS/Telegram sources, DB) must have explicit timeout and retry logic.
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

All deployment targets the **`devbox`** Docker context (`ssh://timamz@100.73.138.67`). Never run `docker compose up` locally with production bot tokens — it will conflict with the remote instance.

```bash
docker --context devbox compose up --build -d          # deploy full stack
docker --context devbox compose up --build -d tgbot     # deploy single service
docker --context devbox logs -f tgbot                   # check logs
```

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
- Use `searxng/searxng:latest` for web search (self-hosted, no API keys).
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
- `SEARXNG_URL` — SearXNG instance URL
- `WEB_SEARCH_PROVIDER` — `searxng` (default) or `openai` (legacy fallback)
- `OPENAI_API_KEY` — read by LiteLLM from environment (not in Settings class)

---

## Database

- Async SQLAlchemy with `asyncpg`. No synchronous DB calls in async context.
- All schema changes via Alembic migrations. Never modify the DB schema manually.
- `pgvector` for embeddings (1536 dimensions by default, configurable).
- The `sources` table (model: `Source`) stores all source types (RSS, Telegram, Reddit, Twitter). The `subscription_sources` join table links subscriptions to their fixed sources; digest/event retrieval must use only those sources.
- `pipeline_events` table records every agent call for observability.
- `evaluation_results` table stores quality scores from the judge per delivery.

---

## Logging

- Structured JSON logs via the standard `logging` module configured in `core/logging.py`.
- Log levels: `DEBUG` locally, `INFO` in production.
- Always include context: `subscription_id`, `source_id`, `user_id` where relevant.
- Never log secrets or PII.

---

When the content of this file becomes outdated (like architecture description), update this file also.
