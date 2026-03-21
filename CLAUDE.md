# AGENTS.md

Development guide for humans and AI agents working on this codebase.

---

## Project Philosophy

**Keep it simple.** Add complexity only when there is no simpler alternative that meets the requirement. Every abstraction, service, and dependency must justify its existence. If in doubt, do less.

**Scalable and reliable without being over-engineered.** The architecture should handle growth, but not be pre-optimised for problems that don't exist yet.

**The backend is agnostic to any frontend.** The backend exposes a generic REST API and delivers digests via webhooks to arbitrary URLs. It has zero knowledge of Telegram, web, mobile, or any other interface. Frontend services are responsible for registering webhook URLs and translating backend payloads into their native format. This separation is a core architectural principle — never introduce frontend-specific logic into the backend.

---

## Monorepo Structure

```
/
  docker-compose.yml   — orchestrates all services (single entry point)
  AGENTS.md            — this file
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

| Agent | File | Trigger | Input | Output |
|---|---|---|---|---|
| **Subscription Parser** | `agents/subscription_parser.py` | Conversational subscription setup | Multi-turn message history | `AgentTurnOutput` with agent message, choices, and `FinalizedSubscriptionConfig` (including `canonical_prompt`) when ready |
| **Schedule Parser** | `agents/schedule_parser.py` | Schedule editing | Natural language schedule text | Cron expression |
| **Source Discovery** | `agents/source_discovery.py` | New subscription (no explicit sources) | User prompt + prompt embedding | Scored list of validated source URLs via tool-calling agent with web search |
| **Web Search + Validation** | `agents/discovery.py` | Called by Source Discovery agent | Search queries | Real web search results via OpenAI Responses API `web_search` tool; URL validation and normalization |
| **Source Poller** | `tasks/poll_feeds.py` | Celery Beat (every 30 min) | All active source rows (`sources`) | New `NewsItem` rows + embeddings; queues event notifications for sources with event subscriptions |
| **Event Assessor** | `agents/event.py` | Event notification delivery | News item + subscription prompt + notification history | `EventAssessmentResult`: detects event, judges relevance, checks dedup, composes notification — all in one LLM call |
| **Event Notifier** | `tasks/deliver_events.py` | New item from a source with event subscriptions | `NewsItem` + matching subscriptions | Immediate webhook notifications via Event Assessor (concurrent, bounded by `recent_event_match_concurrency`) |
| **Digest Dispatcher** | `tasks/schedule_digests.py` | Celery Beat (every 1 min) | Active subscriptions with schedule set | Queued digest delivery tasks |
| **Digest Curator** | `agents/digest_curator.py` | Digest delivery task | Subscription context (embedding, sources, exclusions) | Pre-fetches candidates by relevance + recency, ranks by cosine similarity, single LLM call for selection + composition |
| **Digest Orchestrator** | `agents/digest.py` + `tasks/deliver_digest.py` | Dispatcher task | Subscription + unseen news from fixed subscription sources | Delegates to Digest Curator, marks items sent, delivers via webhook |

Source Discovery uses the OpenAI Agents SDK (`openai-agents`) with tool-calling: the SDK manages the agent loop (LLM call → tool execution → result feedback → repeat). The agent has three tools: `search_existing_sources` (vector search in DB), `tool_search_web` (real web search via OpenAI Responses API with `web_search` tool), and `validate_and_score_source` (fetches real posts, embeds, scores cosine similarity). Source Discovery autonomously searches the existing DB, searches the web for new sources across RSS/Telegram/Reddit, validates URLs, and scores content relevance.

Subscription Parser uses the Chat Completions API with manual tool dispatch for multi-turn subscription setup conversations; it asks clarifying questions, validates sources via tools, produces a `canonical_prompt` (user's prompt with orthographical mistakes corrected), and returns a finalized config when done. Conversation state is stored in Redis with a configurable TTL (`conversation_ttl_seconds`).

Digest Curator uses a single-shot Chat Completions call: it pre-fetches candidates from the DB (by relevance and recency), ranks by cosine similarity, fills up to a configurable context budget (`llm_max_context_chars`), and passes them to one LLM call for selection and composition.

Event Assessor combines event detection, subscription relevance matching, dedup against notification history, and notification composition into a single LLM call per (item, subscription) pair — no pre-detection during polling. Event assessments run concurrently (bounded by `recent_event_match_concurrency`, default 8). Pre-LLM dedup uses Jaccard token overlap (≥0.85) and SequenceMatcher similarity (≥0.96) against recent notification history.

All LLM calls use the direct OpenAI API (`gpt-5.4-nano` by default) and are wrapped with `@with_llm_retry()` for exponential backoff on transient errors. Each subscription stores only a `canonical_prompt_embedding` (no separate `raw_prompt_embedding`).

Each fixed source stores a short LLM-generated source description plus an embedding, and prompt-to-source matching uses the canonical-prompt embedding against those source-description embeddings. Source Poller ingests RSS feeds (`feedparser`), public Telegram channels (`t.me/s/<channel>` HTML parsing), Reddit subreddits (`/r/<subreddit>/new/` via headless Firefox + same-origin JSON fetch), and public X/Twitter accounts (`syndication.twitter.com` server-rendered timelines with rate-limit-aware retries). Scheduled digests are evaluated in each user's stored IANA timezone.

---

## Multi-Service Architecture

The system is composed of independent services that communicate over HTTP:

- **Backend** (`backend/`) — the core service. Manages users, subscriptions, source ingestion (RSS + public Telegram channels + Reddit subreddits + public X/Twitter accounts), news items, embeddings, digest generation, and event notifications. Delivers digests and event alerts by POSTing to webhook URLs.
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
- Fail loudly and early. Raise specific exceptions, not bare `Exception`.
- Never silently swallow errors. Log then re-raise or handle explicitly.
- External calls (OpenAI, RSS/Telegram sources, DB) must have explicit timeout and retry logic.

---

## Testing (TDD)

Write the test before or alongside the implementation — never after.

```
backend/tests/
├── unit/         # Pure logic, no I/O. Mock all external calls (LLM, DB, HTTP).
└── integration/  # Real Postgres + Redis via Docker. Mock OpenAI API only.

tgbot/tests/
└── unit/         # Mock backend API calls, bot interactions, and storage.
```

- Unit tests must be fast (<1s per test) and require no running services.
- Every agent function must have a unit test with a mocked LLM response.
- Every API endpoint must have an integration test.
- Aim for meaningful coverage, not 100% line coverage for its own sake.
- Use `pytest`, `pytest-asyncio`, `httpx` for async API tests, `pytest-mock` for mocking.

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

---

## Docker

- Multi-stage `Dockerfile` per service: `builder` stage installs deps, `runtime` stage is minimal.
- Run as a non-root user in the final image.
- All services defined in the root `docker-compose.yml`. `docker compose up` must bring up the full stack.
- Use `pgvector/pgvector:pg16` for Postgres (supports ARM64 + AMD64).
- No secrets in the image. All config via environment variables read from service-local `.env` files.

---

## Configuration

- Backend config lives in `backend/src/news_service/core/config.py` using `pydantic-settings`.
- Bot config lives in `tgbot/src/tgbot/core/config.py` using `pydantic-settings`.
- Required secrets raise an error at startup if missing.
- Each service's `.env.example` documents required variables. `.env` files are gitignored.

---

## Database

- Async SQLAlchemy with `asyncpg`. No synchronous DB calls in async context.
- All schema changes via Alembic migrations. Never modify the DB schema manually.
- `pgvector` for embeddings. Use `text-embedding-3-small` (1536 dimensions).
- The `sources` table (model: `Source`) stores all source types (RSS, Telegram, Reddit, Twitter). The `subscription_sources` join table links subscriptions to their fixed sources; digest/event retrieval must use only those sources.

---

## Logging

- Structured JSON logs via the standard `logging` module configured in `core/logging.py`.
- Log levels: `DEBUG` locally, `INFO` in production.
- Always include context: `subscription_id`, `source_id`, `user_id` where relevant.
- Never log secrets or PII.

---

When the content of this file becomes outdated (like architecture description), update this file also.
