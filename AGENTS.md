# AGENTS.md

Development guide for humans and AI agents working on this codebase.

---

## Project Philosophy

**Keep it simple.** Add complexity only when there is no simpler alternative that meets the requirement. Every abstraction, service, and dependency must justify its existence. If in doubt, do less.

**Scalable and reliable without being over-engineered.** The architecture should handle growth, but not be pre-optimised for problems that don't exist yet.

---

## Multi-Agent Architecture

| Agent | File | Trigger | Input | Output |
|---|---|---|---|---|
| **Parser** | `agents/parser.py` | New subscription | Raw user prompt | `SubscriptionConfig` (topics, cron, format) |
| **Discovery** | `agents/discovery.py` | Topic gap detected | Uncovered topic strings | Valid RSS feed URLs |
| **RSS Poller** | `tasks/poll_feeds.py` | Celery Beat (every 30 min) | All active `RssFeed` rows | New `NewsItem` rows + embeddings |
| **Digest** | `agents/digest.py` | Per-user cron schedule | Subscription + unseen news pool | Formatted digest text |

Parser and Discovery use OpenAI structured output. RSS Poller uses `feedparser` (no LLM). Digest uses RAG (pgvector similarity search) then LLM generation.

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
- External calls (OpenAI, RSS feeds, DB) must have explicit timeout and retry logic.

---

## Testing (TDD)

Write the test before or alongside the implementation — never after.

```
tests/
├── unit/         # Pure logic, no I/O. Mock all external calls (LLM, DB, HTTP).
└── integration/  # Real Postgres + Redis via Docker. Mock OpenAI API only.
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
# pyproject.toml
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
1. `ruff check . && ruff format --check .`
2. `pytest tests/unit`
3. `docker build` smoke test

Merging to `main` additionally runs:
4. `pytest tests/integration` (requires Docker services)
5. Multi-platform image build (`linux/amd64`, `linux/arm64`)

**PRs cannot be merged if CI fails.**

Branch strategy: `main` is always deployable. Feature work happens on short-lived branches merged via PR.

---

## Dependency Management — uv

```bash
uv add <package>          # add runtime dependency
uv add --dev <package>    # add dev/test dependency
uv sync                   # install from uv.lock
uv run pytest             # run in managed environment
```

- `uv.lock` is committed to the repository. Always run `uv sync` after pulling.
- Pin the Python version in `.python-version`.

---

## Docker

- Multi-stage `Dockerfile`: `builder` stage installs deps, `runtime` stage is minimal.
- Run as a non-root user in the final image.
- All services defined in `docker-compose.yml`. `docker-compose up` must bring up the full stack.
- Use `pgvector/pgvector:pg16` for Postgres (supports ARM64 + AMD64).
- No secrets in the image. All config via environment variables read from `.env`.

---

## Configuration

- All config lives in `core/config.py` using `pydantic-settings`.
- Required secrets (`OPENAI_API_KEY`, `DATABASE_URL`, etc.) raise an error at startup if missing.
- `.env.example` documents every variable. `.env` is gitignored.

---

## Database

- Async SQLAlchemy with `asyncpg`. No synchronous DB calls in async context.
- All schema changes via Alembic migrations. Never modify the DB schema manually.
- `pgvector` for embeddings. Use `text-embedding-3-small` (1536 dimensions).

---

## Logging

- Structured JSON logs via the standard `logging` module configured in `core/logging.py`.
- Log levels: `DEBUG` locally, `INFO` in production.
- Always include context: `subscription_id`, `feed_id`, `user_id` where relevant.
- Never log secrets or PII.

## One more important thing:

When the content of this file becomes outdated (like architecture description), update this file also.