"""Source discovery pipeline: a ReAct discovery agent that orchestrates finders.

The discovery agent is a pure ReAct loop with four tools:

- ``spawn_finder(strategy)`` -- launches one per strategy. ADK runs multiple
  ``spawn_finder`` calls emitted in the same model turn concurrently, so the
  agent gets parallelism natively instead of through a custom abstraction.
  Each finder returns its list of scored sources; candidates accumulate in
  a shared pool.
- ``inspect_source(url)`` -- lets the orchestrator pull a content preview of
  any candidate it is considering, so the acceptance decision can be made
  on real material instead of just the cosine score.
- ``submit_selection(urls)`` -- the orchestrator explicitly names the URLs
  it wants to accept. Whatever the agent submits is accepted verbatim; the
  pipeline does not second-guess the selection. The prompt instructs the
  agent to always submit at least one candidate when the pool is non-empty
  and to prefer submitting something over aborting.
- ``abort(reason)`` -- used only when the pool is genuinely empty after
  every strategy has run (no web results, no existing sources). An abort
  when the pool is non-empty is a prompt bug.

Inputs supplied in the user message:

- ``user_spec`` -- the freeform markdown spec the conversational agent wrote.
- ``topic_text`` -- short retrieval seed.
- ``attached_sources`` -- currently-linked sources with kinds + user/auto flag.
- ``reason`` -- why discovery was triggered.
- ``removal_history`` -- recent removals from this subscription.

There is no hard round cap; the orchestrator decides when it is satisfied.
"""

import asyncio
import logging
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent_text
from news_service.core.config import get_settings
from news_service.services.relevance import SourceKind, fetch_source_posts, sample_recent_posts

from .finder import run_finder
from .models import ScoredSource, SourceDiscoveryResult

logger = logging.getLogger(__name__)
settings = get_settings()

type AttachedSource = tuple[str, str, bool]
"""(url, source_kind, is_user_specified) for each currently-linked source."""


DISCOVERY_AGENT_PROMPT = """\
You are a news source discovery orchestrator. Your job is to assemble a \
high-quality set of sources for a user's subscription by spawning search \
strategies, reviewing what they found, optionally inspecting candidates, \
and selecting the final set yourself.

Inputs you will see in the user message:
- The user's full spec (what they want followed, tone, exclusions, language).
- The short retrieval topic seed.
- The list of currently-attached sources with kinds and user/auto flag.
- A reason explaining why discovery was triggered right now.

Your tools:
1. **spawn_finder(strategy)** -- Launch one search strategy. You can emit \
SEVERAL spawn_finder calls in a single turn; they will run in parallel. The \
finder searches the web and the existing source database, validates real \
posts against the retrieval embedding, and returns scored candidates.
2. **inspect_source(url)** -- Fetch sample recent content from a candidate \
URL so you can judge whether it actually matches the user's intent. Use \
this when the score alone is not enough to decide (borderline scores, \
suspicious titles, possible off-topic feeds).

Score calibration (each candidate carries a score=X.XX value -- cosine \
similarity on the user's topic embedding, range 0.0-1.0):
- 0.00-0.15: noise / unrelated -- reject unless you have a strong \
non-score reason and have inspected the content.
- 0.15-0.30: marginal -- accept only if the source-kind/language fit \
clearly compensates; prefer inspect_source before accepting.
- 0.30-0.55: on-topic -- prefer these.
- 0.55+: strong -- always prefer when available.
Higher is better. The score is informational, not enforced at selection -- \
anything you submit is accepted. Do not burn slots on noise-range \
candidates.
3. **submit_selection(urls)** -- Finalize discovery with the comma-separated \
list of URLs you accept. Only URLs that finders returned are eligible; \
submit_selection validates this. Whatever you submit is accepted verbatim \
-- there is no post-hoc backfill or rescoring. This ends discovery.
4. **abort(reason)** -- Use ONLY when the pool is genuinely empty after \
every strategy has run (zero candidates across all rounds). If the pool \
has ANY candidates at all, submit the best ones instead of aborting -- \
even one mediocre source is better than zero.

Orchestration guidance:
- Target roughly {target_count} final sources, but use judgment. Fewer \
great sources beats many mediocre ones.
- Start by planning 3 diverse strategies, not more. Emit them as \
parallel spawn_finder calls in the SAME turn. Each strategy should \
cover a different source kind or angle.
- Source-kind coverage is a PREFERENCE, not a requirement. Pick the \
kinds the user's spec actually implies. If the topic is visual/ \
international (anime, sports, games) include Telegram + Reddit + RSS. \
Do not burn a strategy on a kind that obviously will not fit.
- Prefer finishing after ONE round. Only spawn a second round if the \
first round's pool is clearly insufficient (fewer than \
target_count / 2 candidates, or missing a kind the spec explicitly \
needs). Do not re-run rounds to pad the count.
- If you do run a second round, cap it at 3 strategies and make them \
MATERIALLY different from round one (different phrasings, different \
kinds you have not tried, or broader queries).
- When you pick the final set, avoid stacking many finds from a single \
strategy -- prefer cross-strategy diversity unless one strategy's \
results are genuinely much stronger.
- Honour exclusions and language constraints stated in the user's spec.
- Do not select sources that are already attached; those are listed for \
your awareness so you can diversify around them.
- Let the reason guide strategy: a "user pivoted from biotech to AI" \
reason means the old sources are stale and the new focus matters more \
than raw diversity.
- Finder return values may include soft errors like "Search \
rate-limited" or "validation timed out"; treat those as transient \
and move on rather than retrying endlessly.
- Persistence rule: if round one returns zero candidates across every \
strategy, spawn ONE more round (max 3 strategies) with broader \
queries. Abort only if both rounds come back empty.
- The pipeline accepts whatever you submit -- there is no backfill or \
score gate that will rescue dropped candidates. If the pool has ANY \
entries at all, submit them (ordered best-first) instead of aborting. \
A sparse subscription is recoverable; an empty selection with a \
non-empty pool is a prompt failure.
- Low scores alone are NOT a reason to drop a candidate. If the best \
thing the finders found scores 0.15, and nothing better exists, \
submit it anyway and let the user refine later.
- Never emit Markdown bold syntax (**...**) in any text you produce. \
The frontend does not render it and the asterisks appear literally. \
Use plain text -- no bold markers at all.
{removal_context}\
"""


def _normalize_url(url: str) -> str:
    return url.rstrip("/").lower()


_PROGRESS_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "starting": "🔎 Looking for sources...",
        "planning": "🧭 Planning search strategies...",
        "searching": "🌐 Searching: {strategy}",
        "searching_generic": "🌐 Searching for sources...",
        "finished": "✓ Discovery finished: {count} source(s) selected",
        "finished_empty": "✓ Discovery finished: no sources matched",
        "aborted": "⚠️ Discovery stopped: {reason}",
    },
    "ru": {
        "starting": "🔎 Ищу источники...",
        "planning": "🧭 Планирую стратегии поиска...",
        "searching": "🌐 Ищу: {strategy}",
        "searching_generic": "🌐 Ищу источники...",
        "finished": "✓ Поиск завершён: выбрано источников — {count}",
        "finished_empty": "✓ Поиск завершён: подходящих источников не найдено",
        "aborted": "⚠️ Поиск остановлен: {reason}",
    },
}


def _progress_text(language: str, key: str, **fmt: Any) -> str:
    table = _PROGRESS_STRINGS.get(language) or _PROGRESS_STRINGS["en"]
    template = table.get(key) or _PROGRESS_STRINGS["en"].get(key) or ""
    try:
        return template.format(**fmt)
    except (KeyError, IndexError):
        return template


def _emit_progress(
    queue: asyncio.Queue[dict[str, Any]] | None,
    language: str,
    phase: str,
    **extra: Any,
) -> None:
    if queue is None:
        return
    key = extra.pop("_text_key", phase)
    text = _progress_text(language, key, **extra)
    event: dict[str, Any] = {
        "event": "discovery_progress",
        "phase": phase,
        "display_text": text,
    }
    event.update(extra)
    queue.put_nowait(event)


def _format_scored(sources: list[ScoredSource]) -> str:
    """Compact one-line-per-source listing for a finder return value."""
    if not sources:
        return "no sources found"
    lines = [
        f"- {src.url} ({src.source_kind}, score={src.relevance_score:.2f})"
        for src in sorted(sources, key=lambda s: s.relevance_score, reverse=True)
    ]
    return "\n".join(lines)


def _format_attached(attached: list[AttachedSource]) -> str:
    if not attached:
        return "Currently attached sources: none."
    lines = ["Currently attached sources:"]
    for url, kind, is_user in attached:
        label = "user-specified" if is_user else "auto-discovered"
        lines.append(f"  - {url} ({kind}, {label})")
    return "\n".join(lines)


def _build_discovery_input(
    *,
    topic_text: str,
    user_spec: str,
    attached: list[AttachedSource],
    reason: str,
) -> str:
    parts: list[str] = []
    if user_spec.strip():
        parts.append(f"User spec:\n{user_spec.strip()}")
    parts.append(f"Retrieval topic seed:\n{topic_text.strip() or '(none)'}")
    parts.append(_format_attached(attached))
    if reason.strip():
        parts.append(f"Reason discovery was triggered:\n{reason.strip()}")
    parts.append(
        "Spawn 3 finders in parallel, inspect candidates if needed, then submit "
        "your chosen selection. Prefer one round unless the pool is clearly thin."
    )
    return "\n\n".join(parts)


async def run_source_discovery(
    *,
    session: AsyncSession,
    topic_text: str,
    prompt_embedding: list[float],
    user_spec: str = "",
    attached_sources: list[AttachedSource] | None = None,
    reason: str = "",
    removal_history: str = "",
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    display_language: str = "en",
) -> SourceDiscoveryResult:
    """Run the source discovery ReAct agent and return its chosen selection.

    The orchestrator spawns finders (in parallel when it wants), optionally
    inspects candidates, and submits the final URL list via submit_selection.
    Returns whatever the orchestrator picked -- there is no post-hoc cosine
    top-N cut.
    """
    attached = attached_sources or []
    exclude_urls = {_normalize_url(url) for url, _, _ in attached}

    candidate_pool: dict[str, ScoredSource] = {}
    selected_urls: list[str] = []
    aborted: dict[str, str] = {"reason": ""}

    _emit_progress(status_queue, display_language, "starting")
    _emit_progress(status_queue, display_language, "planning")

    async def spawn_finder(strategy: str) -> str:
        """Launch one search strategy against the user's retrieval embedding.

        The finder searches the web and existing source DB, validates
        candidates by fetching real recent posts and scoring them against
        the subscription's retrieval embedding, and returns the scored
        list. Emit multiple spawn_finder calls in the same turn to run
        strategies in parallel.

        Args:
            strategy: One search strategy, e.g. "arxiv RSS feeds on
                transformer architectures" or "Telegram channels covering
                AI safety in Russian".

        Returns:
            A listing of this strategy's scored sources, or a note that
            nothing passed validation.
        """
        trimmed = strategy.strip()
        if not trimmed:
            return "empty strategy; nothing spawned."

        logger.info("Discovery spawning finder for strategy '%s'", trimmed[:80])
        _emit_progress(
            status_queue,
            display_language,
            "searching",
            strategy=trimmed[:80],
        )

        try:
            found = await run_finder(
                strategy=trimmed,
                session=session,
                prompt_embedding=prompt_embedding,
                exclude_urls=list(exclude_urls),
                status_queue=status_queue,
            )
        except Exception as exc:
            logger.exception("Finder crashed for strategy '%s'", trimmed[:80])
            return f"finder crashed: {exc}"

        fresh: list[ScoredSource] = []
        for src in found:
            key = _normalize_url(src.url)
            if key in exclude_urls:
                continue
            if key in candidate_pool:
                continue
            candidate_pool[key] = src
            fresh.append(src)

        if not fresh:
            return f"strategy '{trimmed[:80]}': {len(found)} found, 0 new after dedupe."
        return f"strategy '{trimmed[:80]}' returned:\n{_format_scored(fresh)}"

    async def inspect_source(url: str) -> str:
        """Fetch sample recent content from a candidate URL.

        Only candidates returned by a prior spawn_finder call can be
        inspected. Returns a short preview of sampled recent posts so the
        orchestrator can judge actual content before accepting.

        Args:
            url: Canonical URL of a candidate from the pool.

        Returns:
            Content preview or an error/"not in pool" message.
        """
        key = _normalize_url(url)
        candidate = candidate_pool.get(key)
        if candidate is None:
            return f"{url}: not in candidate pool. Spawn a finder that discovers it first."

        logger.info("Discovery inspecting %s", candidate.url)
        try:
            posts = await asyncio.wait_for(
                fetch_source_posts(candidate.url, candidate.source_kind),
                timeout=settings.source_validation_timeout_seconds,
            )
        except TimeoutError:
            return f"{url}: inspection timed out (host too slow)."
        except Exception as exc:
            return f"{url}: fetch failed ({exc})."
        if not posts:
            return f"{url}: no recent posts available."

        samples = sample_recent_posts(
            posts,
            sample_size=min(5, settings.content_sample_size),
            window_days=settings.content_sample_window_days,
        )
        if not samples:
            return f"{url}: sampling returned nothing."
        preview = "\n---\n".join(s[:400] for s in samples)
        return f"{url} ({candidate.source_kind}, score={candidate.relevance_score:.2f}):\n{preview}"

    async def submit_selection(urls: str) -> str:
        """Finalize discovery with the URLs you have chosen to accept.

        Only URLs present in the candidate pool (i.e. returned by some
        finder this run) are valid. Invalid entries are rejected and the
        call is a no-op in that case; rerun with a corrected list.

        Args:
            urls: Comma-separated canonical URLs from the pool.

        Returns:
            Confirmation with the accepted URLs, or an error naming the
            unknown entries.
        """
        picked_raw = [u.strip() for u in urls.split(",") if u.strip()]
        if not picked_raw:
            return "no URLs provided."

        valid: list[str] = []
        unknown: list[str] = []
        for raw in picked_raw:
            key = _normalize_url(raw)
            if key in candidate_pool:
                valid.append(candidate_pool[key].url)
            else:
                unknown.append(raw)

        if unknown:
            return (
                f"these URLs are not in the candidate pool: {', '.join(unknown)}. "
                "Submit again with only URLs returned by the finders."
            )

        selected_urls[:] = valid
        return f"selection submitted: {len(valid)} source(s)."

    async def abort(reason: str) -> str:
        """Give up on this discovery run with an explanation.

        Use only when the run cannot sensibly continue (all finders
        failed, nothing useful returned, spec self-contradicts).

        Args:
            reason: Short explanation of what went wrong.

        Returns:
            Confirmation that discovery will end with no selections.
        """
        cleaned = reason.strip() or "unspecified"
        aborted["reason"] = cleaned
        return f"aborted: {cleaned}."

    removal_context = ""
    if removal_history:
        removal_context = (
            f"\n\nRecently removed sources (use judgment about re-adding):\n{removal_history}\n"
        )

    prompt = DISCOVERY_AGENT_PROMPT.format(
        target_count=settings.source_target_count,
        removal_context=removal_context,
    )

    agent = Agent(
        name="discovery_agent",
        model=LiteLlm(model=settings.litellm_model),
        instruction=prompt,
        tools=[spawn_finder, inspect_source, submit_selection, abort],
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )

    await run_agent_text(
        agent=agent,
        message=_build_discovery_input(
            topic_text=topic_text,
            user_spec=user_spec,
            attached=attached,
            reason=reason,
        ),
    )

    if aborted["reason"]:
        logger.warning(
            "Discovery aborted for topic '%s': %s",
            topic_text[:60],
            aborted["reason"],
        )
        _emit_progress(
            status_queue,
            display_language,
            "aborted",
            reason=aborted["reason"],
        )
        if candidate_pool:
            logger.warning(
                "Discovery aborted with a non-empty pool (%d candidates). "
                "The prompt forbids this; treat as zero sources and surface "
                "the abort reason to the caller.",
                len(candidate_pool),
            )

    selected = [
        candidate_pool[_normalize_url(u)]
        for u in selected_urls
        if _normalize_url(u) in candidate_pool
    ]

    logger.info(
        "Discovery orchestrator finished for '%s': pool=%d, selected=%d",
        topic_text[:60],
        len(candidate_pool),
        len(selected),
    )
    _emit_progress(
        status_queue,
        display_language,
        "finished",
        _text_key="finished" if selected else "finished_empty",
        count=len(selected),
    )
    return SourceDiscoveryResult(sources=selected, abort_reason=aborted["reason"])


__all__ = [
    "AttachedSource",
    "DISCOVERY_AGENT_PROMPT",
    "ScoredSource",
    "SourceKind",
    "run_source_discovery",
]
