"""Source discovery pipeline: single looped ADK agent.

The discovery agent analyzes the topic, runs parallel search strategies via
run_parallel_search(), reviews results, optionally refines, and finalizes
via submit_results(). Replaces the old 3-component orchestrator/finder/aggregator
architecture with a single agent that loops until satisfied.

Inputs beyond the topic seed:

- ``user_spec`` -- the freeform markdown spec the conversational agent wrote.
  Gives the finder nuance (tone, exclusions, language, angle) that a bare
  topic string cannot convey.
- ``attached_sources`` -- everything currently linked to the subscription, so
  the agent can diversify across source kinds and avoid proposing near-
  duplicates. Includes the ``is_user_specified`` flag so the agent knows
  which sources are pinned by the user.
- ``reason`` -- a freeform string from whoever triggered discovery
  (conversational agent, reflector, manual), explaining why now and what
  just changed. Used to steer strategy.
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

from .finder import run_finder
from .models import ScoredSource, SourceDiscoveryResult

logger = logging.getLogger(__name__)
settings = get_settings()

type AttachedSource = tuple[str, str, bool]
"""(url, source_kind, is_user_specified) for each currently-linked source."""


DISCOVERY_AGENT_PROMPT = """\
You are a news source discovery agent. Your job is to find high-quality sources \
for a user's subscription.

Inputs you will see in the user message:
- The user's full spec (what they want followed, tone, exclusions, language).
- The short retrieval query / topic seed.
- The list of currently-attached sources with kinds and whether each is \
user-specified or auto-discovered.
- A reason explaining why discovery was triggered right now.

You have these tools:
1. **run_parallel_search(strategies)** -- Runs multiple search strategies in parallel. \
Each strategy is executed by a separate finder that searches the web and existing \
database, validates sources, and scores them by relevance. Returns results from all \
strategies combined.
2. **submit_results()** -- Call this when you have enough good sources. \
Finalizes the discovery process.

Workflow:
1. Read the spec, attached sources, and reason carefully. Notice which source \
kinds are already covered well and which are underrepresented -- prefer to \
diversify rather than stack more of the same kind.
2. Decide on 2-5 initial search strategies. Target different source types: \
RSS feeds, Telegram channels, Reddit subreddits, Twitter/X accounts.
3. Call run_parallel_search with your strategies.
4. Review the results:
   - How many sources were found? (target: {target_count})
   - Are the relevance scores good? (aim for >0.5)
   - Is there diversity across source types?
5. If not enough sources or missing a source type, run another round with refined strategies.
6. When satisfied (or after 2 rounds max), call submit_results.

Guidelines:
- Use specific, targeted search queries: "arxiv RSS feeds about transformer architectures" \
is better than "academic sources".
- Adapt to the topic domain: academic -> arxiv/papers, consumer -> social/news, tech -> mixed.
- Honour exclusions and language constraints stated in the user's spec.
- Do not propose sources that are already attached (they will be filtered, but \
strategies should aim elsewhere).
- Let the reason guide you: if the reason says "user just switched focus from \
biotech to AI", the old sources are likely stale and AI-focused strategies \
matter more than round-robin diversity.
- Maximum 2 rounds of searching.
{removal_context}\
"""


def _deduplicate(sources: list[ScoredSource]) -> list[ScoredSource]:
    """Deduplicate sources by normalized URL, keeping the first occurrence."""
    seen_urls: set[str] = set()
    unique: list[ScoredSource] = []
    for source in sources:
        normalized = source.url.rstrip("/").lower()
        if normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        unique.append(source)
    return unique


def _format_attached(attached: list[AttachedSource]) -> str:
    """Render the attached-sources block for the discovery agent's input."""
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
    """Assemble the user-message payload passed to the discovery agent."""
    parts: list[str] = []
    if user_spec.strip():
        parts.append(f"User spec:\n{user_spec.strip()}")
    parts.append(f"Retrieval topic seed:\n{topic_text.strip() or '(none)'}")
    parts.append(_format_attached(attached))
    if reason.strip():
        parts.append(f"Reason discovery was triggered:\n{reason.strip()}")
    parts.append("Find high-quality sources that complement what is already attached.")
    return "\n\n".join(parts)


def _format_summary(sources: list[ScoredSource]) -> str:
    """Build a human-readable summary of discovered sources for the agent."""
    if not sources:
        return "No sources found yet."
    type_counts: dict[str, int] = {}
    for src in sources:
        type_counts[src.source_kind] = type_counts.get(src.source_kind, 0) + 1
    types_str = ", ".join(f"{count} {kind}" for kind, count in type_counts.items())
    top_scores = sorted((s.relevance_score for s in sources), reverse=True)[:5]
    scores_str = ", ".join(f"{s:.2f}" for s in top_scores)
    lines = [
        f"Found {len(sources)} sources. Types: {types_str}. Top scores: {scores_str}.",
        "",
        "Sources:",
    ]
    for src in sorted(sources, key=lambda s: s.relevance_score, reverse=True):
        lines.append(f"  - {src.url} ({src.source_kind}, score={src.relevance_score:.2f})")
    return "\n".join(lines)


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
) -> SourceDiscoveryResult:
    """Run the single-agent source discovery pipeline.

    The discovery agent loops: it plans strategies, runs parallel finders,
    reviews results, and optionally refines until satisfied or 2 rounds pass.

    Returns a SourceDiscoveryResult with scored, validated sources.
    """
    attached = attached_sources or []
    exclude_urls = [url for url, _, _ in attached]

    shared_state: dict[str, Any] = {
        "sources": [],
        "completed": False,
    }

    if status_queue is not None:
        status_queue.put_nowait({"event": "status", "status_key": "status_planning_discovery"})

    async def run_parallel_search(strategies: str) -> str:
        """Run multiple search strategies in parallel.

        Args:
            strategies: Newline-separated list of search strategies to execute.

        Returns:
            Combined results from all strategies.
        """
        strategy_list = [s.strip() for s in strategies.strip().split("\n") if s.strip()]
        if not strategy_list:
            return "No strategies provided. Please provide at least one strategy."

        logger.info(
            "Discovery agent launching %d finder(s) for topic '%s'",
            len(strategy_list),
            topic_text[:60],
        )

        if status_queue is not None:
            status_queue.put_nowait({"event": "status", "status_key": "status_searching_sources"})

        finder_tasks = [
            run_finder(
                strategy=strategy,
                session=session,
                prompt_embedding=prompt_embedding,
                exclude_urls=exclude_urls,
                status_queue=status_queue,
            )
            for strategy in strategy_list
        ]

        finder_results = await asyncio.gather(*finder_tasks, return_exceptions=True)

        round_sources: list[ScoredSource] = []
        for i, result in enumerate(finder_results):
            if isinstance(result, Exception):
                logger.exception(
                    "Finder failed for strategy '%s': %s",
                    strategy_list[i][:60],
                    result,
                )
            else:
                round_sources.extend(result)

        all_sources = shared_state["sources"] + round_sources
        shared_state["sources"] = _deduplicate(all_sources)

        return _format_summary(shared_state["sources"])

    async def submit_results() -> str:
        """Finalize the discovery with current results.

        Call this when you have enough good sources or after 2 search rounds.

        Returns:
            Confirmation with final source count.
        """
        shared_state["completed"] = True
        return f"Discovery complete: {len(shared_state['sources'])} sources found."

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
        tools=[run_parallel_search, submit_results],
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

    sources = shared_state["sources"]
    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    selected = sources[: settings.source_target_count]

    logger.info(
        "Discovery agent finished for '%s': %d total unique, %d selected",
        topic_text[:60],
        len(sources),
        len(selected),
    )

    return SourceDiscoveryResult(sources=selected)
