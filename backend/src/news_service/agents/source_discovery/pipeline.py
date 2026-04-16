"""Source discovery pipeline: single looped ADK agent.

The discovery agent analyzes the topic, runs parallel search strategies via
run_parallel_search(), reviews results, optionally refines, and finalizes
via submit_results(). Replaces the old 3-component orchestrator/finder/aggregator
architecture with a single agent that loops until satisfied.
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

DISCOVERY_AGENT_PROMPT = """\
You are a news source discovery agent. Your job is to find high-quality sources \
for a user's subscription topic.

You have these tools:
1. **run_parallel_search(strategies)** — Runs multiple search strategies in parallel. \
Each strategy is executed by a separate finder that searches the web and existing \
database, validates sources, and scores them by relevance. Returns results from all \
strategies combined.
2. **submit_results()** — Call this when you have enough good sources. \
Finalizes the discovery process.

Workflow:
1. Analyze the topic and decide on 2-5 initial search strategies. Target different \
source types: RSS feeds, Telegram channels, Reddit subreddits, Twitter/X accounts.
2. Call run_parallel_search with your strategies.
3. Review the results:
   - How many sources were found? (target: {target_count})
   - Are the relevance scores good? (aim for >0.5)
   - Is there diversity across source types?
4. If not enough sources or missing a source type, run another round with refined strategies.
5. When satisfied (or after 2 rounds max), call submit_results.

Guidelines:
- Use specific, targeted search queries: "arxiv RSS feeds about transformer architectures" \
is better than "academic sources".
- Adapt to the topic domain: academic -> arxiv/papers, consumer -> social/news, tech -> mixed.
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
    raw_prompt: str,
    prompt_embedding: list[float],
    exclude_urls: list[str] | None = None,
    removal_history: str = "",
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> SourceDiscoveryResult:
    """Run the single-agent source discovery pipeline.

    The discovery agent loops: it plans strategies, runs parallel finders,
    reviews results, and optionally refines until satisfied or 2 rounds pass.

    Returns a SourceDiscoveryResult with scored, validated sources.
    """
    if exclude_urls is None:
        exclude_urls = []

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
            raw_prompt[:60],
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
        message=f"Find sources for this topic:\n{raw_prompt}",
    )

    sources = shared_state["sources"]
    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    selected = sources[: settings.source_target_count]

    logger.info(
        "Discovery agent finished for '%s': %d total unique, %d selected",
        raw_prompt[:60],
        len(sources),
        len(selected),
    )

    return SourceDiscoveryResult(sources=selected)
