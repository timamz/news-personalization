"""Agentic source discovery using OpenAI Agents SDK with tool-calling."""

import asyncio
import logging
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
    Tool,
    function_tool,
)
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.discovery import (
    DiscoveredSourceItem,
    SourceKind,
    discover_reddit_subreddits,
    discover_rss_feeds,
    discover_telegram_channels,
    discover_twitter_accounts,
)
from news_service.core.config import get_settings
from news_service.core.openai_client import agents_model
from news_service.db.vector_store import embed_text, find_similar_feeds
from news_service.services.relevance import score_candidate

logger = logging.getLogger(__name__)
settings = get_settings()


class ScoredSource(BaseModel):
    url: str = Field(..., description="Canonical source URL")
    title: str = Field(default="", description="Human-readable source title")
    source_kind: SourceKind = Field(..., description="Source type")
    relevance_score: float = Field(
        ..., description="0.0-1.0 content relevance score (higher is better)"
    )


class SourceDiscoveryResult(BaseModel):
    sources: list[ScoredSource] = Field(..., description="Selected sources ranked by relevance")


SOURCE_DISCOVERY_PROMPT = """\
You are a news source discovery agent. Given a user's news subscription request, \
find the best sources to cover their interests.

Strategy:
1. First, search the existing source database for relevant matches.
2. Discover new sources across multiple source types: RSS feeds, Telegram channels, \
and Reddit subreddits. Call several discovery tools to get broad coverage.
3. Validate and score the most promising candidates to measure content relevance.
4. Select the top {target_count} sources by relevance score.

Efficiency rules:
- Do NOT call the same discovery tool twice with the same query.
- Validate and score only your top candidates, not every single discovery result.
- Once you have {target_count} validated sources with scores above 0.5, stop and return.

Quality criteria:
- A relevance score above 0.7 is good, above 0.5 is acceptable.
- Diversify across source types (RSS, Telegram, Reddit) when possible.

Return the selected sources with their scores. Include only sources you have validated \
and scored. If no sources could be validated, return an empty list.
"""


def _format_discovered_sources(items: list[DiscoveredSourceItem]) -> str:
    if not items:
        return "No sources found."
    lines: list[str] = []
    for item in items:
        lines.append(f"- {item.url} ({item.source_kind}, title: {item.title or 'unknown'})")
    return "\n".join(lines)


@function_tool
async def tool_discover_rss_feeds(query: str) -> str:
    """Discover RSS/Atom feeds relevant to the query from the web.

    Args:
        query: The user's news subscription request describing their interests.
    """
    items = await discover_rss_feeds(query)
    return f"Discovered RSS feeds:\n{_format_discovered_sources(items)}"


@function_tool
async def tool_discover_telegram_channels(query: str) -> str:
    """Discover public Telegram channels relevant to the query.

    Args:
        query: The user's news subscription request describing their interests.
    """
    items = await discover_telegram_channels(query)
    return f"Discovered Telegram channels:\n{_format_discovered_sources(items)}"


@function_tool
async def tool_discover_reddit_subreddits(query: str) -> str:
    """Discover Reddit subreddits relevant to the query.

    Args:
        query: The user's news subscription request describing their interests.
    """
    items = await discover_reddit_subreddits(query)
    return f"Discovered subreddits:\n{_format_discovered_sources(items)}"


@function_tool
async def tool_discover_twitter_accounts(query: str) -> str:
    """Discover Twitter/X accounts relevant to the query.

    Args:
        query: The user's news subscription request describing their interests.
    """
    items = await discover_twitter_accounts(query)
    return f"Discovered Twitter accounts:\n{_format_discovered_sources(items)}"


def _create_source_discovery_agent(
    session: AsyncSession,
    prompt_embedding: list[float],
) -> Agent[None]:
    @function_tool
    async def search_existing_sources(query: str) -> str:
        """Search the existing source database for sources matching the query.

        Args:
            query: The search query to find relevant existing sources.
        """
        query_embedding = await embed_text(query)
        feeds = await find_similar_feeds(
            session,
            query_embedding,
            threshold=settings.content_db_candidate_threshold,
            limit=settings.source_target_count * 2,
        )
        if not feeds:
            return "No existing sources found in database."
        lines: list[str] = []
        for feed in feeds:
            desc = (feed.source_description or "")[:120]
            lines.append(f"- {feed.url} (title: {feed.title}, description: {desc})")
        return f"Existing sources in database:\n{'\n'.join(lines)}"

    @function_tool
    async def validate_and_score_source(url: str, source_kind: str) -> str:
        """Validate a source URL and score its content relevance to the subscription.

        Fetches real posts from the source, embeds them, and computes similarity
        to the user's request. Returns a relevance score between 0.0 and 1.0.

        Args:
            url: The canonical source URL to validate and score.
            source_kind: One of: rss, telegram_channel, reddit_subreddit, twitter_account.
        """
        kind: SourceKind = source_kind  # type: ignore[assignment]
        relevance, sampled = await score_candidate(url, kind, prompt_embedding)
        if not sampled:
            return f"Source {url}: could not fetch posts (score: 0.0)"
        preview = sampled[0][:200] if sampled else ""
        return (
            f"Source {url}: relevance_score={relevance:.3f}, "
            f"sampled {len(sampled)} posts. Preview: {preview}"
        )

    instructions = SOURCE_DISCOVERY_PROMPT.format(target_count=settings.source_target_count)

    return Agent(
        name="source_discovery",
        instructions=instructions,
        tools=[
            search_existing_sources,
            tool_discover_rss_feeds,
            tool_discover_telegram_channels,
            tool_discover_reddit_subreddits,
            # tool_discover_twitter_accounts,  # disabled until Twitter rate limits stabilize
            validate_and_score_source,
        ],
        model=agents_model,
        output_type=SourceDiscoveryResult,
        model_settings=ModelSettings(temperature=0.1),
    )


_TOOL_STATUS: dict[str, str] = {
    "search_existing_sources": "Searching known sources...",
    "tool_discover_rss_feeds": "Discovering RSS feeds...",
    "tool_discover_telegram_channels": "Discovering Telegram channels...",
    "tool_discover_reddit_subreddits": "Discovering Reddit communities...",
    "tool_discover_twitter_accounts": "Discovering X/Twitter accounts...",
    "validate_and_score_source": "Validating source...",
}


class StatusRunHooks(RunHooks[None]):
    """Pushes status events to a queue on each tool invocation."""

    def __init__(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._queue = queue

    async def on_tool_start(
        self,
        context: RunContextWrapper[None],
        agent: Agent[None],
        tool: Tool,
    ) -> None:
        message = _TOOL_STATUS.get(tool.name, f"Running {tool.name}...")
        await self._queue.put({"event": "status", "status_message": message})


async def run_source_discovery(
    *,
    session: AsyncSession,
    raw_prompt: str,
    prompt_embedding: list[float],
    hooks: RunHooks[None] | None = None,
) -> SourceDiscoveryResult:
    """Run the agentic source discovery flow.

    Returns a SourceDiscoveryResult with scored, validated sources.
    """
    agent = _create_source_discovery_agent(session, prompt_embedding)
    result = await Runner.run(
        agent,
        input=f"Find the best news sources for this subscription request: {raw_prompt}",
        run_config=RunConfig(tracing_disabled=True),
        max_turns=40,
        hooks=hooks,
    )
    return result.final_output  # type: ignore[return-value]
