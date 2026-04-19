"""Generic Source Finder — act-mode agent that executes a single search strategy.

Each finder instance receives one strategy string from the orchestrator and uses
tools to search for, validate, and score relevant sources. Multiple finders run
in parallel, each with a different strategy.

The finder follows the ReAct pattern: reason about what to search for, execute
a search or validation tool, observe the results, and repeat until enough
good sources are found.
"""

import asyncio
import logging
import uuid
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent_text
from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text, find_similar_sources
from news_service.services.relevance import score_candidate
from news_service.services.search import search_web

from .models import ScoredSource, SourceKind

logger = logging.getLogger(__name__)
settings = get_settings()

FINDER_PROMPT = """\
You are a news source finder. Execute the search strategy you've been given.

Use your tools to:
1. Search the existing source database for matches.
2. Search the web for new sources using varied, specific queries.
3. Validate and score the most promising candidates.

Rules:
- Focus on the source types mentioned in your strategy.
- Validate only your top candidates, not every search result.
- Stop once you have 3-4 validated sources with scores above 0.5.
- Skip sources that are in the exclude list.
- Never emit Markdown bold syntax (**...**) in any text you produce. \
The frontend does not render it and the asterisks appear literally. \
Use plain text -- no bold markers at all.

Search-query guidance (tool_search_web):
- DO NOT use operators like site:, inurl:, filetype:. The backing meta- \
search returns empty or noisy results for those. Use natural language.
- For Telegram, search things like: "best Telegram channels about X", \
"X news Telegram channel", "Telegram @... X" (but NOT site:t.me).
- For Reddit, search: "best subreddits for X", "r/... X community" \
(but NOT site:reddit.com).
- For X/Twitter, search: "best Twitter accounts for X news", \
"X official account Twitter" (but NOT site:twitter.com or site:x.com).
- For RSS, search: "X RSS feed", "X Atom feed", "list of RSS feeds \
for X", or go to a known site and look for its feed URL in your \
head knowledge before searching (e.g. Crunchyroll's RSS is \
/rss/anime, ArsTechnica's is /feed/).
- Vary phrasing across searches. If one query returns empty or \
nothing relevant, rephrase before trying again -- do not just append \
qualifiers.

URL-quality guidance (validate_and_score_source):
- For source_kind="rss", only submit URLs that are ACTUAL feed \
endpoints. Telltales: path ending in .xml / .rss / /feed / /feed/ / \
/rss / /rss/ / /atom.xml / /index.xml. A bare landing page \
(example.com/news) is almost certainly NOT a feed and will return \
"could not fetch posts" -- do not submit it as rss.
- If you only have a landing-page URL, try common feed suffixes \
(+"/feed", +"/rss", +"/feed.xml") and submit THOSE to the validator \
rather than the landing page.
- If the validator reports "could not fetch posts", try one feed-URL \
variant of the same domain before discarding the candidate.

Persistence:
- Do NOT return empty-handed after a single failed query. Try at \
least 3 different search phrasings before concluding the strategy \
produced nothing.
- A source with a low score (<0.5) is still better than nothing if \
the topic/source-kind is right; mention it in your summary so the \
orchestrator can decide.

When done, summarize what you found -- or, if you found nothing, \
briefly say which queries you tried and what kinds of URL you saw \
(so the orchestrator can adjust its next strategy).
"""


async def run_finder(
    *,
    strategy: str,
    session: AsyncSession,
    prompt_embedding: list[float],
    exclude_urls: list[str],
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> list[ScoredSource]:
    """Execute a single search strategy and return discovered sources."""
    discovered: list[ScoredSource] = []

    async def search_existing_sources(query: str) -> str:
        """Search the existing source database for sources matching the query.

        Args:
            query: The search query to find relevant existing sources.

        Returns:
            Formatted list of existing sources found in the database.
        """
        query_embedding = await embed_text(query)
        sources = await find_similar_sources(
            session,
            query_embedding,
            threshold=settings.content_db_candidate_threshold,
            limit=settings.source_target_count * 2,
        )
        if not sources:
            return "No existing sources found in database."
        lines: list[str] = []
        for src in sources:
            if src.url in exclude_urls:
                continue
            desc = (src.source_description or "")[:120]
            lines.append(f"- {src.url} (title: {src.title}, description: {desc})")
        return (
            f"Existing sources in database:\n{'\n'.join(lines)}"
            if lines
            else ("All matching sources are already in the exclude list.")
        )

    async def tool_search_web(query: str) -> str:
        """Search the web for news sources relevant to the query.

        Try queries like "best RSS feeds about [topic]",
        "Telegram channels for [topic] news", etc.

        Args:
            query: Search query to find relevant news sources.

        Returns:
            Formatted search results with URLs and descriptions.
        """
        if status_queue is not None:
            status_queue.put_nowait(
                {
                    "event": "status",
                    "status_key": "status_searching_web",
                    "status_text": f"Searching the web: {query[:60]}...",
                }
            )
        return await search_web(query)

    async def validate_and_score_source(url: str, source_kind: str) -> str:
        """Validate a source URL and score its content relevance.

        Fetches real posts, embeds them, and computes similarity to the subscription.

        Args:
            url: The canonical source URL to validate and score.
            source_kind: One of: rss, telegram_channel, reddit_subreddit, twitter_account.

        Returns:
            Validation result with relevance score and sample content preview.
        """
        if url in exclude_urls:
            return f"Source {url}: skipped (already subscribed)"
        if status_queue is not None:
            status_queue.put_nowait(
                {
                    "event": "status",
                    "status_key": "status_validating_source",
                    "status_text": f"Checking {url[:60]}...",
                }
            )
        kind: SourceKind = source_kind  # type: ignore[assignment]
        relevance, sampled = await score_candidate(url, kind, prompt_embedding)
        if not sampled:
            return f"Source {url}: could not fetch posts (score: 0.0)"
        if relevance >= 0.0:
            discovered.append(
                ScoredSource(url=url, title="", source_kind=kind, relevance_score=relevance)
            )
        preview = sampled[0][:200] if sampled else ""
        return (
            f"Source {url}: relevance_score={relevance:.3f}, "
            f"sampled {len(sampled)} posts. Preview: {preview}"
        )

    exclude_note = ""
    if exclude_urls:
        exclude_note = "\n\nExclude these URLs (already subscribed):\n" + "\n".join(
            f"- {u}" for u in exclude_urls
        )

    agent = Agent(
        name=f"finder_{uuid.uuid4().hex[:6]}",
        model=LiteLlm(model=settings.litellm_model),
        instruction=FINDER_PROMPT + exclude_note,
        tools=[search_existing_sources, tool_search_web, validate_and_score_source],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    await run_agent_text(
        agent=agent,
        message=f"Execute this search strategy:\n{strategy}",
    )

    logger.info(
        "Finder completed strategy '%s' — found %d sources",
        strategy[:60],
        len(discovered),
    )
    return discovered
