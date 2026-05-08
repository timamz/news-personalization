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
from news_service.agents.web_tools import fetch_page as _fetch_page
from news_service.core.config import get_settings
from news_service.core.llm_usage import agent_tag
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
2. Search the web to find pages that RECOMMEND sources on this topic \
(curated listicles, "best X" posts, community round-ups) and fetch \
the most promising of those pages to read the full list.
3. Harvest concrete source URLs (feed URLs, Telegram channel links, \
subreddit URLs) from the fetched pages.
4. Validate and score the most promising candidates.

Rules:
- Obey the source kind your strategy names. If the strategy targets \
Telegram channels, every web search and every validate_and_score_source \
call MUST be for t.me/... channel URLs with source_kind="telegram_channel"; \
do not drift into RSS. Same for Reddit: reddit.com/r/... URLs with \
source_kind="reddit_subreddit". Same for RSS. A strategy that names a \
kind is a hard constraint, not a hint. If no candidate of the requested \
kind exists after two phrasings + one curator page, return empty for \
THAT kind rather than switching kinds -- the orchestrator combines \
kinds across finders, you do not.
- Validate only your top candidates, not every search result.
- Target 3 validated sources and stop. The orchestrator runs several \
strategies; it does not need every strategy to saturate. If you \
already have 3 decent sources, do NOT keep validating more.
- Budget: at most 2 distinct search phrasings AND at most 4 \
validate_and_score_source calls per strategy. If you hit either \
limit, summarize what you have and stop, even if you have fewer \
than 3 sources. Returning 1 good source + a short note beats \
burning the search budget on dead leads.
- A tool result that starts with "Search rate-limited" or "Search \
temporarily unavailable" is a transient signal -- move on to a \
different phrasing or stop; do not keep retrying the same query.
- Skip sources that are in the exclude list.
- Never emit Markdown bold syntax (**...**) in any text you produce. \
The frontend does not render it and the asterisks appear literally. \
Use plain text -- no bold markers at all.

Primary tactic (curator harvesting) -- use this first:
- Instead of guessing at URLs, let the web recommend them. Search for \
curated lists: "best RSS feeds for X", "top X subreddits", "best \
Telegram channels about X", "awesome X sources", "X news feed roundup".
- Call fetch_page on the 1-3 most promising results (listicles from \
blogs, GitHub "awesome-lists", Reddit threads, Medium articles, \
category pages on aggregators).
- Harvest every concrete source URL or handle from the fetched text: \
feed URLs ending in .xml/.rss/feed/atom, t.me/... Telegram links, \
reddit.com/r/... subreddit URLs.
- Submit the harvested URLs to validate_and_score_source with the \
right source_kind. This is MUCH better than guessing feed paths -- \
the curator has already verified the URL works.

Search-query guidance (tool_search_web):
- DO NOT use operators like site:, inurl:, filetype:. The backing meta- \
search returns empty or noisy results for those. Use natural language.
- Prefer curator-list queries over direct-source queries. "best X \
Telegram channels" outperforms "X Telegram channel"; "list of RSS \
feeds for X" outperforms "X RSS".
- Vary phrasing across searches. If one query returns empty or \
nothing relevant, rephrase before trying again -- do not just append \
qualifiers.

URL-quality guidance (validate_and_score_source):
- For source_kind="rss", only submit URLs that are ACTUAL feed \
endpoints. Telltales: path ending in .xml / .rss / /feed / /feed/ / \
/rss / /rss/ / /atom.xml / /index.xml. A bare landing page \
(example.com/news) is almost certainly NOT a feed and will return \
"could not fetch posts" -- do not submit it as rss.
- If a candidate is a landing page, try fetch_page on it first: \
many sites link to their feed via <link rel="alternate"> or a visible \
"RSS" link. Harvest the feed URL from that HTML rather than guessing.
- If fetch_page does not reveal a feed URL, probe common suffixes \
(/feed, /rss, /feed.xml) and submit THOSE to validate_and_score_source \
-- never the landing page itself.
- If the validator reports "could not fetch posts", try one feed-URL \
variant of the same domain before discarding the candidate.

Score calibration (relevance_score is cosine similarity on the topic \
embedding, range 0.0-1.0):
- 0.00-0.15: noise / unrelated -- do not submit.
- 0.15-0.30: marginal -- submit only if the source-kind/language fit \
clearly compensates, and flag it as marginal in your summary.
- 0.30-0.55: on-topic -- prefer these.
- 0.55+: strong -- always prefer.
Higher is better. Do NOT submit noise-range sources hoping the \
orchestrator will sort it out; it relies on your filtering.

Persistence:
- Do not return empty-handed after a single failed query. Try one \
alternative phrasing OR fetch at least one curator page before \
concluding the strategy produced nothing. Two phrasings is the cap.
- A marginal source (0.15-0.30) is still better than nothing if the \
topic/kind fit clearly compensates; mention it as marginal in your \
summary so the orchestrator can decide.

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
            limit=settings.source_soft_cap * 2,
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
            source_kind: One of: rss, telegram_channel, reddit_subreddit.

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
        try:
            relevance, sampled, is_dormant = await asyncio.wait_for(
                score_candidate(url, kind, prompt_embedding),
                timeout=settings.source_validation_timeout_seconds,
            )
        except TimeoutError:
            logger.info("Validation timed out for %s", url)
            return f"Source {url}: validation timed out (host too slow)"
        if is_dormant:
            return (
                f"Source {url}: REJECTED as dormant -- no posts newer than "
                f"{settings.news_item_max_age_days} days. Do not submit this URL; "
                "find a different source."
            )
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
        tools=[
            search_existing_sources,
            tool_search_web,
            _fetch_page,
            validate_and_score_source,
        ],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    with agent_tag("finder"):
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
