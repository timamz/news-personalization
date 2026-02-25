import logging

import feedparser
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
_client = AsyncOpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """\
You are an RSS feed discovery agent. Given a list of news topics, find real, working RSS feed \
URLs that cover those topics.

Rules:
- Return only well-known, major news sources with RSS feeds.
- For each topic, try to find 2-3 relevant RSS feeds.
- Prefer feeds from sources like Reuters, BBC, TechCrunch, ArsTechnica, MIT Technology Review, \
The Verge, etc.
- Return the full URL to the RSS/Atom feed (not the website homepage).
- Common RSS feed URL patterns: /feed, /rss, /rss.xml, /feed/rss, /feeds/all.atom.xml
"""


class DiscoveredFeedItem(BaseModel):
    url: str = Field(..., description="RSS feed URL")
    topic_tags: list[str] = Field(..., description="Topics this feed covers")
    title: str = Field(default="", description="Feed title")


class DiscoveredFeedList(BaseModel):
    feeds: list[DiscoveredFeedItem] = Field(..., description="List of discovered RSS feeds")


async def discover_feeds(topics: list[str]) -> list[DiscoveredFeedItem]:
    topics_str = ", ".join(topics)

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Find RSS feeds for these topics: {topics_str}"},
        ],
        response_format=DiscoveredFeedList,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"LLM returned empty response for topics: {topics_str}")

    validated = []
    for feed in result.feeds:
        if await validate_feed_url(feed.url):
            validated.append(feed)
            logger.info("Validated RSS feed: %s (%s)", feed.url, feed.title)
        else:
            logger.warning("Invalid RSS feed URL discarded: %s", feed.url)

    return validated


async def validate_feed_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code != 200:
                return False

        parsed = feedparser.parse(response.text)
        return len(parsed.entries) > 0
    except (httpx.HTTPError, Exception):
        logger.exception("Feed validation failed for %s", url)
        return False
