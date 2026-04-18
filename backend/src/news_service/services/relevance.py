"""Score source relevance by sampling real content and comparing to the prompt."""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import feedparser
import httpx

from news_service.core.config import get_settings
from news_service.db.vector_store import embed_texts
from news_service.services.article_fetch import fetch_article_text
from news_service.services.reddit import extract_reddit_subreddit_from_url, fetch_reddit_posts
from news_service.services.telegram import extract_telegram_channel_from_url, fetch_telegram_posts
from news_service.services.twitter import extract_twitter_account_from_url, fetch_twitter_posts

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit", "twitter_account"]

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(slots=True)
class DatedPost:
    """Text content plus the publication timestamp we know about it, if any."""

    text: str
    published_at: datetime | None


async def fetch_source_posts(url: str, source_kind: SourceKind) -> list[DatedPost]:
    """Fetch posts from a source and return their text + published_at."""
    if source_kind == "telegram_channel":
        channel = extract_telegram_channel_from_url(url)
        if channel is None:
            return []
        posts = await fetch_telegram_posts(channel)
        return [
            DatedPost(text=post.body, published_at=post.published_at)
            for post in posts
            if post.body.strip()
        ]

    if source_kind == "reddit_subreddit":
        subreddit = extract_reddit_subreddit_from_url(url)
        if subreddit is None:
            return []
        posts = await fetch_reddit_posts(subreddit)
        return [
            DatedPost(
                text=f"{post.title} {post.body}".strip(),
                published_at=post.published_at,
            )
            for post in posts
            if post.title.strip()
        ]

    if source_kind == "twitter_account":
        account = extract_twitter_account_from_url(url)
        if account is None:
            return []
        posts = await fetch_twitter_posts(account)
        return [
            DatedPost(
                text=f"{post.title} {post.body}".strip(),
                published_at=post.published_at,
            )
            for post in posts
            if post.body.strip()
        ]

    return await _fetch_rss_posts(url)


async def _fetch_rss_posts(url: str) -> list[DatedPost]:
    try:
        async with httpx.AsyncClient(
            timeout=settings.http_timeout_seconds,
            proxy=settings.proxy_url,
        ) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code != 200:
                return []
        parsed = feedparser.parse(response.text)
    except Exception:
        logger.debug("Failed to fetch RSS posts from %s", url)
        return []

    entries: list[tuple[str, str, str, datetime | None]] = []
    for entry in parsed.entries:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        link = getattr(entry, "link", "") or ""
        published = None
        raw = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if raw is not None:
            try:
                published = datetime(*raw[:6], tzinfo=UTC)
            except (TypeError, ValueError):
                published = None
        entries.append((link, title, summary, published))

    sem = asyncio.Semaphore(settings.article_fetch_concurrency)

    async def _enrich(link: str) -> str | None:
        if not link:
            return None
        async with sem:
            return await fetch_article_text(
                link,
                timeout_seconds=settings.article_fetch_timeout_seconds,
                max_chars=settings.article_body_max_chars,
            )

    article_bodies = await asyncio.gather(*(_enrich(link) for link, *_ in entries))

    dated: list[DatedPost] = []
    for (_link, title, summary, published), article in zip(entries, article_bodies, strict=True):
        body = article if article else summary
        text = f"{title}\n\n{body}".strip()
        if not text:
            continue
        dated.append(DatedPost(text=text, published_at=published))
    return dated


def sample_recent_posts(
    posts: list[DatedPost],
    sample_size: int,
    *,
    window_days: int,
    now: datetime | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Pick up to ``sample_size`` random posts from the last ``window_days``.

    Falls back to random sampling across the full pool when the recent
    window yields fewer items than the sample size (either because the
    source is sparse or because timestamps are missing). Posts without a
    ``published_at`` are treated as eligible under the fallback but not
    the window, since we cannot confirm they are recent.
    """
    if not posts:
        return []
    rng = rng or random.Random()
    cutoff = (now or datetime.now(UTC)) - timedelta(days=window_days)

    recent = [p for p in posts if p.published_at is not None and p.published_at >= cutoff]
    if len(recent) >= sample_size:
        return [p.text for p in rng.sample(recent, sample_size)]

    if len(posts) <= sample_size:
        return [p.text for p in posts]
    return [p.text for p in rng.sample(posts, sample_size)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def score_source_relevance(
    post_texts: list[str],
    prompt_embedding: list[float],
    top_k: int,
) -> float:
    """Embed posts and return mean of top-K cosine similarities vs prompt."""
    if not post_texts:
        return 0.0
    embeddings = await embed_texts(post_texts)
    similarities = [cosine_similarity(emb, prompt_embedding) for emb in embeddings]
    similarities.sort(reverse=True)
    top = similarities[:top_k]
    return sum(top) / len(top) if top else 0.0


async def score_candidate(
    url: str,
    source_kind: SourceKind,
    prompt_embedding: list[float],
) -> tuple[float, list[str]]:
    """Fetch, sample, and score a candidate source.

    Returns (relevance_score, sampled_texts).
    """
    try:
        posts = await fetch_source_posts(url, source_kind)
    except Exception:
        logger.debug("Failed to fetch posts from %s for relevance scoring", url)
        return 0.0, []

    if not posts:
        return 0.0, []

    sampled = sample_recent_posts(
        posts,
        sample_size=settings.content_sample_size,
        window_days=settings.content_sample_window_days,
    )
    if not sampled:
        return 0.0, []

    score = await score_source_relevance(
        sampled, prompt_embedding, settings.content_relevance_top_k
    )
    logger.info("Relevance score for %s: %.3f (sampled %d posts)", url, score, len(sampled))
    return score, sampled
