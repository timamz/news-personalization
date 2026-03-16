"""Score source relevance by sampling real content and comparing to the prompt."""

import logging
from typing import Literal

import feedparser
import httpx

from news_service.core.config import get_settings
from news_service.db.vector_store import embed_texts
from news_service.services.reddit import extract_reddit_subreddit_from_url, fetch_reddit_posts
from news_service.services.telegram import extract_telegram_channel_from_url, fetch_telegram_posts
from news_service.services.twitter import extract_twitter_account_from_url, fetch_twitter_posts

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit", "twitter_account"]

logger = logging.getLogger(__name__)
settings = get_settings()


async def fetch_source_posts(url: str, source_kind: SourceKind) -> list[str]:
    """Fetch posts from a source and return their text content."""
    if source_kind == "telegram_channel":
        channel = extract_telegram_channel_from_url(url)
        if channel is None:
            return []
        posts = await fetch_telegram_posts(channel)
        return [post.body for post in posts if post.body.strip()]

    if source_kind == "reddit_subreddit":
        subreddit = extract_reddit_subreddit_from_url(url)
        if subreddit is None:
            return []
        posts = await fetch_reddit_posts(subreddit)
        return [f"{post.title} {post.body}".strip() for post in posts if post.title.strip()]

    if source_kind == "twitter_account":
        account = extract_twitter_account_from_url(url)
        if account is None:
            return []
        posts = await fetch_twitter_posts(account)
        return [f"{post.title} {post.body}".strip() for post in posts if post.body.strip()]

    # RSS
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code != 200:
                return []
        parsed = feedparser.parse(response.text)
        texts: list[str] = []
        for entry in parsed.entries:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            text = f"{title} {summary}".strip()
            if text:
                texts.append(text)
        return texts
    except Exception:
        logger.debug("Failed to fetch RSS posts from %s", url)
        return []


def sample_posts(posts: list[str], sample_size: int) -> list[str]:
    """Evenly sample posts across the list, not just the latest."""
    if len(posts) <= sample_size:
        return list(posts)
    step = len(posts) / sample_size
    return [posts[int(i * step)] for i in range(sample_size)]


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

    sampled = sample_posts(posts, settings.content_sample_size)
    score = await score_source_relevance(
        sampled, prompt_embedding, settings.content_relevance_top_k
    )
    logger.info("Relevance score for %s: %.3f (sampled %d posts)", url, score, len(sampled))
    return score, sampled
