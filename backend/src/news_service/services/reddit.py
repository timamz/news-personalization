from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from news_service.core.config import get_settings

REDDIT_USER_AGENT = "Mozilla/5.0 (compatible; news-digest-bot/1.0; +https://example.com/bot)"

SUBREDDIT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{1,20}$")
SUBREDDIT_MENTION_PATTERN = re.compile(
    r"(?<![\w/])/?r/([A-Za-z0-9][A-Za-z0-9_]{1,20})\b",
    re.IGNORECASE,
)
SUBREDDIT_URL_PATTERN = re.compile(
    r"https?://(?:www\.|old\.)?reddit\.com/r/[A-Za-z0-9_]{2,21}(?:/[^\s?#]*)?",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(slots=True)
class RedditPost:
    url: str
    title: str
    body: str
    published_at: datetime | None
    external_url: str | None = None


def extract_reddit_subreddits(prompt: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for match in SUBREDDIT_URL_PATTERN.finditer(prompt):
        subreddit = extract_reddit_subreddit_from_url(match.group(0))
        if subreddit is not None:
            matches.append((match.start(), subreddit))

    for match in SUBREDDIT_MENTION_PATTERN.finditer(prompt):
        matches.append((match.start(), match.group(1).lower()))

    subreddits: list[str] = []
    seen: set[str] = set()
    for _, subreddit in sorted(matches):
        if subreddit in seen:
            continue
        seen.add(subreddit)
        subreddits.append(subreddit)

    return subreddits


def normalize_reddit_subreddit(value: str) -> str:
    candidate = value.strip().rstrip("/")
    extracted = extract_reddit_subreddit_from_url(candidate)
    if extracted is not None:
        return extracted

    normalized = candidate.lstrip("/")
    if normalized.lower().startswith("r/"):
        normalized = normalized[2:]

    if not SUBREDDIT_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid Reddit subreddit identifier: {value}")

    return normalized.lower()


def build_reddit_subreddit_url(subreddit: str) -> str:
    normalized = normalize_reddit_subreddit(subreddit)
    return f"https://www.reddit.com/r/{normalized}/new/"


def extract_reddit_subreddit_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"reddit.com", "www.reddit.com", "old.reddit.com"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0].lower() != "r":
        return None

    try:
        return normalize_reddit_subreddit(parts[1])
    except ValueError:
        return None


async def fetch_reddit_posts(
    subreddit: str,
    timeout_seconds: float | None = None,
    limit: int | None = None,
) -> list[RedditPost]:
    normalized_subreddit = normalize_reddit_subreddit(subreddit)
    request_timeout = timeout_seconds or settings.reddit_fetch_timeout_seconds
    listing_limit = limit or settings.reddit_listing_limit
    url = (
        f"https://www.reddit.com/r/{normalized_subreddit}/new/.json"
        f"?raw_json=1&limit={listing_limit}"
    )
    client_kwargs: dict[str, object] = {
        "timeout": request_timeout,
        "headers": {"User-Agent": REDDIT_USER_AGENT},
        "follow_redirects": True,
    }
    if settings.proxy_url:
        client_kwargs["proxy"] = settings.proxy_url

    last_error: Exception | None = None
    for attempt in range(1, settings.reddit_fetch_attempts + 1):
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(url)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Reddit returned status {response.status_code} for r/{normalized_subreddit}"
                )
            return parse_reddit_posts(response.json())
        except Exception as exc:
            last_error = exc
            if attempt == settings.reddit_fetch_attempts:
                break
            logger.warning(
                "Reddit fetch attempt %d/%d failed for r/%s; retrying",
                attempt,
                settings.reddit_fetch_attempts,
                normalized_subreddit,
            )

    if last_error is None:
        raise RuntimeError(f"Reddit fetch failed without an explicit error for r/{subreddit}")
    raise last_error


def parse_reddit_posts(payload: object) -> list[RedditPost]:
    if not isinstance(payload, dict):
        return []

    listing_data = payload.get("data")
    if not isinstance(listing_data, dict):
        return []

    children = listing_data.get("children")
    if not isinstance(children, list):
        return []

    posts: list[RedditPost] = []
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t3":
            continue

        post_data = child.get("data")
        if not isinstance(post_data, dict):
            continue

        title = str(post_data.get("title") or "").strip()
        permalink = post_data.get("permalink")
        if not title or not isinstance(permalink, str) or not permalink:
            continue

        created_utc = post_data.get("created_utc")
        published_at = None
        if isinstance(created_utc, int | float):
            published_at = datetime.fromtimestamp(created_utc, tz=UTC)

        is_self = bool(post_data.get("is_self"))
        external = None
        if not is_self:
            raw_external = post_data.get("url")
            if isinstance(raw_external, str) and raw_external.strip():
                external = raw_external.strip()

        posts.append(
            RedditPost(
                url=f"https://www.reddit.com{permalink}",
                title=title,
                body=str(post_data.get("selftext") or "").strip(),
                published_at=published_at,
                external_url=external,
            )
        )

    return posts
