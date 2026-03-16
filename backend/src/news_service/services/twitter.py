from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Final
from urllib.parse import urlparse

import httpx

from news_service.core.config import get_settings

TWITTER_ACCOUNT_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,15}$")
TWITTER_URL_PATTERN = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/[A-Za-z0-9_]{1,15}(?:/[^\s?#]*)?",
    re.IGNORECASE,
)
TWITTER_CONTEXT_BEFORE_PATTERN = re.compile(
    r"\b(?:twitter|x(?:\.com)?)\s*(?:account|handle|profile)?\s*[:\-]?\s*@?([A-Za-z0-9_]{1,15})\b",
    re.IGNORECASE,
)
TWITTER_CONTEXT_AFTER_PATTERN = re.compile(
    r"@([A-Za-z0-9_]{1,15})\b\s+(?:on|from)\s+(?:twitter|x(?:\.com)?)\b",
    re.IGNORECASE,
)
TWITTER_TOKEN_PATTERN = re.compile(
    r"(?<![\w/])(?:x|twitter)/@?([A-Za-z0-9_]{1,15})\b",
    re.IGNORECASE,
)
NEXT_DATA_PATTERN = re.compile(
    r"<script[^>]+id=[\"']__NEXT_DATA__[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
RESERVED_PATH_SEGMENTS: Final[set[str]] = {
    "compose",
    "explore",
    "home",
    "i",
    "intent",
    "login",
    "messages",
    "notifications",
    "search",
    "settings",
    "share",
    "signup",
    "tos",
}
TWITTER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(slots=True)
class TwitterPost:
    url: str
    title: str
    body: str
    published_at: datetime | None


class TwitterRateLimitError(RuntimeError):
    def __init__(self, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__("Twitter syndication endpoint is rate-limited")


def extract_twitter_accounts(prompt: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for match in TWITTER_URL_PATTERN.finditer(prompt):
        account = extract_twitter_account_from_url(match.group(0))
        if account is not None:
            matches.append((match.start(), account))

    for pattern in (
        TWITTER_CONTEXT_BEFORE_PATTERN,
        TWITTER_CONTEXT_AFTER_PATTERN,
        TWITTER_TOKEN_PATTERN,
    ):
        for match in pattern.finditer(prompt):
            matches.append((match.start(1), match.group(1).lower()))

    accounts: list[str] = []
    seen: set[str] = set()
    for _, account in sorted(matches):
        if account in seen:
            continue
        seen.add(account)
        accounts.append(account)
    return accounts


def normalize_twitter_account(value: str) -> str:
    candidate = value.strip().rstrip("/")
    extracted = extract_twitter_account_from_url(candidate)
    if extracted is not None:
        return extracted

    normalized = candidate
    lowered = normalized.lower()
    for prefix in ("twitter:", "twitter/", "x:", "x/", "twitter.com/", "x.com/"):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    if normalized.startswith("@"):
        normalized = normalized[1:]

    if not TWITTER_ACCOUNT_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid Twitter account identifier: {value}")
    return normalized.lower()


def build_twitter_account_url(account: str) -> str:
    normalized = normalize_twitter_account(account)
    return f"https://x.com/{normalized}"


def extract_twitter_account_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {
        "twitter.com",
        "www.twitter.com",
        "mobile.twitter.com",
        "x.com",
        "www.x.com",
        "mobile.x.com",
    }:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None

    account = parts[0]
    if account.lower() in RESERVED_PATH_SEGMENTS:
        return None

    try:
        return normalize_twitter_account(account)
    except ValueError:
        return None


async def fetch_twitter_posts(
    account: str,
    timeout_seconds: float | None = None,
    limit: int | None = None,
) -> list[TwitterPost]:
    normalized_account = normalize_twitter_account(account)
    request_timeout = timeout_seconds or settings.twitter_fetch_timeout_seconds
    listing_limit = limit or settings.twitter_listing_limit
    deadline = time.monotonic() + request_timeout * settings.twitter_fetch_attempts
    last_error: Exception | None = None

    async with httpx.AsyncClient(
        headers={"user-agent": TWITTER_USER_AGENT},
        follow_redirects=True,
    ) as client:
        for attempt in range(1, settings.twitter_fetch_attempts + 1):
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                break

            attempt_timeout = min(request_timeout, remaining_budget)
            try:
                html = await _request_twitter_timeline_html(
                    client,
                    normalized_account,
                    timeout_seconds=attempt_timeout,
                )
                return parse_twitter_posts(html, normalized_account, limit=listing_limit)
            except TwitterRateLimitError as exc:
                last_error = exc
                if attempt == settings.twitter_fetch_attempts:
                    break
                delay = _compute_retry_delay(
                    retry_after_seconds=exc.retry_after_seconds,
                    attempt=attempt,
                    remaining_budget=deadline - time.monotonic(),
                )
                if delay <= 0:
                    break
                logger.warning(
                    "Twitter fetch attempt %d/%d hit a rate limit for @%s; retrying in %.1fs",
                    attempt,
                    settings.twitter_fetch_attempts,
                    normalized_account,
                    delay,
                )
                await asyncio.sleep(delay)
            except (httpx.HTTPError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt == settings.twitter_fetch_attempts:
                    break
                delay = _compute_retry_delay(
                    retry_after_seconds=None,
                    attempt=attempt,
                    remaining_budget=deadline - time.monotonic(),
                )
                if delay <= 0:
                    break
                logger.warning(
                    "Twitter fetch attempt %d/%d failed for @%s; retrying in %.1fs",
                    attempt,
                    settings.twitter_fetch_attempts,
                    normalized_account,
                    delay,
                )
                await asyncio.sleep(delay)

    if last_error is None:
        raise RuntimeError(f"Twitter fetch failed without an explicit error for @{account}")
    raise last_error


def parse_twitter_posts(html: str, account: str, *, limit: int) -> list[TwitterPost]:
    match = NEXT_DATA_PATTERN.search(html)
    if match is None:
        raise ValueError(f"Twitter syndication payload missing __NEXT_DATA__ for @{account}")

    payload = json.loads(unescape(match.group(1)))
    entries = payload.get("props", {}).get("pageProps", {}).get("timeline", {}).get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Twitter timeline payload is malformed for @{account}")

    posts: list[TwitterPost] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "tweet":
            continue
        tweet = entry.get("content", {}).get("tweet")
        if not isinstance(tweet, dict):
            continue

        id_str = str(tweet.get("id_str") or "").strip()
        body = str(tweet.get("full_text") or tweet.get("text") or "").strip()
        screen_name = str(tweet.get("user", {}).get("screen_name") or account).strip().lower()
        if not id_str or not body:
            continue

        title = body.splitlines()[0][:200]
        posts.append(
            TwitterPost(
                url=f"https://x.com/{screen_name}/status/{id_str}",
                title=title or f"Post from @{screen_name}",
                body=body,
                published_at=_parse_twitter_datetime(tweet.get("created_at")),
            )
        )
        if len(posts) >= limit:
            break

    return posts


async def _request_twitter_timeline_html(
    client: httpx.AsyncClient,
    account: str,
    *,
    timeout_seconds: float,
) -> str:
    response = await client.get(
        f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{account}",
        timeout=_build_timeout(timeout_seconds),
    )
    if response.status_code == 429:
        raise TwitterRateLimitError(_extract_retry_after_seconds(response))
    response.raise_for_status()
    return response.text


def _build_timeout(timeout_seconds: float) -> httpx.Timeout:
    connect_timeout = min(5.0, timeout_seconds)
    return httpx.Timeout(
        connect=connect_timeout,
        read=timeout_seconds,
        write=connect_timeout,
        pool=connect_timeout,
    )


def _compute_retry_delay(
    *,
    retry_after_seconds: float | None,
    attempt: int,
    remaining_budget: float,
) -> float:
    fallback = settings.twitter_fetch_retry_backoff_seconds * attempt
    proposed = retry_after_seconds if retry_after_seconds is not None else fallback
    max_wait = min(
        settings.twitter_fetch_max_rate_limit_wait_seconds,
        max(0.0, remaining_budget - 0.1),
    )
    return max(0.0, min(proposed, max_wait))


def _extract_retry_after_seconds(response: httpx.Response) -> float | None:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            try:
                reset_at = parsedate_to_datetime(retry_after).timestamp()
            except (TypeError, ValueError):
                reset_at = None
            if reset_at is not None:
                return max(reset_at - time.time(), 0.0)

    rate_limit_reset = response.headers.get("x-rate-limit-reset")
    if rate_limit_reset:
        try:
            return max(float(rate_limit_reset) - time.time(), 0.0)
        except ValueError:
            return None

    return None


def _parse_twitter_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y").astimezone(UTC)
    except ValueError:
        return None
