from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

CHANNEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{4,31}$")
CHANNEL_MENTION_PATTERN = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_]{4,31})\b")


@dataclass(slots=True)
class TelegramPost:
    url: str
    body: str
    published_at: datetime | None


def extract_telegram_channels(prompt: str) -> list[str]:
    channels: list[str] = []
    seen: set[str] = set()
    for match in CHANNEL_MENTION_PATTERN.finditer(prompt):
        channel = match.group(1).lower()
        if channel in seen:
            continue
        seen.add(channel)
        channels.append(channel)
    return channels


def normalize_telegram_channel(value: str) -> str:
    candidate = value.strip()
    if candidate.startswith("@"):
        candidate = candidate[1:]
    if not CHANNEL_PATTERN.fullmatch(candidate):
        raise ValueError(f"Invalid Telegram channel identifier: {value}")
    return candidate.lower()


def build_telegram_channel_url(channel: str) -> str:
    normalized = normalize_telegram_channel(channel)
    return f"https://t.me/s/{normalized}"


def extract_telegram_channel_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"t.me", "www.t.me"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None

    handle = parts[1] if parts[0] == "s" and len(parts) >= 2 else parts[0]

    try:
        return normalize_telegram_channel(handle)
    except ValueError:
        return None


async def fetch_telegram_posts(channel: str, timeout_seconds: float = 10.0) -> list[TelegramPost]:
    channel_url = build_telegram_channel_url(channel)
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        response = await client.get(channel_url)
        response.raise_for_status()
    return parse_telegram_posts(response.text)


def parse_telegram_posts(html_text: str) -> list[TelegramPost]:
    soup = BeautifulSoup(html_text, "html.parser")
    wrappers = soup.select("div.tgme_widget_message_wrap")

    posts: list[TelegramPost] = []
    for wrapper in wrappers:
        date_link = wrapper.select_one("div.tgme_widget_message_info a.tgme_widget_message_date")
        body_node = wrapper.select_one("div.tgme_widget_message_text")
        if date_link is None or body_node is None:
            continue

        post_url = date_link.get("href")
        if post_url is None:
            continue

        body = body_node.get_text("\n", strip=True)
        if not body:
            continue

        datetime_str = None
        time_node = date_link.select_one("time")
        if time_node is not None:
            datetime_str = time_node.get("datetime")

        published_at = None
        if datetime_str:
            try:
                published_at = datetime.fromisoformat(datetime_str)
            except ValueError:
                published_at = None

        posts.append(TelegramPost(url=post_url, body=body, published_at=published_at))

    return posts
