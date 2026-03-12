import re
from urllib.parse import urlparse

CHANNEL_MENTION_PATTERN = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_]{4,31})\b")
CHANNEL_URL_PATTERN = re.compile(r"https?://(?:www\.)?t\.me/(?:s/)?([A-Za-z][A-Za-z0-9_]{4,31})\b")
SUBREDDIT_MENTION_PATTERN = re.compile(
    r"(?<![\w/])/?r/([A-Za-z0-9][A-Za-z0-9_]{1,20})\b",
    re.IGNORECASE,
)
SUBREDDIT_URL_PATTERN = re.compile(
    r"https?://(?:www\.|old\.)?reddit\.com/r/([A-Za-z0-9][A-Za-z0-9_]{1,20})\b",
    re.IGNORECASE,
)


def extract_telegram_channels(text: str) -> list[str]:
    channels: list[str] = []
    seen: set[str] = set()

    for pattern in (CHANNEL_MENTION_PATTERN, CHANNEL_URL_PATTERN):
        for match in pattern.finditer(text):
            channel = match.group(1).lower()
            if channel in seen:
                continue
            seen.add(channel)
            channels.append(channel)

    return channels


def parse_telegram_channel_tokens(text: str) -> list[str]:
    channels: list[str] = []
    seen: set[str] = set()

    for token in text.replace(",", " ").split():
        candidate = token.strip()
        if not candidate:
            continue

        if candidate.startswith("@"):
            candidate = candidate[1:]
        elif candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlparse(candidate)
            parts = [part for part in parsed.path.split("/") if part]
            if not parts:
                continue
            candidate = parts[1] if parts[0] == "s" and len(parts) >= 2 else parts[0]

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{4,31}", candidate) is None:
            continue

        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        channels.append(normalized)

    return channels


def extract_reddit_subreddits(text: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for pattern in (SUBREDDIT_URL_PATTERN, SUBREDDIT_MENTION_PATTERN):
        for match in pattern.finditer(text):
            matches.append((match.start(), match.group(1).lower()))

    subreddits: list[str] = []
    seen: set[str] = set()
    for _, subreddit in sorted(matches):
        if subreddit in seen:
            continue
        seen.add(subreddit)
        subreddits.append(subreddit)

    return subreddits


def parse_source_tokens(text: str) -> tuple[list[str], list[str]]:
    telegram_channels = parse_telegram_channel_tokens(text)
    reddit_subreddits = extract_reddit_subreddits(text)
    if reddit_subreddits:
        return telegram_channels, reddit_subreddits

    subreddits: list[str] = []
    seen: set[str] = set()

    for token in text.replace(",", " ").split():
        candidate = token.strip().rstrip("/")
        if not candidate:
            continue

        if candidate.lower().startswith("/r/"):
            candidate = candidate[3:]
        elif candidate.lower().startswith("r/"):
            candidate = candidate[2:]
        elif candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlparse(candidate)
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0].lower() == "r":
                candidate = parts[1]
            else:
                continue

        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{1,20}", candidate) is None:
            continue

        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        subreddits.append(normalized)

    return telegram_channels, subreddits
