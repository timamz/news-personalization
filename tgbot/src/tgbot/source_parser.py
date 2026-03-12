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
TWITTER_URL_PATTERN = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:x\.com|twitter\.com)/([A-Za-z0-9_]{1,15})\b",
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
            if parsed.netloc.lower() not in {"t.me", "www.t.me"}:
                continue
            parts = [part for part in parsed.path.split("/") if part]
            if not parts:
                continue
            candidate = parts[1] if parts[0] == "s" and len(parts) >= 2 else parts[0]
        else:
            lowered = candidate.lower()
            if lowered.startswith("t.me/") or lowered.startswith("www.t.me/"):
                path = candidate.split("/", maxsplit=1)[1]
                parts = [part for part in path.split("/") if part]
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


def extract_twitter_accounts(text: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for match in TWITTER_URL_PATTERN.finditer(text):
        matches.append((match.start(1), match.group(1).lower()))
    for pattern in (TWITTER_CONTEXT_BEFORE_PATTERN, TWITTER_CONTEXT_AFTER_PATTERN):
        for match in pattern.finditer(text):
            matches.append((match.start(1), match.group(1).lower()))

    accounts: list[str] = []
    seen: set[str] = set()
    for _, account in sorted(matches):
        if account in seen:
            continue
        seen.add(account)
        accounts.append(account)
    return accounts


def parse_twitter_account_tokens(text: str) -> list[str]:
    accounts: list[str] = []
    seen: set[str] = set()

    for token in text.replace(",", " ").split():
        candidate = token.strip().rstrip("/")
        if not candidate:
            continue

        if candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlparse(candidate)
            host = parsed.netloc.lower()
            if host not in {
                "x.com",
                "www.x.com",
                "mobile.x.com",
                "twitter.com",
                "www.twitter.com",
                "mobile.twitter.com",
            }:
                continue
            parts = [part for part in parsed.path.split("/") if part]
            if not parts:
                continue
            candidate = parts[0]
        else:
            lowered = candidate.lower()
            if lowered.startswith(("x.com/", "www.x.com/", "twitter.com/", "www.twitter.com/")):
                path = candidate.split("/", maxsplit=1)[1]
                parts = [part for part in path.split("/") if part]
                if not parts:
                    continue
                candidate = parts[0]
            else:
                prefix = None
                for possible_prefix in ("x:", "x/", "twitter:", "twitter/"):
                    if lowered.startswith(possible_prefix):
                        prefix = possible_prefix
                        break
                if prefix is None:
                    continue
                candidate = candidate[len(prefix) :]

        if re.fullmatch(r"[A-Za-z0-9_]{1,15}", candidate) is None:
            continue

        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        accounts.append(normalized)

    return accounts


def parse_source_tokens(text: str) -> tuple[list[str], list[str], list[str]]:
    telegram_channels = parse_telegram_channel_tokens(text)
    reddit_subreddits = extract_reddit_subreddits(text)
    twitter_accounts = extract_twitter_accounts(text)
    if reddit_subreddits or twitter_accounts:
        return telegram_channels, reddit_subreddits, twitter_accounts

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

    return telegram_channels, subreddits, parse_twitter_account_tokens(text)
