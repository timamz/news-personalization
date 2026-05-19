"""Small pure-Python helpers used across the conversational package."""

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.models.subscription import Subscription

_CONVERSATION_SUMMARY_BYTE_LIMIT = 2048
_SUBSCRIPTION_PREVIEW_CHARS = 80


def _spec_preview(spec: str) -> str:
    """Return a one-line preview of a user_spec for display summaries.

    Picks the first non-empty line that is NOT a markdown heading (i.e.
    does not start with #). The full spec is fetched on demand via
    get_subscriptions; this is only a hint the agent uses to disambiguate
    which subscription the user means.
    """
    for raw_line in (spec or "").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line[:_SUBSCRIPTION_PREVIEW_CHARS]
    return ""


def _parse_csv_identifiers(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _clean_identifier(identifier: str) -> str:
    cleaned = identifier.strip().lstrip("@").lstrip("#")
    if cleaned.startswith("r/"):
        cleaned = cleaned[2:]
    return cleaned


def _source_display_name(url: str, source_kind: str) -> str:
    """Extract a user-friendly name from a source URL (for status messages)."""
    if source_kind == "telegram_channel":
        name = url.rstrip("/").split("/")[-1]
        return f"@{name}"
    if source_kind == "reddit_subreddit":
        parts = url.rstrip("/").split("/")
        for i, p in enumerate(parts):
            if p == "r" and i + 1 < len(parts):
                return f"r/{parts[i + 1]}"
        return url
    return url


def _append_conversation_summary(existing: str, fact: str) -> str:
    """Dedup by content hash and cap at ~2KB. Evicts oldest lines when over.

    Entries are prefixed with ISO timestamp so boundary flush + remember can
    coexist peacefully in one text field.
    """
    fact = fact.strip()
    if not fact:
        return existing
    lines = [line for line in existing.split("\n") if line.strip()]
    fact_hash = hashlib.sha1(fact.lower().encode("utf-8")).hexdigest()[:8]
    tagged = f"{datetime.now(UTC).strftime('%Y-%m-%d')} [{fact_hash}] {fact}"
    lines = [line for line in lines if f"[{fact_hash}]" not in line]
    lines.append(tagged)
    serialized = "\n".join(lines)
    while len(serialized.encode("utf-8")) > _CONVERSATION_SUMMARY_BYTE_LIMIT and len(lines) > 1:
        lines.pop(0)
        serialized = "\n".join(lines)
    return serialized


async def _load_subscription_summaries(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> list[str]:
    """Fetch compact one-line descriptions of the user's subscriptions.

    Returns running AND stopped (paused) subscriptions; soft-deleted
    rows are excluded. Stopped subscriptions are tagged ``[STOPPED]``
    so the LLM knows their state and can offer to resume them. An
    empty list signals a first-time interaction. The agent calls
    get_subscriptions when it needs the full spec of a specific one.
    """
    result = await session.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.is_active.is_(True),
        )
    )
    subs = list(result.scalars().all())
    lines: list[str] = []
    for sub in subs:
        schedule = sub.schedule_cron or (
            "event mode" if sub.delivery_mode == "event" else "on demand"
        )
        state = "[STOPPED] " if sub.paused_at is not None else ""
        header = f"[{sub.id}] {state}{sub.delivery_mode} | {schedule} | {sub.digest_language}"
        spec = (sub.user_spec or "").strip()
        lines.append(f"{header}\n{spec}" if spec else header)
    return lines


def _status_for_tool_call(event: dict[str, Any]) -> dict[str, Any] | None:
    """Map an ADK tool_call event to a status message for the UI, or None."""
    tool_name = event.get("name", "")
    args = event.get("args", {})
    if tool_name == "add_source":
        return {
            "event": "status",
            "status_key": "status_adding_source",
            "source": args.get("identifier", ""),
            "source_kind": args.get("source_kind", ""),
        }
    if tool_name == "remove_source":
        return {
            "event": "status",
            "status_key": "status_removing_source",
            "source": args.get("identifier", ""),
            "source_kind": args.get("source_kind", ""),
        }
    if tool_name == "set_user_timezone":
        return {
            "event": "status",
            "status_key": "status_resolving_timezone",
            "query": args.get("query", ""),
        }
    if tool_name in {"create_subscription", "update_subscription"}:
        return {
            "event": "status",
            "status_key": "status_saving_subscription",
            "subscription_id": args.get("subscription_id", ""),
        }
    if tool_name == "trigger_digest_now":
        return {
            "event": "status",
            "status_key": "status_queuing_digest",
            "subscription_id": args.get("subscription_id", ""),
        }
    if tool_name == "trigger_source_discovery":
        return {
            "event": "status",
            "status_key": "status_queuing_discovery",
            "subscription_id": args.get("subscription_id", ""),
        }
    return None
