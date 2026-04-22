"""Event Verifier -- weekly outcome-check reflector for event subscriptions.

Mirrors the Digest Pipeline Reflector in agency and tool shape. Where the
digest reflector inspects the subscription's source pool against digest
quality signals, the event verifier checks the outside world against the
subscription's recent notification history: did anything matching the
user_spec actually happen recently, and did we notify the user?

The agent runs as an ADK ReAct loop with full autonomy: it decides when to
inspect a source, when to declare a miss, and when to queue source
discovery. The surrounding task reads the shared_state the agent's tools
wrote into and translates it into side effects (webhook deliveries,
discovery queue, status logs).

Tools:
- web_search: targeted web queries against the user's topic
- fetch_source_items: inspect a linked source's recent items to distinguish
  "source did not cover it" from "assessor missed it"
- trigger_source_discovery: queue a discovery run with an agent-authored
  reason (including full event-sub context: spec excerpt, what was missed,
  which source should have covered it)
- emit_missed_event: record a confirmed missed event for catch-up delivery
- emit_status: free-text progress log (non-user-visible in this context)
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.adk_runner import run_agent_text
from news_service.agents.tools.source_inspection import build_fetch_source_items_tool
from news_service.core.config import get_settings
from news_service.models.subscription import Subscription
from news_service.services.search import search_web

logger = logging.getLogger(__name__)
settings = get_settings()


VERIFIER_PROMPT = """\
# Context

This is a personalized news-notification service. The user maintains an \
EVENT subscription: they want a push notification the moment a specific \
kind of event happens (for example: an official anime episode release \
date announcement, a court ruling, a product launch). The user authored \
a freeform spec describing exactly what counts and what does not.

Every 30 minutes the backend polls a small pool of RSS feeds, Telegram \
channels, and subreddits linked to the subscription, embeds new items, \
and runs a Batch Event Assessor. The assessor decides per-item whether \
to deliver a notification. The assessor can fail silently: it may \
under-flag a genuine announcement, or the linked sources may not have \
covered the announcement at all.

You run WEEKLY as a safety net. Your job is to check whether anything \
matching the spec actually happened in the last few days, and if so, \
whether the user was notified. If something slipped through, you act.

# Your role

You are the Event Verifier. You do not rewrite the assessor and you do \
not touch the user_spec. You investigate the outside world via targeted \
web searches, compare against the recent notification history, and \
decide per finding:

1. Event happened AND is in notification history -> healthy, move on.
2. Event happened AND is NOT in history -> this is a miss. Before \
declaring it, call fetch_source_items on each linked source to check \
whether that source actually covered the event (and the assessor missed \
it) or whether no source covered it (and you need better sources). Then:
   - Call emit_missed_event so the task can deliver a catch-up \
notification to the user.
   - If the miss was caused by source coverage gap (no linked source \
covered this), call trigger_source_discovery with a reason that includes \
WHAT was missed and WHICH source should have covered it -- be specific, \
the discovery agent reads this verbatim.
   - A single confirmed miss caused by coverage gap is enough to trigger \
discovery. Do not wait for a quota.
3. Event did not happen (or no evidence) -> do nothing.

# Inputs

You are given:
- The user_spec (freeform markdown describing exactly what the user \
wants to be notified about).
- Recent notification_history: the titles, summaries, sources, and \
timestamps of notifications delivered to the user in the last \
lookback window.
- Linked sources: one line per source with URL, type (user-specified or \
auto-discovered), last published item timestamp, and item count in the \
lookback window.
- Lookback window in days.

# Tools

1. web_search(query) -- Issue a targeted web search. Keep queries \
specific (include entity names, "official announcement", date ranges). \
You have a soft budget of a few searches -- do not burn them on \
duplicates or vague wording.
2. fetch_source_items(source_id, since_days_ago, limit) -- Read recent \
items from a specific linked source. Call this after finding a \
candidate miss to determine whether the source covered the event or \
not. Do not skip this step -- it is how you distinguish an assessor \
failure from a coverage gap.
3. trigger_source_discovery(reason) -- Queue a discovery run. Use only \
when the miss was caused by coverage gap. Reason string must include: \
the user_spec (brief), what was missed (title + source URL), and which \
linked source should have covered it.
4. emit_missed_event(title, summary, source_url, happened_at) -- \
Record a confirmed miss for catch-up delivery. Only call this for \
misses you have verified via fetch_source_items AND for events that \
have clear authoritative source URLs.
5. emit_status(message) -- Short progress note for logs.

# Guardrails

- Only count OFFICIAL announcements. Rumors, leaks, fan speculation, \
clickbait aggregators -> ignore.
- Only count events within the lookback window.
- If history already contains anything that plausibly matches a search \
hit (same entity, same event type, close timestamp), it is NOT a miss.
- Never emit markdown bold (**...**) in any text you produce. Plain \
text only. The frontend renders it verbatim.
- If you find no misses, just return a short final text summarizing \
what you searched for and that nothing was missed. Do not invent \
problems.
"""


@dataclass(slots=True)
class VerifierSourceContext:
    """Minimal per-source context the verifier reads before searching."""

    source_id: uuid.UUID
    url: str
    title: str
    is_user_specified: bool
    last_published_at: datetime | None
    items_in_window: int


@dataclass(slots=True)
class MissedEvent:
    """A confirmed missed event the task should catch-up-deliver to the user."""

    title: str
    summary: str
    source_url: str
    happened_at: str


def _format_source_contexts(contexts: list[VerifierSourceContext]) -> str:
    if not contexts:
        return "(no sources linked)"
    lines: list[str] = []
    for ctx in contexts:
        label = "user-specified" if ctx.is_user_specified else "auto-discovered"
        last = (
            f"last item: {ctx.last_published_at.isoformat()}"
            if ctx.last_published_at is not None
            else "last item: never"
        )
        display = f"{ctx.title} ({ctx.url})" if ctx.title else ctx.url
        lines.append(
            f"- [source_id={ctx.source_id}] {display} [{label}] | "
            f"items in window: {ctx.items_in_window} | {last}"
        )
    return "\n".join(lines)


def _format_history(history_strings: list[str]) -> str:
    if not history_strings:
        return "(no recent notifications)"
    return "\n\n".join(f"Notification {i + 1}:\n{entry}" for i, entry in enumerate(history_strings))


async def run_event_verifier(
    *,
    db_session: AsyncSession,
    subscription: Subscription,
    user_spec: str,
    history_strings: list[str],
    source_contexts: list[VerifierSourceContext],
    lookback_days: int,
) -> dict[str, Any]:
    """Run the Event Verifier ADK agent.

    Returns shared state:
        missed_events: list[MissedEvent]
        discovery_reasons: list[str]
        status_messages: list[str]
        observations: str  -- agent's final text response
    """
    shared_state: dict[str, Any] = {
        "missed_events": [],
        "discovery_reasons": [],
        "status_messages": [],
        "search_budget_used": 0,
    }
    allowed_source_ids = {ctx.source_id for ctx in source_contexts}
    # Shared lock for DB-touching tools -- ADK may dispatch parallel
    # fetch_source_items_tool calls in a single turn, and asyncpg does
    # not allow overlapping operations on one connection.
    session_lock = asyncio.Lock()

    async def web_search_tool(query: str) -> str:
        """Search the web via Yandex. Returns formatted results (title, URL, snippet)."""
        shared_state["search_budget_used"] += 1
        if shared_state["search_budget_used"] > settings.event_verifier_max_searches:
            return (
                f"Search budget exhausted ({settings.event_verifier_max_searches} calls). "
                "Decide with what you have."
            )
        return await search_web(query)

    fetch_source_items_tool = build_fetch_source_items_tool(
        db_session=db_session,
        allowed_source_ids=allowed_source_ids,
        topic_embedding=None,
        name="fetch_source_items_tool",
        session_lock=session_lock,
    )

    async def trigger_source_discovery_tool(reason: str) -> str:
        """Queue a source discovery run for this subscription."""
        reason = (reason or "").strip()
        if not reason:
            return "Refusing to trigger discovery without a reason."
        shared_state["discovery_reasons"].append(reason)
        return f"Source discovery queued: {reason}"

    async def emit_missed_event_tool(
        title: str,
        summary: str,
        source_url: str,
        happened_at: str,
    ) -> str:
        """Record a confirmed missed event for catch-up delivery."""
        title = (title or "").strip()
        source_url = (source_url or "").strip()
        if not title or not source_url:
            return "Refusing to record miss without title and source_url."
        shared_state["missed_events"].append(
            MissedEvent(
                title=title,
                summary=(summary or "").strip(),
                source_url=source_url,
                happened_at=(happened_at or "").strip() or "unknown",
            )
        )
        return f"Missed event recorded: {title}"

    async def emit_status_tool(message: str) -> str:
        """Log a short progress status."""
        message = (message or "").strip()
        if message:
            shared_state["status_messages"].append(message)
        return "Status recorded."

    input_message = (
        f"Subscription id: {subscription.id}\n"
        f"Target language: {subscription.digest_language}\n"
        f"Lookback window: {lookback_days} days\n\n"
        f"User spec:\n{user_spec}\n\n"
        f"Linked sources:\n{_format_source_contexts(source_contexts)}\n\n"
        f"Recent notification history:\n{_format_history(history_strings)}\n\n"
        f"Search budget: up to {settings.event_verifier_max_searches} web_search calls."
    )

    agent = Agent(
        name=f"event_verifier_{uuid.uuid4().hex[:6]}",
        model=LiteLlm(model=settings.litellm_model),
        instruction=VERIFIER_PROMPT,
        tools=[
            web_search_tool,
            fetch_source_items_tool,
            trigger_source_discovery_tool,
            emit_missed_event_tool,
            emit_status_tool,
        ],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    response = await run_agent_text(
        agent=agent,
        message=input_message,
        user_id=str(subscription.user_id),
    )
    shared_state["observations"] = response
    logger.info(
        "Event verifier completed for subscription %s: misses=%d discovery_queued=%d searches=%d",
        subscription.id,
        len(shared_state["missed_events"]),
        len(shared_state["discovery_reasons"]),
        shared_state["search_budget_used"],
    )
    return shared_state
