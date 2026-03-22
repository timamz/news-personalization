import logging
from datetime import datetime

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry
from news_service.core.openai_client import openai_client

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

ASSESS_AND_COMPOSE_PROMPT = """\
You decide whether to deliver a post to the user based on their subscription request.

The subscription request defines exactly what the user wants. Only deliver posts that \
directly match the request. Being in the same general topic or category is NOT enough — \
the post must be about something the user specifically asked for.

If the user listed specific titles, names, people, or entities, the post must be about \
one of those. For example, if the user asked for "One Piece, Jujutsu Kaisen", a post \
about Naruto is NOT a match.

If the user has already been notified about the same thing (see notification history), \
return is_relevant_event=false.

If is_relevant_event=true, compose a short notification_body in the target language: \
a concise title, timing if known, 1-2 sentence summary, and the source URL. \
Keep it suitable for a chat message.

Always fill in the reason field explaining your decision.
"""

RECENT_EVENTS_PREVIEW_PROMPT = """\
You select the relevant missed events for a user and write a single preview message.

Rules:
- The original subscription request is the source of truth.
- Candidate events may include duplicates, reminders, or irrelevant events.
- Select only events that genuinely match the request.
- Exclude events that are already covered by recent notification history.
- If multiple candidate events describe the same underlying event, keep only one.
- Return selected_item_ids in the same order the events should appear in the message.
- Write both subject and body fully in the target language.
- Keep the subject short and useful.
- Make the body suitable for a chat message: one short intro sentence, then a compact bullet list.
- For each selected event, keep only the most relevant details:
  title, timing if known, why it matters, and URL.
- Keep URLs exactly as provided.
- Do not mention Telegram channel names or source labels before the link.
- Do not add facts that are not present in the input.
- Do not repeat the subject as the first line of the body.
- If no candidate events should be shown, return an empty selected_item_ids list
  and empty subject/body.
"""


class EventAssessmentResult(BaseModel):
    is_relevant_event: bool = Field(
        ..., description="Whether this post matches the subscription request"
    )
    notification_body: str = Field(
        default="", description="Formatted notification text ready to send"
    )
    reason: str = Field(..., min_length=3, description="Short explanation for the decision")


class RecentEventsPreviewDecision(BaseModel):
    selected_item_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the candidate events that should be shown",
    )
    subject: str = Field(default="", description="Short preview subject")
    body: str = Field(default="", description="Single body covering all missed events")


@with_llm_retry()
async def assess_and_compose_event_notification(
    *,
    headline: str,
    body: str,
    url: str,
    published_at: datetime | None,
    raw_prompt: str,
    target_language: str,
    recent_notification_history: list[str],
    max_history_chars: int,
) -> EventAssessmentResult:
    history_text = "\n\n".join(
        f"Notification {index}:\n{entry}"
        for index, entry in enumerate(recent_notification_history, start=1)
    )
    if len(history_text) > max_history_chars:
        history_text = history_text[:max_history_chars] + "\n... (truncated)"

    history_block = history_text if history_text else "No recent notification history."

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": ASSESS_AND_COMPOSE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target language: {target_language}\n\n"
                    f"Subscription request:\n{raw_prompt}\n\n"
                    f"Post headline:\n{headline}\n\n"
                    f"Post body:\n{body}\n\n"
                    f"Post URL:\n{url}\n\n"
                    "Notification history:\n"
                    f"{history_block}"
                ),
            },
        ],
        response_format=EventAssessmentResult,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for event assessment")

    logger.info(
        "Event assessment: is_relevant=%s reason=%s",
        result.is_relevant_event,
        result.reason,
    )
    return result


@with_llm_retry()
async def render_recent_events_preview(
    *,
    raw_prompt: str,
    target_language: str,
    lookback_days: int,
    candidate_events: list[str],
    recent_notifications: list[str],
) -> RecentEventsPreviewDecision:
    normalized_language = target_language.strip().lower().split("-", maxsplit=1)[0]
    if normalized_language not in {"en", "ru"}:
        normalized_language = "en"

    history_block = (
        "\n\n".join(
            f"Notification {index}:\n{entry}"
            for index, entry in enumerate(recent_notifications, start=1)
        )
        if recent_notifications
        else "No recent notification history."
    )
    candidates_block = "\n\n".join(
        f"Candidate event {index}:\n{summary}"
        for index, summary in enumerate(candidate_events, start=1)
    )
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": RECENT_EVENTS_PREVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target language: {normalized_language}\n"
                    f"Lookback window: last {lookback_days} days\n\n"
                    f"Original subscription request:\n{raw_prompt}\n\n"
                    "Recent notification history:\n"
                    f"{history_block}\n\n"
                    "Candidate events:\n"
                    f"{candidates_block}"
                ),
            },
        ],
        response_format=RecentEventsPreviewDecision,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for recent events preview")
    return result
