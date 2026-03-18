import logging
from datetime import datetime

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry
from news_service.core.openai_client import openai_client

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

MAX_EVENT_TEXT_CHARS = 4000

ASSESS_AND_COMPOSE_PROMPT = """\
You assess whether a news item describes an upcoming real-world event that is relevant to a
user's subscription and, if so, compose a notification message.

Step 1 — Event detection:
Return is_relevant_event=false unless the text clearly announces something expected to happen
in the future, such as:
- a new episode, season, film, album, or product release
- a concert, tour date, festival, conference, or livestream
- a launch, premiere, match, or scheduled public appearance
Ignore general news without a concrete future event.
Ignore events that have already happened.

Step 2 — Relevance matching:
The original subscription request is the source of truth.
Return is_relevant_event=false if the event does not match what the user asked for.
Respect exclusions like "only", "not", "except", and exact-person requirements.
Match by meaning, not by exact wording.

Step 3 — Deduplication:
Compare the new candidate event against the recent notification history provided.
If the user has already been notified about substantially the same event (including reposts,
reminders, and differently worded announcements of the same underlying event), return
is_relevant_event=false.
The same event announced by a different source can still count as already notified.
If the new event is a different occurrence, date, speaker, episode, release, or otherwise
materially new, it is NOT a duplicate.

Step 4 — Composition (only if is_relevant_event=true):
Compose notification_body in the target language. Include:
- A concise event title
- Timing if known
- 1-2 sentence summary
- The source URL
Do not include source labels like "Telegram @channel" before the link.
Keep it suitable for a chat message.

Always fill in the reason field with a short explanation for your decision.
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
        ..., description="Whether this item is a relevant upcoming event for notification"
    )
    title: str | None = Field(default=None, description="Short event title")
    summary: str | None = Field(default=None, description="Short description of the event")
    when: str | None = Field(default=None, description="Human-readable timing")
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


def _trim_text(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:MAX_EVENT_TEXT_CHARS]


@with_llm_retry()
async def assess_and_compose_event_notification(
    *,
    headline: str,
    body: str,
    published_at: datetime | None,
    raw_prompt: str,
    target_language: str,
    recent_notification_history: list[str],
    max_history_chars: int,
) -> EventAssessmentResult:
    history_text = "\n\n".join(
        f"Notification {index}:\n{_trim_text(entry)}"
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
                    "Reference timestamp: "
                    f"{published_at.isoformat() if published_at else 'unknown'}\n\n"
                    f"Original subscription request:\n{_trim_text(raw_prompt)}\n\n"
                    f"Headline:\n{_trim_text(headline)}\n\n"
                    f"Body:\n{_trim_text(body)}\n\n"
                    "Recent notification history:\n"
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
    event_matching_mode: str,
    lookback_days: int,
    candidate_events: list[str],
    recent_notifications: list[str],
) -> RecentEventsPreviewDecision:
    normalized_language = target_language.strip().lower().split("-", maxsplit=1)[0]
    if normalized_language not in {"en", "ru"}:
        normalized_language = "en"

    history_block = (
        "\n\n".join(
            f"Notification {index}:\n{_trim_text(entry)}"
            for index, entry in enumerate(recent_notifications, start=1)
        )
        if recent_notifications
        else "No recent notification history."
    )
    candidates_block = "\n\n".join(
        f"Candidate event {index}:\n{_trim_text(summary)}"
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
                    f"Event matching mode: {event_matching_mode}\n"
                    f"Lookback window: last {lookback_days} days\n\n"
                    f"Original subscription request:\n{_trim_text(raw_prompt)}\n\n"
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
