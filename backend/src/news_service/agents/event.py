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

SYSTEM_PROMPT = """\
You detect upcoming real-world events mentioned in news items.

Return is_upcoming_event=true only when the text clearly announces something that is expected
to happen in the future, such as:
- a new episode, season, film, album, or product release
- a concert, tour date, festival, conference, or livestream
- a launch, premiere, match, or scheduled public appearance

Rules:
- Ignore general news without a concrete future event.
- Ignore events that have already happened.
- If the text says the event is upcoming but does not provide an exact date, keep starts_at null.
- Keep title concise and specific.
- Keep summary to 1-2 short sentences.
- Write title and summary in the requested target language.
- Resolve relative dates (for example "tomorrow")
  against the provided reference timestamp when possible.
"""

EVENT_MATCH_PROMPT = """\
You decide whether a candidate upcoming event should trigger a user's event notification.

Rules:
- The original subscription request is the source of truth.
- Return matches=true only when the post clearly satisfies what the user asked for.
- Respect exclusions like "only", "not", "except", and exact-person requirements.
- Match by meaning, not by exact wording. Do not require literal words like "announcement"
  when the text is clearly announcing an event.
- If key details are missing or the post is ambiguous, return matches=false.
- Keep the reason short, concrete, and based on the text.
"""

NOTIFICATION_DUPLICATE_PROMPT = """\
You decide whether a user has already been notified about substantially the same event.

Rules:
- Compare the new candidate event against the recent notification history.
- Treat reposts, reminders, and differently worded announcements of the same underlying event
  as already notified.
- The same event announced by a different source can still count as already notified.
- If the new event is a different occurrence, date, speaker, episode, release, or otherwise
  materially new, return already_notified=false.
- If the history does not contain a substantially same notification, return already_notified=false.
- Keep the reason short, concrete, and based on the provided text.
"""

LOCALIZE_EVENT_PROMPT = """\
You rewrite a detected upcoming event into a requested output language.

Rules:
- Keep the meaning faithful to the provided event details.
- Translate both title and summary fully into the target language.
- Keep title concise and specific.
- Keep summary to 1-2 short sentences.
- Do not add facts that are not present in the provided text.
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


class UpcomingEventCandidate(BaseModel):
    is_upcoming_event: bool = Field(..., description="Whether the item announces a future event")
    title: str | None = Field(default=None, description="Short event title")
    summary: str | None = Field(default=None, description="Short description of the event")
    starts_at: datetime | None = Field(
        default=None,
        description="When the event is expected to happen",
    )


class EventMatchDecision(BaseModel):
    matches: bool = Field(..., description="Whether this event should notify the user")
    reason: str = Field(..., min_length=3, description="Short explanation for the decision")


class NotificationDuplicateDecision(BaseModel):
    already_notified: bool = Field(
        ...,
        description="Whether the user has already received a notification about the same event",
    )
    reason: str = Field(..., min_length=3, description="Short explanation for the decision")


class LocalizedEventText(BaseModel):
    title: str = Field(..., min_length=1, description="Localized event title")
    summary: str = Field(..., min_length=1, description="Localized event summary")


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
async def extract_upcoming_event(
    headline: str,
    body: str,
    published_at: datetime | None,
    *,
    target_language: str | None = None,
) -> UpcomingEventCandidate | None:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target language: {target_language or 'same as the source text'}\n\n"
                    "Reference timestamp: "
                    f"{published_at.isoformat() if published_at else 'unknown'}\n\n"
                    f"Headline:\n{_trim_text(headline)}\n\n"
                    f"Body:\n{_trim_text(body)}"
                ),
            },
        ],
        response_format=UpcomingEventCandidate,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for event extraction")
    if not result.is_upcoming_event:
        return None

    result.title = result.title or headline
    result.summary = result.summary or headline

    logger.info(
        "Detected upcoming event: title=%s starts_at=%s",
        result.title,
        result.starts_at,
    )
    return result


@with_llm_retry()
async def localize_event_text(
    *,
    headline: str,
    body: str,
    event_title: str | None,
    event_summary: str | None,
    event_starts_at: datetime | None,
    target_language: str,
) -> LocalizedEventText:
    fallback_title = event_title or headline
    fallback_summary = event_summary or headline
    normalized_language = target_language.strip().lower().split("-", maxsplit=1)[0]
    if normalized_language not in {"en", "ru"}:
        return LocalizedEventText(title=fallback_title, summary=fallback_summary)

    event_lines = [f"Target language: {normalized_language}"]
    if event_title:
        event_lines.extend(["", f"Detected event title:\n{_trim_text(event_title)}"])
    if event_summary:
        event_lines.extend(["", f"Detected event summary:\n{_trim_text(event_summary)}"])
    if event_starts_at is not None:
        event_lines.extend(["", f"Detected event start:\n{event_starts_at.isoformat()}"])
    event_lines.extend(
        [
            "",
            f"Headline:\n{_trim_text(headline)}",
            "",
            f"Body:\n{_trim_text(body)}",
        ]
    )

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": LOCALIZE_EVENT_PROMPT},
            {"role": "user", "content": "\n".join(event_lines)},
        ],
        response_format=LocalizedEventText,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for event localization")

    result.title = result.title or fallback_title
    result.summary = result.summary or fallback_summary
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


@with_llm_retry()
async def judge_event_match(
    *,
    headline: str,
    body: str,
    published_at: datetime | None,
    raw_prompt: str,
    event_title: str | None = None,
    event_summary: str | None = None,
    event_starts_at: datetime | None = None,
) -> EventMatchDecision:
    event_lines = [
        f"Reference timestamp: {published_at.isoformat() if published_at else 'unknown'}",
    ]
    if event_title:
        event_lines.extend(["", f"Detected event title:\n{_trim_text(event_title)}"])
    if event_summary:
        event_lines.extend(["", f"Detected event summary:\n{_trim_text(event_summary)}"])
    if event_starts_at is not None:
        event_lines.extend(["", f"Detected event start:\n{event_starts_at.isoformat()}"])
    event_lines.extend(
        [
            "",
            f"Headline:\n{_trim_text(headline)}",
            "",
            f"Body:\n{_trim_text(body)}",
        ]
    )
    event_block = "\n".join(event_lines)

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": EVENT_MATCH_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original subscription request:\n{_trim_text(raw_prompt)}\n\n"
                    "Candidate event:\n"
                    f"{event_block}"
                ),
            },
        ],
        response_format=EventMatchDecision,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for event match")

    logger.info("Judged event match: matches=%s reason=%s", result.matches, result.reason)
    return result


@with_llm_retry()
async def judge_notification_duplicate(
    *,
    headline: str,
    body: str,
    published_at: datetime | None,
    recent_notifications: list[str],
    event_title: str | None = None,
    event_summary: str | None = None,
    event_starts_at: datetime | None = None,
) -> NotificationDuplicateDecision:
    if not recent_notifications:
        return NotificationDuplicateDecision(
            already_notified=False,
            reason="No recent notification history to compare.",
        )

    event_lines = [
        f"Reference timestamp: {published_at.isoformat() if published_at else 'unknown'}",
    ]
    if event_title:
        event_lines.extend(["", f"Detected event title:\n{_trim_text(event_title)}"])
    if event_summary:
        event_lines.extend(["", f"Detected event summary:\n{_trim_text(event_summary)}"])
    if event_starts_at is not None:
        event_lines.extend(["", f"Detected event start:\n{event_starts_at.isoformat()}"])
    event_lines.extend(
        [
            "",
            f"Headline:\n{_trim_text(headline)}",
            "",
            f"Body:\n{_trim_text(body)}",
        ]
    )
    event_block = "\n".join(event_lines)
    history_block = "\n\n".join(
        f"Notification {index}:\n{_trim_text(entry)}"
        for index, entry in enumerate(recent_notifications, start=1)
    )

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": NOTIFICATION_DUPLICATE_PROMPT},
            {
                "role": "user",
                "content": (
                    "Recent notifications already shown to the user:\n"
                    f"{history_block}\n\n"
                    "New candidate event:\n"
                    f"{event_block}"
                ),
            },
        ],
        response_format=NotificationDuplicateDecision,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for notification duplicate check")

    logger.info(
        "Judged notification duplicate: already_notified=%s reason=%s",
        result.already_notified,
        result.reason,
    )
    return result
