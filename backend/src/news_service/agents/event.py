import logging
from datetime import datetime

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
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
- Resolve relative dates (for example "tomorrow")
  against the provided reference timestamp when possible.
"""


class UpcomingEventCandidate(BaseModel):
    is_upcoming_event: bool = Field(..., description="Whether the item announces a future event")
    title: str | None = Field(default=None, description="Short event title")
    summary: str | None = Field(default=None, description="Short description of the event")
    starts_at: datetime | None = Field(
        default=None,
        description="When the event is expected to happen",
    )


def _trim_text(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:MAX_EVENT_TEXT_CHARS]


async def extract_upcoming_event(
    headline: str,
    body: str,
    published_at: datetime | None,
) -> UpcomingEventCandidate | None:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
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
