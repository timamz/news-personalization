import logging
from datetime import datetime

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.schemas.subscription import EventConstraint

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

CONSTRAINT_MATCH_PROMPT = """\
You fill a per-subscription event validation schema for a candidate event.

Rules:
- Use exactly the provided keys.
- Respect each constraint's description and value_type.
- For string constraints, fill only string_value.
- For boolean constraints, fill only boolean_value.
- For list constraints, fill only list_value.
- Leave the other value fields empty.
- If the event does not provide enough information for a constraint, return the closest honest
  value you can infer from the text; do not guess beyond the text.
"""


class UpcomingEventCandidate(BaseModel):
    is_upcoming_event: bool = Field(..., description="Whether the item announces a future event")
    title: str | None = Field(default=None, description="Short event title")
    summary: str | None = Field(default=None, description="Short description of the event")
    starts_at: datetime | None = Field(
        default=None,
        description="When the event is expected to happen",
    )


class ParsedEventConstraintValue(BaseModel):
    key: str = Field(..., description="Constraint key being filled")
    string_value: str | None = Field(default=None, description="Used for string constraints")
    boolean_value: bool | None = Field(default=None, description="Used for boolean constraints")
    list_value: list[str] = Field(default_factory=list, description="Used for list constraints")


class ParsedEventConstraintValues(BaseModel):
    values: list[ParsedEventConstraintValue] = Field(
        default_factory=list,
        description="Constraint values extracted from the event",
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


async def parse_event_constraint_values(
    *,
    headline: str,
    body: str,
    published_at: datetime | None,
    raw_prompt: str,
    constraints: list[EventConstraint],
) -> dict[str, str | bool | list[str] | None]:
    if not constraints:
        return {}

    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CONSTRAINT_MATCH_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original subscription request:\n{_trim_text(raw_prompt)}\n\n"
                    f"Constraint schema:\n{_constraint_schema_block(constraints)}\n\n"
                    "Reference timestamp: "
                    f"{published_at.isoformat() if published_at else 'unknown'}\n\n"
                    f"Headline:\n{_trim_text(headline)}\n\n"
                    f"Body:\n{_trim_text(body)}"
                ),
            },
        ],
        response_format=ParsedEventConstraintValues,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for event constraint values")

    values_by_key = {value.key: _extract_constraint_value(value) for value in result.values}
    logger.info("Parsed %d constraint value(s) for event matching", len(values_by_key))
    return values_by_key


def _constraint_schema_block(constraints: list[EventConstraint]) -> str:
    lines: list[str] = []
    for constraint in constraints:
        lines.append(f"- key: {constraint.key}")
        lines.append(f"  description: {constraint.description}")
        lines.append(f"  value_type: {constraint.value_type}")
        lines.append(f"  match_mode: {constraint.match_mode}")
    return "\n".join(lines)


def _extract_constraint_value(
    parsed_value: ParsedEventConstraintValue,
) -> str | bool | list[str] | None:
    if parsed_value.boolean_value is not None:
        return parsed_value.boolean_value
    if parsed_value.list_value:
        return parsed_value.list_value
    return parsed_value.string_value
