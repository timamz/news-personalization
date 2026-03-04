import logging

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client
from news_service.schemas.subscription import SubscriptionConfig

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

SYSTEM_PROMPT = """\
You are a news subscription parser. Given a user's natural language description of how they \
want to receive news, extract structured information.

Rules for schedule_cron:
- Use standard 5-field cron syntax (minute hour day-of-month month day-of-week).
- Only set schedule_cron when the user explicitly requests automatic timing.
- If the prompt has no explicit schedule request, set schedule_cron to null.
- "every morning" → "0 8 * * *"
- "every evening at 9pm" → "0 21 * * *"
- "every Saturday morning" → "0 8 * * 6"
- "every third day" → "0 8 */3 * *"
- "every hour" → "0 * * * *"
- "breaking news immediately" → "*/15 * * * *"

Rules for schedule_was_explicit:
- true if user explicitly asked for automatic schedule/timing in this prompt.
- false if schedule was not explicitly specified.

Rules for topics:
- Extract specific, searchable topic keywords.
- "AI news" → ["artificial intelligence", "machine learning"]
- "politics" → ["politics", "government"]

Rules for delivery_mode:
- Use "event" when the user wants notifications about upcoming releases, premieres, concerts,
  tours, launches, or asks to be notified when something will happen.
- Use "digest" for general news summaries and recurring news monitoring.
- If unclear, default to "digest".

Rules for event_matching_mode:
- Use "basic" for broad event requests where any relevant event from the chosen source
  is acceptable.
- Use "strict_with_prefilter" when the user includes exclusions, exact-person requirements,
  "only", "not", "except", or other wording that requires strict compliance with the prompt.
- For non-event subscriptions, use "basic".

Rules for event_constraints:
- Only create these when event_matching_mode is "strict_with_prefilter".
- Create a compact per-subscription schema with arbitrary snake_case keys.
- Each constraint must include:
  - key
  - description
  - value_type ("string", "boolean", or "list")
  - match_mode ("exact", "contains", "equals", or "intersects")
  - the correct required_* field for that value_type
  - prefilter_terms: cheap substrings that can screen obviously irrelevant posts
- The constraints should capture the user's exact requirements and exclusions.
- For "basic", return an empty list.

Rules for format_instructions:
- If the user specifies a format, use their wording.
- If not specified, default to "brief summary".

Rules for digest_language:
- Detect the language of the user's prompt.
- Return a short language code like "en", "ru", "es", "de", "fr".
- The digest must be generated in this same language.
"""

SCHEDULE_PARSER_PROMPT = """\
You parse natural-language schedule preferences into cron.

Rules:
- Return a valid 5-field cron expression.
- Output only the cron data in the schema.
- Examples:
  - "every morning" -> "0 8 * * *"
  - "every weekday at 9" -> "0 9 * * 1-5"
  - "every hour" -> "0 * * * *"
  - "every 3 days at 8am" -> "0 8 */3 * *"
"""


class ParsedSchedule(BaseModel):
    schedule_cron: str = Field(..., description="Cron expression for delivery schedule")


async def parse_subscription(prompt: str) -> SubscriptionConfig:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=SubscriptionConfig,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for subscription prompt")

    logger.info(
        (
            "Parsed subscription: topics=%s, mode=%s, event_matching=%s, "
            "constraints=%d, cron=%s, explicit=%s, format=%s, language=%s"
        ),
        result.topics,
        result.delivery_mode,
        result.event_matching_mode,
        len(result.event_constraints),
        result.schedule_cron,
        result.schedule_was_explicit,
        result.format_instructions,
        result.digest_language,
    )
    return result


async def parse_schedule_preference(schedule_text: str) -> str:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SCHEDULE_PARSER_PROMPT},
            {"role": "user", "content": schedule_text},
        ],
        response_format=ParsedSchedule,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty parsed response for schedule")
    return result.schedule_cron
