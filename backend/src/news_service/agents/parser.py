import logging

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
- "every morning" → "0 8 * * *"
- "every evening at 9pm" → "0 21 * * *"
- "every Saturday morning" → "0 8 * * 6"
- "every third day" → "0 8 */3 * *"
- "every hour" → "0 * * * *"
- "breaking news immediately" → "*/15 * * * *"

Rules for topics:
- Extract specific, searchable topic keywords.
- "AI news" → ["artificial intelligence", "machine learning"]
- "politics" → ["politics", "government"]

Rules for format_instructions:
- If the user specifies a format, use their wording.
- If not specified, default to "brief summary".

Rules for digest_language:
- Detect the language of the user's prompt.
- Return a short language code like "en", "ru", "es", "de", "fr".
- The digest must be generated in this same language.
"""


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
        "Parsed subscription: topics=%s, cron=%s, format=%s, language=%s",
        result.topics,
        result.schedule_cron,
        result.format_instructions,
        result.digest_language,
    )
    return result
