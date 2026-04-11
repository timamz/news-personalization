import logging

from pydantic import BaseModel, Field

from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

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


@with_llm_retry()
async def parse_schedule_preference(schedule_text: str) -> str:
    completion = await chat_completion(
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
