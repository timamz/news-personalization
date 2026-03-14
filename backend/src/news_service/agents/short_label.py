"""Generate ultra-short 2-3 word labels for subscriptions."""

import logging

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

SYSTEM_PROMPT = """\
Generate an ultra-short 2-3 word label for a news subscription.
Think of it as a category or topic name that fits on a small button.

Rules:
- Maximum 3 words, ideally 2.
- Preserve the language of the input.
- Be specific: "AI Research" is better than "News".
- Examples: "AI News", "Tech Events", "Crypto Prices", "ML Research", \
"Кино новинки", "Спорт события".
"""


class ShortLabel(BaseModel):
    label: str = Field(..., min_length=1, max_length=30, description="Ultra-short 2-3 word label")


async def generate_short_label(prompt_summary: str) -> str:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_summary},
        ],
        response_format=ShortLabel,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for short label generation")
    logger.info("Generated short label: %s -> %s", prompt_summary, result.label)
    return result.label
