import logging
from typing import Literal

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.openai_client import openai_client

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit", "twitter_account"]

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

SYSTEM_PROMPT = """\
You summarize news sources for source matching.

Given a source type, source title, and canonical URL, write one short paragraph describing
what this source likely covers. Keep it factual, neutral, and concise.

Rules:
- Base the summary only on the source type, title, and URL.
- Do not mention subscribers, popularity, or unverifiable claims.
- Do not address the user directly.
- Keep it under 60 words.
"""


class SourceDescription(BaseModel):
    description: str = Field(..., min_length=10, description="Short source coverage summary")


async def describe_source(
    *,
    source_kind: SourceKind,
    title: str,
    url: str,
) -> str:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source type: {source_kind}\n"
                    f"Source title: {title or '(missing)'}\n"
                    f"Canonical URL: {url}"
                ),
            },
        ],
        response_format=SourceDescription,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"LLM returned empty source description for {url}")
    description = " ".join(result.description.split())
    logger.info("Generated source description for %s", url)
    return description
