import logging
from typing import Literal

from pydantic import BaseModel, Field

from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

type SourceKind = Literal["rss", "telegram_channel", "reddit_subreddit"]

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You summarize news sources for source matching.

Given a source type, source title, canonical URL, and optionally sample content from the source,
write one short paragraph describing what this source covers. Keep it factual, neutral, and concise.

Rules:
- If sample content is provided, base the summary on the actual content.
- If no sample content is provided, infer from the source type, title, and URL.
- Do not mention subscribers, popularity, or unverifiable claims.
- Do not address the user directly.
- Keep it under 60 words.
"""


class SourceDescription(BaseModel):
    description: str = Field(..., min_length=10, description="Short source coverage summary")


@with_llm_retry()
async def describe_source(
    *,
    source_kind: SourceKind,
    title: str,
    url: str,
    sample_content: list[str] | None = None,
) -> str:
    user_lines = [
        f"Source type: {source_kind}",
        f"Source title: {title or '(missing)'}",
        f"Canonical URL: {url}",
    ]
    if sample_content:
        truncated = [text[:300] for text in sample_content[:5]]
        user_lines.append("\nSample content from this source:\n" + "\n---\n".join(truncated))

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_lines)},
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
