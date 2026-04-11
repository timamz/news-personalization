"""Digest Composer — writes the actual digest following the planner's outline."""

import logging

from pydantic import BaseModel, Field

from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

COMPOSER_PROMPT = """\
You are a news digest writer. Create a well-structured, readable digest \
following the plan provided.

Quality criteria:
- Prioritize the most substantive items.
- Skip stale items, low-signal community chatter, personal requests, \
endorsement requests, generic questions, and self-promotional posts.
- If multiple items cover the same story, include only the most informative one.
- For every item, end with the exact line '{source_label}: <original link>' \
using exactly that label.
- Never switch to a different language for the source label.
- Do not mention feed names, channel names, site names, or labels \
other than the required '{source_label}:' line.
- Return only the digest. No introductions, closings, commentary, or offers to help.

IMPORTANT: In used_item_ids, list the UUIDs of every news item you included.
"""


def _is_russian_language(digest_language: str) -> bool:
    return digest_language.strip().lower().split("-", maxsplit=1)[0] == "ru"


class DigestComposition(BaseModel):
    digest_text: str = Field(..., description="The formatted news digest")
    used_item_ids: list[str] = Field(..., description="UUIDs of news items included in the digest")


@with_llm_retry()
async def compose_digest(
    *,
    plan: str,
    items_text: str,
    user_spec: str,
    digest_language: str,
    format_instructions: str,
    feedback: str = "",
) -> DigestComposition:
    """Write a digest following the plan, optionally incorporating judge feedback."""
    source_label = "Источник" if _is_russian_language(digest_language) else "Source"
    system = COMPOSER_PROMPT.format(source_label=source_label)

    user_parts = [
        f"Digest plan:\n{plan}",
        f"Language: {digest_language}",
        f"Format: {format_instructions}",
    ]
    if user_spec:
        user_parts.append(f"User preferences:\n{user_spec}")
    if feedback:
        user_parts.append(f"REVISION REQUESTED — address this feedback:\n{feedback}")
    user_parts.append(f"Candidate news items:\n\n{items_text}")

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        response_format=DigestComposition,
        temperature=0.3,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for digest composition")

    logger.info("Digest composed with %d items", len(result.used_item_ids))
    return result
