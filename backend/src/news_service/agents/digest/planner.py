"""Digest Planner — creates a digest outline from user_spec and available candidates."""

import logging

from pydantic import BaseModel, Field

from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry
from news_service.orchestration.guardrails import sanitize_for_llm_prompt

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """\
You are a digest planner. Given the user's preferences and available news candidates, \
create a brief outline for the digest.

Your outline should specify:
- How many items to include (respect user preferences on length)
- Which themes/sections to cover
- What to prioritize and what to skip
- Any exclusions from user preferences

Keep the plan concise — 3-5 bullet points. The composer will follow your plan.
"""


class DigestPlan(BaseModel):
    plan: str = Field(..., description="Brief outline for the digest composer to follow")
    target_item_count: int = Field(
        ..., ge=1, le=20, description="Target number of items for this digest"
    )


@with_llm_retry()
async def plan_digest(
    *,
    user_spec: str,
    items_text: str,
    digest_language: str,
    format_instructions: str,
) -> DigestPlan:
    """Create a digest plan from user preferences and available candidates."""
    user_message = (
        f"User preferences:\n{sanitize_for_llm_prompt('user-preferences', user_spec)}\n\n"
        f"Digest language: {digest_language}\n"
        f"Format: {format_instructions}\n\n"
        f"Available candidates:\n{items_text}"
    )

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=DigestPlan,
        temperature=0.2,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for digest planning")

    logger.info("Digest plan: %d target items", result.target_item_count)
    return result
