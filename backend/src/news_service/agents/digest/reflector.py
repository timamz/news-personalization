"""Pipeline Reflector — reviews pipeline health and self-heals.

Runs after each digest delivery. Can silently:
- Remove sources that contributed 0 items for weeks
- Strengthen user_spec when composer didn't comply with preferences
- Trigger source discovery for replacements

Notifies user only for major issues it cannot fix alone.
"""

import logging

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)
settings = get_settings()

REFLECTOR_PROMPT = """\
You are a pipeline quality reflector. After a digest was generated and delivered, \
review how the pipeline performed.

You receive:
- The digest that was delivered
- The user's preferences (user_spec)
- Quality scores from the judge
- Source contribution data (which sources contributed items)

Analyze and produce:
1. **observations**: Notes about pipeline health (always written to user_spec).
2. **user_spec_patch**: If the composer didn't follow user preferences (e.g., wrong length, \
ignored exclusions, wrong format), write a strengthened version of the relevant user_spec \
section. Empty string if no patch needed.
3. **sources_to_remove**: URLs of sources that have been consistently useless. \
Empty list if none.
4. **should_notify_user**: Only true if there's a major issue you cannot fix silently \
(e.g., all sources degraded, quality below threshold for 3+ digests).
5. **user_notification**: Message to send the user if should_notify_user is true.

Be conservative with removals — only remove sources that contributed 0 items for 3+ weeks.
Be proactive with spec patches — if the composer ignored a preference, fix the spec.
"""


class ReflectionResult(BaseModel):
    observations: str = Field(..., description="Pipeline health notes for user_spec")
    user_spec_patch: str = Field(default="", description="Strengthened user_spec section, or empty")
    sources_to_remove: list[str] = Field(
        default_factory=list, description="Source URLs to silently remove"
    )
    should_notify_user: bool = Field(
        default=False, description="Whether to send a proactive notification"
    )
    user_notification: str = Field(
        default="", description="Message for user if should_notify_user is true"
    )


@with_llm_retry()
async def reflect_on_pipeline(
    *,
    digest_text: str,
    user_spec: str,
    quality_scores: dict,
    source_contributions: str,
) -> ReflectionResult:
    """Review pipeline run and produce self-healing actions."""
    user_message = (
        f"User preferences (user_spec):\n{user_spec}\n\n"
        f"Quality scores: {quality_scores}\n\n"
        f"Source contributions:\n{source_contributions}\n\n"
        f"Delivered digest:\n{digest_text}"
    )

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": REFLECTOR_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=ReflectionResult,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for pipeline reflection")

    logger.info(
        "Reflector: %d sources to remove, notify_user=%s, has_patch=%s",
        len(result.sources_to_remove),
        result.should_notify_user,
        bool(result.user_spec_patch),
    )
    return result
