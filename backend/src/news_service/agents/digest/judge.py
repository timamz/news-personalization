"""Quality Judge — scores a digest and decides PASS or REVISE."""

import logging

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)
settings = get_settings()

JUDGE_PROMPT = """\
You are a quality judge for news digests. Score the digest on these criteria:

1. **Relevance** (1-5): Do the items match the user's stated interests?
2. **Coverage** (1-5): Are there obvious gaps given the available candidates?
3. **Dedup** (1-5): Are any items about the same story?
4. **Quality** (1-5): Is the writing clear, well-formatted, and appropriate length?

Then decide:
- If ALL scores are >= 3 and the average is >= 3.5: verdict = "PASS"
- Otherwise: verdict = "REVISE" and provide specific feedback for improvement.

Be strict but fair. A score of 3 means acceptable, 4 means good, 5 means excellent.
"""


class QualityScores(BaseModel):
    relevance: int = Field(..., ge=1, le=5, description="Relevance to user interests")
    coverage: int = Field(..., ge=1, le=5, description="Coverage of available topics")
    dedup: int = Field(..., ge=1, le=5, description="Deduplication quality")
    quality: int = Field(..., ge=1, le=5, description="Writing and formatting quality")
    verdict: str = Field(..., description="PASS or REVISE")
    feedback: str = Field(default="", description="Specific improvement feedback if REVISE")


@with_llm_retry()
async def judge_digest(
    *,
    digest_text: str,
    user_spec: str,
    candidates_summary: str,
) -> QualityScores:
    """Score a digest and decide whether to pass or request revision."""
    user_message = (
        f"User preferences:\n{user_spec}\n\n"
        f"Available candidates summary:\n{candidates_summary}\n\n"
        f"Digest to evaluate:\n{digest_text}"
    )

    completion = await chat_completion(
        model=settings.litellm_judge_model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=QualityScores,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for digest judging")

    logger.info(
        "Digest judge: relevance=%d coverage=%d dedup=%d quality=%d verdict=%s",
        result.relevance,
        result.coverage,
        result.dedup,
        result.quality,
        result.verdict,
    )
    return result
