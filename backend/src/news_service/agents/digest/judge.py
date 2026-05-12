"""Quality Judge — scores a digest and decides PASS or REVISE."""

import logging

from pydantic import BaseModel, Field

from news_service.core.config import get_settings
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry
from news_service.core.llm_usage import agent_tag

logger = logging.getLogger(__name__)
settings = get_settings()

JUDGE_PROMPT = """\
You are a quality judge for news digests. Score the digest on these criteria:

1. **Relevance** (1-5): Do the items match the user's stated interests?
2. **Format** (1-5): Does the digest follow the user's format, length, and style \
preferences? Check against their user_spec instructions.
3. **Conciseness** (1-5): Is every item substantive? No filler, no redundancy, \
no low-signal items included just to pad the digest.

Then decide:
- If ALL scores are >= 3 and the average is >= 3.5: verdict = "PASS"
- Otherwise: verdict = "REVISE" and provide specific feedback for improvement.

Be strict but fair. A score of 3 means acceptable, 4 means good, 5 means excellent.

Never emit Markdown bold syntax (**...**) in your feedback text. The writer \
reads it verbatim and would copy the markers into the user-visible digest, \
where the frontend does not render them.

# Output format

You MUST respond with a JSON object matching this EXACT schema and nothing \
else -- no prose, no explanations before or after, no markdown code fences:

{
  "relevance": integer 1-5,
  "format_score": integer 1-5,
  "conciseness": integer 1-5,
  "verdict": "PASS" or "REVISE",
  "feedback": string (empty when verdict is PASS)
}
"""


class QualityScores(BaseModel):
    relevance: int = Field(..., ge=1, le=5, description="Relevance to user interests")
    format_score: int = Field(..., ge=1, le=5, description="Adherence to user format preferences")
    conciseness: int = Field(..., ge=1, le=5, description="No filler, no redundancy")
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

    with agent_tag("digest_judge"):
        completion = await chat_completion(
            model=settings.litellm_judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=QualityScores,
            temperature=0.1,
            reasoning=True,
        )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for digest judging")

    logger.info(
        "Digest judge: relevance=%d format=%d conciseness=%d verdict=%s",
        result.relevance,
        result.format_score,
        result.conciseness,
        result.verdict,
    )
    return result
