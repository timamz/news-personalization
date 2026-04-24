"""Batch Assessor Judge -- per-item critic over event notifications.

Reviews the Batch Event Assessor's structured output before delivery. For
each item the assessor marked, the judge decides whether the notification
is fit to send to the user: is `is_relevant` the right call against the
user_spec, does the `notification_body` match the assessor's own reason
without duplicating a recent alert, and does it respect format
constraints (chat-message length, plain text, no markdown bold).

The judge returns per-item verdicts so the caller can re-run the assessor
only for REVISE items on the next turn, leaving PASS items untouched. The
caller maintains the critic loop; this module is a single structured
LLM call.

Per CLAUDE.md error-handling tier 2 (quality gate): the caller must treat
failures as non-blocking and fall through with the unreviewed assessor
output.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from news_service.agents.event.batch_assessor import BatchAssessmentResult
from news_service.core.config import get_settings
from news_service.core.guardrails import sanitize_for_llm_prompt
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry
from news_service.core.llm_usage import agent_tag

logger = logging.getLogger(__name__)
settings = get_settings()

JUDGE_PROMPT = """\
You review the Batch Event Assessor's output before notifications are \
delivered to the user. The assessor produced, per news item, an \
is_relevant decision and (for relevant items) a notification_body written \
in the user's target language.

For each item, decide PASS or REVISE:

- PASS if all of the following hold:
  * is_relevant correctly reflects the user's subscription request -- \
relevant items genuinely match what the user asked for; irrelevant items \
genuinely do not.
  * When is_relevant is True: notification_body summarizes the same item \
the assessor's reason describes, is chat-message length (1-3 short \
sentences), plain text (no markdown bold **...**, no markdown lists), \
includes a source URL or equivalent pointer, and is NOT a near-duplicate \
of anything in the recent notification history.
  * When is_relevant is False: no further checks needed, PASS.

- REVISE otherwise. In the feedback field give the assessor a SHORT, \
ACTIONABLE note on what to fix. Examples:
  * "Body uses markdown bold -- rewrite in plain text."
  * "Near-duplicate of a notification sent 2 days ago about the same \
announcement; mark is_relevant=False."
  * "is_relevant=True but the post is a rumor; spec excludes rumors -- \
flip to False."
  * "Body describes item X but the assessor's reason is about item Y; \
align the body to the actual post."

Keep feedback under 200 characters. Never emit markdown bold in feedback. \
Be strict but fair: when in doubt, PASS; only REVISE for concrete \
issues you can name.

Return verdicts for every item_id the assessor returned.

# Output format

You MUST respond with a JSON object matching this EXACT schema and nothing \
else -- no prose, no explanations before or after, no markdown code fences, \
no bare array:

{
  "per_item": [
    {"item_id": "<uuid>", "verdict": "PASS" or "REVISE", "feedback": "<short note or empty>"}
  ],
  "overall": "PASS" or "REVISE"
}

The per_item array must contain one entry for every item_id in the input \
-- never fewer, never more. The overall field must be "REVISE" if any \
per_item entry has verdict "REVISE", else "PASS".
"""


class ItemVerdict(BaseModel):
    item_id: str = Field(..., description="UUID of the assessed news item")
    verdict: Literal["PASS", "REVISE"] = Field(..., description="PASS or REVISE")
    feedback: str = Field(
        default="",
        description="Short actionable fix note when verdict is REVISE; empty on PASS",
    )


class BatchJudgeResult(BaseModel):
    per_item: list[ItemVerdict] = Field(..., description="Verdict for every input item")
    overall: Literal["PASS", "REVISE"] = Field(
        ...,
        description="REVISE iff any item is REVISE; PASS only if all items PASS",
    )


@with_llm_retry()
async def judge_batch_events(
    *,
    assessment: BatchAssessmentResult,
    user_spec: str,
    recent_notification_history: list[str],
    max_history_chars: int,
) -> BatchJudgeResult:
    """Judge the assessor's per-item output and return PASS/REVISE verdicts."""
    if not assessment.assessments:
        raise ValueError("Cannot judge an empty assessment batch")

    assessments_block = "\n\n".join(
        f"Item {i + 1} [ID: {a.item_id}]:\n"
        f"is_relevant: {a.is_relevant}\n"
        f"assessor_reason: {sanitize_for_llm_prompt('reason', a.reason)}\n"
        f"notification_body: {sanitize_for_llm_prompt('body', a.notification_body)}"
        for i, a in enumerate(assessment.assessments)
    )

    history_text = "\n\n".join(
        f"Notification {i + 1}:\n{entry}" for i, entry in enumerate(recent_notification_history)
    )
    if len(history_text) > max_history_chars:
        history_text = history_text[:max_history_chars] + "\n... (truncated)"
    history_block = history_text if history_text else "No recent notification history."

    with agent_tag("event_judge"):
        completion = await chat_completion(
            model=settings.litellm_judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Subscription request:\n"
                        f"{sanitize_for_llm_prompt('user-preferences', user_spec)}\n\n"
                        f"Assessor output to review "
                        f"({len(assessment.assessments)} items):\n\n{assessments_block}\n\n"
                        f"Recent notification history:\n{history_block}"
                    ),
                },
            ],
            response_format=BatchJudgeResult,
            temperature=0.1,
        )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for batch event judging")

    returned_ids = {v.item_id for v in result.per_item}
    expected_ids = {a.item_id for a in assessment.assessments}
    if returned_ids != expected_ids:
        raise ValueError(
            f"Judge returned verdicts for {len(returned_ids)} items but assessor had "
            f"{len(expected_ids)}; missing={expected_ids - returned_ids} "
            f"extra={returned_ids - expected_ids}"
        )

    revised = sum(1 for v in result.per_item if v.verdict == "REVISE")
    logger.info(
        "Batch judge: %d/%d items REVISE (overall=%s)",
        revised,
        len(result.per_item),
        result.overall,
    )
    return result
