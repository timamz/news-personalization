"""Batch Event Assessor — evaluates multiple items for one subscription in a single LLM call.

Instead of N independent LLM calls (one per item), the batch assessor sees all
new items at once. This enables:
- Cross-item deduplication (3 articles about the same event -> pick the best one)
- Better relevance judgment (context about what else was published)
- Significant cost reduction (M calls instead of N*M)
"""

import logging

from pydantic import BaseModel, Field

from news_service.core.guardrails import sanitize_for_llm_prompt
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

BATCH_ASSESS_PROMPT = """\
You decide which posts to deliver to the user based on their subscription request.

The subscription request defines exactly what the user wants. Only deliver posts that \
directly match the request. Being in the same general topic or category is NOT enough — \
the post must be about something the user specifically asked for.

If the user listed specific titles, names, people, or entities, the post must be about \
one of those.

Rules:
- Evaluate ALL items together. If multiple items cover the same event, pick only the best one.
- Check notification history — skip anything the user was already notified about.
- For each relevant item, compose a short notification_body in the target language: \
a concise title, timing if known, 1-2 sentence summary, and the source URL.
- Keep each notification suitable for a chat message.
- Return assessments for ALL items (both relevant and not relevant).
- Never emit Markdown bold syntax (**...**) anywhere in notification_body. \
The frontend does not render it and the asterisks appear literally. \
Use plain text -- no bold markers at all.

Revision mode: if the user message includes a "Critic feedback" section, this \
is a revision turn. The assessor output you produced earlier failed a quality \
review for the listed items. Address the feedback per item in this turn and \
return refined assessments only for the items listed there.
"""


class ItemAssessment(BaseModel):
    item_id: str = Field(..., description="UUID of the news item")
    is_relevant: bool = Field(..., description="Whether this item matches the subscription")
    notification_body: str = Field(default="", description="Notification text if relevant")
    reason: str = Field(..., min_length=3, description="Why relevant or not")


class BatchAssessmentResult(BaseModel):
    assessments: list[ItemAssessment] = Field(..., description="Assessment for each input item")


@with_llm_retry()
async def assess_batch_events(
    *,
    items: list[dict],
    user_spec: str,
    target_language: str,
    recent_notification_history: list[str],
    max_history_chars: int,
    critic_feedback_per_item: dict[str, str] | None = None,
) -> BatchAssessmentResult:
    """Assess multiple news items for one subscription in a single LLM call.

    Args:
        items: List of dicts with keys: item_id, headline, body, url, published_at
        user_spec: The user's subscription spec (preferences, exclusions, etc.)
        target_language: Language for notification text
        recent_notification_history: Formatted history entries
        max_history_chars: Max chars for history block
        critic_feedback_per_item: Optional {item_id -> feedback} from the judge.
            When set, this is a revision turn: the prompt includes the feedback
            and asks the assessor to refine only the listed items.
    """
    items_block = "\n\n".join(
        f"Item {i + 1} [ID: {item['item_id']}]:\n"
        f"Headline: {sanitize_for_llm_prompt('headline', item['headline'])}\n"
        f"Body: {sanitize_for_llm_prompt('body', item['body'])}\n"
        f"URL: {item['url']}\n"
        f"Published: {item.get('published_at', 'unknown')}"
        for i, item in enumerate(items)
    )

    history_text = "\n\n".join(
        f"Notification {i + 1}:\n{entry}" for i, entry in enumerate(recent_notification_history)
    )
    if len(history_text) > max_history_chars:
        history_text = history_text[:max_history_chars] + "\n... (truncated)"
    history_block = history_text if history_text else "No recent notification history."

    feedback_block = ""
    if critic_feedback_per_item:
        entries = "\n".join(
            f"- Item [ID: {item_id}]: {sanitize_for_llm_prompt('critic-feedback', feedback)}"
            for item_id, feedback in critic_feedback_per_item.items()
        )
        feedback_block = (
            "\n\nCritic feedback (revision turn -- address these specific issues "
            "and return refined assessments for exactly these items):\n" + entries
        )

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": BATCH_ASSESS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target language: {target_language}\n\n"
                    f"Subscription request:\n"
                    f"{sanitize_for_llm_prompt('user-preferences', user_spec)}\n\n"
                    f"Posts to evaluate ({len(items)} items):\n\n{items_block}\n\n"
                    f"Notification history:\n{history_block}"
                    f"{feedback_block}"
                ),
            },
        ],
        response_format=BatchAssessmentResult,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for batch event assessment")

    relevant_count = sum(1 for a in result.assessments if a.is_relevant)
    logger.info(
        "Batch assessment: %d/%d items relevant%s",
        relevant_count,
        len(items),
        " (revision turn)" if critic_feedback_per_item else "",
    )
    return result
