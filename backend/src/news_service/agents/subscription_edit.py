import logging

from news_service.core.config import get_settings
from news_service.core.llm_retry import with_llm_retry
from news_service.core.openai_client import openai_client
from news_service.schemas.subscription import SubscriptionEditProposalResponse

logger = logging.getLogger(__name__)

settings = get_settings()
_client = openai_client

SYSTEM_PROMPT = """\
You update existing news subscriptions.

You will receive:
- the current canonical subscription request
- the current format instructions
- a user's free-form edit request

Return the updated subscription state.

Rules:
- Treat the user's message as an edit to the existing subscription, not as a brand-new request.
- Preserve the existing intent unless the user explicitly changes it.
- Apply additive changes incrementally when possible.
- Do not change sources, schedule, language, or delivery settings.
- `canonical_prompt` should be a clean, self-contained request that reflects the updated meaning.
- `prompt_summary` should be a short user-facing paraphrase,
  usually 3-8 words, in the user's language.
- `format_instructions` should only change when the user asked to change writing style or format.
- `change_summary` should briefly explain what changed in one sentence.
"""


@with_llm_retry()
async def propose_subscription_edit(
    *,
    canonical_prompt: str,
    format_instructions: str,
    change_request: str,
) -> SubscriptionEditProposalResponse:
    completion = await _client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current canonical request:\n{canonical_prompt}\n\n"
                    f"Current format instructions:\n{format_instructions}\n\n"
                    f"User edit request:\n{change_request}"
                ),
            },
        ],
        response_format=SubscriptionEditProposalResponse,
        temperature=0.1,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError("LLM returned empty response for subscription edit proposal")

    logger.info(
        "Proposed subscription edit: summary=%s",
        result.prompt_summary,
    )
    return result
