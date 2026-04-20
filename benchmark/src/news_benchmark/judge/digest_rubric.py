"""
LLM-as-judge rubric for delivered digests.

Runs at temperature=0 on the configured LITELLM_JUDGE_MODEL, reading the
delivered digest text + the subscription's user_spec + the article
corpus that was available to the Writer. Scores are secondary signals
— treat them as relative regression indicators, not absolute truth.
"""

from __future__ import annotations

import json

import litellm
from pydantic import BaseModel, Field

from news_benchmark.tagging import agent_tag


class DigestScores(BaseModel):
    goal_relevance: int = Field(ge=1, le=5)
    format_adherence: int = Field(ge=1, le=5)
    factual_grounding: int = Field(ge=1, le=5)
    language_match: bool
    rationale: str


SYSTEM = (
    "You are a quality judge for a news-digest service. Given the user's "
    "subscription spec and a delivered digest, rate the digest on four axes. "
    "Return strict JSON matching the schema. Be conservative: 5 is reserved "
    "for clearly exemplary work."
)


async def judge_digest(
    *,
    user_spec: str,
    digest_text: str,
    delivered_article_corpus: list[dict[str, str]],
    judge_model: str,
) -> DigestScores:
    corpus_block = "\n".join(
        f"- [{a['url']}] {a['headline']}" for a in delivered_article_corpus[:30]
    )
    user = (
        f"USER_SPEC:\n{user_spec}\n\n"
        f"DIGEST:\n{digest_text}\n\n"
        f"AVAILABLE_ARTICLE_CORPUS:\n{corpus_block}\n\n"
        "Return JSON with keys goal_relevance, format_adherence, "
        "factual_grounding, language_match, rationale."
    )
    async with agent_tag("judge.digest"):
        resp = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
    raw = resp["choices"][0]["message"]["content"] or "{}"
    return DigestScores.model_validate(json.loads(raw))
