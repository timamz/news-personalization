"""Discovery Orchestrator — plan-mode agent that decomposes a topic into search strategies.

The orchestrator analyzes the user's subscription topic and produces a DiscoveryPlan:
a list of 2-5 independent search strategies that GenericFinders will execute in parallel.
The orchestrator has no tools — it only reasons about what to search for.
"""

import logging

from news_service.core.config import get_settings
from news_service.core.llm import chat_completion
from news_service.core.llm_retry import with_llm_retry

from .models import DiscoveryPlan

logger = logging.getLogger(__name__)
settings = get_settings()

ORCHESTRATOR_PROMPT = """\
You are a search strategy planner for a news source discovery system.

Given a user's subscription topic, decompose it into 2-5 independent search strategies. \
Each strategy should target a different type or angle of source:

Source types available:
- RSS feeds (blogs, news sites, arxiv, etc.)
- Telegram channels
- Reddit subreddits
- Twitter/X accounts

Guidelines:
- Each strategy is a self-contained instruction for a search agent.
- Include the source type to focus on in each strategy.
- Use specific, targeted language: "Find arxiv RSS feeds about transformer architectures" \
is better than "Find academic sources".
- Adapt the number of strategies to the topic: a narrow topic needs 2-3, a broad one needs 4-5.
- Consider the topic's domain: academic topics need arxiv/papers, consumer topics need \
social/news, tech topics need a mix.

Return only the list of strategy strings. No commentary.
"""


@with_llm_retry()
async def plan_discovery(topic: str, removal_history: str = "") -> DiscoveryPlan:
    """Analyze a topic and produce a list of search strategies."""
    user_content = f"Topic: {topic}"
    if removal_history:
        user_content += (
            f"\n\nRecently removed sources (use judgment about re-adding):\n{removal_history}"
        )

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=DiscoveryPlan,
        temperature=0.2,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"Orchestrator returned empty plan for topic: {topic}")
    logger.info(
        "Discovery orchestrator produced %d strategies for topic: %s",
        len(result.strategies),
        topic[:80],
    )
    return result
