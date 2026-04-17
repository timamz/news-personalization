"""Digest Writer -- ADK agent that plans, researches, and composes news digests.

Replaces the separate planner and composer with a single agent that can
fetch full articles and search the web for additional context before writing.

The agent follows a three-phase workflow:
1. Review candidates and user preferences to plan what to include.
2. Optionally fetch full articles or search the web for context.
3. Write the digest and submit it via the submit_digest tool.

Example usage::

    composition = await write_digest(
        items_text="[ID: abc] Headline: GPT-5...",
        user_spec="## Topic\\nAI news\\n\\n## Preferences\\nBrief summary",
        digest_language="en",
        recent_digest_summaries="",
    )
    print(composition.digest_text)
"""

import logging
import uuid
from typing import Any

import httpx
from bs4 import BeautifulSoup
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, Field

from news_service.agents.adk_runner import run_agent_text
from news_service.core.config import get_settings
from news_service.core.guardrails import sanitize_for_llm_prompt
from news_service.models.user_spec import parse_user_spec
from news_service.services.search import search_web as _search_web

_DEFAULT_FORMAT_GUIDANCE = "brief summary"

logger = logging.getLogger(__name__)
settings = get_settings()


class DigestComposition(BaseModel):
    """Result of the digest writer: the formatted text and the IDs of items used.

    Example::

        comp = DigestComposition(
            digest_text="## AI News\\n...",
            used_item_ids=["abc-123", "def-456"],
        )
    """

    digest_text: str = Field(..., description="The formatted news digest")
    used_item_ids: list[str] = Field(..., description="UUIDs of news items included in the digest")


_WRITER_PROMPT = """\
You are a news digest writer. Your job is to select the most important items
from the candidates, research them if needed, and write a well-structured digest.

Workflow:
1. Review all candidate items and the user's preferences.
2. For items with thin or unclear content, use fetch_article to read the full text.
3. For items referencing something you need context on, use search_web.
4. Write the digest and call submit_digest with the final text and item IDs.

Quality criteria:
- Prioritize the most substantive items.
- Skip stale items, low-signal chatter, self-promotional posts, generic questions.
- If multiple items cover the same story, include only the most informative one.
- For every item, end with '{source_label}: <original link>'.
- Never switch to a different language for the source label.
- Do not mention feed names, channel names, site names, or labels \
other than the required '{source_label}:' line.
- Return only the digest. No introductions, closings, commentary.
- Budget: fetch up to {max_fetches} articles, do up to {max_searches} web searches.
  Use them on items that need it most -- do not fetch articles that already have \
good summaries.

IMPORTANT: In submit_digest, list the UUIDs of every news item you included
as a comma-separated string in used_item_ids.
"""


def _is_russian_language(digest_language: str) -> bool:
    return digest_language.strip().lower().split("-", maxsplit=1)[0] == "ru"


async def write_digest(
    *,
    items_text: str,
    user_spec: str,
    digest_language: str,
    recent_digest_summaries: str,
    feedback: str = "",
) -> DigestComposition:
    """Run the Digest Writer ADK agent and return the composed digest.

    The agent plans, optionally researches (fetch articles / web search),
    and composes a digest in a single agentic loop. Returns the same
    DigestComposition type as the old composer for pipeline compatibility.

    Raises RuntimeError if the agent finishes without calling submit_digest.
    """
    max_fetches = settings.writer_max_article_fetches
    max_searches = settings.writer_max_web_searches
    fetch_timeout = settings.writer_article_fetch_timeout_seconds
    max_article_chars = settings.writer_article_max_chars

    fetch_counter = 0
    search_counter = 0

    shared_state: dict[str, Any] = {
        "completed": False,
        "digest_text": "",
        "used_item_ids": [],
    }

    async def fetch_article(url: str) -> str:
        """Fetch the full text of an article from its URL.

        Downloads the page and extracts readable text content.
        Useful for items whose summaries are too short or unclear.

        Args:
            url: The article URL to fetch and extract text from.

        Returns:
            The extracted article text, or an error message on failure.
        """
        nonlocal fetch_counter
        if fetch_counter >= max_fetches:
            return "Fetch budget exhausted"
        fetch_counter += 1
        try:
            async with httpx.AsyncClient(
                timeout=fetch_timeout,
                proxy=settings.proxy_url,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) > max_article_chars:
                text = text[:max_article_chars]
            return text if text else "No readable text extracted from the page"
        except Exception as exc:
            return f"Failed to fetch article: {exc}"

    async def search_web(query: str) -> str:
        """Search the web for additional context on a news topic.

        Useful when an item references something you need background on,
        or when you want to verify a claim before including it.

        Args:
            query: The search query string.

        Returns:
            Formatted search results, or an error/budget message.
        """
        nonlocal search_counter
        if search_counter >= max_searches:
            return "Search budget exhausted"
        search_counter += 1
        try:
            return await _search_web(query)
        except Exception as exc:
            return f"Web search failed: {exc}"

    async def submit_digest(digest_text: str, used_item_ids: str) -> str:
        """Submit the final digest text and the IDs of items included.

        Call this exactly once when the digest is ready. The used_item_ids
        must be a comma-separated string of UUID strings corresponding to
        the [ID: ...] markers in the candidate items.

        Args:
            digest_text: The complete, formatted digest text.
            used_item_ids: Comma-separated UUIDs of included news items.

        Returns:
            Confirmation message.
        """
        shared_state["digest_text"] = digest_text
        shared_state["used_item_ids"] = [
            uid.strip() for uid in used_item_ids.split(",") if uid.strip()
        ]
        shared_state["completed"] = True
        return "Digest submitted successfully."

    is_ru = _is_russian_language(digest_language)
    source_label = "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a" if is_ru else "Source"
    system_prompt = _WRITER_PROMPT.format(
        source_label=source_label,
        max_fetches=max_fetches,
        max_searches=max_searches,
    )

    format_guidance = _DEFAULT_FORMAT_GUIDANCE
    if user_spec:
        try:
            preferences = parse_user_spec(user_spec).preferences
        except Exception:
            preferences = ""
        if preferences.strip():
            format_guidance = preferences.strip()

    user_parts = [
        f"Language: {digest_language}",
        f"Format: {format_guidance}",
    ]
    if user_spec:
        user_parts.append(
            f"User preferences:\n{sanitize_for_llm_prompt('user-preferences', user_spec)}"
        )
    if recent_digest_summaries:
        user_parts.append(recent_digest_summaries)
    if feedback:
        user_parts.append(f"REVISION REQUESTED -- address this feedback:\n{feedback}")
    user_parts.append(f"Candidate news items:\n\n{items_text}")

    input_message = "\n\n".join(user_parts)

    agent = Agent(
        name=f"digest_writer_{uuid.uuid4().hex[:6]}",
        model=LiteLlm(model=settings.litellm_model),
        instruction=system_prompt,
        tools=[fetch_article, search_web, submit_digest],
        generate_content_config=types.GenerateContentConfig(temperature=0.3),
    )

    await run_agent_text(
        agent=agent,
        message=input_message,
        user_id="digest-pipeline",
    )

    if not shared_state["completed"]:
        raise RuntimeError("Digest Writer agent finished without calling submit_digest")

    logger.info(
        "Digest Writer composed digest with %d items (fetched=%d, searched=%d)",
        len(shared_state["used_item_ids"]),
        fetch_counter,
        search_counter,
    )

    return DigestComposition(
        digest_text=shared_state["digest_text"],
        used_item_ids=shared_state["used_item_ids"],
    )
