"""ADK agent wiring and streaming runner for the conversational agent."""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from news_service.agents.adk_runner import make_adk_model, run_agent, run_agent_text
from news_service.agents.conversational.helpers import (
    _load_subscription_summaries,
    _status_for_tool_call,
)
from news_service.agents.conversational.prompt import _build_instruction
from news_service.agents.conversational.tools import build_tools
from news_service.core.config import get_settings
from news_service.core.llm_usage import agent_tag
from news_service.db.session import async_session_factory
from news_service.models.user import User
from news_service.schemas.conversation import AgentTurnOutput

logger = logging.getLogger(__name__)
settings = get_settings()


def create_conversational_agent(
    *,
    db_session: AsyncSession,
    user: User,
    conversation_summary: str,
    user_language: str | None = None,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    subscription_summaries: list[str] | None = None,
    compacted_log: list[str] | None = None,
) -> tuple[Agent, dict[str, Any]]:
    """Build a fresh ADK agent bound to this turn's DB session and user.

    Returns the agent and a shared_state dict. All mutation tools open their
    own DB session via session_factory so the model can emit parallel tool
    calls safely. ``shared_state['scenario_close_summary']`` is set when
    the agent calls close_scenario; the caller uses that signal to compact
    the hot transcript after the turn.

    Conversation history is NOT taken here on purpose. Prior turns are
    delivered to the LLM as a real ``messages[]`` array via the ADK session
    pre-population in ``adk_runner.run_agent``; this function only builds
    the stable system instruction (rules, persona, summary cards, closed-
    scenario log).
    """
    scoped_factory = session_factory or async_session_factory
    effective_language = user_language or user.language or "en"
    shared_state: dict[str, Any] = {
        "status": "in_progress",
        "created_subscription_id": None,
        "scenario_close_summary": None,
        "status_queue": status_queue,
        "display_language": effective_language,
    }

    tools = build_tools(
        user=user,
        db_session=db_session,
        scoped_factory=scoped_factory,
        shared_state=shared_state,
    )

    instruction = _build_instruction(
        conversation_summary=conversation_summary,
        user_language=user_language or user.language,
        user_timezone=user.timezone,
        subscription_summaries=subscription_summaries,
        compacted_log=compacted_log,
        has_onboarded=bool(getattr(user, "has_onboarded", False)),
    )

    agent = Agent(
        name="conversational_agent",
        model=make_adk_model(settings.litellm_model, reasoning=False),
        instruction=instruction,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    return agent, shared_state


async def run_conversational_turn(
    *,
    db_session: AsyncSession,
    user: User,
    user_message: str,
    conversation_summary: str,
    user_language: str | None = None,
) -> dict[str, Any]:
    """Run a single non-streaming turn and return a simple result dict.

    Used by tests and non-streaming callers.
    """
    agent, shared_state = create_conversational_agent(
        db_session=db_session,
        user=user,
        conversation_summary=conversation_summary,
        user_language=user_language,
    )
    with agent_tag("conversational"):
        agent_message = await run_agent_text(
            agent=agent,
            message=user_message,
            user_id=str(user.id),
            max_llm_calls=settings.tool_call_budget_conversational,
        )
    return {
        "agent_message": agent_message,
        "created_subscription_id": shared_state["created_subscription_id"],
    }


_ADK_SENTINEL = object()
_SHARE_TOKEN_RE = re.compile(r"SHARE_TOKEN=([A-Za-z0-9_-]+)")


def _extract_share_token(tool_result: str) -> str | None:
    match = _SHARE_TOKEN_RE.search(tool_result)
    if match is None:
        return None
    return match.group(1)


def _message_with_share_token(agent_text: str, token: str, language: str) -> str:
    wrapped = f"```{token}```"
    if wrapped in agent_text:
        return agent_text
    if token in agent_text:
        return agent_text.replace(token, wrapped, 1)
    if language.startswith("ru"):
        return (
            f"Токен для импорта подписки:\n{wrapped}\n\n"
            "Он действует 7 дней и одноразовый: после первого импорта перестанет работать. "
            "Получателю нужно вставить этот токен в свой чат с ассистентом."
        )
    return (
        f"Subscription import token:\n{wrapped}\n\n"
        "It is valid for 7 days and one-shot: after the first import it stops working. "
        "The recipient should paste this token into their own chat with this assistant."
    )


async def run_conversation_turn_streaming(
    messages: list[dict],
    *,
    db_session: AsyncSession,
    user: User,
    conversation_summary: str,
    user_language: str | None = None,
    compacted_log: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming variant: yields status events, then one final done event.

    Uses a single merged queue so progress events emitted by a tool
    mid-execution (e.g. the inline source-discovery run) reach the HTTP
    stream even while ADK is blocked awaiting the tool result. ADK
    events are pushed onto the queue by a background task; tool status
    puts land on the same queue directly via ``shared_state["status_queue"]``.

    Events:
      {"event": "status", "status_key": ..., ...kwargs}
      {"event": "discovery_progress", "phase": ..., "display_text": ..., ...}
      {"event": "done", "output": {...}, "new_messages": [...], "shared_state": {...}}
      {"event": "error", "detail": "..."}
    """
    with agent_tag("conversational"):
        previous_messages = messages[:-1] if len(messages) > 1 else []
        current_message = messages[-1]["content"] if messages else ""

        events_queue: asyncio.Queue[Any] = asyncio.Queue()

        try:
            subscription_summaries = await _load_subscription_summaries(db_session, user.id)
        except Exception:
            logger.exception("Failed to load subscription summaries; continuing without context")
            subscription_summaries = None

        agent, shared_state = create_conversational_agent(
            db_session=db_session,
            user=user,
            conversation_summary=conversation_summary,
            user_language=user_language,
            status_queue=events_queue,
            subscription_summaries=subscription_summaries,
            compacted_log=compacted_log,
        )

        agent_text = ""
        share_token: str | None = None
        agent_error: BaseException | None = None
        confirmation_events: list[dict[str, Any]] = []

        async def pump_adk() -> None:
            nonlocal agent_text, agent_error
            try:
                async for event in run_agent(
                    agent=agent,
                    message=current_message,
                    user_id=str(user.id),
                    conversation_history=previous_messages,
                    max_llm_calls=settings.tool_call_budget_conversational,
                ):
                    await events_queue.put((_ADK_SENTINEL, event))
            except BaseException as exc:
                agent_error = exc
            finally:
                await events_queue.put(None)

        task = asyncio.create_task(pump_adk())

        try:
            while True:
                item = await events_queue.get()
                if item is None:
                    break
                if isinstance(item, tuple) and len(item) == 2 and item[0] is _ADK_SENTINEL:
                    event = item[1]
                    if event["type"] == "tool_call":
                        emitted = _status_for_tool_call(event)
                        if emitted is not None:
                            yield emitted
                    elif event["type"] == "tool_result":
                        if event.get("name") == "share_subscription":
                            extracted = _extract_share_token(str(event.get("result") or ""))
                            if extracted is not None:
                                share_token = extracted
                    elif event["type"] == "final_response":
                        agent_text = event["text"]
                elif isinstance(item, dict):
                    if item.get("event") == "requires_confirmation":
                        confirmation_events.append(item)
                    yield item
        finally:
            await task

        if agent_error is not None:
            logger.exception("Conversational agent streaming failed", exc_info=agent_error)
            yield {"event": "error", "detail": f"Agent error: {agent_error}"}
            return

        if share_token is not None:
            agent_text = _message_with_share_token(
                agent_text,
                share_token,
                str(shared_state.get("display_language") or user.language or "en"),
            )
        if confirmation_events:
            confirmation_texts = [
                str(event.get("message") or "").strip()
                for event in confirmation_events
                if str(event.get("message") or "").strip()
            ]
            if confirmation_texts:
                agent_text = "\n\n".join(confirmation_texts)

        output = AgentTurnOutput(
            message=agent_text,
            status=shared_state["status"],
        )
        yield {
            "event": "done",
            "output": output.model_dump(),
            "new_messages": [{"role": "assistant", "content": agent_text}],
            "shared_state": {
                "scenario_close_summary": shared_state.get("scenario_close_summary"),
            },
        }
