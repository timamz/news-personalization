"""Conversational agent -- the single chat surface for the user.

One ADK agent handles everything: greeting, help, subscription create / edit,
source management, language + timezone setup, digest triggering, deletion.
The agent is respawned fresh each turn; durable state lives in Postgres
(``User``, ``Subscription``) and Redis (conversation transcript, persistent
per user, no conversation ids).

The package is split into:

- ``prompt.py`` -- the system prompt and the per-turn instruction builder.
- ``helpers.py`` -- small pure-Python utilities (identifier parsing,
  subscription summaries, tool-call status mapping, conversation-summary
  deduplication).
- ``tools.py`` -- the 11 tool closures bound to the current turn's user,
  DB session, and shared state.
- ``agent.py`` -- the ADK ``Agent`` factory plus non-streaming and
  streaming runners.

External callers import from the package namespace; this file re-exports
the public surface.
"""

from news_service.agents.conversational.agent import (
    create_conversational_agent,
    run_conversation_turn_streaming,
    run_conversational_turn,
)
from news_service.agents.conversational.helpers import (
    _append_conversation_summary,
    _load_subscription_summaries,
    _source_display_name,
)
from news_service.agents.conversational.prompt import (
    CONVERSATIONAL_AGENT_PROMPT,
    _build_instruction,
)

__all__ = [
    "CONVERSATIONAL_AGENT_PROMPT",
    "_append_conversation_summary",
    "_build_instruction",
    "_load_subscription_summaries",
    "_source_display_name",
    "create_conversational_agent",
    "run_conversation_turn_streaming",
    "run_conversational_turn",
]
