"""
S-discovery-empty-pool: the discovery pipeline surfaces a user-visible ERROR
when no candidates can be found.

Scope
-----
Exercises ``news_service.tasks.discover_sources.run_and_persist_discovery``
end-to-end with a deliberately empty ``World``: no pre-seeded sources, no
web-search corpus, no article-fetch bodies, no polling adapters. The topic
is intentionally absurd ("amateur competitive tiddlywinks tournaments in
Niue") so an LLM cannot reach for plausible priors. Every Finder strategy
returns empty, the candidate pool stays empty, the Discovery Agent aborts,
and the orchestrator maps that to ``status="no_sources_found"``.

The test ALSO calls the Conversational Agent's helper
``_format_discovery_result(result)`` on the returned payload and asserts
that the rendered string is an explicit ERROR prompt -- this is the
contract that tells the LLM to apologize to the user and ask them to
refine the topic instead of claiming the subscription is ready.

Assertions (single behavior: the empty-pool path wires through to a
user-visible ERROR surface)
  * ``status == "no_sources_found"``
  * ``persisted == 0`` and ``discovered == 0``
  * zero ``SubscriptionSource`` rows for the subscription
  * ``_format_discovery_result(result)`` contains both ``ERROR`` and
    ``could not find`` tokens
  * zero ``FailedTask`` rows (tier-1 failure did not sneak in)

Out of scope
------------
Agent wording, exact abort reason text, streaming progress frames, and
the Conversational Agent's routing to ``trigger_source_discovery`` (that
path is covered by ``test_s_conv_trigger_source_discovery.py``).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


@pytest.mark.asyncio
async def test_s_discovery_empty_pool_returns_no_sources_found_and_formats_as_user_error(
    world,
):
    """Empty world -> discovery aborts -> status no_sources_found -> ERROR format."""
    from news_service.agents.conversational.tools import _format_discovery_result
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.tasks.discover_sources import run_and_persist_discovery

    assert not world.search.corpus, (
        "test precondition violated: world.search.corpus must be empty; "
        f"found {len(world.search.corpus)} entries"
    )
    assert not world.adapters, (
        "test precondition violated: world.adapters must be empty; "
        f"found {len(world.adapters)} entries"
    )
    assert not world.article_fetch.bodies, (
        "test precondition violated: world.article_fetch.bodies must be empty; "
        f"found {len(world.article_fetch.bodies)} entries"
    )

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    webhook_url = "https://bench.invalid/webhook/s-discovery-empty-pool"

    user_spec = (
        "# Ultra-niche feed\n\n"
        "I want news strictly about amateur competitive tiddlywinks "
        "tournaments in the South Pacific island nation of Niue. Focus "
        "only on club-level play at Alofi-based venues and official "
        "Niue Tiddlywinks Club fixtures; exclude general board-game "
        "coverage, other Pacific sports, any non-Niuean tiddlywinks "
        "content, and anything not specifically about amateur play.\n"
    )
    retrieval_query = (
        "Niue amateur tiddlywinks tournaments Alofi club play fixtures"
    )
    topic_vector = await embed_text(retrieval_query)

    async with async_session_factory() as s:
        s.add(
            User(
                id=user_id,
                api_key=f"bench-{user_id.hex}",
                language="en",
                timezone="UTC",
                delivery_webhook_url=webhook_url,
                has_onboarded=True,
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                topic_embedding=topic_vector,
                user_spec=user_spec,
                delivery_mode="digest",
                schedule_cron="0 8 * * *",
                digest_language="en",
                delivery_webhook_url=webhook_url,
                is_active=True,
            )
        )
        await s.commit()

    async with async_session_factory() as s:
        result = await run_and_persist_discovery(
            s,
            sub_id,
            reason="initial-discovery-for-test",
        )

    assert result.get("status") == "no_sources_found", (
        "discovery did not surface the empty-pool contract as no_sources_found; "
        f"got status={result.get('status')!r} with payload={result!r}"
    )
    assert result.get("persisted") == 0, (
        "discovery persisted rows despite empty pool; "
        f"persisted={result.get('persisted')!r} in payload={result!r}"
    )
    assert result.get("discovered") == 0, (
        "discovery reported non-zero discovered despite empty pool; "
        f"discovered={result.get('discovered')!r} in payload={result!r}"
    )

    async with async_session_factory() as s:
        link_rows = await s.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == sub_id
            )
        )
        links = list(link_rows.scalars().all())
    assert links == [], (
        "no SubscriptionSource rows should exist for a subscription whose "
        f"discovery found nothing; got {len(links)} row(s): {links!r}"
    )

    formatted = _format_discovery_result(result)
    assert "ERROR" in formatted, (
        "_format_discovery_result must render no_sources_found as an ERROR "
        f"string for the LLM; got: {formatted!r}"
    )
    assert "could not find" in formatted, (
        "_format_discovery_result must tell the LLM the pipeline 'could not "
        f"find' sources so it apologizes to the user; got: {formatted!r}"
    )

    async with async_session_factory() as s:
        failed_rows = await s.execute(select(FailedTask))
        failed = list(failed_rows.scalars().all())
    assert failed == [], (
        "empty-pool discovery must not produce any FailedTask rows; "
        f"got {len(failed)}: {failed!r}"
    )
