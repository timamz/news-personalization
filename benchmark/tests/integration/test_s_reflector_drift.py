"""
S-reflector-drift: reflector removes a drifted auto-discovered source.

Exercises the reflector's "source drift" trigger. A source drifts when
its ``source_description_embedding`` has cosine similarity to the
subscription's ``topic_embedding`` below
``reflector_drift_similarity_threshold`` (default 0.3). When the
pipeline detects drift it invokes the Reflector, which is expected to
call ``remove_source`` on the offending auto-discovered source and
leave the on-topic source linked.

Setup: one digest-mode subscription with two linked sources:

  * ``source_on_topic`` -- auto-discovered Brussels EU energy newswire.
    Description text is aligned with the subscription topic and it is
    seeded with the six canonical on-topic items from
    ``_reflector_common.ON_TOPIC_ITEMS`` so the digest pipeline has
    real material to compose from. Aggregate cosine to the sub's topic
    embedding is high -> does NOT trip drift.
  * ``source_drifted`` -- auto-discovered celebrity-gossip feed.
    Description text is deliberately from an unrelated domain so its
    ``source_description_embedding`` sits well below the 0.3 drift
    threshold relative to the sub's ``topic_embedding``. Seeded with
    one off-topic celebrity-gossip item published 6 hours ago so
    staleness does NOT fire either -- the only reason the reflector
    should have to act is drift.

Assertions:

  1. ``_deliver_digest`` returns ``status == "delivered"``.
  2. Exactly one webhook lands on ``WEBHOOK_URL``.
  3. ``SourceRemovalLog`` has exactly one row for the subscription and
     its ``source_url`` equals ``source_drifted.url`` (reflector
     removed the drifted source, not the on-topic one).
  4. ``read_subscription_sources(sub_id)`` returns exactly one row
     whose ``source_id`` equals the on-topic source id (drifted source
     has been unlinked).
  5. No ``FailedTask`` rows -- the non-blocking reflector did not
     surface as a hard failure.

Out of scope: staleness trigger (``test_s_reflector_staleness.py``),
contribution-streak trigger, REVISE-after-max trigger, and whether the
reflector additionally queues a discovery run (captured by the
discovery stub but intentionally not asserted on, because the drift
trigger alone does not commit us to a specific discovery decision).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from tests.integration._reflector_common import (
    ON_TOPIC_ITEMS,
    WEBHOOK_URL,
    SourceSpec,
    install_discovery_stub,
    read_source_removal_log_for,
    read_subscription_sources,
    seed_reflector_world,
)


@pytest.mark.asyncio
async def test_s_reflector_drift_removes_drifted_source(world):
    """Drifted auto-discovered source is unlinked; on-topic source survives."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.tasks.deliver_digest import _deliver_digest

    on_topic_spec = SourceSpec(
        label="source_on_topic",
        url="https://brussels-energy-policy.invalid/feed.xml",
        description_text=(
            "Brussels newswire on EU energy and climate policy, Council "
            "decisions, ACER and ENTSO-E regulatory publications, "
            "EUR-Lex directives, and Parliament committee votes."
        ),
        is_user_specified=False,
        items=ON_TOPIC_ITEMS,
        items_age_hours=6,
    )
    drifted_spec = SourceSpec(
        label="source_drifted",
        url="https://hollywood-gossip.invalid/feed.xml",
        description_text=(
            "Celebrity gossip, reality TV recaps, Hollywood star "
            "sightings, red-carpet fashion commentary, and dating "
            "rumors from the entertainment industry."
        ),
        is_user_specified=False,
        items=[
            {
                "headline": "Hollywood couple announces summer wedding",
                "body": (
                    "A-list actress and her musician boyfriend confirmed "
                    "on social media that they will marry this summer at "
                    "a private ceremony in Malibu. Celebrity friends "
                    "quickly flooded the comments with congratulations, "
                    "and tabloids speculated about the guest list and "
                    "the designer rumored to be making the dress."
                ),
            }
        ],
        items_age_hours=6,
    )

    _user_id, sub_id, source_id_by_label = await seed_reflector_world(
        world,
        embedding_fn=embed_text,
        sources=[on_topic_spec, drifted_spec],
    )
    on_topic_source_id = source_id_by_label["source_on_topic"]

    _discovery_stub = install_discovery_stub(world)

    result = await _deliver_digest(sub_id)
    assert result.get("status") == "delivered", (
        f"expected delivered status, got {result!r}"
    )

    await world.celery.drain()

    captured = world.delivery.for_url(WEBHOOK_URL)
    assert len(captured) == 1, (
        f"expected exactly 1 digest webhook for {WEBHOOK_URL}, got "
        f"{len(captured)}. Bodies: {[c.body[:120] for c in captured]}"
    )

    removal_rows = await read_source_removal_log_for(sub_id)
    assert len(removal_rows) == 1, (
        f"expected exactly 1 SourceRemovalLog row for sub {sub_id}, got "
        f"{len(removal_rows)}: {[(r.source_url, r.reason) for r in removal_rows]!r}"
    )
    assert removal_rows[0].source_url == drifted_spec.url, (
        f"reflector removed the wrong source: expected "
        f"{drifted_spec.url!r}, got {removal_rows[0].source_url!r}"
    )

    surviving = await read_subscription_sources(sub_id)
    assert len(surviving) == 1, (
        f"expected exactly 1 SubscriptionSource row after drift removal, "
        f"got {len(surviving)}: {[r.source_id for r in surviving]!r}"
    )
    assert surviving[0].source_id == on_topic_source_id, (
        f"surviving linked source should be the on-topic one "
        f"({on_topic_source_id}), got {surviving[0].source_id}"
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, (
        f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    )
