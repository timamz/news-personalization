"""
S-reflector-drift: reflector must *comprehend* fetched content to act.

Exercises the reflector under an adversarial setup where **the
per-source metadata lies**. The drifted source is dressed up so that
its ``source_description`` reads in on-topic EU energy-policy language,
which means ``source_description_embedding`` hugs the subscription's
``topic_embedding`` and the metadata line shows ``cos=high`` -- so the
drift trigger (which fires only when ``cosine_to_topic < 0.3``) does
NOT fire on this source. The only trigger that actually invokes the
reflector here is the **contribution-streak trigger**: the drifted
source is seeded with ``digests_since_last_contribution=12``, above
the threshold of 10, so ``_compute_reflect_reasons`` emits a streak
reason and the pipeline runs the reflector. The per-source metadata
line the reflector then sees is intentionally contradictory: the
description looks on-topic (``cos=high``) yet the streak says "the
writer has been skipping this source for 12 digests". The drift is
only visible inside the recent *items*, whose bodies are celebrity
gossip / reality TV / fashion -- a domain unrelated to EU energy
policy.

Implication: the Reflector cannot legitimately remove this source from
the contradictory metadata alone. The reflector prompt explicitly
guides the agent that trigger reasons can be misleading, and that when
the streak is high but the description still reads on-topic it must
call ``fetch_source_items`` to resolve the contradiction. Correct
removal is only reachable if the agent fetches, reads the returned
item bodies, comprehends that the content is off-topic, and then
decides to remove. The test installs a spy around
``build_fetch_source_items_tool`` to prove the fetch happened, and
asserts that the drifted source is the one that got unlinked and
logged.

Setup: one digest-mode subscription with two linked sources.

  * ``source_healthy`` -- auto-discovered Brussels EU energy newswire.
    Description text is on-topic. Seeded with the six canonical
    on-topic items from ``_reflector_common.ON_TOPIC_ITEMS`` so the
    digest pipeline has real material to compose from. All
    contribution counters are left at defaults (clean streak).
  * ``source_drifted`` -- auto-discovered source whose
    ``source_description`` is deliberately written in on-topic EU
    energy-policy language (so its description embedding hugs the
    topic vector and the metadata line shows ``cos=high``; drift
    trigger does NOT fire). Recent items, however, are celebrity
    gossip / reality TV / fashion -- inlined here, NOT drawn from
    ``ON_TOPIC_ITEMS``. Published 6 hours ago so staleness does not
    fire. ``digests_since_last_contribution=12``,
    ``contribution_rate=0.0`` and ``contributed_last_30_digests=0``
    so that the contribution-streak trigger fires and the reflector
    is invoked in the first place.

Assertions:

  1. ``_deliver_digest`` returns ``status == "delivered"``.
  2. Exactly one webhook lands on ``WEBHOOK_URL``.
  3. The fetch spy recorded at least one
     ``fetch_source_items`` call -- given the on-topic description
     the reflector could not have removed the drifted source on
     metadata alone; it had to read the items to resolve the
     streak-vs-description contradiction.
  4. ``SourceRemovalLog`` has exactly one row for the subscription and
     its ``source_url`` equals ``source_drifted.url`` (reflector
     removed the source whose items were off-topic, not the healthy
     one).
  5. ``read_subscription_sources(sub_id)`` returns exactly one row
     whose ``source_id`` equals the healthy source id (drifted source
     has been unlinked).
  6. No ``FailedTask`` rows -- the non-blocking reflector did not
     surface as a hard failure.

Out of scope: staleness trigger (``test_s_reflector_staleness.py``),
REVISE-after-max trigger, and whether the reflector additionally
queues a discovery run (captured by the discovery stub but
intentionally not asserted on, because the streak trigger alone does
not commit us to a specific discovery decision).
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
    """Reflector must fetch and comprehend items to remove the drifted source."""
    from news_service.agents.digest import reflector as reflector_mod
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.tasks.deliver_digest import _deliver_digest

    fetch_calls: list[dict[str, object]] = []
    original_builder = reflector_mod.build_fetch_source_items_tool

    def _spy_builder(**kwargs):
        real_tool = original_builder(**kwargs)

        async def _spy(source_id, since_days_ago=14, limit=10):
            fetch_calls.append(
                {
                    "source_id": source_id,
                    "since_days_ago": since_days_ago,
                    "limit": limit,
                }
            )
            return await real_tool(source_id, since_days_ago, limit)

        _spy.__name__ = real_tool.__name__
        _spy.__qualname__ = real_tool.__qualname__
        _spy.__doc__ = real_tool.__doc__
        return _spy

    reflector_mod.build_fetch_source_items_tool = _spy_builder
    try:
        healthy_spec = SourceSpec(
            label="source_healthy",
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
            url="https://eu-policy-wire.invalid/feed.xml",
            description_text=(
                "Independent policy wire covering EU energy and climate "
                "regulation: Council of the European Union decisions, "
                "European Commission proposals, ENTSO-E and ACER network "
                "codes, EUR-Lex directives on renewables and methane, and "
                "ENVI / ITRE committee votes on gas storage, LNG, offshore "
                "wind, and methane regulation."
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
                },
                {
                    "headline": "Reality TV villa finale draws record viewership",
                    "body": (
                        "The season finale of the hit reality dating show "
                        "pulled in a record live audience last night. "
                        "Viewers crashed the streaming app in the final "
                        "rose ceremony, and social media lit up with memes "
                        "about the runner-up couple's dramatic exit."
                    ),
                },
                {
                    "headline": "Milan Fashion Week: bold shoulders return",
                    "body": (
                        "Runways at Milan Fashion Week signaled the return "
                        "of exaggerated shoulders and metallic fabrics. "
                        "Front-row celebrities were photographed in "
                        "ruffled gowns, and street-style bloggers hailed "
                        "the comeback of 1980s silhouettes."
                    ),
                },
                {
                    "headline": "Pop star drops surprise album at midnight",
                    "body": (
                        "A chart-topping pop singer released a surprise "
                        "twelve-track album overnight, triggering a frenzy "
                        "across streaming platforms. Fans dissected cryptic "
                        "lyrics for hints about an alleged celebrity feud, "
                        "and the lead single is already trending globally."
                    ),
                },
                {
                    "headline": "Blockbuster sequel breaks opening-weekend record",
                    "body": (
                        "The latest entry in a long-running superhero "
                        "franchise broke opening-weekend records at the "
                        "domestic box office. Studio executives credited "
                        "the returning cast and the viral marketing "
                        "campaign built around the lead actor's red-carpet "
                        "appearances."
                    ),
                },
                {
                    "headline": "Royal couple seen at charity gala in London",
                    "body": (
                        "A senior royal couple attended a charity gala in "
                        "central London last night, posing for photos with "
                        "Hollywood guests and sports stars. Tabloid "
                        "coverage focused on the duchess's gown and a "
                        "rumored feud with a former palace aide."
                    ),
                },
            ],
            items_age_hours=6,
            digests_since_last_contribution=12,
            contribution_rate=0.0,
            contributed_last_30_digests=0,
        )

        _user_id, sub_id, source_id_by_label = await seed_reflector_world(
            world,
            embedding_fn=embed_text,
            sources=[healthy_spec, drifted_spec],
        )
        healthy_source_id = source_id_by_label["source_healthy"]
        drifted_source_id = source_id_by_label["source_drifted"]

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

        assert len(fetch_calls) >= 1, (
            f"expected the reflector to call fetch_source_items at least once "
            f"(drifted source_id={drifted_source_id}) before removing; metadata "
            f"alone does not reveal the drift because the description is "
            f"on-topic. fetch_calls={fetch_calls!r}"
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
        assert surviving[0].source_id == healthy_source_id, (
            f"surviving linked source should be the healthy on-topic one "
            f"({healthy_source_id}), got {surviving[0].source_id}"
        )

        async with async_session_factory() as s:
            failed = list((await s.execute(select(FailedTask))).scalars().all())
        assert not failed, (
            f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
        )
    finally:
        reflector_mod.build_fetch_source_items_tool = original_builder
