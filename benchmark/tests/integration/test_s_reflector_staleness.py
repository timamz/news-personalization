"""
S-reflector-staleness: reflector's per-source staleness trigger.

Seeds one digest-mode subscription linked to two auto-discovered
sources:

  * ``source_fresh`` -- a genuinely active EU energy-policy newswire
    with six on-topic items all published six hours ago. This source
    carries the happy digest path: the writer draws from it to produce
    a real draft so the pipeline still delivers.
  * ``source_stale`` -- a once-on-topic EU energy archive with a single
    item published 45 days ago (> 30-day staleness threshold). The
    item exists so ``max(published_at) for source`` resolves to 45
    days ago rather than NULL (a source with no items at all is
    skipped by ``_compute_reflect_reasons``).

After ``_deliver_digest`` runs the full pipeline -- writer, judge, and
then the reflector (invoked because the staleness trigger fires) --
assertions prove:

  1. The digest still delivered (``result["status"] == "delivered"``)
     and exactly one webhook landed on the subscription's URL. The
     happy path is not blocked by the reflector's verdict on one side
     source.
  2. Exactly one ``SourceRemovalLog`` row was written and its
     ``source_url`` is the stale source's URL -- i.e. the reflector
     targeted the correct source, not the fresh one.
  3. The ``subscription_sources`` table now holds exactly one row for
     this sub, and that row points at the fresh source id -- the stale
     link was unlinked cleanly.
  4. No ``FailedTask`` rows -- the pipeline finished without exceptions.

Out of scope: the drift trigger (covered separately), the
contribution-streak trigger (covered separately), the
REVISE-after-max-revisions trigger (covered separately), and the
reflector's optional ``trigger_source_discovery`` dispatch (the
discovery stub is installed to capture it but whether it fires is not
asserted here -- a confirmed staleness removal alone is sufficient to
prove the trigger wired up).
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
async def test_s_reflector_staleness_removes_stale_source(world):
    """Stale auto-discovered source (45d > 30d threshold) is removed; fresh source survives."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.tasks.deliver_digest import _deliver_digest

    source_fresh = SourceSpec(
        label="source_fresh",
        url="https://brussels-energy-wire.invalid/feed.xml",
        description_text=(
            "Brussels-based newswire on EU energy policy, Council decisions "
            "and Commission proposals. Daily coverage of ENTSO-E and ACER "
            "publications, EUR-Lex directives, and ENVI / ITRE committee "
            "votes."
        ),
        is_user_specified=False,
        items=list(ON_TOPIC_ITEMS),
        items_age_hours=6,
    )
    source_stale = SourceSpec(
        label="source_stale",
        url="https://eu-energy-review-archive.invalid/feed.xml",
        description_text=(
            "EU Energy Review quarterly archive: once-on-topic long-form "
            "analysis of EU electricity and gas policy, Commission white "
            "papers, and Council conclusions."
        ),
        is_user_specified=False,
        items=[
            {
                "headline": "EU electricity market review 2023 annual edition",
                "body": (
                    "The EU Energy Review's 2023 annual edition recaps the "
                    "year's Council conclusions, Commission white papers and "
                    "ACER publications across the electricity and gas files. "
                    "Retrospective long-form analysis with no new primary "
                    "reporting."
                ),
            },
        ],
        items_age_hours=45 * 24,
    )

    _user_id, sub_id, source_id_by_label = await seed_reflector_world(
        world,
        embedding_fn=embed_text,
        sources=[source_fresh, source_stale],
    )

    discovery_stub = install_discovery_stub(world)
    _ = discovery_stub  # captured for completeness; not asserted in this test

    result = await _deliver_digest(sub_id)
    assert result.get("status") == "delivered", (
        f"expected delivered status, got {result!r}"
    )

    await world.celery.drain()

    captured = world.delivery.for_url(WEBHOOK_URL)
    assert len(captured) == 1, (
        f"expected exactly 1 digest webhook for {WEBHOOK_URL}, got {len(captured)}. "
        f"Bodies: {[c.body[:120] for c in captured]}"
    )

    removal_rows = await read_source_removal_log_for(sub_id)
    assert len(removal_rows) == 1, (
        f"expected exactly 1 SourceRemovalLog row for sub {sub_id}, got "
        f"{len(removal_rows)}: {[(r.source_url, r.reason) for r in removal_rows]!r}"
    )
    assert removal_rows[0].source_url == source_stale.url, (
        f"expected the stale source ({source_stale.url}) to be removed, got "
        f"{removal_rows[0].source_url!r}"
    )

    remaining_links = await read_subscription_sources(sub_id)
    assert len(remaining_links) == 1, (
        f"expected exactly 1 SubscriptionSource row remaining after staleness "
        f"removal, got {len(remaining_links)}: "
        f"{[r.source_id for r in remaining_links]!r}"
    )
    fresh_source_id = source_id_by_label[source_fresh.label]
    assert remaining_links[0].source_id == fresh_source_id, (
        f"expected the fresh source ({fresh_source_id}) to be the sole "
        f"surviving link, got {remaining_links[0].source_id!r}"
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
