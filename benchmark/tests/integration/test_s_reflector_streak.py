"""
S-reflector-streak: reflector's contribution-streak trigger.

Exercises the reflector path where an auto-discovered source is removed
because it has failed to contribute to the digest for a long stretch of
runs -- ``SubscriptionSource.digests_since_last_contribution`` has grown
past ``reflector_contribution_streak_threshold`` (default 10).

Setup: one digest-mode subscription linked to two auto-discovered
sources.

  * ``source_contributor``: description aligned with the sub topic, six
    fresh on-topic EU energy-policy items (age 6h),
    ``digests_since_last_contribution = 0``, ``contribution_rate = 0.5``.
    This is the healthy source and must survive.

  * ``source_silent_contributor``: description is plausibly on-topic
    (an EU energy-industry periodical) so the drift trigger does NOT
    fire. One item aged 5h so the staleness trigger does NOT fire
    either. But the item's content is a narrowly-specialised
    mineral-oil-tax clarification the writer is very unlikely to pick,
    and we pre-seed ``digests_since_last_contribution = 12`` (>=
    threshold 10) with ``contribution_rate = 0.0`` and
    ``contributed_last_30_digests = 0``. That is enough -- on its own --
    to make ``_compute_reflect_reasons`` emit a streak reason and hand
    this source to the reflector.

The test asserts:

  * ``_deliver_digest`` returns ``status == "delivered"``;
  * exactly one webhook landed on ``WEBHOOK_URL``;
  * exactly one ``SourceRemovalLog`` row exists and its ``source_url``
    equals the silent-contributor URL (contributor is never removed);
  * exactly one ``SubscriptionSource`` row remains on the sub and it is
    the contributor's source id;
  * no ``FailedTask`` rows were written.

Out of scope: drift trigger (covered by its own reflector test),
staleness trigger (same), REVISE-after-max trigger (same), the
discovery-queue dispatch path (the stub only captures calls; this test
does not assert on it because a streak on a single non-contributing
source does not by itself require queuing a replacement search).
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
async def test_s_reflector_streak_removes_non_contributor(world):
    """Silent source with 12-digest dry streak is removed; healthy contributor survives."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.tasks.deliver_digest import _deliver_digest

    contributor = SourceSpec(
        label="source_contributor",
        url="https://eu-energy-contributor.invalid/feed.xml",
        description_text=(
            "Daily EU energy and climate policy newswire covering Council "
            "decisions, Commission proposals, ENTSO-E and ACER regulatory "
            "publications, EUR-Lex directives, and Parliament ENVI / ITRE "
            "committee votes."
        ),
        is_user_specified=False,
        items=ON_TOPIC_ITEMS,
        items_age_hours=6,
        digests_since_last_contribution=0,
        contribution_rate=0.5,
        contributed_last_30_digests=15,
    )

    silent_contributor = SourceSpec(
        label="source_silent_contributor",
        url="https://eu-energy-silent.invalid/feed.xml",
        description_text=(
            "European energy industry periodical covering policy "
            "developments, regulatory updates, and legislative activity "
            "across the EU-27 energy sector."
        ),
        is_user_specified=False,
        items=[
            {
                "headline": (
                    "EU mineral-oil-tax committee issues clarification on "
                    "diesel excise refunds"
                ),
                "body": (
                    "The Mineral Oil Tax Coordination Committee issued a "
                    "technical clarification on Article 17(3) of the Energy "
                    "Taxation Directive concerning diesel excise refund "
                    "eligibility for commercial hauliers registered in "
                    "multiple Member States. The notice restates the "
                    "calculation basis for partial refunds where the "
                    "vehicle's fuel was purchased in a different Member "
                    "State from the operator's VAT domicile, and reiterates "
                    "that the reference period for reconciliation remains "
                    "the calendar quarter. The committee confirmed that the "
                    "revised Annex II coefficients apply retroactively to "
                    "claims filed after 1 January. No change to the "
                    "underlying directive text is proposed. National tax "
                    "authorities are asked to update their administrative "
                    "guidance accordingly."
                ),
            }
        ],
        items_age_hours=5,
        digests_since_last_contribution=12,
        contribution_rate=0.0,
        contributed_last_30_digests=0,
    )

    _user_id, sub_id, source_id_by_label = await seed_reflector_world(
        world,
        embedding_fn=embed_text,
        sources=[contributor, silent_contributor],
    )

    install_discovery_stub(world)

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
        f"expected exactly 1 SourceRemovalLog row (the silent contributor), "
        f"got {len(removal_rows)}: "
        f"{[(r.source_url, r.reason) for r in removal_rows]!r}"
    )
    assert removal_rows[0].source_url == silent_contributor.url, (
        f"expected SourceRemovalLog.source_url == {silent_contributor.url!r} "
        f"(the silent contributor), got {removal_rows[0].source_url!r}. "
        f"The reflector removed the wrong source."
    )

    remaining = await read_subscription_sources(sub_id)
    assert len(remaining) == 1, (
        f"expected exactly 1 SubscriptionSource row to remain on the sub "
        f"after reflector ran, got {len(remaining)}: "
        f"{[r.source_id for r in remaining]!r}"
    )
    assert remaining[0].source_id == source_id_by_label[contributor.label], (
        f"expected the remaining SubscriptionSource to be the contributor "
        f"({source_id_by_label[contributor.label]!r}), got "
        f"{remaining[0].source_id!r}. Reflector removed the healthy source."
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
