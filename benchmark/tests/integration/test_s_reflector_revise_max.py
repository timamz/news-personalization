"""
S-reflector-revise-max: REVISE-after-max-revisions trigger.

The digest pipeline runs the Writer <-> Judge loop up to
``_MAX_REVISIONS=3`` times. If the judge still returns ``REVISE`` after
the last round, the pipeline delivers the unreviewed final draft and
``_compute_reflect_reasons`` includes a "Final digest verdict was
REVISE after max revisions." entry, which feeds the Reflector prompt.

This test forces that trigger deterministically by monkey-patching
BOTH the Writer and the Judge at BOTH import sites (pipeline module
and source module), mirroring the double-patch pattern from
``test_s_digest_revise.py``. The writer stub returns obviously bad
placeholder drafts on every call; the judge stub returns
``verdict="REVISE"`` with varied feedback strings on every call. After
three bad rounds the loop exits with a REVISE verdict.

World setup: one digest-mode subscription + one linked auto-discovered
source with on-topic description text (so the drift trigger does not
fire), six on-topic items at age 6h (so staleness does not fire), and
``digests_since_last_contribution=0`` (so the contribution-streak
trigger does not fire). Only the REVISE-after-max reason should be
present in the Reflector's trigger list.

The real Reflector agent then runs against that single reason and the
single-source context. It is expected to pick the cascade: remove the
lone auto-discovered source (it clearly cannot help the sub produce a
PASS-worthy draft) and trigger source discovery to repopulate the sub.

Assertions cover the cascade end-to-end:

  1. Final draft is delivered (pipeline delivers even on REVISE-after-max).
  2. Writer ran >= 3 times (the max-revision budget was exhausted).
  3. Judge ran >= 3 times (REVISE every round).
  4. Exactly 1 webhook lands (the last draft).
  5. Exactly 1 ``SourceRemovalLog`` row exists for the sub and its
     ``source_url`` matches the seeded source.
  6. ``read_subscription_sources(sub_id)`` is empty (the source is gone).
  7. Discovery stub captured >= 1 call (reflector queued re-discovery
     after emptying the source list).
  8. No ``FailedTask`` rows (tier-3 non-blocking path stayed clean).

Out of scope: real judge behaviour (always forced REVISE here), real
writer output quality (always forced bad drafts), and the other three
reflector triggers (drift / staleness / contribution-streak -- each
has its own sibling test file).
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
async def test_s_reflector_revise_after_max_removes_and_rediscovers(world):
    """REVISE every round -> reflector removes the lone source and queues discovery."""
    from news_service.agents.digest import judge as judge_mod
    from news_service.agents.digest import pipeline as pipeline_mod
    from news_service.agents.digest import writer as writer_mod
    from news_service.agents.digest.judge import QualityScores
    from news_service.agents.digest.writer import DigestComposition
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.news_item import NewsItem
    from news_service.tasks.deliver_digest import _deliver_digest

    source_single = SourceSpec(
        label="source_single",
        url="https://eu-policy-wire.invalid/feed.xml",
        description_text=(
            "EU-level energy and climate policy newswire covering Council "
            "decisions, Commission proposals, ENTSO-E and ACER regulatory "
            "publications, EUR-Lex directives, and Parliament committee "
            "votes on gas storage, renewables, methane and offshore wind."
        ),
        is_user_specified=False,
        items=ON_TOPIC_ITEMS,
        items_age_hours=6,
        digests_since_last_contribution=0,
        contribution_rate=0.5,
        contributed_last_30_digests=5,
    )

    _user_id, sub_id, source_id_by_label = await seed_reflector_world(
        world,
        embedding_fn=embed_text,
        sources=[source_single],
    )
    source_single_id = source_id_by_label["source_single"]

    async with async_session_factory() as s:
        row = (
            await s.execute(
                select(NewsItem).where(NewsItem.source_id == source_single_id).limit(1)
            )
        ).scalar_one()
        one_item_id = row.id

    writer_call_count = 0
    judge_call_count = 0

    async def _stub_writer(**kwargs) -> DigestComposition:
        """Always returns a bad draft; three distinct bad variants across calls."""
        nonlocal writer_call_count
        writer_call_count += 1
        if writer_call_count == 1:
            text = (
                "Here is your daily digest. Several EU policy items occurred "
                "today and may be of interest to you."
            )
        elif writer_call_count == 2:
            text = "EU energy news updates are included below."
        else:
            text = "Digest placeholder."
        return DigestComposition(
            digest_text=text,
            used_item_ids=[str(one_item_id)],
        )

    async def _stub_judge(**kwargs) -> QualityScores:
        """Always REVISE; varied feedback per call to prove threading."""
        nonlocal judge_call_count
        judge_call_count += 1
        if judge_call_count == 1:
            feedback = (
                "Add specific policy citations (directive numbers, Council "
                "session dates) to every item. None present."
            )
        elif judge_call_count == 2:
            feedback = (
                "The draft still has no numerical figures. Include "
                "percentages, dates, or GW capacities from the source bodies."
            )
        else:
            feedback = (
                "Items remain abstract placeholders rather than summaries of "
                "the actual news. Quote distinguishing facts from each body."
            )
        return QualityScores(
            relevance=2,
            format_score=2,
            conciseness=2,
            verdict="REVISE",
            feedback=feedback,
        )

    original_pipeline_write = pipeline_mod.write_digest
    original_module_write = writer_mod.write_digest
    original_pipeline_judge = pipeline_mod.judge_digest
    original_module_judge = judge_mod.judge_digest

    pipeline_mod.write_digest = _stub_writer  # type: ignore[assignment]
    writer_mod.write_digest = _stub_writer  # type: ignore[assignment]
    pipeline_mod.judge_digest = _stub_judge  # type: ignore[assignment]
    judge_mod.judge_digest = _stub_judge  # type: ignore[assignment]

    discovery_stub = install_discovery_stub(world)

    try:
        result = await _deliver_digest(sub_id)
        assert result.get("status") == "delivered", (
            f"expected delivered status (pipeline delivers even on REVISE-after-max), "
            f"got {result!r}"
        )

        await world.celery.drain()

        assert writer_call_count >= 3, (
            f"expected writer invoked at least 3 times (max-revision budget), "
            f"got {writer_call_count}"
        )
        assert judge_call_count >= 3, (
            f"expected judge invoked at least 3 times (REVISE each round), "
            f"got {judge_call_count}"
        )

        captured = world.delivery.for_url(WEBHOOK_URL)
        assert len(captured) == 1, (
            f"expected exactly 1 webhook delivery (the final REVISE draft), got "
            f"{len(captured)}. Bodies: {[c.body[:120] for c in captured]}"
        )

        removal_rows = await read_source_removal_log_for(sub_id)
        assert len(removal_rows) == 1, (
            f"expected exactly 1 SourceRemovalLog row (reflector removes the "
            f"lone auto-discovered source), got {len(removal_rows)}: "
            f"{[(r.source_url, r.reason) for r in removal_rows]}"
        )
        assert removal_rows[0].source_url == source_single.url, (
            f"removal log source_url mismatch: expected {source_single.url!r}, "
            f"got {removal_rows[0].source_url!r}"
        )

        remaining = await read_subscription_sources(sub_id)
        assert remaining == [], (
            f"expected subscription_sources to be empty after reflector removed "
            f"the lone source, got {len(remaining)} rows: "
            f"{[r.source_id for r in remaining]}"
        )

        assert discovery_stub.call_count() >= 1, (
            f"expected reflector to queue source discovery at least once after "
            f"emptying the source list, got {discovery_stub.call_count()} calls"
        )

        async with async_session_factory() as s:
            failed = list((await s.execute(select(FailedTask))).scalars().all())
        assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    finally:
        pipeline_mod.write_digest = original_pipeline_write  # type: ignore[assignment]
        writer_mod.write_digest = original_module_write  # type: ignore[assignment]
        pipeline_mod.judge_digest = original_pipeline_judge  # type: ignore[assignment]
        judge_mod.judge_digest = original_module_judge  # type: ignore[assignment]
