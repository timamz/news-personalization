"""
S-reflector-revise-max: garbage items force content comprehension.

The digest pipeline runs the Writer <-> Judge loop up to
``_MAX_REVISIONS=3`` times. If the judge still returns ``REVISE`` after
the last round, the pipeline delivers the unreviewed final draft and
``_compute_reflect_reasons`` includes a "Final digest verdict was
REVISE after max revisions." entry, which feeds the Reflector prompt.

This test keeps the Writer/Judge stubs (double-patched at both import
sites, mirroring ``test_s_digest_revise.py``) because the REVISE-after-max
trigger mechanism is the point of the test -- the Writer always produces
a bad draft and the Judge always returns ``REVISE``, exhausting the
budget deterministically.

The difference from earlier revisions of this test is WHAT IS IN THE
SOURCE. Previously the linked source was seeded with six on-topic EU
policy items and only its metadata line + the REVISE trigger reason
carried any signal. A lazy Reflector could then reach "remove" without
ever inspecting content: the trigger reason itself is suggestive.

To close that loophole, the six items seeded here are deliberate
garbage -- short, hollow placeholders like "More news expected. Check
back later." Headlines are vaguely on-topic so ``source_description``
/ ``source_description_embedding`` (still on-topic) keep the metadata
line looking healthy from the outside. The ONLY place the rot shows up
is inside item bodies, which are reachable exclusively via
``fetch_source_items``. A Reflector that blindly trusts the trigger
reason cannot distinguish this from an innocent source; one that
actually reads content will see the placeholders and have concrete
evidence to justify removal.

Judge scores are also tuned to point at content rather than format:
``relevance=2`` with ``format_score=4`` and ``conciseness=4`` plus
feedback like "Items are hollow placeholders; draft lacks substance"
rules out the format/spec path and leaves the source content as the
clear cause.

A fetch spy wraps ``build_fetch_source_items_tool`` so the test can
assert the Reflector actually issued at least one ``fetch_source_items``
call for the suspect source before removing it.

Assertions cover the cascade end-to-end:

  1. Writer ran >= 3 times (the max-revision budget was exhausted).
  2. Judge ran >= 3 times (REVISE every round).
  3. ``fetch_source_items`` was invoked at least once -- the Reflector
     looked at actual items before deciding, not just the trigger reason.
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
    WEBHOOK_URL,
    SourceSpec,
    install_discovery_stub,
    read_source_removal_log_for,
    read_subscription_sources,
    seed_reflector_world,
)


GARBAGE_ITEMS = [
    {
        "headline": "EU updates today",
        "body": "More news expected. Check back later.",
    },
    {
        "headline": "Energy news coming soon",
        "body": "Details will follow shortly.",
    },
    {
        "headline": "Weekly wrap-up",
        "body": "See our main page for the latest stories.",
    },
    {
        "headline": "Climate policy briefing",
        "body": "Updates to be posted. Stay tuned.",
    },
    {
        "headline": "Renewables bulletin",
        "body": "Watch this space for details.",
    },
    {
        "headline": "Gas and LNG notes",
        "body": "Full article unavailable at this time.",
    },
]


@pytest.mark.asyncio
async def test_s_reflector_revise_after_max_removes_and_rediscovers(world):
    """REVISE every round + garbage items -> reflector inspects, removes, rediscovers."""
    from news_service.agents.digest import judge as judge_mod
    from news_service.agents.digest import pipeline as pipeline_mod
    from news_service.agents.digest import reflector as reflector_mod
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
        items=GARBAGE_ITEMS,
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
        """Always REVISE; feedback fingers source content, not format."""
        nonlocal judge_call_count
        judge_call_count += 1
        if judge_call_count == 1:
            feedback = (
                "Items are hollow placeholders; draft lacks substance. No "
                "specific policy facts appear because the source bodies do "
                "not contain any."
            )
        elif judge_call_count == 2:
            feedback = (
                "Draft still has no concrete content. The underlying items "
                "are stubs like 'More news expected. Check back later.' -- "
                "there is nothing to summarise."
            )
        else:
            feedback = (
                "Items remain empty placeholders rather than real news. "
                "Nothing substantive can be distilled from these bodies."
            )
        return QualityScores(
            relevance=2,
            format_score=4,
            conciseness=4,
            verdict="REVISE",
            feedback=feedback,
        )

    original_pipeline_write = pipeline_mod.write_digest
    original_module_write = writer_mod.write_digest
    original_pipeline_judge = pipeline_mod.judge_digest
    original_module_judge = judge_mod.judge_digest
    original_fetch_builder = reflector_mod.build_fetch_source_items_tool

    pipeline_mod.write_digest = _stub_writer  # type: ignore[assignment]
    writer_mod.write_digest = _stub_writer  # type: ignore[assignment]
    pipeline_mod.judge_digest = _stub_judge  # type: ignore[assignment]
    judge_mod.judge_digest = _stub_judge  # type: ignore[assignment]

    fetch_calls: list[dict] = []

    def _spy_builder(**kwargs):
        real_tool = original_fetch_builder(**kwargs)

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

    reflector_mod.build_fetch_source_items_tool = _spy_builder  # type: ignore[assignment]

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

        assert len(fetch_calls) >= 1, (
            f"expected reflector to call fetch_source_items at least once for "
            f"source {source_single_id} (removal must be grounded in item "
            f"content, not just the REVISE trigger reason), got {fetch_calls!r}"
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
        reflector_mod.build_fetch_source_items_tool = original_fetch_builder  # type: ignore[assignment]
