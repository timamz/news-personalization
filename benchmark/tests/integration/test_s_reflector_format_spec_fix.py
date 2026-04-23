"""
S-reflector-format-spec-fix: REVISE-after-max when the problem is FORMAT, not content.

This test complements ``test_s_reflector_revise_max.py``. That sibling
test covers the bad-content removal path: the Judge scores every
dimension low, so the only plausible fix is to drop the sole
auto-discovered source and re-discover. The test below covers the
opposite half of the same trigger -- the content is fine but the
Writer ignores the ``user_spec``'s strict format rules on every
revision round. In that case the correct self-heal is NOT to delete
a perfectly healthy source; it is to tighten the ``user_spec``'s
format section so the next digest run has clearer instructions.

The new Reflector tool exercised here is
``update_user_spec(new_spec, reason)``, which rewrites the
subscription's ``user_spec`` in place for format/presentation only.

Scenario:
    * One digest-mode subscription with an inline custom ``user_spec``
      whose FORMAT section is strict and unambiguous: exactly 5
      numbered short paragraphs, plain prose, no bullets, no markdown
      headers, no bold markers, no emoji.
    * One auto-discovered source linked to the subscription, seeded
      with healthy ``ON_TOPIC_ITEMS`` so the content pool is fine.
    * The Writer stub violates the format contract three different
      ways across three successive calls (8-item hyphen bullet list,
      markdown headers, one giant paragraph).
    * The Judge stub scores ``relevance=5, format_score=2,
      conciseness=5`` and returns ``verdict="REVISE"`` with feedback
      that explicitly names the format violation on every round.
    * Max revisions are exhausted -> ``_compute_reflect_reasons``
      emits the REVISE-after-max reason -> the real Reflector agent
      runs against a healthy single-source context plus a clearly
      format-dominated score signal.

Assertions:
    1. Pipeline delivered the final draft (status=="delivered").
    2. Writer ran >= 3 times (max revision budget exhausted).
    3. Judge ran >= 3 times (REVISE every round).
    4. Exactly 1 webhook delivery captured (the final REVISE draft).
    5. The Subscription's ``user_spec`` was mutated -- proves the
       Reflector called ``update_user_spec``.
    6. No ``SourceRemovalLog`` row -- the healthy source was NOT
       removed even though a REVISE-after-max trigger fired.
    7. ``subscription_sources`` still has 1 row -- the link survived.
    8. Discovery stub was never invoked -- no wasted discovery run.
    9. No ``FailedTask`` rows (tier-3 non-blocking path stayed clean).

Together (5) + (6) + (7) + (8) prove the Reflector picked the right
branch: amend presentation rules, do not churn the source pool.

Out of scope: real Writer or Judge behaviour (both stubbed), the
three other reflector triggers, and the bad-content removal cascade
(covered by ``test_s_reflector_revise_max.py``).
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from tests.integration._reflector_common import (
    ON_TOPIC_ITEMS,
    SUB_RETRIEVAL_QUERY,
    WEBHOOK_URL,
    install_discovery_stub,
    read_source_removal_log_for,
    read_subscription_sources,
)

STRICT_FORMAT_USER_SPEC = (
    "# EU energy policy digest\n"
    "\n"
    "Topic: daily EU energy and climate policy news: Council decisions, "
    "Commission proposals, ENTSO-E/ACER publications, EUR-Lex directives, "
    "Parliament committee votes.\n"
    "\n"
    "FORMAT (STRICT):\n"
    "- Exactly 5 items, numbered 1. to 5.\n"
    "- Each item must be a short paragraph of 2-3 sentences, plain prose.\n"
    "- No bullet points, no markdown headers, no ** bold markers.\n"
    "- Plain English, no emoji.\n"
)


@pytest.mark.asyncio
async def test_s_reflector_format_violation_rewrites_user_spec_without_removing_source(world):
    """Format-only REVISE-after-max -> reflector updates user_spec, keeps the source."""
    from news_service.agents.digest import judge as judge_mod
    from news_service.agents.digest import pipeline as pipeline_mod
    from news_service.agents.digest import writer as writer_mod
    from news_service.agents.digest.judge import QualityScores
    from news_service.agents.digest.writer import DigestComposition
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.news_item import NewsItem
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.tasks.deliver_digest import _deliver_digest

    from news_benchmark.clock import CLOCK

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    source_id = uuid.uuid4()
    source_url = "https://eu-policy-wire.invalid/feed.xml"
    source_description = (
        "EU-level energy and climate policy newswire covering Council "
        "decisions, Commission proposals, ENTSO-E and ACER regulatory "
        "publications, EUR-Lex directives, and Parliament committee "
        "votes on gas storage, renewables, methane and offshore wind."
    )

    topic_embedding = await embed_text(SUB_RETRIEVAL_QUERY)
    description_embedding = await embed_text(source_description)

    async with async_session_factory() as s:
        s.add(
            User(
                id=user_id,
                api_key=f"bench-{user_id.hex}",
                language="en",
                timezone="UTC",
                delivery_webhook_url=WEBHOOK_URL,
                has_onboarded=True,
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=STRICT_FORMAT_USER_SPEC,
                delivery_mode="digest",
                schedule_cron="0 8 * * *",
                digest_language="en",
                delivery_webhook_url=WEBHOOK_URL,
                topic_embedding=topic_embedding,
                is_active=True,
            )
        )
        s.add(
            Source(
                id=source_id,
                url=source_url,
                title="eu_policy_wire",
                source_description=source_description[:500],
                source_description_embedding=description_embedding,
                subscriber_count=1,
            )
        )
        s.add(
            SubscriptionSource(
                subscription_id=sub_id,
                source_id=source_id,
                is_user_specified=False,
                digests_since_last_contribution=0,
                contribution_rate=0.5,
                contributed_last_30_digests=5,
            )
        )

        now = CLOCK.now()
        for idx, item in enumerate(ON_TOPIC_ITEMS):
            item_body = item["body"]
            item_embedding = await embed_text(item["headline"] + "\n" + item_body[:400])
            s.add(
                NewsItem(
                    id=uuid.uuid4(),
                    source_id=source_id,
                    headline=item["headline"],
                    body=item_body,
                    url=f"{source_url.rstrip('/')}/item-{idx:02d}",
                    source="eu_policy_wire",
                    published_at=now - timedelta(hours=6),
                    fetched_at=now,
                    embedding=item_embedding,
                )
            )

        await s.commit()

    async with async_session_factory() as s:
        original_sub = (
            await s.execute(select(Subscription).where(Subscription.id == sub_id))
        ).scalar_one()
        original_spec = original_sub.user_spec

        row = (
            await s.execute(select(NewsItem).where(NewsItem.source_id == source_id).limit(1))
        ).scalar_one()
        one_item_id = row.id

    writer_call_count = 0
    judge_call_count = 0

    async def _stub_writer(**kwargs) -> DigestComposition:
        """Always violates the STRICT FORMAT rules; three distinct violations."""
        nonlocal writer_call_count
        writer_call_count += 1
        if writer_call_count == 1:
            text = (
                "- Council adopts gas storage directive for 2027 winter.\n"
                "- Commission proposes 40% renewable electricity target by 2030.\n"
                "- ACER publishes final balancing network code.\n"
                "- ENTSO-E warns of Central European winter capacity shortfall.\n"
                "- EUR-Lex publishes accelerated offshore wind permitting directive.\n"
                "- ENVI tightens methane-leak limits for imported LNG.\n"
                "- Parliament committee schedules renewables vote next week.\n"
                "- Member States to transpose new code within six months.\n"
            )
        elif writer_call_count == 2:
            text = (
                "# Item 1\nCouncil adopts gas storage directive.\n\n"
                "# Item 2\nCommission proposes 40% renewable electricity target.\n\n"
                "# Item 3\nACER publishes balancing network code.\n\n"
                "# Item 4\nENTSO-E warns of winter capacity shortfall.\n\n"
                "# Item 5\nEUR-Lex offshore wind permitting directive.\n"
            )
        else:
            text = (
                "Council adopts gas storage directive for 2027 winter while the "
                "Commission proposes a 40% renewable electricity target by 2030 "
                "and ACER publishes the final balancing network code as ENTSO-E "
                "warns of Central European winter capacity shortfall and EUR-Lex "
                "publishes an accelerated offshore wind permitting directive and "
                "ENVI tightens methane-leak limits for imported LNG."
            )
        return DigestComposition(
            digest_text=text,
            used_item_ids=[str(one_item_id)],
        )

    async def _stub_judge(**kwargs) -> QualityScores:
        """Relevance high, format low, REVISE every round with format-specific feedback."""
        nonlocal judge_call_count
        judge_call_count += 1
        if judge_call_count == 1:
            feedback = (
                "Draft uses hyphen bullet points and has 8 items. Spec requires "
                "exactly 5 items numbered 1. to 5. as short paragraphs, no bullets."
            )
        elif judge_call_count == 2:
            feedback = (
                "Draft uses markdown '# Item N' headers. Spec forbids markdown "
                "headers and requires numbered short paragraphs only."
            )
        else:
            feedback = (
                "Draft is a single long paragraph with no item separators. "
                "Spec requires 5 distinct numbered items, one short paragraph each."
            )
        return QualityScores(
            relevance=5,
            format_score=2,
            conciseness=5,
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
            f"expected judge invoked at least 3 times (REVISE each round), got {judge_call_count}"
        )

        captured = world.delivery.for_url(WEBHOOK_URL)
        assert len(captured) == 1, (
            f"expected exactly 1 webhook delivery (the final REVISE draft), got "
            f"{len(captured)}. Bodies: {[c.body[:120] for c in captured]}"
        )

        async with async_session_factory() as s:
            refreshed = (
                await s.execute(select(Subscription).where(Subscription.id == sub_id))
            ).scalar_one()
            new_spec = refreshed.user_spec

        assert new_spec != original_spec, (
            f"expected user_spec to be rewritten by reflector's update_user_spec "
            f"tool (format-only REVISE requires a spec tightening, not a source "
            f"churn). Original spec was:\n{original_spec!r}\n\nCurrent spec is:"
            f"\n{new_spec!r}"
        )

        removal_rows = await read_source_removal_log_for(sub_id)
        assert removal_rows == [], (
            f"expected 0 SourceRemovalLog rows (the source is healthy; format "
            f"violations must not cause source removal), got {len(removal_rows)}: "
            f"{[(r.source_url, r.removal_reason) for r in removal_rows]}"
        )

        remaining = await read_subscription_sources(sub_id)
        assert len(remaining) == 1, (
            f"expected subscription_sources to still have 1 row (healthy source "
            f"survives a format-only REVISE), got {len(remaining)} rows: "
            f"{[r.source_id for r in remaining]}"
        )

        assert discovery_stub.call_count() == 0, (
            f"expected 0 discovery calls (format-only REVISE must NOT trigger "
            f"source discovery), got {discovery_stub.call_count()} calls: "
            f"{discovery_stub.calls!r}"
        )

        async with async_session_factory() as s:
            failed = list((await s.execute(select(FailedTask))).scalars().all())
        assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    finally:
        pipeline_mod.write_digest = original_pipeline_write  # type: ignore[assignment]
        writer_mod.write_digest = original_module_write  # type: ignore[assignment]
        pipeline_mod.judge_digest = original_pipeline_judge  # type: ignore[assignment]
        judge_mod.judge_digest = original_module_judge  # type: ignore[assignment]
