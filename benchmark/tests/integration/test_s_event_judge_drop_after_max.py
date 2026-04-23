"""
S-event-judge-drop-after-max: drop-after-max branch of ``_judge_and_revise``.

The event pipeline's Generator/Critic loop in
``news_service.tasks.deliver_events._judge_and_revise`` is bounded. It runs
at most ``event_judge_max_revisions + 1`` judge rounds (default 2 + 1 = 3).
If an item is still flagged REVISE by the final round, its id is returned
in the ``revise_ids`` set and ``_assess_and_deliver_for_subscription``
SKIPS its delivery. Other items that PASS are delivered normally.

This test proves:
  * the loop honours the exact bounded round count (no infinite loop, no
    premature exit),
  * a stubbornly-defective item is dropped rather than force-delivered,
  * clean PASS items alongside it are unaffected and still deliver,
  * drop-after-max is a normal control path, not a failure -- 0
    ``FailedTask`` rows are recorded.

Strategy: stub BOTH the assessor and the judge for full determinism.
Every round the assessor returns the same three relevant items with the
same notification bodies (so the judge's "still broken" verdict is
defensible); every round the judge PASSes two items and REVISEs the
third. The third is the "defective" one that should end up dropped.

Out of scope: real assessor/judge LLM behaviour (both stubbed), the
overall PASS early-exit path, and the unreviewed-fallthrough path on
judge exception. Those live in other tests.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem


SOURCE_URL = "https://example-nova-metals.invalid/feed.xml"
WEBHOOK_URL = "https://bench.invalid/webhook/s-event-judge-drop-after-max"

USER_SPEC = (
    "# Lithium supply-chain alerts\n"
    "\n"
    "Notify me instantly when news breaks about lithium mining, lithium "
    "refining, battery-grade lithium carbonate or hydroxide pricing, or "
    "regulatory actions affecting the lithium supply chain (mining permits, "
    "export restrictions, royalty changes, pricing floors).\n"
    "\n"
    "Do NOT notify me about: electric-vehicle sales or delivery numbers, "
    "Tesla stock moves, downstream battery-cell or pack news, or other "
    "metals (copper, nickel, cobalt) unless the story is explicitly about "
    "their interaction with lithium supply.\n"
)


@pytest.mark.asyncio
async def test_s_event_judge_drop_after_max_revisions(world):
    """Stubbornly-REVISE item is dropped after the bounded loop; PASS items deliver."""
    from news_service.agents.event import batch_assessor as assessor_mod
    from news_service.agents.event import judge as judge_mod
    from news_service.agents.event.batch_assessor import (
        BatchAssessmentResult,
        ItemAssessment,
    )
    from news_service.agents.event.judge import BatchJudgeResult, ItemVerdict
    from news_service.db.session import async_session_factory
    from news_service.models.failed_task import FailedTask
    from news_service.models.sent_item import SentItem
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.tasks import deliver_events as deliver_events_mod

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    source_id = uuid.uuid4()

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
            Source(
                id=source_id,
                url=SOURCE_URL,
                title="Nova Metals Daily",
                source_description="Metals supply-chain newswire.",
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=USER_SPEC,
                delivery_mode="event",
                digest_language="en",
                delivery_webhook_url=WEBHOOK_URL,
                is_active=True,
            )
        )
        s.add(
            SubscriptionSource(
                subscription_id=sub_id,
                source_id=source_id,
                is_user_specified=True,
            )
        )
        await s.commit()

    now = CLOCK.now()
    items = [
        ScenarioItem(
            fake_ts=now - timedelta(hours=2),
            source_url=SOURCE_URL,
            headline="Chile ministry unveils lithium pricing floor for 2026 contracts",
            body=(
                "Chile's economy ministry published draft regulations that set a "
                "minimum floor price for battery-grade lithium carbonate sold "
                "under long-term contracts starting January 2026. Consultations "
                "close in May; SQM and Albemarle are reviewing the draft."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=4),
            source_url=SOURCE_URL,
            headline="Albemarle pauses Kemerton lithium hydroxide expansion",
            body=(
                "Albemarle Corp said on Monday it will pause the third-train "
                "expansion of its Kemerton lithium hydroxide refinery in Western "
                "Australia, citing weaker-than-expected demand through 2026."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=6),
            source_url=SOURCE_URL,
            headline="Argentina publishes revised lithium export royalty schedule",
            body=(
                "Argentina's mining ministry released a revised royalty schedule "
                "for lithium exports on Tuesday, raising the headline rate from "
                "3% to 4.5% for brine operations commissioned after 2027."
            ),
        ),
    ]

    world.adapters[SOURCE_URL] = FakeAdapter(source_url=SOURCE_URL, items=items)

    defective_body = (
        "**Argentina lithium royalty** hike announced -- details at "
        "https://example-nova-metals.invalid/argentina"
    )
    defective_feedback = "Body uses markdown bold -- rewrite in plain text."

    assessor_call_count = 0
    defective_item_id_holder: dict[str, str] = {}

    async def _stub_assess_batch_events(
        *,
        items: list[dict],
        user_spec: str,
        target_language: str,
        recent_notification_history: list[str],
        max_history_chars: int,
        critic_feedback_per_item: dict[str, str] | None = None,
    ) -> BatchAssessmentResult:
        """Return deterministic relevant assessments for whatever items arrive.

        Three relevant items on the initial call; the same defective body
        for the one item that gets re-submitted on each revision turn, so
        the judge can keep REVISEing it.
        """
        nonlocal assessor_call_count
        assessor_call_count += 1

        if critic_feedback_per_item is None:
            assessments = [
                ItemAssessment(
                    item_id=it["item_id"],
                    is_relevant=True,
                    notification_body=(
                        defective_body
                        if i == 2
                        else (
                            f"Lithium alert #{i + 1}: {it['headline']} -- "
                            f"{it['url']}"
                        )
                    ),
                    reason=f"Matches subscription spec item {i + 1}.",
                )
                for i, it in enumerate(items)
            ]
            defective_item_id_holder["id"] = items[2]["item_id"]
            return BatchAssessmentResult(assessments=assessments)

        return BatchAssessmentResult(
            assessments=[
                ItemAssessment(
                    item_id=it["item_id"],
                    is_relevant=True,
                    notification_body=defective_body,
                    reason="Re-submitted for revision but unchanged on purpose.",
                )
                for it in items
            ]
        )

    judge_call_count = 0

    async def _stub_judge_batch_events(
        *,
        assessment: BatchAssessmentResult,
        user_spec: str,
        recent_notification_history: list[str],
        max_history_chars: int,
    ) -> BatchJudgeResult:
        """PASS all items except the defective one; defective stays REVISE forever."""
        nonlocal judge_call_count
        judge_call_count += 1

        defective_id = defective_item_id_holder.get("id")
        per_item = [
            ItemVerdict(
                item_id=a.item_id,
                verdict="REVISE" if a.item_id == defective_id else "PASS",
                feedback=defective_feedback if a.item_id == defective_id else "",
            )
            for a in assessment.assessments
        ]
        overall = "REVISE" if any(v.verdict == "REVISE" for v in per_item) else "PASS"
        return BatchJudgeResult(per_item=per_item, overall=overall)

    original_assessor_module = assessor_mod.assess_batch_events
    original_assessor_caller = deliver_events_mod.assess_batch_events
    original_judge_module = judge_mod.judge_batch_events
    original_judge_caller = deliver_events_mod.judge_batch_events

    assessor_mod.assess_batch_events = _stub_assess_batch_events  # type: ignore[assignment]
    deliver_events_mod.assess_batch_events = _stub_assess_batch_events  # type: ignore[assignment]
    judge_mod.judge_batch_events = _stub_judge_batch_events  # type: ignore[assignment]
    deliver_events_mod.judge_batch_events = _stub_judge_batch_events  # type: ignore[assignment]

    try:
        from news_service.models.news_item import NewsItem
        from news_service.tasks.poll_feeds import _poll_all_feeds

        poll_result = await _poll_all_feeds()
        assert poll_result["new_items"] == 3, (
            f"expected 3 new items ingested, got {poll_result['new_items']}: "
            f"{poll_result!r}"
        )

        await world.celery.drain()

        async with async_session_factory() as s:
            news_rows = list(
                (await s.execute(select(NewsItem).where(NewsItem.source_id == source_id)))
                .scalars()
                .all()
            )
        assert len(news_rows) == 3, (
            f"expected 3 news_items persisted by poller, got {len(news_rows)}"
        )

        assert judge_call_count == 3, (
            f"judge_batch_events should run exactly event_judge_max_revisions + 1 = 3 "
            f"rounds; got {judge_call_count}"
        )
        assert assessor_call_count == 3, (
            f"assess_batch_events should run 1 initial + 2 revision calls = 3 total; "
            f"got {assessor_call_count}"
        )

        defective_id = defective_item_id_holder.get("id")
        assert defective_id is not None, (
            "assessor stub was never invoked with the initial batch; "
            "pipeline did not reach the loop"
        )

        captured = world.delivery.for_url(WEBHOOK_URL)
        captured_bodies = [c.body for c in captured]
        captured_snip = [b[:160] for b in captured_bodies]

        assert len(captured) == 2, (
            f"expected exactly 2 webhooks delivered (the PASS items), got "
            f"{len(captured)}. Bodies: {captured_snip}"
        )

        combined = "\n".join(captured_bodies)
        assert defective_body not in combined, (
            f"defective item's body leaked into delivery despite REVISE-after-max "
            f"drop. Bodies: {captured_snip}"
        )

        async with async_session_factory() as s:
            sent_rows = list(
                (
                    await s.execute(
                        select(SentItem).where(SentItem.subscription_id == sub_id)
                    )
                )
                .scalars()
                .all()
            )
        assert len(sent_rows) == 2, (
            f"expected 2 SentItem rows (one per delivered PASS item), got "
            f"{len(sent_rows)}"
        )

        defective_uuid = uuid.UUID(defective_id)
        sent_ids = {row.news_item_id for row in sent_rows}
        assert defective_uuid not in sent_ids, (
            f"defective item {defective_uuid} should not have a SentItem row but "
            f"was found in {sent_ids}"
        )

        async with async_session_factory() as s:
            failed = list((await s.execute(select(FailedTask))).scalars().all())
        assert not failed, (
            f"drop-after-max is not a failure path; expected 0 FailedTask rows, "
            f"got {len(failed)}: {failed!r}"
        )
    finally:
        assessor_mod.assess_batch_events = original_assessor_module  # type: ignore[assignment]
        deliver_events_mod.assess_batch_events = original_assessor_caller  # type: ignore[assignment]
        judge_mod.judge_batch_events = original_judge_module  # type: ignore[assignment]
        deliver_events_mod.judge_batch_events = original_judge_caller  # type: ignore[assignment]
