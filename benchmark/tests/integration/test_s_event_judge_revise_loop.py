"""
S-event-judge-revise-loop: round-by-round mechanics of ``_judge_and_revise``.

This test exercises the Generator/Critic loop in
``backend/src/news_service/tasks/deliver_events.py::_judge_and_revise``.
It does NOT evaluate the real judge's grading quality. The judge is
stubbed so we can drive the loop deterministically round-by-round; the
real Batch Event Assessor is left in place so the test actually exercises
the parameter-passing surface of the loop.

Specifically verified:
  1. Critic feedback produced on round 1 propagates to the round-2
     ``assess_batch_events`` call as ``critic_feedback_per_item``.
  2. Only the single item the judge flagged REVISE is re-assessed on
     round 2 (passing items are preserved, not re-evaluated).
  3. The loop exits early when the judge returns ``overall="PASS"``
     rather than continuing to the revision cap.

Out of scope: the real judge's PASS/REVISE decisions, notification-body
content quality, assessor prompt engineering, and cross-batch dedup.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem


SOURCE_URL = "https://example-battery-wire.invalid/feed.xml"
WEBHOOK_URL = "https://bench.invalid/webhook/s-event-judge-revise"

USER_SPEC = (
    "# Lithium supply-chain alerts\n"
    "\n"
    "Notify me instantly when news breaks about lithium mining, lithium "
    "refining, battery-grade lithium carbonate or hydroxide pricing, or "
    "regulatory actions affecting the lithium supply chain (mining permits, "
    "export restrictions, royalty changes, pricing floors). Include the "
    "source URL at the end of each notification.\n"
    "\n"
    "Do NOT notify me about: electric-vehicle sales or delivery numbers, "
    "Tesla stock moves, downstream battery-cell or pack news, or other "
    "metals (copper, nickel, cobalt) unless the story is explicitly about "
    "their interaction with lithium supply.\n"
)


@pytest.mark.asyncio
async def test_s_event_judge_revise_loop_rewrites_only_failing_item(world):
    """Round 1 REVISE one item -> round 2 re-assesses only that item with feedback -> PASS."""
    from news_service.agents.event import batch_assessor as batch_assessor_mod
    from news_service.agents.event import judge as judge_mod
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
                title="Battery Wire Daily",
                source_description="Lithium supply-chain newswire.",
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
                "under long-term contracts starting January 2026. The measure is "
                "aimed at protecting state royalties as spot prices recover from "
                "the 2024 trough. Industry consultations close in May. SQM and "
                "Albemarle, the two largest lithium producers in Chile, said "
                "they are reviewing the draft text."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=4),
            source_url=SOURCE_URL,
            headline="Albemarle pauses expansion of Kemerton lithium hydroxide refinery",
            body=(
                "Albemarle Corp said on Monday it will pause the third-train "
                "expansion of its Kemerton lithium hydroxide refinery in Western "
                "Australia, citing weaker-than-expected demand for battery-grade "
                "lithium hydroxide through 2026. Commissioned capacity from "
                "trains 1 and 2 remains on stream; the company declined to give "
                "a restart date for the paused expansion."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=6),
            source_url=SOURCE_URL,
            headline="Argentina approves new lithium brine royalty regime for Jujuy and Salta",
            body=(
                "Argentina's federal mining secretariat approved a revised "
                "royalty regime for lithium brine operations in Jujuy and Salta "
                "provinces. The new schedule indexes royalties to realized "
                "battery-grade lithium carbonate prices and takes effect July "
                "2026. Producers including Livent and Allkem said they are "
                "evaluating the fiscal impact; provincial governors welcomed "
                "the predictability."
            ),
        ),
    ]

    world.adapters[SOURCE_URL] = FakeAdapter(source_url=SOURCE_URL, items=items)

    feedback_text = (
        "This notification is missing the required source URL. Rewrite it to "
        "include the full URL at the end."
    )

    judge_call_count = 0
    revise_item_id_holder: dict[str, str] = {}

    async def _stub_judge(
        *,
        assessment,
        user_spec,
        recent_notification_history,
        max_history_chars,
    ) -> BatchJudgeResult:
        """Round 1: one REVISE, rest PASS. Round 2: all PASS."""
        nonlocal judge_call_count
        judge_call_count += 1
        ids = [a.item_id for a in assessment.assessments]
        if judge_call_count == 1:
            revise_id = ids[0]
            revise_item_id_holder["id"] = revise_id
            per_item = [
                ItemVerdict(
                    item_id=iid,
                    verdict="REVISE" if iid == revise_id else "PASS",
                    feedback=feedback_text if iid == revise_id else "",
                )
                for iid in ids
            ]
            return BatchJudgeResult(per_item=per_item, overall="REVISE")
        per_item = [
            ItemVerdict(item_id=iid, verdict="PASS", feedback="") for iid in ids
        ]
        return BatchJudgeResult(per_item=per_item, overall="PASS")

    real_assess_batch_events = batch_assessor_mod.assess_batch_events
    assessor_calls: list[dict] = []

    async def _spy_assess_batch_events(**kwargs):
        """Record kwargs of each call, then delegate to the real assessor."""
        assessor_calls.append(
            {
                "items": list(kwargs.get("items") or []),
                "critic_feedback_per_item": kwargs.get("critic_feedback_per_item"),
            }
        )
        return await real_assess_batch_events(**kwargs)

    original_module_judge = judge_mod.judge_batch_events
    original_task_judge = deliver_events_mod.judge_batch_events
    original_module_assessor = batch_assessor_mod.assess_batch_events
    original_task_assessor = deliver_events_mod.assess_batch_events

    judge_mod.judge_batch_events = _stub_judge  # type: ignore[assignment]
    deliver_events_mod.judge_batch_events = _stub_judge  # type: ignore[assignment]
    batch_assessor_mod.assess_batch_events = _spy_assess_batch_events  # type: ignore[assignment]
    deliver_events_mod.assess_batch_events = _spy_assess_batch_events  # type: ignore[assignment]

    try:
        from news_service.tasks.poll_feeds import _poll_all_feeds

        result = await _poll_all_feeds()
        assert result["new_items"] == 3, (
            f"expected 3 new items ingested, got {result['new_items']}: {result!r}"
        )

        await world.celery.drain()

        assert judge_call_count == 2, (
            f"judge should have been called exactly twice (REVISE then PASS), "
            f"got {judge_call_count}"
        )

        assert len(assessor_calls) == 2, (
            f"assessor should have been called exactly twice (initial + 1 revision), "
            f"got {len(assessor_calls)}: "
            f"{[{'n_items': len(c['items']), 'has_feedback': c['critic_feedback_per_item'] is not None} for c in assessor_calls]}"
        )

        initial_call = assessor_calls[0]
        assert initial_call["critic_feedback_per_item"] is None, (
            f"initial assessor call must not carry critic feedback, got "
            f"{initial_call['critic_feedback_per_item']!r}"
        )
        assert len(initial_call["items"]) == 3, (
            f"initial assessor call must evaluate all 3 items, got "
            f"{len(initial_call['items'])}"
        )

        revise_item_id = revise_item_id_holder["id"]

        revision_call = assessor_calls[1]
        feedback_map = revision_call["critic_feedback_per_item"]
        assert feedback_map is not None, (
            "revision assessor call must receive critic_feedback_per_item, got None"
        )
        assert revise_item_id in feedback_map, (
            f"revision feedback map must key the REVISE item {revise_item_id!r}, "
            f"got keys {list(feedback_map.keys())!r}"
        )
        assert feedback_map[revise_item_id] == feedback_text, (
            f"revision feedback must match the judge-emitted text exactly; "
            f"got {feedback_map[revise_item_id]!r}"
        )
        assert len(revision_call["items"]) == 1, (
            f"revision assessor call must re-assess ONLY the REVISE item, got "
            f"{len(revision_call['items'])} items"
        )
        revised_item_id = revision_call["items"][0]["item_id"]
        assert revised_item_id == revise_item_id, (
            f"revision assessor must re-assess the flagged item {revise_item_id!r}, "
            f"got {revised_item_id!r}"
        )

        captured = world.delivery.for_url(WEBHOOK_URL)
        assert len(captured) == 3, (
            f"expected 3 webhook deliveries (all items PASS after round 2), got "
            f"{len(captured)}. Bodies: {[c.body[:120] for c in captured]}"
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
        assert len(sent_rows) == 3, (
            f"expected 3 SentItem rows after delivery, got {len(sent_rows)}"
        )

        async with async_session_factory() as s:
            failed = list((await s.execute(select(FailedTask))).scalars().all())
        assert not failed, (
            f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
        )
    finally:
        judge_mod.judge_batch_events = original_module_judge  # type: ignore[assignment]
        deliver_events_mod.judge_batch_events = original_task_judge  # type: ignore[assignment]
        batch_assessor_mod.assess_batch_events = original_module_assessor  # type: ignore[assignment]
        deliver_events_mod.assess_batch_events = original_task_assessor  # type: ignore[assignment]
