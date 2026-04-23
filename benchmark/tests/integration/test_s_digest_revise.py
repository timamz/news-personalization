"""
S-digest-revise: Writer <-> Judge REVISE -> PASS loop.

Same DB seeding as ``test_s_digest_happy.py`` (eight pre-embedded items,
six on-topic, two off-topic). The difference: ``judge_digest`` is
monkey-patched so the first invocation returns ``verdict="REVISE"`` with
specific critic feedback, and the second returns ``verdict="PASS"``.

Assertions prove the pipeline actually honours the REVISE signal:

  1. ``judge_digest`` was invoked at least twice (first REVISE, second
     PASS). If the writer/judge loop were broken the stub would fire
     only once.
  2. Exactly one webhook lands -- the final PASS draft, not the
     rejected first draft. FakeDelivery records every ``deliver`` call,
     so a count of 1 rules out the pipeline delivering the REVISE draft
     by mistake.
  3. The delivered body is non-trivial (>= 200 chars).
  4. ``SentItem`` rows exist for the sub -- bookkeeping survived the
     revision loop.

Out of scope: judge-always-REVISE fallback (writer exhausts
``_MAX_REVISIONS=3`` and delivers the unreviewed last draft), judge
exception (tier-2 fail-open path).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from tests.integration._digest_common import (
    WEBHOOK_URL,
    seed_digest_world,
)


@pytest.mark.asyncio
async def test_s_digest_revise_loop(world):
    """Judge REVISE on call 1, PASS on call 2 -> writer regenerates, delivery fires."""
    from news_service.agents.digest import judge as judge_mod
    from news_service.agents.digest import pipeline as pipeline_mod
    from news_service.agents.digest.judge import QualityScores
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.sent_item import SentItem
    from news_service.tasks.deliver_digest import _deliver_digest

    call_count = 0

    async def _stub_judge(*, digest_text, user_spec, candidates_summary):
        """Two-call stub: REVISE then PASS."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return QualityScores(
                relevance=3,
                format_score=2,
                conciseness=3,
                verdict="REVISE",
                feedback=(
                    "The draft is too abstract. Rewrite so every item includes at "
                    "least one specific numerical figure from the source body "
                    "(percentages, dates, GW capacity, etc.). Keep the same "
                    "number of items."
                ),
            )
        return QualityScores(
            relevance=5,
            format_score=5,
            conciseness=5,
            verdict="PASS",
            feedback="",
        )

    original_pipeline_judge = pipeline_mod.judge_digest
    original_module_judge = judge_mod.judge_digest
    pipeline_mod.judge_digest = _stub_judge  # type: ignore[assignment]
    judge_mod.judge_digest = _stub_judge  # type: ignore[assignment]

    try:
        _user_id, sub_id, _source_id = await seed_digest_world(world, embedding_fn=embed_text)

        result = await _deliver_digest(sub_id)
        assert result.get("status") == "delivered", (
            f"expected delivered status, got {result!r}"
        )

        await world.celery.drain()

        assert call_count >= 2, (
            f"expected judge_digest invoked at least twice (REVISE then PASS), "
            f"got {call_count}"
        )

        captured = world.delivery.for_url(WEBHOOK_URL)
        assert len(captured) == 1, (
            f"expected exactly 1 webhook delivery (the PASS draft), got "
            f"{len(captured)}. Bodies: {[c.body[:120] for c in captured]}"
        )

        body = captured[0].body
        assert len(body) >= 200, (
            f"PASS-draft body should be >= 200 chars, got {len(body)}: {body!r}"
        )

        async with async_session_factory() as s:
            sent_rows = list(
                (await s.execute(select(SentItem).where(SentItem.subscription_id == sub_id)))
                .scalars()
                .all()
            )
        assert len(sent_rows) >= 1, (
            f"expected at least 1 SentItem after delivery, got {len(sent_rows)}"
        )

        async with async_session_factory() as s:
            failed = list((await s.execute(select(FailedTask))).scalars().all())
        assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    finally:
        pipeline_mod.judge_digest = original_pipeline_judge  # type: ignore[assignment]
        judge_mod.judge_digest = original_module_judge  # type: ignore[assignment]
