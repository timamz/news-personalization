"""
S-judge-event-per-item: per-item grading fidelity of the Event Judge.

This test calls the real ``judge_batch_events`` with a crafted
``BatchAssessmentResult`` to verify that the judge correctly grades
per-item across all six check dimensions enumerated in its prompt:

  1. ``is_relevant`` correctness vs ``user_spec`` (rumor exclusion).
  2. ``notification_body`` describes the same item as the assessor's
     ``reason`` (no cross-item swap).
  3. Chat-message length (1-3 short sentences; no essays).
  4. Plain text -- no markdown bold ``**...**``, no markdown lists.
  5. Source URL (or equivalent pointer) present in the body.
  6. Not a near-duplicate of anything in
     ``recent_notification_history``.

Seven items are constructed: one clean (expected PASS) and six each
carrying a single distinct defect (expected REVISE). The judge's
per-item output is looked up by ``item_id`` (not position) because
the judge may reorder.

Out of scope: the assessor itself, the REVISE loop driven by
``tasks/deliver_events.py``, webhook delivery, and ``SentItem``
persistence. Those paths are covered by S-event-assess and the
reflector suites.

Feedback assertions are intentionally soft -- a disjunction of
literal-or-case-insensitive fingerprint tokens per defect, not an
exact-match string. LLM paraphrase variance makes exact matches
brittle, but a well-graded item will always name the defect with at
least one of the canonical tokens (``bold``/``markdown``/``**`` for
a bold violation, ``url``/``link``/``source`` for a missing URL,
``duplicate``/``already``/``recent`` for a duplicate).
"""

from __future__ import annotations

import uuid

import pytest


USER_SPEC = (
    "# Lithium supply-chain alerts\n"
    "\n"
    "Notify me instantly when news breaks about lithium mining, lithium "
    "refining, battery-grade lithium carbonate or hydroxide pricing, or "
    "regulatory actions affecting the lithium supply chain (mining permits, "
    "export restrictions, royalty changes, pricing floors).\n"
    "\n"
    "Do NOT notify me about rumors or unconfirmed speculation -- only "
    "confirmed, officially announced developments. Skip any story whose "
    "lede relies on anonymous sources, 'people familiar with the matter', "
    "or phrases like 'is said to' / 'reportedly considering'.\n"
)


@pytest.mark.asyncio
async def test_s_judge_event_per_item_grades_all_six_dimensions(world):
    """Per-item PASS/REVISE verdicts across all six judge check dimensions."""
    from news_service.agents.event.batch_assessor import (
        BatchAssessmentResult,
        ItemAssessment,
    )
    from news_service.agents.event.judge import judge_batch_events

    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())
    id_c = str(uuid.uuid4())
    id_d = str(uuid.uuid4())
    id_e = str(uuid.uuid4())
    id_f = str(uuid.uuid4())
    id_g = str(uuid.uuid4())

    history_duplicate_entry = (
        "Chile ministry unveils lithium pricing floor for 2026 contracts. "
        "Draft regulations set a minimum floor price for battery-grade "
        "lithium carbonate under long-term contracts starting January 2026. "
        "https://novametals.invalid/2026/chile-pricing-floor"
    )
    recent_notification_history = [
        history_duplicate_entry,
        (
            "Ganfeng expands Mariana brine project -- joint venture with "
            "Lithium Argentina closes USD 400m financing for Stage 2. "
            "https://novametals.invalid/2026/ganfeng-mariana"
        ),
    ]

    item_a = ItemAssessment(
        item_id=id_a,
        is_relevant=True,
        reason=(
            "Australia's Pilbara Minerals announced a signed offtake with "
            "LG Energy for battery-grade lithium hydroxide -- matches spec "
            "(supply-chain, confirmed)."
        ),
        notification_body=(
            "Pilbara Minerals signed a multi-year lithium hydroxide offtake "
            "with LG Energy, covering 2026-2030 deliveries. "
            "https://novametals.invalid/2026/pilbara-lg-offtake"
        ),
    )

    item_b = ItemAssessment(
        item_id=id_b,
        is_relevant=True,
        reason=(
            "Rumor that Albemarle is reportedly considering a lithium "
            "refinery acquisition in Chile, per anonymous sources."
        ),
        notification_body=(
            "Albemarle is reportedly considering acquiring a Chilean "
            "lithium refinery, according to people familiar with the "
            "matter. No official confirmation has been issued. "
            "https://novametals.invalid/2026/albemarle-rumor"
        ),
    )

    item_c = ItemAssessment(
        item_id=id_c,
        is_relevant=True,
        reason=(
            "Argentina's mining secretariat published a new lithium royalty "
            "framework affecting brine operations in Jujuy and Salta."
        ),
        notification_body=(
            "Peru's copper miners at Las Bambas voted to begin an open-ended "
            "strike over wage disputes. Copper futures jumped 2.1% in London "
            "trading. https://novametals.invalid/2026/peru-copper-strike"
        ),
    )

    item_d = ItemAssessment(
        item_id=id_d,
        is_relevant=True,
        reason=(
            "SQM reported Q1 2026 lithium carbonate production up 14% YoY."
        ),
        notification_body=(
            "SQM reported its first-quarter 2026 production results on "
            "Tuesday. Lithium carbonate output reached 48,300 tonnes, up "
            "14% year-on-year. Management attributed the gain to a ramp "
            "at the Salar de Atacama expansion. Realized prices averaged "
            "USD 13,200 per tonne, down from USD 15,100 a year earlier. "
            "The company said margin compression is expected to continue "
            "through the second quarter before stabilizing. Full-year "
            "guidance for lithium volumes was held at 210,000 tonnes. "
            "Shares in Santiago rose 2.3% on the report. "
            "https://novametals.invalid/2026/sqm-q1-results"
        ),
    )

    item_e = ItemAssessment(
        item_id=id_e,
        is_relevant=True,
        reason=(
            "China's MIIT released updated battery-grade lithium carbonate "
            "purity standards taking effect July 2026."
        ),
        notification_body=(
            "**China MIIT** released **updated** battery-grade lithium "
            "carbonate purity standards, effective **July 2026**. "
            "https://novametals.invalid/2026/miit-purity-standards"
        ),
    )

    item_f = ItemAssessment(
        item_id=id_f,
        is_relevant=True,
        reason=(
            "Zimbabwe imposed a ban on raw lithium ore exports, mandating "
            "in-country processing."
        ),
        notification_body=(
            "Zimbabwe's mines ministry banned raw lithium ore exports "
            "effective immediately; all concentrate must now be processed "
            "domestically before shipment."
        ),
    )

    item_g = ItemAssessment(
        item_id=id_g,
        is_relevant=True,
        reason=(
            "Chile published draft lithium carbonate pricing floor "
            "regulations for 2026 long-term contracts."
        ),
        notification_body=(
            "Chile's economy ministry unveiled a draft pricing floor for "
            "battery-grade lithium carbonate on long-term contracts "
            "starting in January 2026. "
            "https://novametals.invalid/2026/chile-pricing-floor"
        ),
    )

    assessment = BatchAssessmentResult(
        assessments=[item_a, item_b, item_c, item_d, item_e, item_f, item_g]
    )

    result = await judge_batch_events(
        assessment=assessment,
        user_spec=USER_SPEC,
        recent_notification_history=recent_notification_history,
        max_history_chars=4000,
    )

    by_id = {v.item_id: v for v in result.per_item}

    assert result.overall == "REVISE", (
        f"overall verdict should be REVISE when any item is REVISE, got "
        f"{result.overall!r}; per_item={[v.model_dump() for v in result.per_item]}"
    )

    verdict_a = by_id[id_a]
    assert verdict_a.verdict == "PASS", (
        f"item A (clean) should PASS; got {verdict_a.model_dump()}"
    )

    verdict_b = by_id[id_b]
    assert verdict_b.verdict == "REVISE", (
        f"item B (rumor marked relevant) should REVISE -- user_spec "
        f"excludes rumors/unconfirmed speculation; got {verdict_b.model_dump()}"
    )
    assert len(verdict_b.feedback.strip()) > 0, (
        f"item B REVISE must carry non-empty feedback; got {verdict_b.model_dump()}"
    )

    verdict_c = by_id[id_c]
    assert verdict_c.verdict == "REVISE", (
        f"item C (body/reason mismatch) should REVISE -- body talks about "
        f"Peru copper strike but reason is about Argentina lithium "
        f"royalties; got {verdict_c.model_dump()}"
    )
    assert len(verdict_c.feedback.strip()) > 0, (
        f"item C REVISE must carry non-empty feedback; got {verdict_c.model_dump()}"
    )

    verdict_d = by_id[id_d]
    assert verdict_d.verdict == "REVISE", (
        f"item D (8-sentence essay body) should REVISE -- chat-message "
        f"length constraint is 1-3 short sentences; got {verdict_d.model_dump()}"
    )
    assert len(verdict_d.feedback.strip()) > 0, (
        f"item D REVISE must carry non-empty feedback; got {verdict_d.model_dump()}"
    )

    verdict_e = by_id[id_e]
    assert verdict_e.verdict == "REVISE", (
        f"item E (markdown bold markers) should REVISE -- prompt forbids "
        f"**...** bold; got {verdict_e.model_dump()}"
    )
    assert len(verdict_e.feedback.strip()) > 0, (
        f"item E REVISE must carry non-empty feedback; got {verdict_e.model_dump()}"
    )
    fb_e_lower = verdict_e.feedback.lower()
    assert ("bold" in fb_e_lower) or ("markdown" in fb_e_lower) or (
        "**" in verdict_e.feedback
    ), (
        f"item E feedback should fingerprint the bold/markdown defect "
        f"with one of 'bold'/'markdown'/'**'; got {verdict_e.model_dump()}"
    )

    verdict_f = by_id[id_f]
    assert verdict_f.verdict == "REVISE", (
        f"item F (missing URL) should REVISE -- prompt requires a source "
        f"URL or equivalent pointer in the body; got {verdict_f.model_dump()}"
    )
    assert len(verdict_f.feedback.strip()) > 0, (
        f"item F REVISE must carry non-empty feedback; got {verdict_f.model_dump()}"
    )
    fb_f_lower = verdict_f.feedback.lower()
    assert ("url" in fb_f_lower) or ("link" in fb_f_lower) or (
        "source" in fb_f_lower
    ), (
        f"item F feedback should fingerprint the missing-URL defect with "
        f"one of 'url'/'link'/'source'; got {verdict_f.model_dump()}"
    )

    verdict_g = by_id[id_g]
    assert verdict_g.verdict == "REVISE", (
        f"item G (near-duplicate of history entry) should REVISE -- the "
        f"Chile pricing-floor story was already notified; got "
        f"{verdict_g.model_dump()}"
    )
    assert len(verdict_g.feedback.strip()) > 0, (
        f"item G REVISE must carry non-empty feedback; got {verdict_g.model_dump()}"
    )
    fb_g_lower = verdict_g.feedback.lower()
    assert ("duplicate" in fb_g_lower) or ("already" in fb_g_lower) or (
        "recent" in fb_g_lower
    ), (
        f"item G feedback should fingerprint the duplicate-of-history "
        f"defect with one of 'duplicate'/'already'/'recent'; got "
        f"{verdict_g.model_dump()}"
    )
