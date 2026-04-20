"""
Deterministic assertions executed after a scenario run.

These are the PRIMARY pass/fail signal; LLM rubrics are secondary. Every
AssertionSpec `kind` maps to a checker that reads scenario expectations
and run state (DB snapshots + webhook capture log) and returns a
Pass/Fail with a rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_benchmark.fakes.delivery import FakeDelivery
from news_benchmark.scenarios.base import AssertionSpec, Scenario


@dataclass
class AssertionOutcome:
    kind: str
    passed: bool
    detail: str = ""


@dataclass
class ActionCorrectnessReport:
    outcomes: list[AssertionOutcome] = field(default_factory=list)

    def passed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    def failed(self) -> list[AssertionOutcome]:
        return [o for o in self.outcomes if not o.passed]

    def overall_pass(self) -> bool:
        return all(o.passed for o in self.outcomes)


async def evaluate(
    *,
    scenario: Scenario,
    session: AsyncSession,
    delivery: FakeDelivery,
) -> ActionCorrectnessReport:
    """Evaluate every assertion in the scenario against live run state."""

    report = ActionCorrectnessReport()

    for spec in scenario.assertions:
        outcome = await _one(spec, scenario, session, delivery)
        report.outcomes.append(outcome)

    return report


async def _one(
    spec: AssertionSpec,
    scenario: Scenario,
    session: AsyncSession,
    delivery: FakeDelivery,
) -> AssertionOutcome:
    from news_service.models import FailedTask, Source, Subscription, SubscriptionSource

    k = spec.kind
    p = spec.payload

    if k == "failed_tasks_zero":
        rows = (await session.execute(select(FailedTask))).scalars().all()
        return AssertionOutcome(
            kind=k,
            passed=len(rows) == 0,
            detail=f"{len(rows)} failed tasks" if rows else "no failed tasks",
        )

    if k == "subscription_exists_matching":
        goal_id = p["goal_id"]
        rows = (await session.execute(select(Subscription))).scalars().all()
        for sub in rows:
            keywords_ok = all(
                kw.lower() in sub.user_spec.lower()
                for kw in p.get("expected_user_spec_keywords", [])
            )
            cron_ok = (
                p.get("expected_schedule_cron") is None
                or sub.schedule_cron == p["expected_schedule_cron"]
            )
            mode_ok = sub.delivery_mode == p.get("expected_delivery_mode", sub.delivery_mode)
            lang_ok = (
                p.get("expected_digest_language") is None
                or (sub.digest_language or "").lower() == p["expected_digest_language"].lower()
            )
            if keywords_ok and cron_ok and mode_ok and lang_ok:
                return AssertionOutcome(
                    kind=k, passed=True, detail=f"matched subscription {sub.id}"
                )
        return AssertionOutcome(
            kind=k,
            passed=False,
            detail=f"no subscription matched all expectations for {goal_id}",
        )

    if k == "digest_webhooks_delivered":
        goal_id = p["goal_id"]
        expected_url = next(
            (g.expected_webhook_url for g in scenario.goals if g.goal_id == goal_id),
            None,
        )
        hits = delivery.for_url(expected_url) if expected_url else []
        lo = p.get("min_count", 1)
        hi = p.get("max_count", 999)
        ok = lo <= len(hits) <= hi
        return AssertionOutcome(
            kind=k,
            passed=ok,
            detail=f"{len(hits)} digest webhooks (expected [{lo}, {hi}])",
        )

    if k == "sources_within_bounds":
        rows = (await session.execute(select(SubscriptionSource))).scalars().all()
        lo = p.get("min", 3)
        hi = p.get("max", 10)
        ok = lo <= len(rows) <= hi
        return AssertionOutcome(
            kind=k,
            passed=ok,
            detail=f"{len(rows)} SubscriptionSource rows (expected [{lo}, {hi}])",
        )

    if k == "sources_are_from_good_pool":
        good_urls = {s.url for s in scenario.source_universe if s.should_be_picked_by_finder}
        rows = (await session.execute(select(Source).join(SubscriptionSource))).scalars().all()
        noise = [s for s in rows if s.url not in good_urls]
        max_noise = p.get("max_noise_sources", 0)
        ok = len(noise) <= max_noise
        return AssertionOutcome(
            kind=k,
            passed=ok,
            detail=(
                f"{len(noise)} noise sources picked "
                f"(max allowed {max_noise}): {[s.url for s in noise]}"
            ),
        )

    return AssertionOutcome(kind=k, passed=False, detail=f"unknown assertion kind {k}")
