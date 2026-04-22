"""
Classification metrics for event-notification delivery.

Every scenario item carries `should_notify_per_sub: {sub_id: bool}`. After
a run, we know which (sub, news_item) pairs produced a captured webhook.
Comparing the two gives TP / FP / FN / TN, precision, recall, F1, and
false-positive rate — per scenario, per subscription, per run.

Dedup correctness is a separate binary: for any two captured notifications
within the same subscription whose source items share ≥ 0.85 headline
similarity (token-Jaccard), we count a dedup failure.

Values live in the final report alongside PASS/FAIL and judge rubrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from news_benchmark.scenarios.base import Scenario


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
            "fpr": round(self.fpr(), 4),
        }


@dataclass
class ClassificationReport:
    per_sub: dict[str, ConfusionMatrix] = field(default_factory=dict)
    dedup_failures_per_sub: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "per_sub": {sub: cm.to_dict() for sub, cm in self.per_sub.items()},
            "dedup_failures_per_sub": dict(self.dedup_failures_per_sub),
        }


def score_classification(
    scenario: Scenario,
    delivered_bodies_per_sub: dict[str, list[str]],
) -> ClassificationReport:
    """Compare scenario gold labels against delivery bodies.

    For each goal, we flatten the captured webhooks into one or more
    "item blocks". Digest deliveries are bullet-list formatted
    ('- headline\\n  body...'), so splitting on '\\n- ' yields one block
    per included item. Event deliveries carry a single item body; the
    same split leaves them intact as one block. A gold timeline entry
    counts as delivered iff any block contains enough of its headline
    tokens.
    """
    report = ClassificationReport()
    for goal in scenario.goals:
        sub = goal.goal_id
        bodies = delivered_bodies_per_sub.get(sub, [])
        blocks: list[str] = []
        for body in bodies:
            blocks.extend(_extract_item_blocks(body))
        is_digest = goal.expected_delivery_mode == "digest"
        cm = ConfusionMatrix()
        for entry in scenario.timeline:
            if is_digest:
                label = entry.should_contribute_to_digest_per_sub.get(sub)
            else:
                label = entry.should_notify_per_sub.get(sub)
            if label is None:
                continue
            was_delivered = any(_match(entry.headline, b) for b in blocks)
            if label and was_delivered:
                cm.tp += 1
            elif label and not was_delivered:
                cm.fn += 1
            elif not label and was_delivered:
                cm.fp += 1
            else:
                cm.tn += 1
        report.per_sub[sub] = cm
        first_lines = [b.splitlines()[0] for b in blocks if b]
        report.dedup_failures_per_sub[sub] = _dedup_failures(first_lines)
    return report


def _extract_item_blocks(body: str) -> list[str]:
    """Split a webhook body into one block per delivered item.

    The digest writer prompt asks for a single format, but different
    models drift to a handful of common list styles:
      * '- <headline>\\n  <body>...'  (markdown dash bullet)
      * '* <headline>\\n  <body>...'  (markdown asterisk)
      * '1) <headline>\\n   <body>...' or '1. <headline>\\n   <body>...'
        (numbered list)
    We split on any of those line-start markers so the per-item matcher
    is not fooled by a single-block numbered digest. Event bodies have
    no bullets; the regex leaves them as one block.
    """
    text = body.strip()
    if not text:
        return []
    import re

    # A line-start boundary before any of: '- ', '* ', or '\\d+[.)] '
    parts = re.split(r"(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)", text)
    return [p.strip() for p in parts if p.strip()]


def _match(expected: str, delivered_block: str) -> bool:
    """Token-containment match: how many expected tokens appear in delivered.

    Symmetric Jaccard is wrong here because digest blocks vary in length
    and a long block dilutes the intersection. Containment
    (|gold ∩ delivered| / |gold|) is the right semantic: "did this gold
    headline show up in the delivery?" We keep the 0.6 threshold from
    the old scorer so callers that tune fixtures do not have to retune.
    """
    a = set(expected.lower().split())
    if not a:
        return False
    b = set(delivered_block.lower().split())
    return (len(a & b) / len(a)) >= 0.6


def _dedup_failures(headlines: list[str]) -> int:
    fails = 0
    for i, h1 in enumerate(headlines):
        for h2 in headlines[i + 1 :]:
            a = set(h1.lower().split())
            b = set(h2.lower().split())
            if not a or not b:
                continue
            union = len(a | b)
            sim = len(a & b) / union if union else 0.0
            if sim >= 0.85:
                fails += 1
    return fails
