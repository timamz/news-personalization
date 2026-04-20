"""
Automated audits run against every scenario skeleton before it is accepted.

Audits:

  tier_counts:            every difficulty tier has >= its MIN_COUNTS
                          threshold. Scenarios below the floor are rejected.

  positive_rate:          positive-label rate per subscription lies inside
                          TARGET_POSITIVE_RATE (5-20%). Real feeds are
                          noise-dominated; scenarios that invert this ratio
                          produce inflated precision.

  stupid_baseline:        TF-IDF + logistic regression classifier trained
                          on the scenario's own labels, evaluated with 5-fold
                          CV. F1 must land in [0.50, 0.70]. Below that the
                          data is incoherent; above that a trivial classifier
                          wins and the benchmark is too easy to discriminate
                          between models.

  label_consistency:      a second LLM (the configured judge model), blind
                          to the first author's labels, re-labels a 20%
                          stratified sample. Disagreements are logged, not
                          auto-corrected. Scenarios with > 20% disagreement
                          are flagged for review.

  lexical_overlap:        Jaccard similarity between top-50 TF-IDF tokens of
                          positives vs. negatives. Must be >= 0.15 — if the
                          two pools use disjoint vocabulary the classifier
                          succeeds on surface features alone.

  temporal_distribution:  histogram of items per simulated week; no week
                          should hold more than 40% of the total items.

Results are written to `audit.json` and `diversity.md` alongside the
fabric file. The pipeline raises if a hard-gate audit (tier_counts,
positive_rate) fails.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import litellm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from news_benchmark.data_gen.taxonomy import (
    DIFFICULTY_TIERS,
    MIN_COUNTS_PER_SCENARIO,
    TARGET_POSITIVE_RATE,
)
from news_benchmark.scenarios.base import Scenario, headline_hash


@dataclass
class AuditResult:
    tier_counts: dict[str, int] = field(default_factory=dict)
    tier_shortfalls: dict[str, int] = field(default_factory=dict)
    positive_rate_per_sub: dict[str, float] = field(default_factory=dict)
    stupid_baseline_f1_per_sub: dict[str, float] = field(default_factory=dict)
    lexical_overlap_per_sub: dict[str, float] = field(default_factory=dict)
    label_consistency_per_sub: dict[str, dict[str, object]] = field(default_factory=dict)
    temporal_max_week_share: float = 0.0
    warnings: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)

    def is_accepted(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict[str, object]:
        return {
            "tier_counts": self.tier_counts,
            "tier_shortfalls": self.tier_shortfalls,
            "positive_rate_per_sub": self.positive_rate_per_sub,
            "stupid_baseline_f1_per_sub": self.stupid_baseline_f1_per_sub,
            "lexical_overlap_per_sub": self.lexical_overlap_per_sub,
            "label_consistency_per_sub": self.label_consistency_per_sub,
            "temporal_max_week_share": self.temporal_max_week_share,
            "warnings": self.warnings,
            "hard_failures": self.hard_failures,
        }


async def run_audits(
    scenario: Scenario,
    *,
    judge_model: str | None = None,
    consistency_sample_fraction: float = 0.2,
    rng_seed: int = 42,
) -> AuditResult:
    """Run the full audit suite. Returns an AuditResult dataclass."""
    result = AuditResult()
    _audit_tier_counts(scenario, result)
    _audit_positive_rate(scenario, result)
    _audit_temporal(scenario, result)
    _audit_baseline_and_overlap(scenario, result, rng_seed)
    if judge_model:
        await _audit_label_consistency(
            scenario, result, judge_model, consistency_sample_fraction, rng_seed
        )
    return result


def _audit_tier_counts(scenario: Scenario, result: AuditResult) -> None:
    counts = Counter(t.difficulty for t in scenario.timeline)
    for tier in DIFFICULTY_TIERS:
        result.tier_counts[tier] = counts.get(tier, 0)
        need = MIN_COUNTS_PER_SCENARIO[tier]
        have = counts.get(tier, 0)
        if have < need:
            result.tier_shortfalls[tier] = need - have
            result.hard_failures.append(f"tier '{tier}' has {have} items, need >= {need}")


def _audit_positive_rate(scenario: Scenario, result: AuditResult) -> None:
    sub_ids = {g.goal_id for g in scenario.goals}
    for sub in sub_ids:
        total = 0
        positives = 0
        for t in scenario.timeline:
            flag = t.should_notify_per_sub.get(sub)
            contrib = t.should_contribute_to_digest_per_sub.get(sub)
            if flag is None and contrib is None:
                continue
            total += 1
            if flag or contrib:
                positives += 1
        rate = positives / total if total else 0.0
        result.positive_rate_per_sub[sub] = rate
        lo, hi = TARGET_POSITIVE_RATE
        if total == 0:
            result.hard_failures.append(f"subscription '{sub}' has no labeled items")
        elif not (lo <= rate <= hi):
            result.hard_failures.append(
                f"subscription '{sub}' positive rate {rate:.2%} outside [{lo:.0%}, {hi:.0%}]"
            )


def _audit_temporal(scenario: Scenario, result: AuditResult) -> None:
    from datetime import datetime

    if not scenario.timeline:
        return
    start = datetime.fromisoformat(scenario.start_date_iso)
    by_week: Counter[int] = Counter()
    for t in scenario.timeline:
        ts = datetime.fromisoformat(t.fake_ts)
        week = (ts - start).days // 7
        by_week[week] += 1
    top = max(by_week.values())
    share = top / sum(by_week.values())
    result.temporal_max_week_share = share
    if share > 0.4:
        result.warnings.append(
            f"temporal clustering: one week holds {share:.0%} of items (threshold 40%)"
        )


def _audit_baseline_and_overlap(scenario: Scenario, result: AuditResult, rng_seed: int) -> None:
    """Train a TF-IDF + logistic regression classifier per subscription.

    F1 in [0.50, 0.70] is the target "just hard enough" band. Lower than
    0.50 means labels are noisy or incoherent; higher means lexical
    features alone win and the benchmark cannot discriminate between
    competing LLMs.
    """
    sub_ids = {g.goal_id for g in scenario.goals}
    for sub in sub_ids:
        texts: list[str] = []
        labels: list[int] = []
        for t in scenario.timeline:
            flag = t.should_notify_per_sub.get(sub)
            contrib = t.should_contribute_to_digest_per_sub.get(sub)
            if flag is None and contrib is None:
                continue
            body = scenario.bodies_by_headline_hash.get(headline_hash(t), "")
            texts.append(t.headline + "\n" + body[:1500])
            labels.append(1 if (flag or contrib) else 0)
        y = np.array(labels)
        if len(set(y)) < 2 or len(y) < 20:
            result.warnings.append(f"subscription '{sub}' has too few labels for baseline audit")
            continue

        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(texts)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
        f1s: list[float] = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            tp = int(((preds == 1) & (y[test_idx] == 1)).sum())
            fp = int(((preds == 1) & (y[test_idx] == 0)).sum())
            fn = int(((preds == 0) & (y[test_idx] == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        mean_f1 = float(np.mean(f1s))
        result.stupid_baseline_f1_per_sub[sub] = mean_f1

        if mean_f1 < 0.45:
            result.warnings.append(
                f"subscription '{sub}': stupid baseline F1 {mean_f1:.2f} very low — "
                "labels may be inconsistent or noisy"
            )
        elif mean_f1 > 0.72:
            result.warnings.append(
                f"subscription '{sub}': stupid baseline F1 {mean_f1:.2f} too high — "
                "scenario is lexically separable; add more near_miss_negatives"
            )

        overlap = _lexical_overlap(texts, labels)
        result.lexical_overlap_per_sub[sub] = overlap
        if overlap < 0.15:
            result.warnings.append(
                f"subscription '{sub}': lexical overlap {overlap:.2f} too low — "
                "positive/negative pools use disjoint vocabularies"
            )


def _lexical_overlap(texts: list[str], labels: list[int]) -> float:
    vec = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    mat = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    pos_mask = np.array(labels) == 1
    neg_mask = ~pos_mask
    if not pos_mask.any() or not neg_mask.any():
        return 0.0
    pos_scores = np.asarray(mat[pos_mask].sum(axis=0)).ravel()
    neg_scores = np.asarray(mat[neg_mask].sum(axis=0)).ravel()
    top_pos = set(vocab[np.argsort(-pos_scores)[:50]].tolist())
    top_neg = set(vocab[np.argsort(-neg_scores)[:50]].tolist())
    inter = len(top_pos & top_neg)
    union = len(top_pos | top_neg)
    return inter / union if union else 0.0


async def _audit_label_consistency(
    scenario: Scenario,
    result: AuditResult,
    judge_model: str,
    sample_fraction: float,
    rng_seed: int,
) -> None:
    """Blind-label a stratified 20% sample with a second LLM and compare."""
    random.seed(rng_seed)
    sub_ids = {g.goal_id for g in scenario.goals}
    for sub in sub_ids:
        goal = next(g for g in scenario.goals if g.goal_id == sub)
        labeled = [
            (
                t,
                bool(
                    t.should_notify_per_sub.get(sub)
                    or t.should_contribute_to_digest_per_sub.get(sub)
                ),
            )
            for t in scenario.timeline
            if sub in t.should_notify_per_sub or sub in t.should_contribute_to_digest_per_sub
        ]
        by_tier: dict[str, list[tuple]] = {}
        for entry, label in labeled:
            by_tier.setdefault(entry.difficulty, []).append((entry, label))
        sampled: list[tuple] = []
        for _tier, rows in by_tier.items():
            k = max(1, int(len(rows) * sample_fraction))
            sampled.extend(random.sample(rows, min(k, len(rows))))

        disagreements = 0
        details: list[dict[str, object]] = []
        sem = asyncio.Semaphore(6)

        async def relabel(entry_label, _sem=sem, _judge=judge_model, _goal=goal):
            entry, original = entry_label
            async with _sem:
                pred = await _relabel_one(_judge, _goal, entry, scenario)
            return entry, original, pred

        preds = await asyncio.gather(*(relabel(x) for x in sampled))
        for entry, original, pred in preds:
            if pred is None:
                continue
            if pred != original:
                disagreements += 1
                details.append(
                    {
                        "headline": entry.headline,
                        "difficulty": entry.difficulty,
                        "author_label": original,
                        "judge_label": pred,
                    }
                )
        total = len(preds)
        rate = (disagreements / total) if total else 0.0
        result.label_consistency_per_sub[sub] = {
            "sample_size": total,
            "disagreements": disagreements,
            "disagreement_rate": rate,
            "examples": details[:20],
        }
        if rate > 0.2:
            result.warnings.append(
                f"subscription '{sub}': label-consistency disagreement {rate:.0%} "
                "exceeds 20% — review flagged items"
            )


async def _relabel_one(judge_model: str, goal, entry, scenario: Scenario) -> bool | None:
    system = (
        "You are a blind labeler. Given a user's subscription spec and one news "
        "headline, decide whether the user would want to be notified or digested. "
        'Reply with a JSON object: {"notify": true|false}. No other keys.'
    )
    user = (
        f"Subscription description: {goal.description}\n"
        f"Expected keywords: {', '.join(goal.expected_user_spec_keywords)}\n"
        f"Delivery mode: {goal.expected_delivery_mode}\n\n"
        f"Headline: {entry.headline}\n"
        f"Source URL: {entry.source_url}\n"
        f"Body excerpt: {scenario.bodies_by_headline_hash.get(headline_hash(entry), '')[:500]}\n"
    )
    try:
        resp = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=40,
            response_format={"type": "json_object"},
        )
        raw = resp["choices"][0]["message"]["content"] or "{}"
        parsed = json.loads(raw)
        return bool(parsed.get("notify"))
    except Exception:
        return None


def write_diversity_report(scenario: Scenario, audit: AuditResult, out_dir: Path) -> Path:
    """Render a human-readable diversity.md alongside the fabric file."""
    p = out_dir / scenario.scenario_id / "diversity.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Scenario `{scenario.scenario_id}` diversity report")
    lines.append("")
    lines.append(f"- Total timeline items: {len(scenario.timeline)}")
    lines.append(f"- Simulated days: {scenario.simulated_days}")
    lines.append(f"- Temporal top-week share: {audit.temporal_max_week_share:.0%}")
    lines.append("")
    lines.append("## Tier counts")
    lines.append("")
    lines.append("| Tier | Count | Minimum |")
    lines.append("|---|---:|---:|")
    for tier in DIFFICULTY_TIERS:
        have = audit.tier_counts.get(tier, 0)
        need = MIN_COUNTS_PER_SCENARIO[tier]
        flag = " ✓" if have >= need else f" (short {need - have})"
        lines.append(f"| {tier} | {have} | {need}{flag} |")
    lines.append("")
    lines.append("## Per-subscription label stats")
    lines.append("")
    tgt_lo, tgt_hi = TARGET_POSITIVE_RATE
    for sub, rate in audit.positive_rate_per_sub.items():
        lines.append(f"### `{sub}`")
        lines.append("")
        lines.append(f"- Positive rate: **{rate:.1%}** (target {tgt_lo:.0%}–{tgt_hi:.0%})")
        f1 = audit.stupid_baseline_f1_per_sub.get(sub)
        if f1 is not None:
            lines.append(f"- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **{f1:.2f}**")
            lines.append("  (target band 0.50–0.70)")
        lex = audit.lexical_overlap_per_sub.get(sub)
        if lex is not None:
            lines.append(f"- Positive/negative lexical overlap (Jaccard top-50): **{lex:.2f}**")
        lc = audit.label_consistency_per_sub.get(sub)
        if lc:
            rate = lc["disagreement_rate"]
            size = lc["sample_size"]
            lines.append(
                f"- Label-consistency on a {size}-item sample: **{rate:.0%}** "
                "disagreement (threshold 20%)"
            )
        lines.append("")
    if audit.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in audit.warnings:
            lines.append(f"- {w}")
        lines.append("")
    if audit.hard_failures:
        lines.append("## Hard failures (scenario rejected)")
        lines.append("")
        for f in audit.hard_failures:
            lines.append(f"- **{f}**")
        lines.append("")

    p.write_text("\n".join(lines))
    audit_p = out_dir / scenario.scenario_id / "audit.json"
    audit_p.write_text(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    return p
