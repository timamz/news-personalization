"""Diploma-relevant metrics for the news-notification benchmark.

The default classification F1 was designed for information retrieval,
not for a consumer notification product. For this diploma the three
things that actually matter are:

  1. SPAM TOLERANCE — the moment the product ships, the user's first
     question is "is this useful or is it flooding me?". Measured by
     precision on delivered items, counted within a realistic
     reading-budget window (not against the full timeline).

  2. COVERAGE OF TOP STORIES — a user accepts missing 100 marginal
     items a week, but not the single most important one. Measured
     by recall at K where K = their reasonable reading budget per
     delivery (~5 per weekly digest, or the absolute gold count for
     events).

  3. DELIVERY HEALTH — did the schedule fire as promised, were
     subscriptions actually created, and how much did it cost per
     delivered notification? These are economic/UX gates: if the
     bot didn't run, even P=R=1.0 is useless.

This script loads v3 result artifacts, re-derives these metrics from
captured_webhooks + scenario fabric, and prints a table.

Usage:
    python scripts/diploma_metrics.py [<result_dir> ...]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCH_ROOT / "data" / "scenarios"
RESULTS_DIR = BENCH_ROOT / "results"

DIGEST_READING_BUDGET = 5  # per-digest K for coverage / precision at K


_BULLET_SPLIT = re.compile(r"(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)")


def _parse_iso(ts: str) -> datetime:
    cleaned = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _blocks(body: str) -> list[str]:
    text = (body or "").strip()
    if not text:
        return []
    parts = _BULLET_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _match(expected_headline: str, block: str) -> bool:
    a = set(expected_headline.lower().split())
    if not a:
        return False
    b = set(block.lower().split())
    return len(a & b) / len(a) >= 0.6


def _gold_items_for(sub_id: str, timeline: list[dict], mode: str) -> list[dict]:
    """Return items labeled positive for this sub."""
    label_key = (
        "should_contribute_to_digest_per_sub"
        if mode == "digest"
        else "should_notify_per_sub"
    )
    return [it for it in timeline if (it.get(label_key) or {}).get(sub_id) is True]


def score_digest(
    sub_id: str,
    webhooks: list[dict],
    timeline: list[dict],
    budget_k: int = DIGEST_READING_BUDGET,
) -> dict:
    """Metrics for one digest-mode sub."""
    gold = _gold_items_for(sub_id, timeline, "digest")
    digest_count = 0
    total_delivered = 0
    total_gold_in_budget_hit = 0  # delivered item matched a gold from its week's top-budget_k
    total_noise = 0
    topk_gold_total = 0  # sum over digests of min(budget_k, gold-in-window)
    topk_recall_hits = 0
    for wh in webhooks:
        blocks = _blocks(wh.get("body") or "")
        if not blocks:
            continue
        digest_count += 1
        total_delivered += len(blocks)
        delivered_at = _parse_iso(wh["fake_clock_iso"]) if wh.get("fake_clock_iso") else None
        if delivered_at is None:
            continue
        window_start = delivered_at - timedelta(days=7)
        window_gold = [
            it for it in gold
            if window_start <= _parse_iso(it["fake_ts"]) <= delivered_at
        ]
        # "top-K by recency" is a proxy for "most relevant"; using ts is
        # scenario-agnostic. Scenarios with cosine scores could rank by
        # that instead.
        topk = sorted(window_gold, key=lambda it: it["fake_ts"], reverse=True)[:budget_k]
        topk_gold_total += len(topk)
        for gold_item in topk:
            if any(_match(gold_item["headline"], b) for b in blocks):
                topk_recall_hits += 1
        for block in blocks:
            if any(_match(it["headline"], block) for it in topk):
                total_gold_in_budget_hit += 1
            else:
                total_noise += 1
    precision_at_k = (
        total_gold_in_budget_hit / total_delivered if total_delivered else 0.0
    )
    coverage_at_k = (topk_recall_hits / topk_gold_total) if topk_gold_total else 0.0
    return {
        "mode": "digest",
        "digests_delivered": digest_count,
        "items_delivered": total_delivered,
        "mean_items_per_digest": (
            total_delivered / digest_count if digest_count else 0.0
        ),
        "precision_at_K": round(precision_at_k, 3),
        "coverage_at_K": round(coverage_at_k, 3),
        "noise_items_total": total_noise,
        "gold_in_budget_total": topk_gold_total,
        "budget_K": budget_k,
    }


def score_event(sub_id: str, webhooks: list[dict], timeline: list[dict]) -> dict:
    """Metrics for one event-mode sub."""
    gold = _gold_items_for(sub_id, timeline, "event")
    if not webhooks:
        return {
            "mode": "event",
            "notifications": 0,
            "true_positives": 0,
            "false_positives": 0,
            "gold_count": len(gold),
            "precision": 0.0,
            "recall": 0.0,
            "false_alarms_per_day": 0.0,
        }
    tp = 0
    fp = 0
    gold_hit: set[int] = set()
    days_span = 30
    for wh in webhooks:
        body = wh.get("body") or ""
        matched_any = False
        for i, gold_item in enumerate(gold):
            if _match(gold_item["headline"], body):
                matched_any = True
                gold_hit.add(i)
                break  # one match per notification
        if matched_any:
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = len(gold_hit) / len(gold) if gold else 0.0
    false_alarms_per_day = fp / days_span
    return {
        "mode": "event",
        "notifications": len(webhooks),
        "true_positives": tp,
        "false_positives": fp,
        "gold_count": len(gold),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_alarms_per_day": round(false_alarms_per_day, 2),
    }


def health(rec: dict) -> dict:
    """Return delivery-health signals independent of classification."""
    ac = rec.get("action_correctness") or {}
    webhooks = rec.get("captured_webhooks") or []
    subs_pass = sum(
        1
        for o in ac.get("outcomes", [])
        if o["kind"] == "subscription_exists_matching" and o["passed"]
    )
    subs_fail = sum(
        1
        for o in ac.get("outcomes", [])
        if o["kind"] == "subscription_exists_matching" and not o["passed"]
    )
    failed_tasks_ok = any(
        o["kind"] == "failed_tasks_zero" and o["passed"]
        for o in ac.get("outcomes", [])
    )
    notes = " | ".join(rec.get("notes") or [])
    ticks_match = re.search(r"(\d+) poll ticks", notes)
    poll_ticks = int(ticks_match.group(1)) if ticks_match else None
    return {
        "conversation_turns": len(rec.get("conversation_transcript") or []),
        "webhooks_total": len(webhooks),
        "subs_created_ok": subs_pass,
        "subs_created_missing": subs_fail,
        "failed_tasks_zero": failed_tasks_ok,
        "poll_ticks_ran": poll_ticks,
        "scheduler_coverage_pct": (
            round(100 * poll_ticks / 1422, 1) if poll_ticks else None
        ),
        "cost_usd": round(rec.get("cost", {}).get("total_usd", 0), 3),
    }


def rescore_record(rec_path: Path) -> dict:
    rec = json.loads(rec_path.read_text())
    sid = rec.get("scenario_id") or rec_path.stem.split("__")[0]
    skel = json.loads((DATA_DIR / sid / "skeleton.json").read_text())
    timeline = skel.get("timeline", [])
    goals = skel.get("goals", [])

    by_url: dict[str, list[dict]] = {}
    for wh in rec.get("captured_webhooks") or []:
        by_url.setdefault(wh.get("url") or "", []).append(wh)

    per_sub = {}
    for goal in goals:
        sub_id = goal["goal_id"]
        mode = goal.get("expected_delivery_mode") or "digest"
        hits = by_url.get(goal.get("expected_webhook_url") or "", [])
        if mode == "digest":
            per_sub[sub_id] = score_digest(sub_id, hits, timeline)
        else:
            per_sub[sub_id] = score_event(sub_id, hits, timeline)

    return {
        "scenario_id": sid,
        "health": health(rec),
        "per_sub": per_sub,
    }


def main() -> int:
    args = sys.argv[1:]
    roots = [Path(a) for a in args] if args else list(RESULTS_DIR.glob("v3_*"))
    rec_paths: list[Path] = []
    for root in roots:
        rec_paths.extend(sorted(root.glob("**/scenarios/*.json")))
    if not rec_paths:
        print("No scenario records found.")
        return 1

    # Dedupe: prefer rerun over original for the same scenario_id.
    best: dict[str, Path] = {}
    for p in rec_paths:
        rec = json.loads(p.read_text())
        sid = rec.get("scenario_id")
        existing = best.get(sid)
        if existing is None:
            best[sid] = p
            continue
        # Heuristic: any path that has more webhooks OR has 'rerun' wins.
        old = json.loads(existing.read_text())
        has_more_webhooks = len(rec.get("captured_webhooks") or []) > len(
            old.get("captured_webhooks") or []
        )
        is_rerun_replacement = "rerun" in str(p) and "rerun" not in str(existing)
        if has_more_webhooks or is_rerun_replacement:
            best[sid] = p

    summaries = [rescore_record(p) for p in best.values()]
    summaries.sort(key=lambda s: s["scenario_id"])

    print("=" * 110)
    print("DIPLOMA METRICS — v3 PARALLEL RUN")
    print("=" * 110)
    for s in summaries:
        h = s["health"]
        print(f"\n[{s['scenario_id']}]  "
              f"turns={h['conversation_turns']}  "
              f"subs_ok={h['subs_created_ok']}/{h['subs_created_ok']+h['subs_created_missing']}  "
              f"webhooks={h['webhooks_total']}  "
              f"sched_cov={h['scheduler_coverage_pct']}%  "
              f"cost=${h['cost_usd']}")
        for sub_id, m in s["per_sub"].items():
            if m["mode"] == "digest":
                print(f"   digest [{sub_id}]")
                print(f"     digests_delivered={m['digests_delivered']}  "
                      f"items/digest={m['mean_items_per_digest']:.1f}")
                print(f"     precision@K={m['precision_at_K']}  "
                      f"coverage@K={m['coverage_at_K']}  "
                      f"K={m['budget_K']}")
                print(f"     gold_in_budget={m['gold_in_budget_total']}  "
                      f"noise_items={m['noise_items_total']}")
            else:
                print(f"   event [{sub_id}]")
                print(f"     notifications={m['notifications']}  "
                      f"TP={m['true_positives']}  FP={m['false_positives']}  "
                      f"gold={m['gold_count']}")
                print(f"     precision={m['precision']}  "
                      f"recall={m['recall']}  "
                      f"false_alarms/day={m['false_alarms_per_day']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
