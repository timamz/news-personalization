"""Rescore v3 digest artifacts with fairer per-digest metrics.

The default classification scorer treats every timeline item as a binary
decision, which over-penalizes the Writer's promiscuity at v3 scale: a
digest that contains 5 gold items plus 30 filler items scores the same
as one that plausibly delivered 35 off-topic items. That F1 answers
"did the labeled items appear?" but not "was the digest curated?".

This script reads existing run artifacts (captured_webhooks + scenario
fabric) and recomputes metrics that match how a human reader would
judge digest quality:

- digest size: items per delivered digest (~5 is ideal)
- filler rate: non-gold items per digest (1 - precision_per_digest)
- per-window recall: fraction of gold items from the 7-day window
  before the digest that ended up in the digest (capped by digest
  capacity)
- top-K recall: of the K newest gold items in the window, how many
  landed in the digest (K = digest size)

Event scenarios are scored with the original per-item F1 unchanged.

Usage:
    python scripts/rescore_digests.py <result_dir> [<result_dir> ...]

With no args, rescans every v3 result dir under results/.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO timestamp; if naive, attach UTC."""
    cleaned = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCH_ROOT / "data" / "scenarios"
RESULTS_DIR = BENCH_ROOT / "results"


_BULLET_SPLIT = re.compile(r"(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)")


def extract_item_blocks(body: str) -> list[str]:
    """Split a digest body into one block per bullet."""
    text = body.strip()
    if not text:
        return []
    parts = _BULLET_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def token_containment(expected: str, candidate: str) -> float:
    """Return |expected_tokens ∩ candidate_tokens| / |expected_tokens|."""
    a = set(expected.lower().split())
    if not a:
        return 0.0
    b = set(candidate.lower().split())
    return len(a & b) / len(a)


def match(expected_headline: str, delivered_block: str) -> bool:
    return token_containment(expected_headline, delivered_block) >= 0.6


@dataclass
class DigestScore:
    sub_id: str
    digest_count: int
    total_items_delivered: int
    total_gold_hits: int
    total_filler: int
    mean_items_per_digest: float
    filler_rate: float
    window_recall: float
    topk_recall: float
    gold_in_all_windows: int

    def to_dict(self) -> dict:
        return {
            "sub_id": self.sub_id,
            "digest_count": self.digest_count,
            "total_items_delivered": self.total_items_delivered,
            "total_gold_hits": self.total_gold_hits,
            "total_filler": self.total_filler,
            "mean_items_per_digest": round(self.mean_items_per_digest, 2),
            "filler_rate": round(self.filler_rate, 3),
            "window_recall": round(self.window_recall, 3),
            "topk_recall": round(self.topk_recall, 3),
            "gold_in_all_windows": self.gold_in_all_windows,
        }


def score_digest_sub(
    *,
    sub_id: str,
    webhooks: list[dict],
    fabric_items: list[dict],
) -> DigestScore:
    """Compute per-digest metrics for a digest-mode subscription.

    fabric_items is the scenario timeline (each dict has headline,
    fake_ts, should_contribute_to_digest_per_sub).
    """
    gold_items = [
        it for it in fabric_items
        if (it.get("should_contribute_to_digest_per_sub") or {}).get(sub_id) is True
    ]

    total_items_delivered = 0
    total_gold_hits = 0
    total_filler = 0
    window_gold = 0
    window_hits = 0
    topk_gold_total = 0
    topk_hits_total = 0
    digest_count = 0

    for wh in webhooks:
        blocks = extract_item_blocks(wh.get("body") or "")
        if not blocks:
            continue
        digest_count += 1
        total_items_delivered += len(blocks)

        fake_clock_iso = wh.get("fake_clock_iso")
        if fake_clock_iso:
            delivered_at = _parse_iso(fake_clock_iso)
            window_start = delivered_at - timedelta(days=7)
            window_gold_items = [
                it for it in gold_items
                if window_start
                <= _parse_iso(it["fake_ts"])
                <= delivered_at
            ]
        else:
            window_gold_items = gold_items

        digest_size = len(blocks)
        topk_gold_items = sorted(
            window_gold_items,
            key=lambda it: it["fake_ts"],
            reverse=True,
        )[:max(digest_size, 1)]

        for block in blocks:
            if any(match(it["headline"], block) for it in window_gold_items):
                total_gold_hits += 1
            else:
                total_filler += 1

        for gold in window_gold_items:
            window_gold += 1
            if any(match(gold["headline"], b) for b in blocks):
                window_hits += 1

        for gold in topk_gold_items:
            topk_gold_total += 1
            if any(match(gold["headline"], b) for b in blocks):
                topk_hits_total += 1

    return DigestScore(
        sub_id=sub_id,
        digest_count=digest_count,
        total_items_delivered=total_items_delivered,
        total_gold_hits=total_gold_hits,
        total_filler=total_filler,
        mean_items_per_digest=(
            total_items_delivered / digest_count if digest_count else 0.0
        ),
        filler_rate=(
            total_filler / total_items_delivered
            if total_items_delivered
            else 0.0
        ),
        window_recall=(window_hits / window_gold) if window_gold else 0.0,
        topk_recall=(topk_hits_total / topk_gold_total) if topk_gold_total else 0.0,
        gold_in_all_windows=window_gold,
    )


def load_fabric(scenario_id: str) -> tuple[list[dict], list[dict]]:
    """Return (timeline_items, goals) from the scenario skeleton.json.

    Goals and timeline live in skeleton.json; fabric.json only stores
    the LLM-generated bodies/fluff keyed by headline hash.
    """
    skel = json.loads((DATA_DIR / scenario_id / "skeleton.json").read_text())
    return skel.get("timeline", []), skel.get("goals", [])


def rescore_record(rec_path: Path) -> dict:
    """Rescore a single scenario result; return a summary dict."""
    rec = json.loads(rec_path.read_text())
    sid = rec.get("scenario_id") or rec_path.stem.split("__")[0]
    timeline, goals = load_fabric(sid)
    webhooks = rec.get("captured_webhooks") or []
    by_url: dict[str, list[dict]] = {}
    for wh in webhooks:
        by_url.setdefault(wh.get("url") or "", []).append(wh)

    cls_default = rec.get("classification", {}).get("per_sub", {})

    per_sub: dict[str, dict] = {}
    for goal in goals:
        sub_id = goal.get("goal_id")
        mode = goal.get("expected_delivery_mode") or "digest"
        expected_url = goal.get("expected_webhook_url") or ""
        hits = by_url.get(expected_url, [])
        entry: dict = {"mode": mode, "webhook_count": len(hits)}
        if mode == "digest":
            digest_score = score_digest_sub(
                sub_id=sub_id, webhooks=hits, fabric_items=timeline
            )
            entry["digest_metrics"] = digest_score.to_dict()
        entry["default_classification"] = cls_default.get(sub_id)
        per_sub[sub_id] = entry

    return {
        "scenario_id": sid,
        "result_path": str(rec_path),
        "cost_usd": rec.get("cost", {}).get("total_usd", 0),
        "per_sub": per_sub,
        "notes": [n[:150] for n in rec.get("notes", [])],
    }


def main() -> int:
    args = sys.argv[1:]
    if args:
        roots = [Path(a) for a in args]
    else:
        roots = list(RESULTS_DIR.glob("v3_*"))
    rec_paths: list[Path] = []
    for root in roots:
        rec_paths.extend(sorted(root.glob("**/scenarios/*.json")))
    if not rec_paths:
        print("No scenario records found.")
        return 1
    summaries = [rescore_record(p) for p in rec_paths]
    out = {"summaries": summaries}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
