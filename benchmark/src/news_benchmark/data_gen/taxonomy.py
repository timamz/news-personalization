"""
Difficulty taxonomy and per-scenario minimum counts.

Every TimelineEntry carries a `difficulty` tag. The data generation
pipeline rejects a scenario that falls below the minimum count for any
tier. Near-miss negatives get the highest floor because that's the band
where models actually differ.

Tiers:
    easy_positive       - headline literally names the topic. Recall floor.
    hard_positive       - relevant via synonyms or implication; needs body.
    easy_negative       - plainly off-topic. Precision floor.
    near_miss_negative  - shares vocabulary with positives, wrong scope.
    adversarial         - clickbait, body contradicts headline, editorial
                          content excluded by user_spec. Tests rule-following.
    duplicate           - near-identical to an earlier item. Tests dedup.
"""

from __future__ import annotations

DIFFICULTY_TIERS = (
    "easy_positive",
    "hard_positive",
    "easy_negative",
    "near_miss_negative",
    "adversarial",
    "duplicate",
)

MIN_COUNTS_PER_SCENARIO: dict[str, int] = {
    "easy_positive": 6,
    "hard_positive": 10,
    "easy_negative": 30,
    "near_miss_negative": 25,
    "adversarial": 4,
    "duplicate": 3,
}

TARGET_POSITIVE_RATE = (0.10, 0.35)
