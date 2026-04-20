# Scenario `s03` diversity report

- Total timeline items: 91
- Simulated days: 30
- Temporal top-week share: 27%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 10 | 10 ✓ |
| easy_negative | 32 | 30 ✓ |
| near_miss_negative | 35 | 25 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `rare_earth_events`

- Positive rate: **22.0%** (target 10%–35%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.77**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.12**
- Label-consistency on a 18-item sample: **0%** disagreement (threshold 20%)

## Warnings

- subscription 'rare_earth_events': stupid baseline F1 0.77 too high — scenario is lexically separable; add more near_miss_negatives
- subscription 'rare_earth_events': lexical overlap 0.12 too low — positive/negative pools use disjoint vocabularies
