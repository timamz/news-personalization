# Scenario `s04` diversity report

- Total timeline items: 221
- Simulated days: 30
- Temporal top-week share: 26%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 13 | 10 ✓ |
| easy_negative | 89 | 80 ✓ |
| near_miss_negative | 105 | 100 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `rare_earth_events`

- Positive rate: **10.4%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.60**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.14**

## Warnings

- subscription 'rare_earth_events': lexical overlap 0.14 too low — positive/negative pools use disjoint vocabularies
