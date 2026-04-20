# Scenario `s03` diversity report

- Total timeline items: 218
- Simulated days: 30
- Temporal top-week share: 26%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 10 | 10 ✓ |
| easy_negative | 89 | 80 ✓ |
| near_miss_negative | 105 | 100 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `rare_earth_events`

- Positive rate: **9.2%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.58**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.12**

## Warnings

- subscription 'rare_earth_events': lexical overlap 0.12 too low — positive/negative pools use disjoint vocabularies
