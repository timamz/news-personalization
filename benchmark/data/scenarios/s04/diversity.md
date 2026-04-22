# Scenario `s04` diversity report

- Total timeline items: 1221
- Simulated days: 30
- Temporal top-week share: 24%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 13 | 10 ✓ |
| easy_negative | 389 | 300 ✓ |
| near_miss_negative | 805 | 400 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `rare_earth_events`

- Positive rate: **1.9%** (target 0%–8%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.29**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.23**

## Warnings

- subscription 'rare_earth_events': stupid baseline F1 0.29 very low — labels may be inconsistent or noisy
