# Scenario `s01` diversity report

- Total timeline items: 2316
- Simulated days: 30
- Temporal top-week share: 24%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 12 | 10 ✓ |
| easy_negative | 536 | 300 ✓ |
| near_miss_negative | 1754 | 400 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `eu_energy_digest`

- Positive rate: **0.9%** (target 0%–8%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.38**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.33**

## Warnings

- subscription 'eu_energy_digest': stupid baseline F1 0.38 very low — labels may be inconsistent or noisy
