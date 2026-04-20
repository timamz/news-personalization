# Scenario `s02` diversity report

- Total timeline items: 248
- Simulated days: 30
- Temporal top-week share: 27%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 12 | 10 ✓ |
| easy_negative | 86 | 80 ✓ |
| near_miss_negative | 136 | 100 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `eu_energy_digest`

- Positive rate: **8.9%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.61**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.16**
