# Scenario `s01` diversity report

- Total timeline items: 100
- Simulated days: 30
- Temporal top-week share: 28%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 12 | 10 ✓ |
| easy_negative | 32 | 30 ✓ |
| near_miss_negative | 42 | 25 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `eu_energy_digest`

- Positive rate: **22.0%** (target 10%–35%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.58**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.27**
- Label-consistency on a 19-item sample: **0%** disagreement (threshold 20%)
