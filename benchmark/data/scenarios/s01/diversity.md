# Scenario `s01` diversity report

- Total timeline items: 216
- Simulated days: 30
- Temporal top-week share: 26%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 7 | 6 ✓ |
| hard_positive | 12 | 10 ✓ |
| easy_negative | 86 | 80 ✓ |
| near_miss_negative | 104 | 100 ✓ |
| adversarial | 4 | 4 ✓ |
| duplicate | 3 | 3 ✓ |

## Per-subscription label stats

### `eu_energy_digest`

- Positive rate: **10.2%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.63**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.19**
- Label-consistency on a 42-item sample: **0%** disagreement (threshold 20%)
