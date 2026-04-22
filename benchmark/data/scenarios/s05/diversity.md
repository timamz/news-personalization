# Scenario `s05` diversity report

- Total timeline items: 4491
- Simulated days: 30
- Temporal top-week share: 24%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 20 | 6 ✓ |
| hard_positive | 32 | 10 ✓ |
| easy_negative | 1219 | 300 ✓ |
| near_miss_negative | 3206 | 400 ✓ |
| adversarial | 8 | 4 ✓ |
| duplicate | 6 | 3 ✓ |

## Per-subscription label stats

### `eu_energy_digest`

- Positive rate: **0.9%** (target 0%–8%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.38**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.33**

### `eu_ai_regulation_digest`

- Positive rate: **1.7%** (target 0%–8%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.74**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.23**

### `rare_earth_events`

- Positive rate: **1.6%** (target 0%–8%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.24**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.19**

## Warnings

- subscription 'eu_energy_digest': stupid baseline F1 0.38 very low — labels may be inconsistent or noisy
- subscription 'eu_ai_regulation_digest': stupid baseline F1 0.74 too high — scenario is lexically separable; add more near_miss_negatives
- subscription 'rare_earth_events': stupid baseline F1 0.24 very low — labels may be inconsistent or noisy
