# Scenario `s05` diversity report

- Total timeline items: 541
- Simulated days: 30
- Temporal top-week share: 26%

## Tier counts

| Tier | Count | Minimum |
|---|---:|---:|
| easy_positive | 20 | 6 ✓ |
| hard_positive | 32 | 10 ✓ |
| easy_negative | 219 | 80 ✓ |
| near_miss_negative | 256 | 100 ✓ |
| adversarial | 8 | 4 ✓ |
| duplicate | 6 | 3 ✓ |

## Per-subscription label stats

### `eu_ai_regulation_digest`

- Positive rate: **15.0%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.75**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.18**

### `eu_energy_digest`

- Positive rate: **10.2%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.63**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.19**

### `rare_earth_events`

- Positive rate: **9.2%** (target 4%–15%)
- Stupid-baseline (TF-IDF + LR, 5-fold CV) F1: **0.72**
  (target band 0.50–0.70)
- Positive/negative lexical overlap (Jaccard top-50): **0.10**

## Warnings

- subscription 'eu_ai_regulation_digest': stupid baseline F1 0.75 too high — scenario is lexically separable; add more near_miss_negatives
- subscription 'rare_earth_events': lexical overlap 0.10 too low — positive/negative pools use disjoint vocabularies
