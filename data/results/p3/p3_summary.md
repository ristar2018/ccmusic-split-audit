# P3 pooled results (2000 recording-cluster bootstrap resamples)

Primary metric: pooled macro-F1 over the three folds; reference arm = A1.

## Point estimates

| arm | backbone | slice-level | recording-level | seed range |
| :-- | :-- | --: | --: | --: |
| A0 | alexnet | 0.8446 | 0.8797 | 0.015 |
| A0 | swin_t | 0.8772 | 0.9041 | 0.008 |
| A1 | alexnet | 0.4483 | 0.4917 | 0.004 |
| A1 | swin_t | 0.5295 | 0.5676 | 0.028 |
| A2 | alexnet | 0.4447 | 0.4714 | 0.005 |
| A2 | swin_t | 0.5012 | 0.5467 | 0.017 |

## Paired Δ (percentage points)

| arm − reference | backbone | Δ slice-level [95% CI] | Δ recording-level [95% CI] | p | confirmatory |
| :-- | :-- | --: | --: | --: | :-- |
| A0 − A1 | alexnet | +39.63 [+36.84, +42.38] | +38.81 [+35.76, +42.50] | 0.0000 | — |
| A0 − A1 | swin_t | +34.77 [+32.00, +38.08] | +33.66 [+30.65, +38.12] | 0.0000 | — |
| A2 − A1 | alexnet | -0.36 [-1.75, +0.91] | -2.03 [-3.74, -0.50] | 0.5550 | yes |
| A2 − A1 | swin_t | -2.83 [-4.81, -0.89] | -2.09 [-4.14, +0.10] | 0.0030 | yes |

## Instrument-identity probe (linear classifier on frozen features)

| arm | backbone | seed | AUC (macro OvR) | top1 | folds |
| :-- | :-- | --: | --: | --: | --: |
| A0 | alexnet | 0 | 0.9383 | 0.5567 | 3 |
| A0 | alexnet | 1 | 0.9264 | 0.5533 | 3 |
| A0 | swin_t | 0 | 0.9425 | 0.632 | 3 |
| A0 | swin_t | 1 | 0.9363 | 0.6333 | 3 |
| A1 | alexnet | 0 | 0.9619 | 0.6657 | 3 |
| A1 | alexnet | 1 | 0.9606 | 0.6706 | 3 |
| A1 | swin_t | 0 | 0.9668 | 0.7308 | 3 |
| A1 | swin_t | 1 | 0.9726 | 0.7333 | 3 |
| A2 | alexnet | 0 | 0.9327 | 0.5969 | 3 |
| A2 | alexnet | 1 | 0.9194 | 0.5935 | 3 |
| A2 | swin_t | 0 | 0.9321 | 0.6674 | 3 |
| A2 | swin_t | 1 | 0.9412 | 0.6711 | 3 |
