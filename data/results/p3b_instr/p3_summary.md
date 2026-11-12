# P3 pooled results (2000 recording-cluster bootstrap resamples)

Primary metric: pooled macro-F1 over the three folds; reference arm = A1r.

## Point estimates

| arm | backbone | slice-level | recording-level | seed range |
| :-- | :-- | --: | --: | --: |
| A0 | alexnet | 0.8446 | 0.8797 | 0.015 |
| A0 | swin_t | 0.8772 | 0.9041 | 0.008 |
| A1 | alexnet | 0.4483 | 0.4917 | 0.004 |
| A1 | swin_t | 0.5295 | 0.5676 | 0.028 |
| A1r | alexnet | 0.8157 | 0.8640 | 0.009 |
| A1r | swin_t | 0.8431 | 0.8799 | 0.003 |
| A2 | alexnet | 0.4447 | 0.4714 | 0.005 |
| A2 | swin_t | 0.5012 | 0.5467 | 0.017 |

## Paired Δ (percentage points)

| arm − reference | backbone | Δ slice-level [95% CI] | Δ recording-level [95% CI] | p | confirmatory |
| :-- | :-- | --: | --: | --: | :-- |
| A0 − A1r | alexnet | +2.90 [+0.32, +5.66] | +1.58 [-1.33, +4.95] | 0.0280 | — |
| A0 − A1r | swin_t | +3.40 [+0.68, +6.36] | +2.43 [-0.61, +5.63] | 0.0220 | — |
| A1 − A1r | alexnet | -36.73 [-39.89, -33.31] | -37.23 [-41.02, -33.72] | 0.0000 | — |
| A1 − A1r | swin_t | -31.37 [-34.97, -28.32] | -31.23 [-35.76, -27.99] | 0.0000 | — |
| A2 − A1r | alexnet | -37.10 [-40.37, -33.77] | -39.26 [-43.38, -35.74] | 0.0000 | yes |
| A2 − A1r | swin_t | -34.20 [-37.67, -31.13] | -33.32 [-37.58, -30.06] | 0.0000 | yes |

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
