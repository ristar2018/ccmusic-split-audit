# E1 summary (A1 − A0)

Recording-cluster bootstrap, 2000 resamples (whole recording clusters drawn with replacement, one index set shared by every arm and backbone in a library); Δ in percentage points; the seed range is the spread of the same-seed paired Δ across the three seeds.

| library | arm | backbone | Δ slice-level [95% CI] | Δ recording-level [95% CI] | seed range | above SESOI (3pt) |
| :-- | :-- | :-- | --: | --: | --: | :-- |
| CNPM | A1 | alexnet | +22.39 [+14.60, +29.99] | +26.45 [+17.82, +38.70] | 7.13 | yes |
| CNPM | A1 | swin_t | +28.71 [+19.75, +37.52] | +29.91 [+21.66, +39.57] | 2.71 | yes |
| CTIS | A1 | alexnet | +2.02 [+0.83, +3.35] | +1.67 [+0.26, +2.83] | 1.86 | no |
| CTIS | A1 | swin_t | +1.70 [+1.28, +2.84] | +1.64 [+0.87, +2.60] | 1.36 | no |
| bel_canto | A1 | alexnet | +8.42 [+5.18, +13.80] | +10.18 [+5.71, +26.23] | 2.46 | yes |
| bel_canto | A1 | swin_t | +13.06 [+7.50, +24.75] | +15.20 [+9.28, +41.28] | 2.96 | yes |
| music_genre | A1 | alexnet | +15.22 [+13.07, +17.53] | +16.07 [+11.97, +18.91] | 0.97 | yes |
| music_genre | A1 | swin_t | +17.99 [+15.29, +21.20] | +16.63 [+12.88, +19.78] | 0.89 | yes |
| pianos | A1 | alexnet | +14.62 [+11.89, +18.28] | +7.63 [+3.90, +12.26] | 2.07 | yes |
| pianos | A1 | swin_t | +11.36 [+8.15, +15.26] | +4.85 [+2.07, +8.43] | 3.21 | yes |

## Confirmatory family (Holm α=0.05)

- CTIS|alexnet|A1: Δ=+2.02 pt, bootstrap p=0.0005, Holm p=0.0010 → reject H0
- CTIS|swin_t|A1: Δ=+1.70 pt, bootstrap p=0.0005, Holm p=0.0010 → reject H0
