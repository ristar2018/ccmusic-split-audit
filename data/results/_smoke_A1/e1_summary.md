# E1 summary (A1 − A0)

Recording-cluster bootstrap, 200 resamples (whole recording clusters drawn with replacement, one index set shared by every arm and backbone in a library); Δ in percentage points; the seed range is the spread of the same-seed paired Δ across the three seeds.

| library | arm | backbone | Δ slice-level [95% CI] | Δ recording-level [95% CI] | seed range | above SESOI (3pt) |
| :-- | :-- | :-- | --: | --: | --: | :-- |
| CNPM | A1 | alexnet | +22.39 [+14.89, +30.09] | +26.45 [+18.36, +38.44] | 7.13 | yes |
| CNPM | A1 | swin_t | +28.71 [+21.30, +37.88] | +29.91 [+23.76, +39.84] | 2.71 | yes |
| CTIS | A1 | alexnet | +2.02 [+0.83, +3.40] | +1.67 [+0.20, +2.86] | 1.86 | no |
| CTIS | A1 | swin_t | +1.70 [+1.29, +2.80] | +1.64 [+0.79, +2.59] | 1.36 | no |
| bel_canto | A1 | alexnet | +8.42 [+4.85, +13.60] | +10.18 [+5.42, +27.10] | 2.46 | yes |
| bel_canto | A1 | swin_t | +13.06 [+6.99, +23.54] | +15.20 [+11.41, +40.67] | 2.96 | yes |
| music_genre | A1 | alexnet | +15.22 [+13.07, +17.81] | +16.07 [+11.83, +18.79] | 0.97 | yes |
| music_genre | A1 | swin_t | +17.99 [+15.25, +21.51] | +16.63 [+13.22, +19.60] | 0.89 | yes |
| pianos | A1 | alexnet | +14.62 [+11.73, +17.73] | +7.63 [+3.89, +11.62] | 2.07 | yes |
| pianos | A1 | swin_t | +11.36 [+8.44, +15.38] | +4.85 [+2.30, +8.26] | 3.21 | yes |

## Confirmatory family (Holm α=0.05)

- CTIS|alexnet|A1: Δ=+2.02 pt, bootstrap p=0.0050, Holm p=0.0100 → reject H0
- CTIS|swin_t|A1: Δ=+1.70 pt, bootstrap p=0.0050, Holm p=0.0100 → reject H0
