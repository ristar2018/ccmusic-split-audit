# Re-check of the single-seed A2−A0 rows of the negative control (2026-11-06)

In `e1_delta.csv` the six rows for chest_falsetto / erhu_playing_tech / GZ_IsoTech are all single-seed (seed0) runs, so `seed_range_pt` is empty.
Among them **chest_falsetto/swin_t A2−A0 = +5.57 pt [95% CI +0.62, +10.31] (the script sets `meets_sesoi` to true) cannot be taken at face value**, for five reasons:

1. n_seeds=1, so there is no seed average and no seed range. The 20 three-seed rows in the same table span 0.87–8.29 pt (median 2.1), and 5.57 sits inside single-seed jitter.
2. A0 and A2 do not share a data pool: A0 = e1f_drv (n_train 768 / n_val 190), A2 = e1f2 (666 / 83). Q_test is the same (n=257, y/records identical element by element), but Δ mixes in a partition difference and is not a pure arm effect.
3. On that group A0 scores 0.8261 on Q_test, 6.8 pt below its own val_best of 0.8940 (a negative-side gap for this batch); the A2 gap is −0.3 pt.
4. The CI is 9.7 pt wide and its lower bound of 0.62 barely clears zero. A2 is not in the confirmatory family (CONFIRM_ARM=A1), so no Holm support applies.
5. Per-class ΔF1 is +3.55 / +3.98 / +4.70 / +10.06: no single class collapses, but class 4 recall moving from 0.717 to 0.804 accounts for roughly half of it.

Verdict: the row is recorded as a single-seed observation pending re-check, and is **not** written up as clearing the SESOI gate. The re-check is to re-run A0 and A2 on the same partition, three seeds each, seal the runs and recompute the row.
The other five rows have |Δ| < 1.6 pt, as the negative control predicts.
