# Round-2 data accounting: CTIS three-arm slice budget (A) and §6.4 fold composition (B)

Accounting date: 2026-11-11 · Rerun script: `code/round2_data_accounting.py` (57 assertions, all PASS)
This file only performs accounting and disclosure and **does not modify paper/**; every number is read directly from the artifacts or recomputed by the script, and the script exit code is 0.

---

## 0. Conclusions first

**A (CTIS three-arm budget)**: the three arms are **not exactly equal per class**. A0 = 25,502, C1 = 25,502, A1 = 25,498; after a class-by-class comparison **only C0296 (唢呐) differs, by 4 slices** (A0/C1 have 241 slices each, A1 has 237). The cause is that 4 slices of C0296 in Q_pool duplicate the `content_hash` of audio in Q_test and were dropped by the global same-hash de-duplication (J4); but the `A1_slice_budget` assertion is settled **before** de-duplication, so `assertions.json` still reports `all_pass=true`. The paper should change "equal" to **approximately equal** and note C0296 and the cause.

**B (§6.4 fold composition)**: the three L-inst folds are mutually exclusive by instrument (135/135 instruments belong to exactly one fold); each fold trains on 4,477–4,528 slices and tests on 2,509–2,510 slices; the training side has 43 classes in every fold, **the test side of fold1 has only 42** (missing "自然泛音"), and fold2/3 have 43. The 43 classes = 49 technique labels minus 6 content classes, requiring "(technique, instrument) cells with ≥2 recordings across ≥3 instruments". The paper's pooled macro-F1 0.4483 / 0.5295 agrees with the A1 row of `data/results/p3/p3_metrics.csv`.

---

## 1. Data sources (files + fields)

### A: CTIS three-arm training manifests

| Purpose | File | Field |
| :-- | :-- | :-- |
| A0 training manifest | `data/splits/v2/CTIS_a0_train.csv` | `label` (class), `sample_id`, `content_hash`, `recording_uid` |
| A1 training manifest | `data/splits/v2/CTIS_a1_train.csv` | same as above |
| C1 training manifest | `data/splits/v2/CTIS_c1_train.csv` | same as above |
| Budget ledger | `data/splits/v2/feasibility.csv` | `m_c`, `qpool_slices`, `n_a0`, `n_a1`, `a0_short`, `a1_short` |
| Q_pool actual manifest | `data/splits/v2/CTIS_qpool.csv` | `label`, `content_hash` |
| Assertion record | `data/splits/v2/assertions.json` | `libraries.CTIS.ok.A1_slice_budget`, `dropped_dup_slices` |
| C1 audit | `data/splits/v2/c1_audit.json` | `assertions.K5_per_class_equal_to_a0`, `detail.n_train` |

Note: under `runs/CTIS/<arm>/<bb>/seed*/` there are only `result.json`, `train_log.csv` and `last.pt`, **no** training manifest, so training samples can only be traced back through the frozen CSVs in `data/splits/v2/`. The `sample_id` values in the three arm CSVs are drawn afresh per arm and **cannot be aligned across arms by sample_id**; this accounting counts by `label` and compares by `content_hash`.

### B: §6.4 fold definition and use

| Purpose | File | Field |
| :-- | :-- | :-- |
| L-inst fold definition | `data/p3/p3_splits.csv` | `fold` (1–3), `technique`, `instrument_canon`, `recording_uid` |
| R-rec (B1) fold definition | `data/p3/p3b_splits.csv` | `fold_b1` |
| Fold description | `data/p3/p3_folds.json`, `data/p3/p3b_folds.json` | `classes`, `per_fold`, `assertions`, `disclosed` |
| Actual train/test | `runs_p3/<arm>/<bb>/seed0/fold<f>_result.json` | `n_train`, `n_val`, `n_test`, `n_classes`, `n_inst` |
| Actual test set | `runs_p3/<arm>/<bb>/seed0/fold<f>.npz` | `records` (recordings), `y` (labels) |
| Class universe | `data/p3/universe_before_drop.csv`, `universe_43.csv`, `P3_universe.json` | `technique`, `recordings`, `instruments` |
| Metrics | `data/results/p3/p3_metrics.csv`, `data/results/p3b_instr/p3_metrics.csv` | `macro_f1_slice` |

---

## 2. A: per-class slice budget of the CTIS three arms

### 2.1 Verdict

| Arm | Training slices | Classes | Difference from A0 |
| :-- | --: | --: | :-- |
| A0 (clean baseline) | 25,502 | 201 | — |
| A1 (equal-size pool substitution) | 25,498 | 201 | **−4**, all in C0296 |
| C1 (control) | 25,502 | 201 | 0 (equal class by class across all 201) |

The full per-class comparison is in `data/results/human_review/round2/slice_budget_by_class.csv` (`lib,arm,class,n_slices`, 603 rows = 3 arms × 201 classes). 4/25,502 = **0.0157%** (about 0.016%, consistent with the reviewer's estimate).

### 2.2 Classes that differ

| Class | A0 | A1 | C1 | Difference |
| :-- | --: | --: | --: | --: |
| C0296 (唢呐) | 241 | 237 | 241 | A1 −4 |
| The other 200 classes | — | — | — | 0 |

### 2.3 Causal chain

1. The per-class budget in `feasibility.csv`: C0296 `m_c=241`, `qpool_slices=39`, `n_a1=241`, `a1_short=0`. All 201 CTIS classes have `a1_short=0`; the ledger gives `Σqpool_slices=4,182` and `Σn_a1=25,502`.
2. The actual `CTIS_qpool.csv` totals **4,178** slices, and only C0296 differs class by class (39 → 35, 4 fewer). A1 = Q_pool plus the C-side top-up, so C0296 in A1 keeps only 237 (35 + 202).
3. De-duplication rule: `code/splits_v2.py:343-367` `trim_global_dups` (J4) keeps only the highest-priority copy of the same `content_hash` across sides, with priority `qtest > qpool > r > c`; the allowed sides for `a1_train` are `{c, qpool}`. This run dropped **12** rows = 4 `content_hash` values × 3 keys (`qpool`, `a1_train`, `a2_train`), all belonging to C0296, recording "C0296/【唢呐】演奏技法 垫音 第二遍", with `qtest` as the surviving side in every case. The 4 hash prefixes: `6b3fd0cdeaefc3bc`, `a13ebc00004afe50`, `91a789f6dae7c286`, `ab51db544d703020`.
4. Assertion timing: the class loop collects `cls_stats` at `code/splits_v2.py:415`; `trim_global_dups` runs only at `:418`; but `A1_slice_budget` at `:476-495` (specifically `:486`) uses the **pre-trim** `cls_stats`. So `assertions.json` records `A1_slice_budget=true`, `all_pass=true`, `dropped_dup_slices=12`, which contradicts the final CSV's 25,498 — not a data error but **an assertion settlement basis that predates de-duplication**.
5. A0 has no Q_pool side, and C1 is drawn back per class by the actual injection count (`code/c1_inject.py:121-126`, `:154`), so both keep 241; `K5_per_class_equal_to_a0=true` and `n_train=25,502` in `c1_audit.json` also confirm that C1 equals A0 class by class.

### 2.4 Assertions (script output, all PASS)

- Row counts of A0/A1/C1 = 25,502 / 25,498 / 25,502; each arm has 201 classes.
- Per-class difference A0−A1 = `{C0296: 4}`; per-class difference A0−C1 = `{}`.
- qpool planned 4,182 vs actual 4,178, difference `{C0296: 4}`; in feasibility `Σn_a1=25,502` and `a1_short` is 0 throughout (pre-de-duplication basis).
- `assertions.json` `A1_slice_budget=true`, `dropped_dup_slices=12`; `c1_audit.json` `K5=true`, `n_train=25,502`.

---

## 3. B: §6.4 fold composition

### 3.1 Fold definition

- **L-inst (paper §6.4 main line, A1/A2)**: the `fold` column of `data/p3/p3_splits.csv`, grouped by instrument, same instrument in the same fold, greedy slice-count balancing; all 135 canonical instruments **appear in exactly one fold** (assertion measured multi=0). Fold sizes 2,509 / 2,509 / 2,510 slices.
- **R-slice (A0, optimistic reference)**: `split_a0` at `code/train_p3.py:85-96`, sorting by `sample_id` within a class, shuffling with `random.Random(20261105)` and assigning folds by `i % 3 + 1`; **slices of the same recording can cross folds** (the script docstring states this is the shortcut under criticism). Measured intersection of fold1 training and test recordings: **819**.
- **R-rec (A1r)**: the `fold_b1` column of `data/p3/p3b_splits.csv`, grouped by recording, allowing 110 instruments to appear in two folds at once. Checking the npz `records` fold by fold, A1r matches `fold_b1` and does not match `fold`.
- val split (identical across arms): shuffle `sorted(recording_uid)` with `random.Random(20261105+fold)` and take the first `max(1, int(0.1*n))` recordings (`code/train_p3.py:282-299`).

### 3.2 Fold composition (`p3_fold_composition.csv`)

| fold | arm | Train recordings | Train slices | Test recordings | Test slices | Train classes | Test classes |
| --: | :-- | --: | --: | --: | --: | --: | --: |
| 1 | A0 | 1,151 | 4,522 | 1,019 | 2,524 | 43 | 43 |
| 2 | A0 | 1,146 | 4,520 | 1,014 | 2,508 | 43 | 43 |
| 3 | A0 | 1,164 | 4,519 | 997 | 2,496 | 43 | 43 |
| 1 | A1 | 829 | 4,528 | 467 | 2,509 | 43 | **42** |
| 2 | A1 | 818 | 4,494 | 480 | 2,509 | 43 | 43 |
| 3 | A1 | 853 | 4,477 | 441 | 2,510 | 43 | 43 |
| 1 | A2 | 829 | 4,528 | 467 | 2,509 | 43 | **42** |
| 2 | A2 | 818 | 4,494 | 480 | 2,509 | 43 | 43 |
| 3 | A2 | 853 | 4,477 | 441 | 2,510 | 43 | 43 |
| 1 | A1r | 835 | 4,511 | 461 | 2,509 | 43 | 43 |
| 2 | A1r | 834 | 4,528 | 462 | 2,509 | 43 | 43 |
| 3 | A1r | 831 | 4,514 | 465 | 2,510 | 43 | 43 |

`fold` means different things per arm (the CSV distinguishes them by the `arm` column): A0 = `fold_a0` (slice-level random folds), A1/A2 = `fold` (L-inst), A1r = `fold_b1` (R-rec); definitions in §3.1. A2 and A1 have identical rows, folds and val, differing only in GRL.

Check: all 12 rows agree digit for digit with `n_train/n_val/n_test` in `fold<f>_result.json`; per fold, the test slice count equals the npz `y` length and the test recording set equals the npz `records` set.

### 3.3 Inclusion rule for the 43 classes (`code/p3_universe.py`)

The unit is `recording_uid`, not slices. First count recordings per `(technique, instrument)` cell; the requirements are:

- `MIN_CELL = 2`: ≥2 recordings in the cell;
- `MIN_INSTR = 3`: at least 3 instruments have ≥2 recordings of that technique;
- additionally exclude 6 content-class labels (not playing techniques): `音阶`, `空弦音`, `筒音`, `单音`, `乐曲片段`, `特有演奏技法`.

Result: **49 classes / 4,037 recordings → 43 classes / 1,388 recordings**; instrument names 212 → 204 (6 groups, 14 names merged), and fold grouping uses the 135 canonical names after merging. All 6 excluded labels are the content classes above, not threshold eliminations (details in `universe_before_drop.csv` and `P3_universe.json`).

### 3.4 Checking the paper's 0.4483 / 0.5295

| Artifact | Field | alexnet | swin_t |
| :-- | :-- | --: | --: |
| `data/results/p3/p3_metrics.csv` | `arm=A1, macro_f1_slice` | 0.4483 | 0.5295 |
| Recomputed by the script from npz (full-precision mean of two seeds) | `_pooled_seed_f1` | 0.44835 → 0.4483 | 0.52945 → 0.5295 |
| `runs_p3/A1/<bb>/seed{0,1}/pooled.json` | `macro_f1_slice` | 0.4503 / 0.4464 | 0.5435 / 0.5154 |

The paper's numbers **agree** with the artifacts. Note: averaging the seed values already rounded to 4 places in `pooled.json` would give 0.4484 / 0.5294; the paper's basis is the mean of full-precision seed F1 followed by `round(...,4)` at `code/analyze_p3.py:142-146,171`, so 0.4483 / 0.5295 is correct and there is no discrepancy.

Also: A0/A1/A2 in `data/results/p3b_instr/p3_metrics.csv` are **identical digit for digit** to `p3/p3_metrics.csv` (the same batch of p3 training artifacts; only A1r is retrained on `fold_b1`), so do not cite them as two independent experiments.

---

## 4. Gaps and recommendations

1. **Paper wording**: wherever "equal per-class budget" appears in §5/§6 it should read approximately equal, with one added sentence: "A1 has 4 fewer slices in C0296, because 4 slices in Q_pool duplicate Q_test audio and were dropped by same-hash de-duplication". The current draft is unchanged (paper/ is read-only).
2. **Assertion basis**: `A1_slice_budget` in `data/splits/v2/assertions.json` settles before trim; add a J5 after `trim_global_dups` (post-trim per-class == m_c), otherwise `all_pass=true` will keep hiding differences of this kind.
3. **L-inst fold1 class coverage**: the test side has 42/43 ("自然泛音" has no test recording), already recorded by `disclosed.A3_missing_class_fold_cells=1` in `p3_folds.json`; the paper body states only that R-rec has 43 classes per fold and does not mention this L-inst gap, so add a sentence to the table note or cite `P3_universe.json`.
4. **Not verifiable**: whether the paper PDF/`.tex` has been updated as above, and whether other places in `paper/` restate 25,502/25,498 — the draft was not edited this round and the authors must make it consistent.
5. This accounting does not cover the §6.4 Δ/CI recomputation (other artifacts such as `a1_minus_c1.csv` come from a parallel task and their values are not cited here).

---

## 5. Rerun

```bat
cd /d D:\VSCode\科研任务\音乐情感研究
D:\XiaBao\AIArtDetect\envs\fapn\python.exe code\round2_data_accounting.py
```

Output: `data/results/human_review/round2/slice_budget_by_class.csv`, `p3_fold_composition.csv`; the last line is `ALL PASS`, and any failing assertion exits with code 1. The script reads only `data/` and `runs_p3/`, writes nothing to `paper/` and does not change the splits.
