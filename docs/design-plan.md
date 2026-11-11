# Design plan (released version)

This document records the design that the artifacts in this repository implement, in the form released with the manuscript. It is deliberately free of machine-specific paths and of working notes.

## 1. Question

A multi-library Chinese music information retrieval framework states an 8:1:1 train/validation/test ratio over *slices*. If that ratio is realised by splitting slices at random, slices cut from the same recording can land on both sides of the split, so the test set contains recordings the model has already seen. This audit asks how large the resulting difference is, whether it survives a content-matched control, and what it costs to remove.

The audit does **not** claim that the audited framework's authors split at the slice level, and does not claim to recover their split files. It reconstructs one slice-level reading of the stated ratios and quantifies the consequence of that reading.

## 2. Libraries

Five libraries form the main study: CTIS, music_genre, pianos, CNPM and bel canto. Three further libraries with one slice per recording (chest_falsetto, erhu_playing_tech, GZ_IsoTech) serve as negative controls, where the recording-level correction is expected to have no effect by construction.

All libraries are used under the terms of their original publishers. Raw audio and image features are not redistributed here.

## 3. Arms

| arm | training side | purpose |
| :-- | :-- | :-- |
| A0 | slices from recordings disjoint from test (recording-level split) | reference |
| A1 | slice-level split; training slices may come from test recordings | the audited reading |
| A2 | random 80/10/10 over slices, with a self-test set | the stated-ratio reading, nearest to the original protocol |
| C1 | A1 plus an equal volume of non-test content | content-matched control isolating the exposure term |
| R-slice / R-rec / L-inst | cross-instrument stages (see §5) | protocol decomposition |

A1−A0 is the confirmatory contrast; A1−C1 separates "more training data" from "training on test content".

## 4. Split construction (four steps, frozen once written)

1. **Class universe** — a class enters a library only if it has at least five recordings.
2. **Partition** — recordings are assigned to Q (20%), R (15%) and C (the remainder), by recording, before any slice is touched.
3. **Q subdivision** — slices of Q recordings are grouped by content hash and adjacent window, so that near-duplicate slices stay together; even groups go to Q_test, odd groups to Q_pool (a single-slice recording goes wholly to Q_test).
4. **Training budget** — `m_c = 0.60 · S_c` slices per class, matched on the recording count `n_c` within a 2% tolerance; `α_c` (the realised exposure share) is reported per class but is **not** forced to a target.

Assertions are written per arm: A0 requires training/validation/test recordings to be mutually disjoint; A1 permits test recordings on the training side; A2 permits recording overlap outright. An assertion failure stops the pipeline; nothing is skipped and patched later.

## 5. Cross-instrument decomposition

On CTIS, a 43-class / 1,388-recording universe is partitioned by instrument into three folds, so that every recording is predicted exactly once. Three stages are reported separately — R-slice, R-rec and L-inst — so that the drop attributable to the protocol can be read apart from the drop attributable to unseen instruments.

## 6. Pre-specified analysis

* Confirmatory family: two tests (CTIS × AlexNet, CTIS × Swin-T) on A1−A0.
* Smallest effect size of interest (SESOI): 3 percentage points, fixed before any Δ was seen.
* Multiplicity: Holm at α = 0.05 across the two confirmatory tests.
* Intervals: 2,000 bootstrap replicates resampling **recordings** (not slices), so that within-recording dependence is respected.
* Seed spread is reported as a range and is not treated as an interval.
* Two metrics are reported side by side — slice-level macro-F1 and recording-level macro-F1 (one vote per recording) — and are never mixed.

## 7. Gates

| gate | criterion |
| :-- | :-- |
| G-A data | recording-id extraction ≥ 99%; A0 recordings mutually disjoint; 100% identity spot-check |
| G-B smoke | two seeds of a smoke run differ by < 1 point macro-F1 |
| G-C effect | lower bound of the 95% CI of the confirmatory Δ above zero |
| G-D writing | every figure and table traceable to a file under `data/results/` |

The SESOI, the class universe and the backbone pair were fixed before the confirmatory results were inspected. Anything that changed afterwards is recorded in `execution-changes.md` and, where it affects interpretation, in the limitations of the manuscript.

## 8. Hard constraints

* Original dataset directories are read-only; no analysis script writes into them.
* Frozen artifacts under `data/splits/` and `runs/sealed_eval/` are not edited by hand.
* Selection uses only the R side; the test side is unsealed after all seeds of an arm complete.
