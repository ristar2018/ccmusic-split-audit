# ccmusic-split-audit

**Within-recording exposure in a slice-level reconstruction: an audit of a multi-library Chinese music information retrieval framework**

> **Unofficial audit.** This repository is an independent re-analysis. It is not affiliated with, endorsed by, or produced in collaboration with the CCMusic authors or the ModelScope `ccmusic-database` organisation. The audited article is identified in `CITATION.cff` and in the manuscript.

This repository holds the analysis code, the frozen splits, the sealed test-set predictions and the verification scripts needed to re-derive every number reported in the manuscript and its supplementary material.

## What is here

| path | contents |
| :-- | :-- |
| `code/` | Every script: manifest and split construction (`manifest.py`, `splits_v2.py`, `negcontrol.py`), feature cache (`ccmusic_cache.py`), training (`train_e1.py`, `train_p3.py`), analysis (`analyze_e1.py`, `analyze_p3.py`, `sealed_eval_delta.py`), round-2 additions (`round2_a1_minus_c1.py`, `round2_macro_denominator.py`, `round2_data_accounting.py`), figure scripts, audits (`audit_*.py`, including `audit_cross_library_duplicates.py`), the inventory and convergence summary (`summarise_library_inventory.py`) and the number verifier (`verify_paper_numbers.py`) |
| `data/manifest/` | Per-slice manifests (`<lib>.csv`) with recording id, class, candidate group and content hash; `summary.json` records per-library counts and the audio-hash status |
| `data/splits/v2/` | Frozen split CSVs, one file per library and side (`<lib>_qtest.csv`, `_qpool.csv`, `_a0_train.csv`, `_a1_train.csv`, `_a2_train.csv`, `_a2_selftest.csv`, `_r.csv`, `_c.csv`), plus `_audit_summary.csv`, `assertions.json` and `feasibility.csv` |
| `data/results/` | Metric tables and bootstrap summaries, `paper_verify.json` (verifier report) and `human_review/round2/` (A1−C1, exposure split, slice budget, fold composition, macro-denominator sensitivity) |
| `data/p3/` | Cross-instrument artifacts and the 43-class universe (`universe_43.csv`, `universe_before_drop.csv`, `P3_universe.json`) |
| `runs/sealed_eval/` | Sealed test-set predictions, per `<library>_<arm>_<backbone>/seed<k>_qtest.npz` (arrays `probs`, `y`, `records`) plus `SEALED.json` |
| `figures/` | The three figures exactly as embedded in the manuscript (PDF and PNG) |
| `paper/` | Manuscript sources (`main_sn.tex`, `supplement_s1.tex`, `sections/*.tex`, `refs.bib`, `bst/sn-basic.bst`), included so that the verifier's text layer can run; compiled PDFs are not included |
| `docs/` | The design plan, the execution-change list and the expected verifier output, in the form released with the manuscript |
| `SHA256SUMS.txt` | sha256 of every file in this repository, in `sha256sum -c` format |

## Verify every number (about one minute)

```bash
python code/verify_paper_numbers.py
```

Expected last lines:

```
checks passed: 337, failed: 0
report: data/results/paper_verify.json
```

The verifier works in four groups: (1) *identity* — re-derives rounded means, deltas, sums, ratios and cost totals from the artifact tables; (2) *text* — rebuilds table rows and reported tokens from the manuscript source and requires each to appear; (3) *round-2 additions* — A1−C1 cells and their transitivity, the exposure split, fold composition, the class-set table, the 43-class universe and the bel canto ceiling; (4) *round-3 additions* — the arm table, the thirteen-library inventory, the threshold-sensitivity sweep, the excluded-class profile, the cross-library duplicate count and the end-of-training losses, each recomputed from `data/results/library_inventory_and_convergence.json` and the split files. It reads the CSVs and JSONs, **not** the `.npz` predictions: the bootstrap intervals are produced by the analysis scripts from the sealed predictions, which the verifier does not re-run.

## Re-derive the bootstrap intervals

```bash
python code/round2_a1_minus_c1.py        # A1-C1 and the exposure split, 2,000 recording-cluster replicates
python code/round2_macro_denominator.py  # recomputed / fixed / full class set
python code/sealed_eval_delta.py         # A1-A0 and other paired contrasts, per library and arm
python code/round2_data_accounting.py    # per-class budgets and fold composition, 57 assertions
```

All read `runs/sealed_eval/` and finish in about a minute on one CPU core.

## What is not here

* **Raw audio and image features** (35.47 GB, 13 libraries, 328,973 files). They are released by the original publisher under its own terms and are **not** redistributed here. Five libraries from the ModelScope organisation `ccmusic-database` were used: CTIS, CNPM, pianos and bel canto ship audio; music genre ships feature images only.
* **Feature cache** (115,847 images: 110,415 study slices + 5,432 control-library slices) and **trained checkpoints**. No reported number needs them; every interval is computed from the sealed predictions in `runs/sealed_eval/`.
* **Compiled manuscript PDFs.** The LaTeX sources are included so that the verifier can run its text layer; rebuilding them requires the Springer Nature `sn-jnl` class from CTAN.

Scripts in `code/` derive the repository root from their own location, so the analysis, verification and round-2 chains run unmodified. A few early data-preparation scripts (`ccmusic_cache.py`, `ccmusic_extract.py`, `datasets_manifest.py` and similar) still carry the absolute dataset path of the machine on which the study ran; point `DATA` / `DATASETS_ROOT` at your own copy before running those.

## Environment used

Python 3.10.22, PyTorch 2.11.0+cu128 (cuDNN 9.19), one NVIDIA RTX 5060 Ti 16 GB, Windows 11 25H2, `pdflatex` + `bibtex` with the Springer Nature `sn-jnl` class.

## Scope and known limits

These are carried in the manuscript itself and are not artefacts of this deposit.

* One split realisation and three training seeds; no bootstrap creates evidence about new splits.
* Compute timing comes from integer-minute driver logs, incomplete and rounded; no exact study-wide GPU-hour total is claimed.
* The reconstruction of the stated 8:1:1 ratios is **one** slice-level reading. The original authors' split files are unavailable and this audit does not claim to recover them.
* A2's self-test score is illustrative and is not the original benchmark score; in one library a small number of self-test images repeat content that also appears in A2 training (11 slices, reported in Supplementary Material S1).
* The instrument-grouped partition covers 42 of 43 classes in its first fold (one technique class has no test recording there); the pooled score averages over the classes present.
* The bel canto recording-level contrast is bounded by a ceiling: the control arm reaches the low 0.9s and the exposure arm saturates at 1.00 for most seeds.

## Data and licensing

Code and derived artifacts (manifests, splits, sealed predictions, result tables) are released under the MIT License — see `LICENSE`. No third-party audio, images or model weights are redistributed; those remain governed by the terms of their original publishers.

## Citation

If you use these artifacts, please cite the manuscript (see `CITATION.cff`). Repository: <https://github.com/ristar2018/ccmusic-split-audit>.
