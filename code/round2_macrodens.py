#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""round2_macrodens.py — sensitivity of the macro-average denominator convention (CTIS, slice-level + recording-level, reads existing sealed outputs only).

Background: two scripts treat "classes absent from a resample" differently —
  * code/analyze_e1.py:96-106 `macro_f1(conf)`: recomputes keep = row sum + column sum > 0 on every call (line 104),
    dropping absent classes (**dynamic keep**). The bootstrap calls that function on each resample (analyze_e1.py:215-218),
    so the CIs in e1_delta.csv use dynamic keep.
  * code/sealed_eval_delta.py:143 `mask = np.bincount(y, minlength=C) > 0` is computed once over the **whole test set**,
    and line 45 `f1[:, mask].mean(1)` reuses that same mask for every resample; an absent class in that resample
    has tp=fp=fn=0 → F1=0 counted in the denominator (**fixed mask / fixed denominator**).
  * `code/round2_a1_minus_c1.py` uses dynamic keep: its `macro_f1_batch` (lines 79-90) recomputes keep
    per resample on line 87, and the nanmean on line 89 averages only over classes that occur; the main loop calls it on line 230.

This script compares the two conventions cell by cell on **the same set of recording-cluster bootstrap indices** (replaying
analyze_e1's default_rng(20261105) stream, the same source as e1_delta.csv): CTIS × {alexnet, swin_t} × {A1−A0, A1−C1} × {slice-level, recording-level},
2000 resamples. Δ at slice and recording level is defined as in round2_a1_minus_c1.py: per-seed paired difference inside each resample,
then averaged over the 3 seeds; 95% CI = 2.5/97.5 percentiles; units pp.

Output:
  data/results/human_review/round2/macrodens_sensitivity.csv           (slice-level, 8 rows)
  data/results/human_review/round2/macrodens_sensitivity_recording.csv (recording-level, 8 rows)
  data/results/human_review/round2/macrodens_sensitivity.json          (detail + attribution reconciliation)

Reads only: runs/sealed_eval/CTIS_*/seed*_qtest.npz, data/results/e1_delta.csv.
Usage: python code/round2_macrodens.py
"""
import csv
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_e1 as ae                     # noqa: E402  source of truth for macro-F1 / splits / bootstrap RNG stream
import round2_a1_minus_c1 as r2             # noqa: E402  reuse of the batched implementation and the legacy p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUTDIR = os.path.join(ROOT, "data", "results", "human_review", "round2")
OUT_SLICE = os.path.join(OUTDIR, "macrodens_sensitivity.csv")
OUT_REC = os.path.join(OUTDIR, "macrodens_sensitivity_recording.csv")
OUT_JSON = os.path.join(OUTDIR, "macrodens_sensitivity.json")
PUB_A1A0 = os.path.join(ROOT, "data", "results", "e1_delta.csv")

BOOT = r2.BOOT          # 2000
SEED = r2.BOOT_SEED     # 20261105
ARMS = ("A0", "A1", "C1")
SEEDS = (0, 1, 2)
BBS = ("alexnet", "swin_t")
CONTRASTS = (("A1", "A0"), ("A1", "C1"))
COLS = ["library", "bb", "contrast", "scheme", "point", "ci_lo", "ci_hi", "width", "excludes_zero"]


def masked_f1_batch(conf, mask):
    """(B,C,C) confusion matrix -> (B,) macro-F1, fixed denominator: an absent class gets F1=0 and stays in the mean (brick convention)."""
    tp = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=2) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    return f1[:, mask].mean(axis=1)


def batch_conf(pred, y, nc, W):
    """(B,nc,nc) confusion matrix: W (B,n) holds each sample's cluster multiplicity (slice level W=slice weight, recording level W=cluster multiplicity)."""
    flat = y * nc + pred
    out = np.empty((W.shape[0], nc * nc))
    for st in range(0, W.shape[0], 200):
        en = min(W.shape[0], st + 200)
        for i, b in enumerate(range(st, en)):
            out[st + i] = np.bincount(flat, weights=W[b], minlength=nc * nc)
    return out.reshape(-1, nc, nc)


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    ae.RUNS = RUNS
    data = ae.discover()

    # ---- replay analyze_e1's RNG stream and take the CTIS cluster multiplicities (shared by slice and recording level) ----
    rng = np.random.default_rng(SEED)
    mult = None
    for lib in sorted(data):
        per_arm = data[lib]
        if ("A0", "alexnet") not in per_arm:        # same condition as analyze_e1.py:155-158
            continue
        z0 = sorted(per_arm[("A0", "alexnet")])[0]
        _, y_l, r_l = ae.load_npz(per_arm[("A0", "alexnet")][z0])
        recs, rec_idx = np.unique(r_l, return_inverse=True)
        pick = rng.integers(0, recs.size, size=(BOOT, recs.size))
        if lib != "CTIS":
            continue
        mult = np.stack([np.bincount(p, minlength=recs.size) for p in pick])
        W = mult[:, rec_idx]                        # (B, n) slice-level cluster weights
        y_ref, rec_ref, nrec = y_l, r_l, recs.size
        rec_idx_ref = rec_idx                       # np.unique for later libraries overwrites rec_idx, so pin it first
    if mult is None:
        sys.exit("no CTIS sealed output found")
    nc = int(y_ref.max()) + 1

    mask_s = np.bincount(y_ref, minlength=nc) > 0      # slice-level fixed mask (sealed_eval_delta.py:143)
    # recording-level fixed mask: fixed by the recording-level label distribution (same convention as the brick recording counts)
    print("[convention] nc=%d, slice-level mask covers %d classes, recording clusters=%d"
          % (nc, int(mask_s.sum()), nrec))

    rows = {"slice": [], "recording": []}
    detail = {"nc": nc, "mask_classes_slice": int(mask_s.sum()), "boot": BOOT,
              "boot_seed": SEED, "n_recordings": int(nrec), "n_slices": int(y_ref.size),
              "levels": {}, "attribution": {}}

    for bb in BBS:
        boot = {"slice": {"dyn": {}, "fix": {}}, "recording": {"dyn": {}, "fix": {}}}
        pt = {"slice": {"dyn": {}, "fix": {}}, "recording": {"dyn": {}, "fix": {}}}
        mask_r = None
        for arm in ARMS:
            for s in SEEDS:
                probs, y, rec = ae.load_npz(os.path.join(RUNS, "CTIS_%s_%s" % (arm, bb),
                                                         "seed%d_qtest.npz" % s))
                assert np.array_equal(y, y_ref) and np.array_equal(rec, rec_ref)
                pred = probs.argmax(1)
                c_s = ae.conf_slice(y, pred, nc)
                pt["slice"]["dyn"][(arm, s)] = ae.macro_f1(c_s)[0]
                pt["slice"]["fix"][(arm, s)] = float(masked_f1_batch(c_s[None], mask_s)[0])
                conf_s = batch_conf(pred, y, nc, W)
                boot["slice"]["dyn"][(arm, s)] = r2.macro_f1_batch(conf_s)
                boot["slice"]["fix"][(arm, s)] = masked_f1_batch(conf_s, mask_s)

                rec_y, rec_pred, _ = ae.per_recording(probs, y, rec_idx_ref, nrec, nc)
                if mask_r is None:
                    mask_r = np.bincount(rec_y, minlength=nc) > 0
                c_r = ae.conf_slice(rec_y, rec_pred, nc)
                pt["recording"]["dyn"][(arm, s)] = ae.macro_f1(c_r)[0]
                pt["recording"]["fix"][(arm, s)] = float(masked_f1_batch(c_r[None], mask_r)[0])
                conf_r = batch_conf(rec_pred, rec_y, nc, mult)
                boot["recording"]["dyn"][(arm, s)] = r2.macro_f1_batch(conf_r)
                boot["recording"]["fix"][(arm, s)] = masked_f1_batch(conf_r, mask_r)
        detail["mask_classes_recording"] = int(mask_r.sum())

        for num, ref in CONTRASTS:
            cell_all = {}
            for level in ("slice", "recording"):
                cell = {}
                for scheme in ("dynamic_keep", "fixed_mask"):
                    key = "dyn" if scheme == "dynamic_keep" else "fix"
                    d_b = np.mean(np.stack([boot[level][key][(num, s)] - boot[level][key][(ref, s)]
                                            for s in SEEDS]), axis=0)
                    pd = [100.0 * (pt[level][key][(num, s)] - pt[level][key][(ref, s)])
                          for s in SEEDS]
                    lo, hi = np.percentile(100.0 * d_b, [2.5, 97.5])
                    cell[scheme] = {"point": float(np.mean(pd)), "ci_lo": float(lo),
                                    "ci_hi": float(hi), "width": float(hi - lo),
                                    "excludes_zero": bool(lo > 0 or hi < 0),
                                    "boot_p_legacy": r2.boot_p_legacy(d_b),
                                    "n_le0": int(np.count_nonzero(d_b <= 0)),
                                    "n_ge0": int(np.count_nonzero(d_b >= 0))}
                    rows[level].append({"library": "CTIS", "bb": bb,
                                        "contrast": "%s-%s" % (num, ref), "scheme": scheme,
                                        "point": round(cell[scheme]["point"], 4),
                                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                        "width": round(cell[scheme]["width"], 4),
                                        "excludes_zero": cell[scheme]["excludes_zero"]})
                cell["width_diff_pp"] = cell["fixed_mask"]["width"] - cell["dynamic_keep"]["width"]
                cell["point_diff_pp"] = cell["fixed_mask"]["point"] - cell["dynamic_keep"]["point"]
                cell["flip"] = (cell["fixed_mask"]["excludes_zero"]
                                != cell["dynamic_keep"]["excludes_zero"])
                cell_all[level] = cell
                print("%-8s %-9s %s-%-2s dyn %+6.2f [%+6.2f,%+6.2f] w=%5.2f | fix %+6.2f "
                      "[%+6.2f,%+6.2f] w=%5.2f | Δw=%+.3f flip=%s"
                      % (bb, level, num, ref, cell["dynamic_keep"]["point"],
                         cell["dynamic_keep"]["ci_lo"], cell["dynamic_keep"]["ci_hi"],
                         cell["dynamic_keep"]["width"], cell["fixed_mask"]["point"],
                         cell["fixed_mask"]["ci_lo"], cell["fixed_mask"]["ci_hi"],
                         cell["fixed_mask"]["width"], cell["width_diff_pp"], cell["flip"]))
            detail["levels"]["%s|%s-%s" % (bb, num, ref)] = cell_all

    # ---- attribution: reconcile the A1−A0 slice-level CI against the published e1_delta.csv, confirming the published values use dynamic keep ----
    pub = {}
    with io.open(PUB_A1A0, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["library"] == "CTIS" and r["contrast"].replace("−", "-") == "A1-A0":
                pub[r["bb"]] = (float(r["delta_slice_pt"]), float(r["ci_lo_slice_pt"]),
                                float(r["ci_hi_slice_pt"]))
    for bb, (dp, lo, hi) in sorted(pub.items()):
        dyn = detail["levels"]["%s|A1-A0" % bb]["slice"]["dynamic_keep"]
        fix = detail["levels"]["%s|A1-A0" % bb]["slice"]["fixed_mask"]
        detail["attribution"][bb] = {
            "published": [dp, lo, hi],
            "dynamic_keep": [round(dyn["point"], 4), round(dyn["ci_lo"], 4), round(dyn["ci_hi"], 4)],
            "fixed_mask": [round(fix["point"], 4), round(fix["ci_lo"], 4), round(fix["ci_hi"], 4)],
            "matches_dynamic": bool(abs(dyn["point"] - dp) < 5e-3 and abs(dyn["ci_lo"] - lo) < 5e-3
                                    and abs(dyn["ci_hi"] - hi) < 5e-3)}
    detail["max_abs_width_diff_pp"] = max(abs(c["width_diff_pp"]) for lv in detail["levels"].values()
                                          for c in lv.values())
    detail["any_flip"] = any(c["flip"] for lv in detail["levels"].values() for c in lv.values())
    detail["elapsed_sec"] = round(time.time() - t0, 1)

    for level, path in (("slice", OUT_SLICE), ("recording", OUT_REC)):
        with io.open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows[level])
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    print("[attribution] published A1−A0 slice CI identical to dynamic_keep bit for bit: %s"
          % {k: v["matches_dynamic"] for k, v in detail["attribution"].items()})
    print("wrote: %s | %s" % (OUT_SLICE, OUT_REC))


if __name__ == "__main__":
    main()
