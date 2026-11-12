#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_e1.py — read the per-sample predictions of sealed_eval and compute the metrics and intervals fixed by §3.5.

Design basis (design note v2.8)
  §3.5 :273-277  primary metric macro-F1 in two readings (slice-level + recording-level 'average the probabilities within a recording, then one vote'), recording-level takes precedence on disagreement;
                  the three interval sources are reported separately, never merged into a single CI:
                    1) 2,000 resamples of the test-recording clusters (whole clusters kept or dropped, never split; one cluster index runs through every arm and backbone)
                    2) between-seed spread = range over the 3 seeds, reported as a number only, not tested
                    3) split-seed spread (not computed here, it belongs to the appendix)
                  confirmatory family = 2 items = Δ1 of CTIS × {AlexNet, Swin-T}, Holm α=0.05;
                  other libraries/backbones are labelled replication; SESOI 3 points (agreed value);
                  not done: testing the 3 seeds as independent samples, McNemar on macro-F1, using the slice count as n.
  §3.4 :269      no early stopping on Q_test and no seed picking — this script only reads sealed Q_test predictions and writes back no training artifact.

Usage
  python code/analyze_e1.py                       # default A1-A0, all libraries and backbones
  python code/analyze_e1.py --arms A2 --base A0   # E1c: A2-A0
  python code/analyze_e1.py --boot 500            # quick self-test

Output (data/results/)
  e1_metrics.csv        slice-level / recording-level metrics per (lib, arm, bb, seed)
  e1_delta.csv          Δ per (lib, bb) (mean over the 3 seeds) + cluster bootstrap CI + seed range
  e1_holm.json          Holm correction of the confirmatory family (CTIS × both backbones)
  e1_per_class.csv      per-class F1 (mean over the 3 seeds, feeds the class-level decomposition figure of E1d)
  e1_summary.md         human-readable summary
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUT = os.path.join(ROOT, "data", "results")
BOOT = 2000
SESOI = 3.0            # points
LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto",
        "chest_falsetto", "erhu_playing_tech", "GZ_IsoTech"]  # the last three are the E1f negative-control libraries
CONFIRM_ARM = "A1"          # the confirmatory family counts Δ1 only (design note §3.5, "Δ1 of CTIS×{AlexNet,Swin-T}")
CONFIRM_LIBS = ["CTIS"]
CONFIRM_BBS = ["alexnet", "swin_t"]
LIBS = sorted(LIBS, key=len, reverse=True)   # longest first: a short library name must not match ahead of a long one


# ---------------------------------------------------------------- read the sealed artifacts
def discover():
    """{lib: {(arm, bb): {seed: (probs, y, records, n_classes)}}}"""
    data = {}
    pend = os.path.join(RUNS, "pending")
    if os.path.isdir(pend) and os.listdir(pend):
        print("!! pending/ still holds unsealed groups, these numbers are provisional: %s" % ", ".join(sorted(os.listdir(pend))))
    for d in sorted(glob.glob(os.path.join(RUNS, "*_*_*"))):
        if not os.path.isdir(d) or os.path.basename(d) == "pending":
            continue
        # directory name = <lib>_<arm>_<bb>; lib (music_genre/bel_canto) and bb (swin_t) both contain underscores,
        # so match on known vocabulary prefixes, never split directly
        name = os.path.basename(d)
        lib = next((L for L in LIBS if name.startswith(L + "_")), None)
        if lib is None:
            print("!! cannot identify the library from the directory name: %s" % name)
            continue
        rest = name[len(lib) + 1:]
        arm = next((A for A in ("A0", "A1", "A2", "C1") if rest.startswith(A + "_")), None)
        if arm is None:
            print("!! cannot identify the arm from the directory name: %s" % name)
            continue
        bb = rest[len(arm) + 1:]
        seeds = {}
        for z in sorted(glob.glob(os.path.join(d, "seed*_qtest.npz"))):
            try:
                s = int(os.path.basename(z)[4:-10])
            except ValueError:
                print("!! cannot parse the seed file: %s" % z)
                continue
            seeds[s] = z
        miss = [s for s in (0, 1, 2) if s not in seeds]
        if miss:
            print("!! missing sealed seed %s: %s" % (miss, name))
        if seeds:
            data.setdefault(lib, {})[(arm, bb)] = seeds
        else:
            print("!! this group has no seed*_qtest.npz: %s" % name)
    return data


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["probs"].astype(np.float32), z["y"].astype(np.int64), z["records"].astype(str)


# ---------------------------------------------------------------- metrics
def macro_f1(conf):
    """conf[c, p] -> macro-F1; empty classes are excluded (same reading as sklearn average='macro' with zero_division=0)."""
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    keep = conf.sum(1) + conf.sum(0) > 0      # the class occurs in this sample
    f1 = f1[keep]
    return float(np.nanmean(f1)) if f1.size else float("nan"), f1, keep


def conf_slice(y, pred, nc, w=None):
    flat = y * nc + pred
    return np.bincount(flat, weights=w, minlength=nc * nc).reshape(nc, nc)


def per_recording(probs, y, rec_idx, nrec, nc):
    """Average the probabilities within a recording, then one vote. Returns (rec_y, rec_pred, rec_n_slices)."""
    cnt = np.bincount(rec_idx, minlength=nrec).astype(np.float64)
    acc = np.zeros((nrec, nc), dtype=np.float64)
    np.add.at(acc, rec_idx, probs)
    mean = acc / np.maximum(cnt, 1)[:, None]
    rec_pred = mean.argmax(1)
    # recording label: majority over the slice labels of that recording (labels within a recording should agree; this is a fallback outside the assertion)
    rec_y = np.zeros(nrec, dtype=np.int64)
    for c in range(nc):
        m = np.bincount(rec_idx[y == c], minlength=nrec)
        rec_y[m > 0] = c
    return rec_y, rec_pred, cnt


# ---------------------------------------------------------------- main flow
def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A1", help="experiment arms, comma separated (A1 or A2)")
    ap.add_argument("--base", default="A0")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    os.makedirs(a.out, exist_ok=True)

    RUNS = a.runs
    data = discover()
    if not data:
        print("!! no sealed group under %s" % a.runs)
        return 2

    rng = np.random.default_rng(a.seed)
    rows_m, rows_d, rows_c = [], [], []
    boot_store = {}

    for lib in sorted(data):
        per_arm = data[lib]
        key0 = (a.base, "alexnet")
        if key0 not in per_arm:
            print("!! %s is missing the %s/alexnet seal, skipping" % (lib, a.base))
            continue
        z0 = sorted(per_arm[key0])[0]
        _, y_ref, rec_ref = load_npz(per_arm[key0][z0])
        n = y_ref.size
        nc = int(y_ref.max()) + 1
        recs, rec_idx = np.unique(rec_ref, return_inverse=True)
        nrec = recs.size
        # slice indices grouped by recording (the cluster bootstrap samples whole clusters)
        order = np.argsort(rec_idx, kind="stable")
        bounds = np.searchsorted(rec_idx[order], np.arange(nrec + 1))

        combos = sorted(per_arm)
        # sampling-order consistency assertion: every arm/backbone/seed must share the exact slice order (otherwise Δ is not paired)
        for (arm, bb), seeds in per_arm.items():
            for s, z in seeds.items():
                _, y1, r1 = load_npz(z)
                if not (y1.shape == y_ref.shape and np.array_equal(y1, y_ref)
                        and np.array_equal(r1, rec_ref)):
                    raise SystemExit("the Q_test order of %s/%s/%s/seed%d differs from the reference" % (lib, arm, bb, s))

        # ---- precompute pred and the recording-level aggregate for every (arm,bb,seed)
        pre = {}
        for (arm, bb), seeds in sorted(per_arm.items()):
            for s, z in sorted(seeds.items()):
                probs, y, _ = load_npz(z)
                pred = probs.argmax(1)
                rec_y, rec_pred, rec_n = per_recording(probs, y, rec_idx, nrec, nc)
                pre[(arm, bb, s)] = {"pred": pred, "rec_pred": rec_pred,
                                     "rec_y": rec_y, "rec_n": rec_n}
                conf = conf_slice(y, pred, nc)
                f1s, _, _ = macro_f1(conf)
                f1r, f1r_c, keep = macro_f1(conf_slice(rec_y, rec_pred, nc))
                rows_m.append({"library": lib, "arm": arm, "bb": bb, "seed": s,
                               "n_slices": n, "n_recordings": nrec, "n_classes": nc,
                               "macro_f1_slice": round(f1s, 4),
                               "macro_f1_recording": round(f1r, 4),
                               "acc_slice": round(float((pred == y).mean()), 4),
                               "acc_recording": round(float((rec_pred == rec_y).mean()), 4)})
                # per-class F1 (slice-level): computed straight from the confusion-matrix columns, nan means the class is absent from the sample
                tp = np.diag(conf).astype(float)
                fp = conf.sum(0) - tp
                fn = conf.sum(1) - tp
                den = 2 * tp + fp + fn
                per_class = np.where(den > 0, 2 * tp / den, np.nan)
                rows_c.append({"library": lib, "arm": arm, "bb": bb, "seed": s,
                               "f1_per_class": per_class})

        # ---- cluster bootstrap: one set of recording indices runs through every arm and backbone of this library
        pick = rng.integers(0, nrec, size=(a.boot, nrec))
        mult = np.zeros((a.boot, nrec), dtype=np.float64)
        for b in range(a.boot):
            mult[b] = np.bincount(pick[b], minlength=nrec)
        w_slice = mult[:, rec_idx]                     # (boot, n)
        boot_f1 = {}
        for (arm, bb, s), p in pre.items():
            flat = y_ref * nc + p["pred"]
            wsum = np.zeros((a.boot, nc * nc))
            for b in range(a.boot):
                wsum[b] = np.bincount(flat, weights=w_slice[b], minlength=nc * nc)
            conf_s = wsum.reshape(a.boot, nc, nc)
            f1s = np.array([macro_f1(conf_s[b])[0] for b in range(a.boot)])
            f1r = np.array([macro_f1(conf_slice(p["rec_y"], p["rec_pred"], nc, w=mult[b]))[0]
                            for b in range(a.boot)])
            boot_f1[(arm, bb, s)] = (f1s, f1r)

        # ---- Δ (paired within seed, then averaged over seeds)
        bbs = sorted({bb for (_, bb) in per_arm})
        for bb in bbs:
            base_s = sorted(pre_k[2] for pre_k in pre if pre_k[0] == a.base and pre_k[1] == bb)
            if not base_s:
                print("!! %s/%s/%s is missing the baseline seal, skipping this backbone" % (lib, arm, bb))
                continue
            for arm in arms:
                arm_s = sorted(pre_k[2] for pre_k in pre if pre_k[0] == arm and pre_k[1] == bb)
                if not arm_s:
                    print("!! %s/%s/%s has no seal" % (lib, arm, bb))
                    continue
                seeds = sorted(set(base_s) & set(arm_s))
                if not seeds:
                    continue
                if len(seeds) < 3:
                    print("!! %s/%s/%s has only %d paired seeds: %s" % (lib, arm, bb, len(seeds), seeds))
                # bootstrap distribution: within each resample, take the per-seed (F1_arm − F1_base) first, then average over seeds
                d_slice = np.mean(np.stack([
                    boot_f1[(arm, bb, s)][0] - boot_f1[(a.base, bb, s)][0] for s in seeds]), axis=0)
                d_rec = np.mean(np.stack([
                    boot_f1[(arm, bb, s)][1] - boot_f1[(a.base, bb, s)][1] for s in seeds]), axis=0)
                # point estimate: Δ averaged over the 3 seeds
                p_slice = float(np.mean([
                    macro_f1(conf_slice(y_ref, pre[(arm, bb, s)]["pred"], nc))[0]
                    - macro_f1(conf_slice(y_ref, pre[(a.base, bb, s)]["pred"], nc))[0]
                    for s in seeds]))
                p_rec = float(np.mean([
                    macro_f1(conf_slice(pre[(arm, bb, s)]["rec_y"], pre[(arm, bb, s)]["rec_pred"],
                                        nc))[0]
                    - macro_f1(conf_slice(pre[(a.base, bb, s)]["rec_y"],
                                          pre[(a.base, bb, s)]["rec_pred"], nc))[0]
                    for s in seeds]))
                lo_s, hi_s = np.percentile(d_slice, [2.5, 97.5])
                lo_r, hi_r = np.percentile(d_rec, [2.5, 97.5])
                smaller_tail_count = min(int(np.count_nonzero(d_slice <= 0)),
                                         int(np.count_nonzero(d_slice >= 0)))
                p_lo = min(1.0, 2.0 * (smaller_tail_count + 1) / (a.boot + 1))
                # seed range (number only: range of the per-seed Δ, in points)
                per_seed = [100 * (
                    macro_f1(conf_slice(y_ref, pre[(arm, bb, s)]["pred"], nc))[0]
                    - macro_f1(conf_slice(y_ref, pre[(a.base, bb, s)]["pred"], nc))[0])
                    for s in seeds]
                rows_d.append({
                    "library": lib, "arm": arm, "bb": bb,
                    "contrast": "%s-%s" % (arm, a.base),
                    "delta_slice_pt": round(100 * p_slice, 2),
                    "ci_lo_slice_pt": round(100 * lo_s, 2), "ci_hi_slice_pt": round(100 * hi_s, 2),
                    "delta_rec_pt": round(100 * p_rec, 2),
                    "ci_lo_rec_pt": round(100 * lo_r, 2), "ci_hi_rec_pt": round(100 * hi_r, 2),
                    "boot_p_slice": round(p_lo, 4),
                    "seed_range_pt": round(max(per_seed) - min(per_seed), 2) if len(per_seed) > 1 else None,
                    "per_seed_delta_pt": "|".join("%.2f" % v for v in per_seed),
                    "n_recordings": nrec, "n_slices": n, "boot": a.boot,
                    "meets_sesoi": bool(100 * p_slice >= SESOI),
                    "confirmatory": bool(arm == CONFIRM_ARM and lib in CONFIRM_LIBS and bb in CONFIRM_BBS),
                })
                boot_store["%s|%s|%s" % (lib, bb, arm)] = {
                    "delta_slice": d_slice.tolist(), "delta_rec": d_rec.tolist()}

    # ---- Holm (confirmatory family)
    fam = [r for r in rows_d if r["confirmatory"]]
    holm = {}
    if fam:
        fam_sorted = sorted(fam, key=lambda r: r["boot_p_slice"])
        m = len(fam_sorted)
        prev = 0.0
        for i, r in enumerate(fam_sorted):
            adj = min(1.0, max(prev, (m - i) * r["boot_p_slice"]))
            prev = adj
            holm["%s|%s|%s" % (r["library"], r["bb"], r["arm"])] = {
                "delta_slice_pt": r["delta_slice_pt"], "boot_p": r["boot_p_slice"],
                "holm_p": round(adj, 4), "reject_at_0.05": bool(adj < 0.05)}

    # ---- write to disk
    import csv as _csv

    def w(path, rows, cols):
        with open(path, "w", encoding="utf-8", newline="") as f:
            wr = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow(r)

    w(os.path.join(a.out, "e1_metrics.csv"), rows_m,
      ["library", "arm", "bb", "seed", "n_slices", "n_recordings", "n_classes",
       "macro_f1_slice", "macro_f1_recording", "acc_slice", "acc_recording"])
    w(os.path.join(a.out, "e1_delta.csv"), rows_d,
      ["library", "bb", "contrast", "delta_slice_pt", "ci_lo_slice_pt", "ci_hi_slice_pt",
       "delta_rec_pt", "ci_lo_rec_pt", "ci_hi_rec_pt", "boot_p_slice", "seed_range_pt",
       "per_seed_delta_pt", "n_recordings", "n_slices", "boot", "meets_sesoi", "confirmatory"])

    # per class: average over seeds for each (lib, arm, bb), then expand to one row per class
    if rows_c:
        agg = {}
        for r in rows_c:
            k = (r["library"], r["arm"], r["bb"])
            agg.setdefault(k, []).append(r["f1_per_class"])
        pc_rows = []
        for (lib, arm, bb), mats in sorted(agg.items()):
            M = np.vstack(mats)
            mean = np.nanmean(M, axis=0)
            lo = np.nanmin(M, axis=0)
            hi = np.nanmax(M, axis=0)
            for c in range(mean.size):
                pc_rows.append({"library": lib, "arm": arm, "bb": bb, "class_idx": c,
                                "f1_mean": round(float(mean[c]), 4),
                                "f1_min": round(float(lo[c]), 4),
                                "f1_max": round(float(hi[c]), 4),
                                "n_seeds": len(mats)})
        w(os.path.join(a.out, "e1_per_class.csv"), pc_rows,
          ["library", "arm", "bb", "class_idx", "f1_mean", "f1_min", "f1_max", "n_seeds"])

    with open(os.path.join(a.out, "e1_holm.json"), "w", encoding="utf-8") as f:
        json.dump({"family": "confirmatory family = CTIS × {alexnet, swin_t} for %s−%s (%s arm only)"
                          % (CONFIRM_ARM, a.base, CONFIRM_ARM),
                   "alpha": 0.05, "sesoi_pt": SESOI, "boot": a.boot,
                   "note": "boot_p is the two-sided percentile p of the recording-cluster bootstrap; Holm takes (m−i)·p item by item and enforces monotonicity",
                   "tests": holm}, f, ensure_ascii=False, indent=1)

    # ---- summary (human-readable)
    lines = ["# E1 result summary (%s − %s)" % ("/".join(arms), a.base), "",
             "cluster bootstrap, %d resamples (whole recording clusters drawn with replacement, one set of indices running through every arm and backbone of a library); "
             "Δ is in percentage points; the seed range is the range of the same-seed paired Δ across the 3 seeds." % a.boot, ""]
    if rows_d:
        # keep the arm column: with --arms A1,A2 one library and backbone yields two rows, and dropping the arm column makes them read as duplicates/contradictions
        lines.append("| library | arm | backbone | Δ slice-level [95% CI] | Δ recording-level [95% CI] | seed range | meets SESOI (3pt) |")
        lines.append("| :-- | :-- | :-- | --: | --: | --: | :-- |")
        for r in rows_d:
            head = "| %s | %s | %s |" % (r["library"], r.get("arm", ""), r["bb"])
            lines.append("%s %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] | %s | %s |" % (
                head, r["delta_slice_pt"], r["ci_lo_slice_pt"], r["ci_hi_slice_pt"],
                r["delta_rec_pt"], r["ci_lo_rec_pt"], r["ci_hi_rec_pt"],
                r["seed_range_pt"] if r["seed_range_pt"] is not None else "n/a",
                "yes" if r["meets_sesoi"] else "no"))
    if holm:
        lines += ["", "## Confirmatory family (Holm α=0.05)", ""]
        for k, v in sorted(holm.items()):
            lines.append("- %s: Δ=%+.2f pt, boot p=%.4f, Holm p=%.4f → %s" % (
                k, v["delta_slice_pt"], v["boot_p"], v["holm_p"],
                "reject H0" if v["reject_at_0.05"] else "do not reject"))
    with open(os.path.join(a.out, "e1_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\n→ %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
