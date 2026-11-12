"""Recording-level macro-F1 caliber correction (review follow-up item).

Where the problem is (`code/analyze_e1.py`):
  :189  point  = macro_f1(conf_slice(rec_y, rec_pred, nc, w=rec_n))    # weighted by "this recording's slice count"
  :219  CI     = macro_f1(conf_slice(rec_y, rec_pred, nc, w=mult[b]))  # weighted by "times the recording was drawn" = one vote per recording
  :243  d_rec  = mean_over_seeds(boot_f1[...][1] - boot_f1[...][1])    # built from :219 -> the interval is "one vote per recording"
  :250  p_rec  = mean_over_seeds(f1r) uses w=rec_n                     # the point estimate is "slice-weighted"
So the recording column of the old table mixes a slice-weighted point with a one-vote-per-recording interval: two different estimators.
**The old interval is itself internally consistent (unweighted) and is kept**; only the point estimate is recomputed as unweighted.

Writes (reads the sealed predictions, never overwrites the old CSVs):
  data/results/human_review/corrected_recording_metrics.csv  per (lib,bb,arm,seed):
      f1_rec_unweighted / f1_rec_sliceweighted_legacy / f1_slice
  data/results/human_review/corrected_recording_delta.csv    A1-A0 and A2-A0 (all libraries)
      with the corrected point (unweighted), the old table CI (unweighted, quoted as is), the legacy point (slice-weighted),
      the seed range, and the conditional interval that is "**given the three trained seeds** and resamples test recordings only"
      (`conditional_ci_*`; not a joint bootstrap: **no seed is resampled**). The true joint seed x recording bootstrap lives in
      the recording-level unweighted implementation of `code/audit_ctis_bootstrap.py`.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUT = os.path.join(ROOT, "data", "results", "human_review")
RES = os.path.join(ROOT, "data", "results")
LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]
BBS = ("alexnet", "swin_t")
SEEDS = (0, 1, 2)
BOOT = 2000


def load(lib, arm, bb, seed):
    z = np.load(os.path.join(RUNS, "%s_%s_%s" % (lib, arm, bb), "seed%d_qtest.npz" % seed),
                allow_pickle=True)
    return z["probs"].astype(np.float32), z["y"].astype(np.int64), z["records"].astype(str)


def per_recording(probs, y, rec_idx, nrec, nc):
    sums = np.zeros((nrec, nc), dtype=np.float64)
    np.add.at(sums, rec_idx, probs.astype(np.float64))
    cnt = np.bincount(rec_idx, minlength=nrec).astype(np.float64)
    rec_pred = (sums / np.maximum(cnt, 1)[:, None]).argmax(1)
    rec_y = np.zeros(nrec, dtype=np.int64)
    first = np.full(nrec, -1, dtype=np.int64)
    for i in range(len(y)):
        r = rec_idx[i]
        if first[r] < 0:
            first[r] = y[i]
        elif first[r] != y[i]:
            raise SystemExit("slice labels disagree for recording %d" % r)
    rec_y[:] = first
    return rec_y, rec_pred, cnt


def weighted_macro_f1(conf):
    tp = np.diag(conf).astype(float)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2 * tp / den, np.nan)
    return float(np.nanmean(f1))


def conf_from(rec_y, rec_pred, w, nc):
    conf = np.zeros((nc, nc), dtype=np.float64)
    np.add.at(conf, (rec_y, rec_pred), w)
    return conf


class Batch:
    def __init__(self, M, true_idx, nc):
        self.M, self.nc = M, nc
        self.groups = [np.where(true_idx == c)[0] for c in range(nc)]

    def f1(self, W):
        B = W.shape[0]
        tp = np.zeros((B, self.nc)); row = np.zeros((B, self.nc)); col = np.zeros((B, self.nc))
        for c, idx in enumerate(self.groups):
            if idx.size == 0:
                continue
            Cc = W[:, idx] @ self.M[idx]
            tp[:, c] = Cc[:, c]
            row[:, c] = Cc.sum(1)
            col += Cc
        den = 2 * tp + (col - tp) + (row - tp)
        keep = (row + col) > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(den > 0, 2 * tp / den, np.nan)
        f1 = np.where(keep, f1, np.nan)
        return np.nanmean(f1, axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    # old table (its recording CI is quoted verbatim)
    # A1-A0 lives in data/results/e1_delta.csv, A2-A0 in data/results/delta_A2/e1_delta.csv (two separate tables)
    old = {}
    for src in (os.path.join(RES, "e1_delta.csv"),
                os.path.join(RES, "delta_A2", "e1_delta.csv")):
        with open(src, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                old[(r["library"], r["bb"], r["contrast"])] = r

    metric_rows, delta_rows, arms_meta = [], [], {}
    rng = np.random.default_rng(20261105)

    for lib in sorted(LIBS):
        arms = set()
        for d in os.listdir(RUNS):
            if d.startswith(lib + "_"):
                parts = d[len(lib) + 1:].split("_")
                if len(parts) >= 2:
                    arms.add(parts[0])
        arms = sorted(a for a in arms if a in ("A0", "A1", "A2", "C1"))
        if "A0" not in arms:
            print("!! %s has no A0" % lib)
            continue
        probs0, y0, rec0 = load(lib, "A0", "alexnet", 0)
        recs = list(dict.fromkeys(rec0.tolist()))
        rmap = {r: i for i, r in enumerate(recs)}
        rec_idx = np.array([rmap[r] for r in rec0], dtype=np.int64)
        nrec = len(recs)
        nc = int(max(y0.max(), probs0.shape[1] - 1)) + 1

        pre = {}
        for arm in arms:
            for bb in BBS:
                for s in SEEDS:
                    try:
                        probs, y, rec = load(lib, arm, bb, s)
                    except FileNotFoundError:
                        continue
                    if not np.array_equal(y, y0) or not np.array_equal(rec.astype(str), rec0.astype(str)):
                        raise SystemExit("%s/%s/%s/%d panel differs from the reference" % (lib, arm, bb, s))
                    rec_y, rec_pred, cnt = per_recording(probs, y, rec_idx, nrec, nc)
                    onehot = np.zeros((nrec, nc)); onehot[np.arange(nrec), rec_pred] = 1.0
                    f1_unw = weighted_macro_f1(conf_from(rec_y, rec_pred, np.ones(nrec), nc))
                    f1_sw = weighted_macro_f1(conf_from(rec_y, rec_pred, cnt, nc))
                    f1_sl = weighted_macro_f1(conf_from(y, probs.argmax(1), np.ones(len(y)), nc))
                    pre[(arm, bb, s)] = {"M": onehot, "rec_y": rec_y, "cnt": cnt,
                                         "f1_unw": f1_unw, "f1_sw": f1_sw}
                    metric_rows.append({
                        "library": lib, "bb": bb, "arm": arm, "seed": s,
                        "n_slices": int(len(y)), "n_recordings": int(nrec), "n_classes": int(nc),
                        "f1_rec_unweighted": f1_unw,
                        "f1_rec_sliceweighted_legacy": f1_sw,
                        "f1_slice": f1_sl})

        pick = rng.integers(0, nrec, size=(BOOT, nrec))
        mult = np.zeros((BOOT, nrec))
        for b in range(BOOT):
            mult[b] = np.bincount(pick[b], minlength=nrec)

        for bb in BBS:
            for arm in arms:
                if arm == "A0":
                    continue
                seeds = [s for s in SEEDS if (arm, bb, s) in pre and ("A0", bb, s) in pre]
                if not seeds:
                    continue
                per_unw = [100.0 * (pre[(arm, bb, s)]["f1_unw"] - pre[("A0", bb, s)]["f1_unw"]) for s in seeds]
                per_sw = [100.0 * (pre[(arm, bb, s)]["f1_sw"] - pre[("A0", bb, s)]["f1_sw"]) for s in seeds]
                d_list = []
                for s in seeds:
                    fa = Batch(pre[(arm, bb, s)]["M"], pre[(arm, bb, s)]["rec_y"], nc).f1(mult)
                    fb = Batch(pre[("A0", bb, s)]["M"], pre[("A0", bb, s)]["rec_y"], nc).f1(mult)
                    d_list.append(fa - fb)
                d_seed = np.mean(np.stack(d_list), axis=0)
                lo, hi = np.percentile(100 * d_seed, [2.5, 97.5])
                key = (lib, bb, "%s-A0" % arm)
                o = old.get(key, {})
                delta_rows.append({
                    "library": lib, "bb": bb, "contrast": "%s-A0" % arm, "n_seeds": len(seeds),
                    "point_unw_pt": float(np.mean(per_unw)),
                    "ci_lo_unw_orig_pt": o.get("ci_lo_rec_pt", ""),
                    "ci_hi_unw_orig_pt": o.get("ci_hi_rec_pt", ""),
                    "legacy_point_sw_pt": float(np.mean(per_sw)),
                    "correction_pt": float(np.mean(per_unw) - np.mean(per_sw)),
                    "per_seed_unw_pt": "|".join("%.2f" % v for v in per_unw),
                    "seed_range_unw_pt": float(max(per_unw) - min(per_unw)),
                    "conditional_ci_lo_pt": round(float(lo), 6),
                    "conditional_ci_hi_pt": round(float(hi), 6),
                    "conditional_note": "conditional on the three trained seeds; recordings resampled only",
                })
                print("%-11s %-8s %-7s unw %+6.2f (legacy %+6.2f) oldCI [%s,%s] cond [%+5.2f,%+5.2f]"
                      % (lib, bb, arm, delta_rows[-1]["point_unw_pt"], delta_rows[-1]["legacy_point_sw_pt"],
                         o.get("ci_lo_rec_pt"), o.get("ci_hi_rec_pt"), lo, hi))
        arms_meta[lib] = {"arms": arms, "n_recordings": nrec, "n_slices": int(len(y0)), "n_classes": nc}

    p1 = os.path.join(OUT, "corrected_recording_metrics.csv")
    cols1 = ["library", "bb", "arm", "seed", "n_slices", "n_recordings", "n_classes",
             "f1_rec_unweighted", "f1_rec_sliceweighted_legacy", "f1_slice"]
    with open(p1, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols1, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in metric_rows:
            w.writerow({k: ("%.10f" % r[k] if isinstance(r[k], float) else r[k]) for k in cols1})
    p2 = os.path.join(OUT, "corrected_recording_delta.csv")
    cols2 = ["library", "bb", "contrast", "n_seeds", "point_unw_pt", "ci_lo_unw_orig_pt",
             "ci_hi_unw_orig_pt", "legacy_point_sw_pt", "correction_pt", "per_seed_unw_pt",
             "seed_range_unw_pt", "conditional_ci_lo_pt", "conditional_ci_hi_pt", "conditional_note"]
    with open(p2, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols2, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in delta_rows:
            w.writerow({k: ("%.10f" % r[k] if isinstance(r[k], float) else r[k]) for k in cols2})
    p3 = os.path.join(OUT, "corrected_recording_meta.json")
    with open(p3, "w", encoding="utf-8") as f:
        json.dump({"arms_meta": arms_meta, "boot": BOOT,
                   "old_ci_source": "A1-A0: data/results/e1_delta.csv; A2-A0: data/results/delta_A2/e1_delta.csv (delta_rec_pt / ci_lo_rec_pt / ci_hi_rec_pt)",
                   "bug": "analyze_e1.py:189 weights the point estimate by slice count, :219 weights the interval one vote per recording; the two are not the same estimator",
                   "fix": "point estimate switched to one vote per recording; the interval keeps the unweighted version from the old table, no re-run",
                   "conditional_note": "conditional CI = recordings resampled only, seeds held at the three trained ones; NOT a joint seed x recording bootstrap"},
                  f, ensure_ascii=False, indent=1)
    print("\nwrote", p1); print("wrote", p2); print("wrote", p3)


if __name__ == "__main__":
    main()
