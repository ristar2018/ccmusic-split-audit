# -*- coding: utf-8 -*-
"""P3 analysis: three-fold pooled metrics -> paired Δ between arms (recording-cluster bootstrap) + probe AUC comparison.

Conventions per the P3 pre-registration draft (v1, frozen):
  - Main metric = pooled macro-F1 over the **merged** held-out predictions of the three folds (slice-level primary, recording-level alongside).
  - Confirmatory family has 1 item = the pooled macro-F1 difference A1 vs A2 (no Holm correction needed, single test).
  - A0 vs A1 is reported as a point estimate and CI only, labelled "upper bound on the shortcut contribution", not in the confirmatory family.
  - Diagnostic = linear probe AUC for instrument identity on frozen features (A0 vs A2), with A1 as the reference arm.

Bootstrap: 2000 resamples by **recording cluster**, the same drawn recording indices spanning every arm and backbone
(this is the only way to pair them — A0 uses slice-level random folds, A1/A2 use leave-instrument-out folds, so slice order differs,
but after merging the three folds the slice set and the recording set covered are the same).

Usage:
  python code/analyze_p3.py --arms A0,A1 --base A1 --boot 2000
Output: data/results/p3/{p3_metrics.csv, p3_delta.csv, p3_probe.csv, p3_summary.md}
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs_p3")
OUT = os.path.join(HERE, "data", "results", "p3")
LIBS = ["CTIS"]
ARM_BASE = "A1"          # pre-registered: the main conclusion uses the leave-instrument-out A1 as reference
FOLDS = [1, 2, 3]


def macro_f1(y, pred, n_classes):
    """Average the per-class F1; only classes that appear in the truth or the prediction are counted."""
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else float("nan")


def load_run(arm, bb, seed):
    """Read the three fold npz files and merge them; returns (probs, y, records) or None."""
    d = os.path.join(RUNS, arm, bb, "seed%d" % seed)
    ps, ys, rs = [], [], []
    for f in FOLDS:
        p = os.path.join(d, "fold%d.npz" % f)
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        ps.append(z["probs"].astype(np.float32))
        ys.append(z["y"].astype(np.int64))
        rs.append(np.asarray(z["records"], dtype=object))
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(rs)


def recording_index(records):
    """recording -> column index array, ordered by recording name (every arm shares one cluster index)."""
    uniq = sorted(set(records.tolist()))
    pos = {r: i for i, r in enumerate(uniq)}
    return uniq, np.array([pos[r] for r in records], dtype=np.int64)


def per_recording(prob, y, rec_idx, n_rec, n_classes):
    """Average the probabilities within a recording, then take one vote."""
    sums = np.zeros((n_rec, n_classes), dtype=np.float64)
    np.add.at(sums, rec_idx, prob)
    pred = sums.argmax(1)
    ys, cnt, first = np.zeros(n_rec, dtype=np.int64), np.zeros(n_rec, dtype=np.int64), np.full(n_rec, -1, dtype=np.int64)
    for i, r in enumerate(rec_idx):
        if first[r] < 0:
            first[r] = i
        ys[r] = y[first[r]]
        cnt[r] += 1
    keep = cnt > 0
    return ys[keep], pred[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,A2")
    ap.add_argument("--base", default=ARM_BASE)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", default="", help="output directory; defaults to data/results/p3")
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    base = a.base
    out_dir = a.out or OUT
    os.makedirs(out_dir, exist_ok=True)

    # ---- inventory ----
    runs = defaultdict(dict)          # (arm,bb) -> seed -> (prob,y,rec)
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "*", "seed*", "fold1.npz"))):
        arm, bb, sd = p.split(os.sep)[-4:-1]
        seed = int(sd.replace("seed", ""))
        got = load_run(arm, bb, seed)
        if got is not None:
            runs[(arm, bb)][seed] = got
    if not runs:
        raise SystemExit("no three-fold output found under runs_p3/")
    n_classes = int(max(v[1].max() for d in runs.values() for v in d.values())) + 1

    # ---- class names ----
    cls_names = {}
    fu = os.path.join(HERE, "data", "p3", "p3_folds.json")
    if os.path.exists(fu):
        j = json.load(open(fu, encoding="utf-8"))
        inv = {v: k for k, v in j.get("class_to_idx", {}).items()}
        cls_names = inv

    # common order of recording clusters: the union over all arms/backbones/seeds (identical once the folds are merged)
    all_recs = set()
    for d in runs.values():
        for prob, y, rec in d.values():
            all_recs |= set(rec.tolist())
    uniq = sorted(all_recs)
    pos = {r: i for i, r in enumerate(uniq)}
    n_rec = len(uniq)
    print("[clusters] %d recordings" % n_rec)

    rng = np.random.default_rng(a.seed)
    picks = [rng.integers(0, n_rec, n_rec) for _ in range(a.boot)]

    rows_m, rows_d, rows_p = [], [], []
    for arm, bb in sorted(runs):
        d = runs[(arm, bb)]
        seeds = sorted(d)
        # point estimate
        s_slice, s_rec = [], []
        for s in seeds:
            prob, y, rec = d[s]
            pred = prob.argmax(1)
            s_slice.append(macro_f1(y, pred, n_classes))
            _, ridx = recording_index(rec)
            ys, pr = per_recording(prob, y, ridx, n_rec, n_classes)
            s_rec.append(macro_f1(ys, pr, n_classes))
        pt_slice = float(np.mean(s_slice))
        pt_rec = float(np.mean(s_rec))

        # bootstrap: within each resample compute F1 per seed, then average over seeds
        boot_slice, boot_rec = np.zeros(a.boot), np.zeros(a.boot)
        cached = []
        for s in seeds:
            prob, y, rec = d[s]
            ridx = np.array([pos[r] for r in rec.tolist()], dtype=np.int64)
            ys, pr = per_recording(prob, y, ridx, n_rec, n_classes)
            cached.append((prob, y, ridx, ys, pr))
        for b, pick in enumerate(picks):
            mult = np.bincount(pick, minlength=n_rec).astype(np.float64)
            v_s, v_r = [], []
            for prob, y, ridx, ys, pr in cached:
                w_s = mult[ridx]
                keep = w_s > 0
                v_s.append(macro_f1(y[keep], prob.argmax(1)[keep], n_classes))
                keep_r = mult > 0
                v_r.append(macro_f1(ys[keep_r], pr[keep_r], n_classes))
            boot_slice[b] = float(np.mean(v_s))
            boot_rec[b] = float(np.mean(v_r))

        rows_m.append({"lib": "CTIS", "arm": arm, "bb": bb, "seeds": ",".join(map(str, seeds)),
                       "n_seeds": len(seeds),
                       "macro_f1_slice": round(pt_slice, 4),
                       "macro_f1_recording": round(pt_rec, 4),
                       "slice_ci_lo": round(float(np.percentile(boot_slice, 2.5)), 4),
                       "slice_ci_hi": round(float(np.percentile(boot_slice, 97.5)), 4),
                       "rec_ci_lo": round(float(np.percentile(boot_rec, 2.5)), 4),
                       "rec_ci_hi": round(float(np.percentile(boot_rec, 97.5)), 4),
                       "seed_min_slice": round(min(s_slice), 4), "seed_max_slice": round(max(s_slice), 4),
                       "seed_spread_slice": round(max(s_slice) - min(s_slice), 4)})

        # ---- Δ vs base ----
        key = (base, bb)
        if arm != base and key in runs:
            bd = runs[key]
            bs = sorted(set(bd) & set(seeds))
            # per seed, lay out the slice-level predictions and the recording-level aggregates up front; the bootstrap only reweights
            prep = []
            for s in bs:
                pa, ya, ra = d[s]
                pb, yb, rb = bd[s]
                ia = np.array([pos[r] for r in ra.tolist()], dtype=np.int64)
                ib = np.array([pos[r] for r in rb.tolist()], dtype=np.int64)
                ya_r, pa_r = per_recording(pa, ya, ia, n_rec, n_classes)
                yb_r, pb_r = per_recording(pb, yb, ib, n_rec, n_classes)
                prep.append((pa.argmax(1), ya, ia, pa_r, ya_r,
                             pb.argmax(1), yb, ib, pb_r, yb_r))
            d_slice, d_rec = [], []
            for pick in picks:
                mult = np.bincount(pick, minlength=n_rec).astype(np.float64)
                dif_s, dif_r = [], []
                kr = mult > 0
                for (pra, ya, ia, pa_r, ya_r, prb, yb, ib, pb_r, yb_r) in prep:
                    ka, kb = mult[ia] > 0, mult[ib] > 0
                    dif_s.append(macro_f1(ya[ka], pra[ka], n_classes)
                                 - macro_f1(yb[kb], prb[kb], n_classes))
                    dif_r.append(macro_f1(ya_r[kr], pa_r[kr], n_classes)
                                 - macro_f1(yb_r[kr], pb_r[kr], n_classes))
                d_slice.append(float(np.mean(dif_s)))
                d_rec.append(float(np.mean(dif_r)))
            d_slice, d_rec = np.array(d_slice), np.array(d_rec)
            pt = pt_slice - float(np.mean([macro_f1(runs[key][s][1], runs[key][s][0].argmax(1), n_classes)
                                           for s in bs]))
            pt_r = pt_rec - float(np.mean([macro_f1(
                *per_recording(runs[key][s][0], runs[key][s][1],
                               np.array([pos[r] for r in runs[key][s][2].tolist()], dtype=np.int64),
                               n_rec, n_classes), n_classes) for s in bs]))
            lo, hi = float(np.percentile(d_slice, 2.5)), float(np.percentile(d_slice, 97.5))
            lo_r, hi_r = float(np.percentile(d_rec, 2.5)), float(np.percentile(d_rec, 97.5))
            p_sign = 2 * min((d_slice <= 0).mean(), (d_slice >= 0).mean())
            rows_d.append({"lib": "CTIS", "arm": arm, "base": base, "bb": bb,
                           "delta_slice": round(pt * 100, 3), "ci_lo": round(lo * 100, 3),
                           "ci_hi": round(hi * 100, 3), "p_boot": round(float(min(1.0, p_sign)), 4),
                           "delta_rec": round(pt_r * 100, 3), "rec_ci_lo": round(lo_r * 100, 3),
                           "rec_ci_hi": round(hi_r * 100, 3),
                           "confirmatory": bool(arm == "A2" and bb in ("alexnet", "swin_t")),
                           "n_seeds": len(bs)})
            print("[delta] %s − %s | %s : %+.2f [%+.2f, %+.2f]  (rec %+.2f [%+.2f, %+.2f])"
                  % (arm, base, bb, pt * 100, lo * 100, hi * 100, pt_r * 100, lo_r * 100, hi_r * 100))

        # ---- probe ----
        for s in seeds:
            pj = os.path.join(RUNS, arm, bb, "seed%d" % s)
            aucs, tops = [], []
            for f in FOLDS:
                # prefer the new-convention probe back-filled by p3_probe.py (8:2 evaluation on the training side)
                fp = os.path.join(pj, "fold%d_probe.json" % f)
                if os.path.exists(fp):
                    pr = json.load(open(fp, encoding="utf-8"))
                else:
                    fp = os.path.join(pj, "fold%d_result.json" % f)
                    if not os.path.exists(fp):
                        continue
                    pr = json.load(open(fp, encoding="utf-8")).get("probe") or {}
                if pr.get("probe_auc_ovr_macro") is not None:
                    aucs.append(pr["probe_auc_ovr_macro"])
                if pr.get("probe_top1") is not None:
                    tops.append(pr["probe_top1"])
            if aucs or tops:
                rows_p.append({"lib": "CTIS", "arm": arm, "bb": bb, "seed": s,
                               "probe_auc_mean": round(float(np.mean(aucs)), 4) if aucs else "",
                               "probe_top1_mean": round(float(np.mean(tops)), 4) if tops else "",
                               "folds": len(aucs)})

    def dump(path, rows):
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("[out] %s (%d rows)" % (os.path.relpath(path, HERE), len(rows)))

    dump(os.path.join(out_dir, "p3_metrics.csv"), rows_m)
    dump(os.path.join(out_dir, "p3_delta.csv"), rows_d)
    dump(os.path.join(out_dir, "p3_probe.csv"), rows_p)

    # ---- summary ----
    lines = ["# P3 pooled results (%d recording-cluster bootstraps)" % a.boot, "",
             "Main metric pooled macro-F1 (three folds merged), reference arm = %s." % base, "",
             "## Point estimates", "", "| arm | backbone | slice-level | recording-level | seed spread |", "| :-- | :-- | --: | --: | --: |"]
    for r in rows_m:
        lines.append("| %s | %s | %.4f | %.4f | %.3f |" % (r["arm"], r["bb"], r["macro_f1_slice"],
                                                           r["macro_f1_recording"], r["seed_spread_slice"]))
    if rows_d:
        lines += ["", "## Paired Δ (percentage points)", "",
                  "| arm − base | backbone | Δ slice-level [95% CI] | Δ recording-level [95% CI] | p | confirmatory |",
                  "| :-- | :-- | --: | --: | --: | :-- |"]
        for r in rows_d:
            lines.append("| %s − %s | %s | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] | %.4f | %s |"
                         % (r["arm"], r["base"], r["bb"], r["delta_slice"], r["ci_lo"], r["ci_hi"],
                            r["delta_rec"], r["rec_ci_lo"], r["rec_ci_hi"], r["p_boot"],
                            "yes" if r["confirmatory"] else "—"))
    if rows_p:
        lines += ["", "## Instrument-identity probe (linear classifier on frozen features)", "",
                  "| arm | backbone | seed | AUC(macro OvR) | top1 | folds |", "| :-- | :-- | --: | --: | --: | --: |"]
        for r in rows_p:
            lines.append("| %s | %s | %d | %s | %s | %d |" % (r["arm"], r["bb"], r["seed"],
                                                               r["probe_auc_mean"], r["probe_top1_mean"], r["folds"]))
    open(os.path.join(out_dir, "p3_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[out] data/results/p3/p3_summary.md")


if __name__ == "__main__":
    main()
