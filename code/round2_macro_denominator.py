#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sensitivity of the macro-average denominator caliber (round 2 review #21.3): one batch of sealed predictions, A1-C1 / A1-A0 recomputed under three denominator calibers.

The three calibers
  fixed    : the class set is frozen once from the full-sample observation panel (sealed_eval_delta.py:143 mask); classes absent from a resample count as 0
  dynamic  : keep is recomputed inside every resample (analyze_e1.py:104 keep = seen in training or prediction); absent classes are dropped
  fullC    : average over all C classes, with f1=0 for absent classes (the point-estimate caliber of sealed_eval_delta.py :21-26 / :164)

Reads runs/sealed_eval/*.npz; writes data/results/human_review/round2/macro_denominator.{csv,json}
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("runs/sealed_eval")
OUT_DIR = Path("data/results/human_review/round2")
LIB, BBS = "CTIS", ["alexnet", "swin_t"]
CONTRASTS = [("A1", "A0"), ("A1", "C1")]
N_BOOT, BOOT_SEED, SEEDS = 2000, 20261105, [0, 1, 2]


def load(lib, arm, bb, s):
    z = np.load(ROOT / f"{lib}_{arm}_{bb}" / f"seed{s}_qtest.npz", allow_pickle=True)
    return z["probs"].astype(np.float64), z["y"].astype(np.int64), z["records"].astype(str)


def counts_by_cluster(y, pred, gidx, R, C):
    M = np.zeros((R, C * C))
    codes = y.astype(np.int64) * C + pred.astype(np.int64)
    np.add.at(M, (gidx, codes), 1.0)
    return M


def tpfpfn(A, codes, C):
    B = A.shape[0]
    tp = np.zeros((B, C)); fp = np.zeros((B, C)); fn = np.zeros((B, C))
    for j, c in enumerate(codes):
        t, p = divmod(int(c), C)
        if t == p:
            tp[:, t] += A[:, j]
        else:
            fp[:, p] += A[:, j]; fn[:, t] += A[:, j]
    return tp, fp, fn


def f1mat(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def score(tp, fp, fn, C, mode):
    f1 = f1mat(tp, fp, fn)
    if mode == "fixed":
        return f1.mean(1), None
    if mode == "fullC":
        return f1.mean(1), None
    sup = (tp + fn > 0) | (fp + tp > 0)
    n = sup.sum(1)
    return (f1 * sup).sum(1) / np.maximum(n, 1), n


def main():
    rows, detail = [], {}
    for bb in BBS:
        # balance gate: y / records must be element-wise identical across every arm and seed
        base = None
        probs = {}
        for arm in ("A0", "A1", "C1"):
            for s in SEEDS:
                p, y, r = load(LIB, arm, bb, s)
                if base is None:
                    base = (y, r)
                elif not (np.array_equal(y, base[0]) and np.array_equal(r, base[1])):
                    sys.exit(f"[audit failed] {LIB}/{bb}/{arm}/seed{s}: y/records differ from the baseline")
                probs[(arm, s)] = p
        y, rec = base
        C = probs[("A0", 0)].shape[1]
        rid, gidx = np.unique(rec, return_inverse=True)
        R = rid.size
        mask_fixed = np.bincount(y, minlength=C) > 0
        rng = np.random.default_rng(
            int(hashlib.md5(f"{LIB}|{bb}|denom".encode()).hexdigest()[:8], 16) ^ BOOT_SEED)
        W = np.zeros((N_BOOT, R))
        np.add.at(W, (np.repeat(np.arange(N_BOOT), R), rng.integers(0, R, size=(N_BOOT, R)).ravel()), 1.0)

        for hi, lo in CONTRASTS:
            # build the cluster x pair counts for each (arm,seed) and apply the weights
            A = {}
            codes = sorted({int(a) * C + int(p) for arm in (hi, lo) for s in SEEDS
                            for a, p in zip(y, probs[(arm, s)].argmax(1))})
            c2j = {c: j for j, c in enumerate(codes)}
            for arm in (hi, lo):
                for s in SEEDS:
                    M = counts_by_cluster(y, probs[(arm, s)].argmax(1), gidx, R, C)[:, codes]
                    A[(arm, s)] = W @ M
            # point estimate (full-sample counts)
            Pt = {}
            for arm in (hi, lo):
                for s in SEEDS:
                    M1 = counts_by_cluster(y, probs[(arm, s)].argmax(1), gidx, R, C)
                    Pt[(arm, s)] = M1.sum(0)[codes].reshape(1, -1)
            out = {}
            for mode in ("fixed", "dynamic", "fullC"):
                boots, pts = {}, {}
                for arm in (hi, lo):
                    for s in SEEDS:
                        tp, fp, fn = tpfpfn(A[(arm, s)], codes, C)
                        if mode == "fixed":
                            boots[(arm, s)] = f1mat(tp, fp, fn)[:, mask_fixed].mean(1)
                            tp1, fp1, fn1 = tpfpfn(Pt[(arm, s)], codes, C)
                            pts[(arm, s)] = f1mat(tp1, fp1, fn1)[:, mask_fixed].mean()
                        elif mode == "fullC":
                            boots[(arm, s)] = f1mat(tp, fp, fn).mean(1)
                            tp1, fp1, fn1 = tpfpfn(Pt[(arm, s)], codes, C)
                            pts[(arm, s)] = f1mat(tp1, fp1, fn1).mean()
                        else:
                            b, n = score(tp, fp, fn, C, mode)
                            boots[(arm, s)] = b
                            tp1, fp1, fn1 = tpfpfn(Pt[(arm, s)], codes, C)
                            pts[(arm, s)] = score(tp1, fp1, fn1, C, mode)[0][0]
                d_boot = np.mean([boots[(hi, s)] - boots[(lo, s)] for s in SEEDS], axis=0)
                seed_pt = [pts[(hi, s)] - pts[(lo, s)] for s in SEEDS]
                lo95, hi95 = np.percentile(d_boot * 100, [2.5, 97.5])
                out[mode] = dict(d=float(np.mean(seed_pt)) * 100, lo=float(lo95), hi=float(hi95),
                                 sr=(max(seed_pt) - min(seed_pt)) * 100)
            # count absent classes (how many are dropped per resample under the dynamic caliber)
            tp, fp, fn = tpfpfn(A[(hi, 0)], codes, C)
            sup = ((tp + fn > 0) | (fp + tp > 0)).sum(1)
            out["_dropped_classes_mean"] = float((C - sup).mean())
            out["_dropped_classes_max"] = int((C - sup).max())
            out["_classes_lt5_rec"] = int((np.bincount(y, minlength=C) < 5).sum())
            out["_C"] = int(C); out["_n_slice"] = int(y.size); out["_n_rec"] = int(R)
            detail[f"{hi}-{lo}"] = out
            for mode in ("fixed", "dynamic", "fullC"):
                r0 = out[mode]
                rows.append(dict(bb=bb, contrast=f"{hi}-{lo}", mode=mode,
                                 d_pp=round(r0["d"], 3), lo_pp=round(r0["lo"], 3),
                                 hi_pp=round(r0["hi"], 3), seed_range_pp=round(r0["sr"], 3),
                                 ci_width_pp=round(r0["hi"] - r0["lo"], 3),
                                 dropped_classes_mean=round(out["_dropped_classes_mean"], 2),
                                 dropped_classes_max=out["_dropped_classes_max"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(OUT_DIR / "macro_denominator.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    (OUT_DIR / "macro_denominator.json").write_text(
        json.dumps(dict(lib=LIB, n_boot=N_BOOT, boot_seed=BOOT_SEED, seeds=SEEDS, detail=detail),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'bb':9s}{'contrast':10s}{'mode':8s}{'d':>8s}{'95%CI':>20s}{'width':>8s}")
    for r in rows:
        print(f"{r['bb']:9s}{r['contrast']:10s}{r['mode']:8s}{r['d_pp']:+8.3f}"
              f"  [{r['lo_pp']:+7.3f},{r['hi_pp']:+7.3f}]{r['ci_width_pp']:8.3f}")
    for k, v in detail.items():
        print(f"[{k}] C={v['_C']} nSlice={v['_n_slice']} nRec={v['_n_rec']} "
              f"dropped/mean={v['_dropped_classes_mean']:.2f} max={v['_dropped_classes_max']} "
              f"classes<5rec={v['_classes_lt5_rec']}")
    print("wrote:", OUT_DIR / "macro_denominator.csv")


if __name__ == "__main__":
    main()
