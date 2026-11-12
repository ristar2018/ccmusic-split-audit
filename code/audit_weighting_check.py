"""A/B check: under one random stream, do the two weightings of the recording-level CI give the same interval?

Goal: corroborating evidence, **checkable but not claimed to be bit-exact**, for which weighting `analyze_e1.py:219` uses.
  - The decisive evidence for the weighting is the code semantics: `mult[b] = bincount(draw)` is an integer count,
    so a recording drawn once contributes 1; hence the :219 distribution is weighted one vote per recording, i.e. unweighted.
  - This script only corroborates: under the same random stream it computes the endpoints of both weightings and asks
    which one is closer to the published values [0.26, 2.83] / [0.87, 2.60]. Note that exact agreement with the
    analyze_e1 stream cannot be guaranteed, so "close" is corroboration only, **not** a bit-exact reproduction.
Read-only; writes data/results/human_review/weighting_check.json.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUT = os.path.join(ROOT, "data", "results", "human_review")
B = 2000
SEED_STREAM = 20261105          # same as audit_recording_metric.py
PUBLISHED = {"alexnet": (0.26, 2.83), "swin_t": (0.87, 2.60)}


def load(arm, bb, seed):
    z = np.load(os.path.join(RUNS, "CTIS_%s_%s" % (arm, bb), "seed%d_qtest.npz" % seed),
                allow_pickle=True)
    return z["probs"].astype(np.float64), z["y"].astype(np.int64), z["records"].astype(str)


def per_recording(probs, y, rec_idx, nrec, nc):
    sums = np.zeros((nrec, nc))
    np.add.at(sums, rec_idx, probs)
    cnt = np.bincount(rec_idx, minlength=nrec).astype(float)
    pred = (sums / np.maximum(cnt, 1)[:, None]).argmax(1)
    ry = np.zeros(nrec, dtype=np.int64)
    for i, r in enumerate(rec_idx):
        ry[r] = y[i]
    return ry, pred, cnt


class Batch:
    def __init__(self, M, ry, nc):
        self.M, self.nc = M, nc
        self.groups = [np.where(ry == c)[0] for c in range(nc)]

    def f1(self, W):
        Bn = W.shape[0]
        tp = np.zeros((Bn, self.nc)); row = np.zeros((Bn, self.nc)); col = np.zeros((Bn, self.nc))
        for c, idx in enumerate(self.groups):
            if idx.size == 0:
                continue
            Cc = W[:, idx] @ self.M[idx]
            tp[:, c] = Cc[:, c]; row[:, c] = Cc.sum(1); col += Cc
        den = 2 * tp + (col - tp) + (row - tp)
        keep = (row + col) > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(den > 0, 2 * tp / den, np.nan)
        return np.nanmean(np.where(keep, f1, np.nan), axis=1)


def main():
    probs, y, rec = load("A0", "alexnet", 0)
    recs = list(dict.fromkeys(rec.tolist()))
    rmap = {r: i for i, r in enumerate(recs)}
    ridx = np.array([rmap[r] for r in rec], dtype=np.int64)
    nrec = len(recs)
    nc = int(max(y.max(), probs.shape[1] - 1)) + 1

    # draw one (B, nrec) matrix then bincount each row -- matches the stream of analyze_e1/audit_recording_metric;
    # drawing size=nrec inside a for loop instead changes the stream and shifts the interval by a fraction of a percent.
    rng = np.random.default_rng(SEED_STREAM)
    pick = rng.integers(0, nrec, size=(B, nrec))
    mult = np.stack([np.bincount(row, minlength=nrec) for row in pick]).astype(float)

    out = {"stream": "default_rng(%d), recordings drawn with replacement, B=%d" % (SEED_STREAM, B),
           "note": "corroboration only: same stream, two weightings; close endpoints do not prove bit-exact agreement",
           "arms": {}}
    for bb in ("alexnet", "swin_t"):
        per = {}
        for arm in ("A0", "A1"):
            d = []
            for s in (0, 1, 2):
                p, yy, rr = load(arm, bb, s)
                ry, pred, cnt = per_recording(p, yy, ridx, nrec, nc)
                Mw = np.zeros((nrec, nc)); Mw[np.arange(nrec), pred] = cnt
                Mu = np.zeros((nrec, nc)); Mu[np.arange(nrec), pred] = 1.0
                d.append((Batch(Mw, ry, nc).f1(mult), Batch(Mu, ry, nc).f1(mult)))
            per[arm] = d
        res = {}
        for tag, k in (("sliceweighted_cnt", 0), ("unweighted_onevote", 1)):
            # take A1 minus A0 for this backbone, averaged over the three seeds (as in the main text)
            d0 = []
            for s in (0, 1, 2):
                p0, y0, _ = load("A0", bb, s)
                ry0, pred0, cnt0 = per_recording(p0, y0, ridx, nrec, nc)
                Mw0 = np.zeros((nrec, nc)); Mw0[np.arange(nrec), pred0] = cnt0
                Mu0 = np.zeros((nrec, nc)); Mu0[np.arange(nrec), pred0] = 1.0
                d0.append((Batch(Mw0, ry0, nc).f1(mult), Batch(Mu0, ry0, nc).f1(mult)))
            d = np.mean(np.stack([per["A1"][s][k] - d0[s][k] for s in range(3)], axis=0), axis=0)
            lo, hi = np.percentile(100 * d, [2.5, 97.5])
            res[tag] = [round(float(lo), 3), round(float(hi), 3)]
        res["published"] = list(PUBLISHED[bb])
        res["unweighted_matches_published"] = (
            abs(res["unweighted_onevote"][0] - PUBLISHED[bb][0]) < 0.01
            and abs(res["unweighted_onevote"][1] - PUBLISHED[bb][1]) < 0.01)
        res["sliceweighted_matches_published"] = (
            abs(res["sliceweighted_cnt"][0] - PUBLISHED[bb][0]) < 0.01
            and abs(res["sliceweighted_cnt"][1] - PUBLISHED[bb][1]) < 0.01)
        out["arms"][bb] = res
        print("%-8s unweighted %s  sliceweighted %s  published %s  -> unweighted match %s"
              % (bb, res["unweighted_onevote"], res["sliceweighted_cnt"], list(PUBLISHED[bb]),
                 res["unweighted_matches_published"]))

    with open(os.path.join(OUT, "weighting_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", os.path.join(OUT, "weighting_check.json"))


if __name__ == "__main__":
    main()
