"""A/B 检查：同一随机流下，录音级 CI 的两种加权是否给出同一区间。

目的：给「`analyze_e1.py:219` 用的是哪种权重」一个**可复核但不宣称逐位复现**的旁证。
  - 判定权重的决定性证据是代码语义：`mult[b] = bincount(draw)` 是整数计数，被抽中一次的录音
    贡献 1，所以：219 的分布按「每录音一票」加权，是 unweighted。
  - 本脚本只做旁证：在同一随机流下算两种权重的端点，看哪一种更接近已发表值
    [0.26, 2.83] / [0.87, 2.60]。注意我无法保证与 analyze_e1 的抽样流完全一致，
    所以「接近」只能当旁证，**不能**当逐位复现。
只读；结果写 data/results/human_review/weighting_check.json。
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUT = os.path.join(ROOT, "data", "results", "human_review")
B = 2000
SEED_STREAM = 20261105          # 与 audit_recording_metric.py 相同
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

    # 一次抽 (B, nrec) 矩阵再按行 bincount —— 与 analyze_e1/audit_recording_metric 的抽样流一致；
    # 若改成 for 循环逐次抽 size=nrec，随机流不同，区间会偏离几百分之一。
    rng = np.random.default_rng(SEED_STREAM)
    pick = rng.integers(0, nrec, size=(B, nrec))
    mult = np.stack([np.bincount(row, minlength=nrec) for row in pick]).astype(float)

    out = {"stream": "default_rng(%d), recordings drawn with replacement, B=%d" % (SEED_STREAM, B),
           "note": "corroboration only: same stream, two weightings; 端点接近不能证明逐位一致",
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
            # 取 A1 与该骨干 A0 的差，按三种子平均（与正文同法）
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
