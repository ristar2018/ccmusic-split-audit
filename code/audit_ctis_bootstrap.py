"""CTIS A1-A0 封存预测：种子 × 类分层录音自助敏感性分析（审稿意见 B2 / B4 / B10 核验用）。

只读 `runs/sealed_eval/`，不写任何训练产物，不覆盖既有 CSV；结果落 `data/results/human_review/`。

四个重抽样方案，全部在**同一批抽到的录音**上配对计算 A1 与 A0（与正文口径一致）：
  L  legacy      录音无分层重抽样，种子固定           → 复现正文口径
  S  stratified  按真实类别分层重抽样，种子固定        → 审稿人 B4 建议
  SS stratified + 种子重抽样（3 个有放回）            → 审稿人 B2 建议
  LS legacy      + 种子重抽样（3 个有放回）

指标：切片级 macro-F1（主）与录音级 macro-F1（稳健性）。Δ = f(A1) − f(A0)。
"""
import argparse
import csv
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUT = os.path.join(ROOT, "data", "results", "human_review")
LIB = "CTIS"
ARMS = ("A0", "A1")
BBS = ("alexnet", "swin_t")
SEEDS = (0, 1, 2)
TAGS = ("slice", "recording", "recording_sliceweighted_legacy")
SCHEMES = ("legacy_seedfixed", "stratified_seedfixed", "stratified_seedboot", "legacy_seedboot")
BOOT = 2000
SESOI = 3.0


def load(arm, bb, seed):
    p = os.path.join(RUNS, "%s_%s_%s" % (LIB, arm, bb), "seed%d_qtest.npz" % seed)
    z = np.load(p, allow_pickle=True)
    return z["probs"].astype(np.float32), z["y"].astype(np.int64), z["records"].astype(str)


def macro_f1_from_conf(conf):
    """conf (B, nc, nc) → macro-F1 (B,)；空类（该次样本里没出现）不参与。"""
    tp = np.einsum("bii->bi", conf)
    row = conf.sum(2)
    col = conf.sum(1)
    denom = 2 * tp + (col - tp) + (row - tp)
    keep = (row + col) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    f1 = np.where(keep, f1, np.nan)
    return np.nanmean(f1, axis=1)


class Metric:
    """把 (nrec, nc) 的「每个录音的预测计数」矩阵变成加权 macro-F1 的批量计算器。"""

    def __init__(self, M, true_idx, nc):
        self.M = M                    # (nrec, nc) float64
        self.true_idx = true_idx      # (nrec,) int
        self.nc = nc
        self.groups = [np.where(true_idx == c)[0] for c in range(nc)]

    def f1(self, W):
        """W (B, nrec) 录音权重 → (B,) macro-F1。"""
        B = W.shape[0]
        tp = np.zeros((B, self.nc))
        row = np.zeros((B, self.nc))
        col = np.zeros((B, self.nc))
        for c, idx in enumerate(self.groups):
            if idx.size == 0:
                continue
            Cc = W[:, idx] @ self.M[idx]          # (B, nc)
            tp[:, c] = Cc[:, c]
            row[:, c] = Cc.sum(1)
            col += Cc
        denom = 2 * tp + (col - tp) + (row - tp)
        keep = (row + col) > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
        f1 = np.where(keep, f1, np.nan)
        return np.nanmean(f1, axis=1)


def weights_legacy(rng, nrec, B):
    W = np.zeros((B, nrec))
    idx = rng.integers(0, nrec, size=(B, nrec))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=nrec)
    return W


def weights_stratified(rng, true_idx, nc, B):
    W = np.zeros((B, true_idx.size))
    for c in range(nc):
        idx = np.where(true_idx == c)[0]
        if idx.size == 0:
            continue
        draw = rng.integers(0, idx.size, size=(B, idx.size))
        for b in range(B):
            W[b, idx] = np.bincount(draw[b], minlength=idx.size)
    return W


SESOI_PT = SESOI  # 3 点


def summarise(d_frac, observed_frac):
    """输入为比例的配对 Δ；输出统一换成「点」。

    point 用观测到的配对 Δ（三个种子的均值），不是重抽样分布的中心；
    CI 用重抽样分位数；p 记录原始值与分辨率下限是否触发钳位。
    """
    d = np.asarray(d_frac, dtype=np.float64) * 100.0
    lo, hi = np.percentile(d, [2.5, 97.5])
    B = d.size
    p_lo = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    p_res = 2.0 / (B + 1)
    return {
        "point_pt": float(observed_frac * 100.0),
        "bootstrap_mean_pt": float(d.mean()),
        "bootstrap_median_pt": float(np.median(d)),
        "sd_pt": float(d.std(ddof=1)),
        "ci_lo_pt": float(lo),
        "ci_hi_pt": float(hi),
        "ci_halfwidth_pt": float((hi - lo) / 2.0),
        "p_raw": float(p_lo),
        "p_reported": float(max(p_lo, p_res)),
        "p_resolution": float(p_res),
        "p_clamped": bool(p_lo < p_res),
        "n_le0": int((d <= 0).sum()),
        "ci_lo_above_sesoi": bool(lo > SESOI_PT),
        "frac_draws_above_sesoi": float((d > SESOI_PT).mean()),
        "boot": int(B),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    B = a.boot

    # ---------- 参照面板：以 A0/alexnet/seed0 定义切片顺序 ----------
    _, y_ref, rec_ref = load("A0", "alexnet", 0)
    n = y_ref.size
    nc = int(y_ref.max()) + 1
    recs, rec_idx = np.unique(rec_ref, return_inverse=True)
    nrec = recs.size
    true_idx = np.zeros(nrec, dtype=np.int64)
    for r, c in enumerate(y_ref):
        true_idx[rec_idx[r]] = c

    # 配对断言：所有臂/骨干/种子必须逐切片同序、同标签、同录音
    for bb in BBS:
        for arm in ARMS:
            for s in SEEDS:
                _, y, rc = load(arm, bb, s)
                assert y.size == n, (bb, arm, s, "切片数不一致")
                assert np.array_equal(y, y_ref), (bb, arm, s, "标签不一致")
                assert np.array_equal(rc, rec_ref), (bb, arm, s, "录音序列不一致")

    print("[%5.1fs] 面板 %d 切片 / %d 录音 / %d 类（201 类占位）" % (time.time() - t0, n, nrec, nc))

    # ---------- 每 (arm, bb, seed) 的两种预测计数矩阵 ----------
    M_slice, M_rec, M_rec_sw = {}, {}, {}
    for bb in BBS:
        for arm in ARMS:
            for s in SEEDS:
                probs, y, rc = load(arm, bb, s)
                pred = probs.argmax(1)
                M_slice[(arm, bb, s)] = np.zeros((nrec, nc))
                np.add.at(M_slice[(arm, bb, s)], (rec_idx, pred), 1.0)
                # 录音级：同录音概率均值后一票
                acc = np.zeros((nrec, nc))
                np.add.at(acc, rec_idx, probs)
                cnt = np.bincount(rec_idx, minlength=nrec).astype(np.float64)
                rec_pred = (acc / np.maximum(cnt, 1)[:, None]).argmax(1)
                onehot = np.zeros((nrec, nc))
                onehot[np.arange(nrec), rec_pred] = 1.0   # 每录音一票（正确口径，配 :219 的 CI）
                M_rec[(arm, bb, s)] = onehot
                onehot_sw = np.zeros((nrec, nc))
                onehot_sw[np.arange(nrec), rec_pred] = cnt  # 旧点估计口径（w=rec_n），仅用于对照
                M_rec_sw[(arm, bb, s)] = onehot_sw
    print("[%5.1fs] 预测矩阵就绪" % (time.time() - t0))

    # ---------- 四套抽样权重 ----------
    W_leg = weights_legacy(np.random.default_rng(a.seed + 1), nrec, B)
    W_str = weights_stratified(np.random.default_rng(a.seed + 2), true_idx, nc, B)
    seed_draws = np.random.default_rng(a.seed + 3).integers(0, len(SEEDS), size=(B, len(SEEDS)))
    print("[%5.1fs] 权重就绪（legacy 行和 %d，stratified 行和 %d）"
          % (time.time() - t0, int(W_leg[0].sum()), int(W_str[0].sum())))

    out = {"lib": LIB, "boot": B, "rng_seed": a.seed, "sesoi_pt": SESOI,
           "panel": {"n_slices": int(n), "n_recordings": int(nrec), "n_classes": int(nc)},
           "by_backbone": {}}

    for tag, MM in tuple(zip(TAGS, (M_slice, M_rec, M_rec_sw))):
        for bb in BBS:
            f1 = {}
            for arm in ARMS:
                for s in SEEDS:
                    f1[(arm, bb, s)] = Metric(MM[(arm, bb, s)], true_idx, nc)

            # 点估计（不重抽样）= 全 1 权重
            W1 = np.ones((1, nrec))
            point = {}
            for s in SEEDS:
                point[s] = float(f1[("A1", bb, s)].f1(W1)[0] - f1[("A0", bb, s)].f1(W1)[0])

            obs = float(np.mean(list(point.values())))
            res = {"per_seed_delta_pt": {int(k): v * 100.0 for k, v in point.items()},
                   "seed_mean_pt": obs * 100.0,
                   "seed_sd_pt": float(np.std(list(point.values()), ddof=1)) * 100.0,
                   "seed_range_pt": float(max(point.values()) - min(point.values())) * 100.0}

            for name, W in (("legacy_seedfixed", W_leg), ("stratified_seedfixed", W_str)):
                d = np.mean([f1[("A1", bb, s)].f1(W) - f1[("A0", bb, s)].f1(W)
                             for s in SEEDS], axis=0)
                res[name] = summarise(d, obs)

            # 种子重抽样：第 b 次抽 3 个（有放回）种子，取这 3 个的均值
            for name, W in (("stratified_seedboot", W_str), ("legacy_seedboot", W_leg)):
                per_s = {s: (f1[("A1", bb, s)].f1(W) - f1[("A0", bb, s)].f1(W)) for s in SEEDS}
                acc = np.take_along_axis(
                    np.stack([per_s[s] for s in SEEDS], axis=1), seed_draws, axis=1).sum(axis=1)
                d = acc / len(SEEDS)
                res[name] = summarise(d, obs)

            out["by_backbone"].setdefault(bb, {})[tag] = res
            print("[%5.1fs] %s/%s 完成" % (time.time() - t0, tag, bb))

    # ---------- 只重抽种子（3 个值上的 3 次有放回均值）----------
    for bb in BBS:
        per_s_pt = np.array([out["by_backbone"][bb]["slice"]["per_seed_delta_pt"][s] for s in SEEDS])
        d = per_s_pt[seed_draws].mean(1) / 100.0
        out["by_backbone"][bb]["seedonly"] = summarise(d, per_s_pt.mean() / 100.0)

    # ---------- 落盘 ----------
    json.dump(out, open(os.path.join(a.out, "ctis_seed_stratified_bootstrap.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    csv_path = os.path.join(a.out, "ctis_seed_stratified_bootstrap.csv")
    cols = ["metric", "backbone", "scheme", "point_pt", "ci_lo_pt", "ci_hi_pt",
            "ci_halfwidth_pt", "bootstrap_mean_pt", "bootstrap_median_pt", "sd_pt",
            "seed_range_pt", "p_raw", "p_reported", "p_resolution", "p_clamped",
            "n_le0", "ci_lo_above_sesoi", "frac_draws_above_sesoi", "boot"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for bb in BBS:
            for tag in TAGS:
                r = out["by_backbone"][bb][tag]
                for scheme in SCHEMES:
                    if scheme not in r:
                        continue
                    x = r[scheme]
                    row = {"metric": tag, "backbone": bb, "scheme": scheme, "boot": x["boot"],
                           "seed_range_pt": r["seed_range_pt"] if tag == "slice" else "",
                           "p_clamped": int(bool(x["p_clamped"])),
                           "ci_lo_above_sesoi": int(bool(x["ci_lo_above_sesoi"]))}
                    for k in ("point_pt", "ci_lo_pt", "ci_hi_pt", "ci_halfwidth_pt",
                              "bootstrap_mean_pt", "bootstrap_median_pt", "sd_pt",
                              "p_raw", "p_reported", "p_resolution",
                              "frac_draws_above_sesoi", "n_le0"):
                        row[k] = "%.10g" % float(x[k])
                    w.writerow(row)
    print("[%5.1fs] 写出 %s" % (time.time() - t0, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
