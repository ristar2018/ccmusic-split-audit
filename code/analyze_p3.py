# -*- coding: utf-8 -*-
"""P3 分析：三折 pooled 指标 → 臂间配对 Δ（录音簇自助）+ 探针 AUC 对照。

口径依据 worklog/exp-plan/P3-预注册草案.md（v1 冻结稿）：
  - 主指标 = 三折 held-out 预测**合并**后的 pooled macro-F1（切片级为主、录音级并列）。
  - 确认性族 1 项 = A1 vs A2 的 pooled macro-F1 差（Holm 不需要，单条）。
  - A0 vs A1 只报点估计与 CI，标「捷径贡献上界」，不进确认性族。
  - 诊断指标 = 冻结特征上乐器身份线性探针 AUC（A0 vs A2），配合 A1 作参照。

自助法：按**录音簇**重采样 2000 次，同一份被抽到的录音索引贯穿所有臂与骨干
（这是唯一的配对方式——A0 用切片级随机折、A1/A2 用留乐器折，切片顺序不同，
但三折合并后覆盖的切片集合与录音集合相同）。

用法：
  python code/analyze_p3.py --arms A0,A1 --base A1 --boot 2000
输出：data/results/p3/{p3_metrics.csv, p3_delta.csv, p3_probe.csv, p3_summary.md}
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
ARM_BASE = "A1"          # 预注册：主结论以「留乐器」A1 为参照
FOLDS = [1, 2, 3]


def macro_f1(y, pred, n_classes):
    """逐类 F1 后取平均；只统计「真值或预测至少出现一次」的类。"""
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
    """读三折 npz 并合并；返回 (probs, y, records) 或 None。"""
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
    """录音 → 列的索引数组，并按录音名字典序固定顺序（各臂共用同一簇索引）。"""
    uniq = sorted(set(records.tolist()))
    pos = {r: i for i, r in enumerate(uniq)}
    return uniq, np.array([pos[r] for r in records], dtype=np.int64)


def per_recording(prob, y, rec_idx, n_rec, n_classes):
    """同录音概率平均后一票。"""
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
    ap.add_argument("--out", default="", help="输出目录；缺省 data/results/p3")
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    base = a.base
    out_dir = a.out or OUT
    os.makedirs(out_dir, exist_ok=True)

    # ---- 盘点 ----
    runs = defaultdict(dict)          # (arm,bb) -> seed -> (prob,y,rec)
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "*", "seed*", "fold1.npz"))):
        arm, bb, sd = p.split(os.sep)[-4:-1]
        seed = int(sd.replace("seed", ""))
        got = load_run(arm, bb, seed)
        if got is not None:
            runs[(arm, bb)][seed] = got
    if not runs:
        raise SystemExit("没有找到 runs_p3/ 下的三折产物")
    n_classes = int(max(v[1].max() for d in runs.values() for v in d.values())) + 1

    # ---- 类别名 ----
    cls_names = {}
    fu = os.path.join(HERE, "data", "p3", "p3_folds.json")
    if os.path.exists(fu):
        j = json.load(open(fu, encoding="utf-8"))
        inv = {v: k for k, v in j.get("class_to_idx", {}).items()}
        cls_names = inv

    # 录音簇的公共顺序：取所有臂/骨干/seed 的并集（三折合并后应完全相同）
    all_recs = set()
    for d in runs.values():
        for prob, y, rec in d.values():
            all_recs |= set(rec.tolist())
    uniq = sorted(all_recs)
    pos = {r: i for i, r in enumerate(uniq)}
    n_rec = len(uniq)
    print("[clusters] %d 个录音" % n_rec)

    rng = np.random.default_rng(a.seed)
    picks = [rng.integers(0, n_rec, n_rec) for _ in range(a.boot)]

    rows_m, rows_d, rows_p = [], [], []
    for arm, bb in sorted(runs):
        d = runs[(arm, bb)]
        seeds = sorted(d)
        # 点估计
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

        # 自助：每次重采样内先逐 seed 算 F1 再对 seed 取均值
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
            # 每 seed 预先把「切片级预测 / 录音级聚合」都摊好，自助里只做加权
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

        # ---- 探针 ----
        for s in seeds:
            pj = os.path.join(RUNS, arm, bb, "seed%d" % s)
            aucs, tops = [], []
            for f in FOLDS:
                # 优先读 p3_probe.py 回填的新口径探针（训练侧 8:2 评估）
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
        print("[out] %s (%d 行)" % (os.path.relpath(path, HERE), len(rows)))

    dump(os.path.join(out_dir, "p3_metrics.csv"), rows_m)
    dump(os.path.join(out_dir, "p3_delta.csv"), rows_d)
    dump(os.path.join(out_dir, "p3_probe.csv"), rows_p)

    # ---- 摘要 ----
    lines = ["# P3 pooled 结果（%d 次录音簇自助）" % a.boot, "",
             "主指标 pooled macro-F1（三折合并），基准臂 = %s。" % base, "",
             "## 点估计", "", "| 臂 | 骨干 | 切片级 | 录音级 | seed 极差 |", "| :-- | :-- | --: | --: | --: |"]
    for r in rows_m:
        lines.append("| %s | %s | %.4f | %.4f | %.3f |" % (r["arm"], r["bb"], r["macro_f1_slice"],
                                                           r["macro_f1_recording"], r["seed_spread_slice"]))
    if rows_d:
        lines += ["", "## 配对 Δ（百分点）", "",
                  "| 臂 − 基准 | 骨干 | Δ 切片级 [95% CI] | Δ 录音级 [95% CI] | p | 确认性 |",
                  "| :-- | :-- | --: | --: | --: | :-- |"]
        for r in rows_d:
            lines.append("| %s − %s | %s | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] | %.4f | %s |"
                         % (r["arm"], r["base"], r["bb"], r["delta_slice"], r["ci_lo"], r["ci_hi"],
                            r["delta_rec"], r["rec_ci_lo"], r["rec_ci_hi"], r["p_boot"],
                            "是" if r["confirmatory"] else "—"))
    if rows_p:
        lines += ["", "## 乐器身份探针（冻结特征上的线性分类器）", "",
                  "| 臂 | 骨干 | seed | AUC(macro OvR) | top1 | 折数 |", "| :-- | :-- | --: | --: | --: | --: |"]
        for r in rows_p:
            lines.append("| %s | %s | %d | %s | %s | %d |" % (r["arm"], r["bb"], r["seed"],
                                                               r["probe_auc_mean"], r["probe_top1_mean"], r["folds"]))
    open(os.path.join(out_dir, "p3_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[out] data/results/p3/p3_summary.md")


if __name__ == "__main__":
    main()
