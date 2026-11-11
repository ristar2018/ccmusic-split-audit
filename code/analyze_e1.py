#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_e1.py — 读 sealed_eval 的逐样本预测，算 §3.5 规定的指标与区间。

设计依据（report/聚焦版实验设计方案v2.8.md）
  §3.5 :273-277  主指标 macro-F1 双口径（切片级 + 录音级「同录音概率平均后一票」），不一致以录音级为主；
                 区间三源分开报、不合成单一 CI：
                   ① 测试录音簇重采样 2,000 次（整簇保留/剔除，不拆簇；同一簇索引贯穿所有臂与骨干）
                   ② seed 间离散 = 3 seed 极差，只报数不检验
                   ③ 划分种子离散（本脚本不算，属附录）
                 确认性族 2 项 = CTIS × {AlexNet, Swin-T} 的 Δ1，Holm α=0.05；
                 其余库/骨干标 replication；SESOI 3 点（约定值）；
                 不做：3 seed 当独立样本检验、McNemar 比 macro-F1、用切片数当 n。
  §3.4 :269      禁止用 Q_test 早停/挑 seed —— 本脚本只读已封存的 Q_test 预测，不回写任何训练产物。

用法
  python code/analyze_e1.py                       # 默认 A1-A0，全库全骨干
  python code/analyze_e1.py --arms A2 --base A0   # E1c：A2-A0
  python code/analyze_e1.py --boot 500            # 快速自检

输出（data/results/）
  e1_metrics.csv        每 (lib, arm, bb, seed) 的切片级/录音级指标
  e1_delta.csv          每 (lib, bb) 的 Δ（3 seed 平均）+ 簇自助 CI + seed 极差
  e1_holm.json          确认性族 Holm 校正（CTIS × 两骨干）
  e1_per_class.csv      逐类 F1（3 seed 平均，喂 E1d 的类级分解图）
  e1_summary.md         人读摘要
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
SESOI = 3.0            # 点
LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto",
        "chest_falsetto", "erhu_playing_tech", "GZ_IsoTech"]  # 后三个是 E1f 负对照库
CONFIRM_ARM = "A1"          # 确认性族只认 Δ1（设计稿 §3.5「CTIS×{AlexNet,Swin-T} 的 Δ1」）
CONFIRM_LIBS = ["CTIS"]
CONFIRM_BBS = ["alexnet", "swin_t"]
LIBS = sorted(LIBS, key=len, reverse=True)   # 最长优先：短库名不得抢先匹配长库名


# ---------------------------------------------------------------- 读封存产物
def discover():
    """{lib: {(arm, bb): {seed: (probs, y, records, n_classes)}}}"""
    data = {}
    pend = os.path.join(RUNS, "pending")
    if os.path.isdir(pend) and os.listdir(pend):
        print("!! pending/ 仍有未封存组，本表为暂时结果：%s" % ", ".join(sorted(os.listdir(pend))))
    for d in sorted(glob.glob(os.path.join(RUNS, "*_*_*"))):
        if not os.path.isdir(d) or os.path.basename(d) == "pending":
            continue
        # 目录名 = <lib>_<arm>_<bb>；lib（music_genre/bel_canto）与 bb（swin_t）都含下划线，
        # 只能按已知词表前缀匹配，不能直接 split
        name = os.path.basename(d)
        lib = next((L for L in LIBS if name.startswith(L + "_")), None)
        if lib is None:
            print("!! 目录名无法识别库: %s" % name)
            continue
        rest = name[len(lib) + 1:]
        arm = next((A for A in ("A0", "A1", "A2", "C1") if rest.startswith(A + "_")), None)
        if arm is None:
            print("!! 目录名无法识别臂: %s" % name)
            continue
        bb = rest[len(arm) + 1:]
        seeds = {}
        for z in sorted(glob.glob(os.path.join(d, "seed*_qtest.npz"))):
            try:
                s = int(os.path.basename(z)[4:-10])
            except ValueError:
                print("!! 无法解析 seed 文件：%s" % z)
                continue
            seeds[s] = z
        miss = [s for s in (0, 1, 2) if s not in seeds]
        if miss:
            print("!! 缺封存 seed %s：%s" % (miss, name))
        if seeds:
            data.setdefault(lib, {})[(arm, bb)] = seeds
        else:
            print("!! 该组无 seed*_qtest.npz：%s" % name)
    return data


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["probs"].astype(np.float32), z["y"].astype(np.int64), z["records"].astype(str)


# ---------------------------------------------------------------- 指标
def macro_f1(conf):
    """conf[c, p] → macro-F1；空类不参与（与 sklearn zero_division=0 的 average='macro' 同口径）。"""
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    keep = conf.sum(1) + conf.sum(0) > 0      # 该类在本次样本里出现过
    f1 = f1[keep]
    return float(np.nanmean(f1)) if f1.size else float("nan"), f1, keep


def conf_slice(y, pred, nc, w=None):
    flat = y * nc + pred
    return np.bincount(flat, weights=w, minlength=nc * nc).reshape(nc, nc)


def per_recording(probs, y, rec_idx, nrec, nc):
    """同录音概率平均后一票。返回 (rec_y, rec_pred, rec_n_slices)。"""
    cnt = np.bincount(rec_idx, minlength=nrec).astype(np.float64)
    acc = np.zeros((nrec, nc), dtype=np.float64)
    np.add.at(acc, rec_idx, probs)
    mean = acc / np.maximum(cnt, 1)[:, None]
    rec_pred = mean.argmax(1)
    # 录音标签：取该录音切片标签的多数（同录音标签应一致，做个断言外的兜底）
    rec_y = np.zeros(nrec, dtype=np.int64)
    for c in range(nc):
        m = np.bincount(rec_idx[y == c], minlength=nrec)
        rec_y[m > 0] = c
    return rec_y, rec_pred, cnt


# ---------------------------------------------------------------- 主流程
def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A1", help="实验臂，逗号分隔（A1 或 A2）")
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
        print("!! %s 下没有已封存的组" % a.runs)
        return 2

    rng = np.random.default_rng(a.seed)
    rows_m, rows_d, rows_c = [], [], []
    boot_store = {}

    for lib in sorted(data):
        per_arm = data[lib]
        key0 = (a.base, "alexnet")
        if key0 not in per_arm:
            print("!! %s 缺 %s/alexnet 封存，跳过" % (lib, a.base))
            continue
        z0 = sorted(per_arm[key0])[0]
        _, y_ref, rec_ref = load_npz(per_arm[key0][z0])
        n = y_ref.size
        nc = int(y_ref.max()) + 1
        recs, rec_idx = np.unique(rec_ref, return_inverse=True)
        nrec = recs.size
        # 切片索引按录音分组（簇自助要整簇抽样）
        order = np.argsort(rec_idx, kind="stable")
        bounds = np.searchsorted(rec_idx[order], np.arange(nrec + 1))

        combos = sorted(per_arm)
        # 采样顺序一致性断言：所有臂/骨干/seed 的切片顺序必须完全相同（否则 Δ 不是配对的）
        for (arm, bb), seeds in per_arm.items():
            for s, z in seeds.items():
                _, y1, r1 = load_npz(z)
                if not (y1.shape == y_ref.shape and np.array_equal(y1, y_ref)
                        and np.array_equal(r1, rec_ref)):
                    raise SystemExit("%s/%s/%s/seed%d 的 Q_test 顺序与参照不一致" % (lib, arm, bb, s))

        # ---- 预计算每个 (arm,bb,seed) 的 pred 与录音级聚合
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
                # 逐类 F1（切片级）：直接由混淆矩阵按列算，nan 表示该类未出现在样本里
                tp = np.diag(conf).astype(float)
                fp = conf.sum(0) - tp
                fn = conf.sum(1) - tp
                den = 2 * tp + fp + fn
                per_class = np.where(den > 0, 2 * tp / den, np.nan)
                rows_c.append({"library": lib, "arm": arm, "bb": bb, "seed": s,
                               "f1_per_class": per_class})

        # ---- 簇自助：同一套录音索引贯穿本库所有臂与骨干
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

        # ---- Δ（同 seed 配对，再对 seed 取平均）
        bbs = sorted({bb for (_, bb) in per_arm})
        for bb in bbs:
            base_s = sorted(pre_k[2] for pre_k in pre if pre_k[0] == a.base and pre_k[1] == bb)
            if not base_s:
                print("!! %s/%s/%s 缺基线封存，跳过该骨干" % (lib, arm, bb))
                continue
            for arm in arms:
                arm_s = sorted(pre_k[2] for pre_k in pre if pre_k[0] == arm and pre_k[1] == bb)
                if not arm_s:
                    print("!! %s/%s/%s 无封存" % (lib, arm, bb))
                    continue
                seeds = sorted(set(base_s) & set(arm_s))
                if not seeds:
                    continue
                if len(seeds) < 3:
                    print("!! %s/%s/%s 只有 %d 个配对 seed：%s" % (lib, arm, bb, len(seeds), seeds))
                # 自助分布：每次重采样下，先算各 seed 的 (F1_arm − F1_base)，再对 seed 取平均
                d_slice = np.mean(np.stack([
                    boot_f1[(arm, bb, s)][0] - boot_f1[(a.base, bb, s)][0] for s in seeds]), axis=0)
                d_rec = np.mean(np.stack([
                    boot_f1[(arm, bb, s)][1] - boot_f1[(a.base, bb, s)][1] for s in seeds]), axis=0)
                # 点估计：3 seed 平均的 Δ
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
                # seed 极差（只报数：逐 seed 的 Δ 极差，单位点）
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

    # ---- Holm（确认性族）
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

    # ---- 落盘
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

    # 逐类：按 (lib, arm, bb) 对 seed 取平均后展开成「每类一行」
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
        json.dump({"family": "确认性族 = CTIS × {alexnet, swin_t} 的 %s−%s（只收 %s 臂）"
                          % (CONFIRM_ARM, a.base, CONFIRM_ARM),
                   "alpha": 0.05, "sesoi_pt": SESOI, "boot": a.boot,
                   "note": "boot_p 为录音簇自助的双侧百分位 p；Holm 逐项取 (m−i)·p 并单调化",
                   "tests": holm}, f, ensure_ascii=False, indent=1)

    # ---- 摘要（人读）
    lines = ["# E1 结果摘要（%s − %s）" % ("/".join(arms), a.base), "",
             "簇自助 %d 次（整录音簇有放回抽样，同一套索引贯穿同库所有臂与骨干）；"
             "Δ 单位为百分点；seed 极差是同 seed 配对的 Δ 在 3 seed 间的极差。" % a.boot, ""]
    if rows_d:
        # 带 arm 列：--arms A1,A2 时同库同骨干会出现两行，缺 arm 列会被读成「重复/矛盾」
        lines.append("| 库 | 臂 | 骨干 | Δ 切片级 [95% CI] | Δ 录音级 [95% CI] | seed 极差 | 过 SESOI(3pt) |")
        lines.append("| :-- | :-- | :-- | --: | --: | --: | :-- |")
        for r in rows_d:
            head = "| %s | %s | %s |" % (r["library"], r.get("arm", ""), r["bb"])
            lines.append("%s %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] | %s | %s |" % (
                head, r["delta_slice_pt"], r["ci_lo_slice_pt"], r["ci_hi_slice_pt"],
                r["delta_rec_pt"], r["ci_lo_rec_pt"], r["ci_hi_rec_pt"],
                r["seed_range_pt"] if r["seed_range_pt"] is not None else "n/a",
                "是" if r["meets_sesoi"] else "否"))
    if holm:
        lines += ["", "## 确认性族（Holm α=0.05）", ""]
        for k, v in sorted(holm.items()):
            lines.append("- %s：Δ=%+.2f pt，boot p=%.4f，Holm p=%.4f → %s" % (
                k, v["delta_slice_pt"], v["boot_p"], v["holm_p"],
                "拒绝 H0" if v["reject_at_0.05"] else "不拒绝"))
    with open(os.path.join(a.out, "e1_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\n→ %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
