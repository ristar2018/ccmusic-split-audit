# -*- coding: utf-8 -*-
"""设计审核用计算：机时核算 + 最小可检测效应（MDE）+ 类频次统计。

不训练任何模型，只用实测吞吐、切片数与泄漏审计结果做算术。
输出 results/design/design_audit.json 与若干 CSV。

用法：
    python code/design_audit.py            # 全部
    python code/design_audit.py budget     # 只算机时
    python code/design_audit.py mde        # 只算 MDE
"""
import argparse
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\VSCode\Datasets"
OUT = os.path.join(ROOT, "results", "design")

# 实测吞吐（img/s，224²，AMP fp16，bs 32/64）——来自 fapn 实测，见计划文档 §6.1
THROUGHPUT = {"alexnet": 339.0, "vgg19_bn": 167.0, "swin_t": 341.0}

# 各库 mel 切片数（= data/eval/<feature> 下的图片数；pianos 只有一种特征）
SLICES = {
    "CTIS": 43054, "music_genre": 36375, "pianos": 18352, "bel_canto": 9910,
    "GZ_IsoTech": 2899, "CNPM": 2724, "chest_falsetto": 1280,
    "erhu_playing_tech": 1253, "Guzheng_Tech99": 99,
}
# 有 cqt/chroma 三套特征的库
MULTI_FEATURE = ["CTIS", "music_genre", "CNPM", "bel_canto", "chest_falsetto", "GZ_IsoTech"]

EPOCHS = 10


def per_config_hours(datasets, backbones, epochs=EPOCHS, repeats=1):
    """跑完给定库 × 给定骨干各 1 次（repeats 个配置，如 3 种子 × 2 协议）需多少 GPU·h。"""
    imgs = sum(SLICES[d] for d in datasets) * epochs
    hours = {}
    for b in backbones:
        hours[b] = imgs / THROUGHPUT[b] / 3600.0 * repeats
    return hours


def budget():
    rows = []
    # ---- v1.0 的网格：9 库 × 3 骨干 × (3 seed × 2 协议) = 18 配置 ----
    mel_all = list(SLICES)
    h = per_config_hours(mel_all, list(THROUGHPUT), repeats=18)
    rows.append(("v1.0 E1 主网格（mel，9 库 × 3 骨干 × 6 配置）", sum(h.values())))
    # CQT 追加（CNPM、CTIS）
    h = per_config_hours(["CTIS", "CNPM"], list(THROUGHPUT), repeats=18)
    rows.append(("v1.0 E1 CQT 追加（同上 18 配置）", sum(h.values())))
    # 等量对照（P-orig 抽样到 P-source 规模，3 骨干 × 3 种子 = 9 配置）
    h = per_config_hours(mel_all, list(THROUGHPUT), repeats=9)
    rows.append(("v1.0 E1 同数据量对照（9 配置）", sum(h.values())))
    # P-proxy（4 个 K≈1 库 × 3 骨干 × 3 种子 = 9 配置）
    proxy = ["GZ_IsoTech", "chest_falsetto", "erhu_playing_tech", "Guzheng_Tech99"]
    h = per_config_hours(proxy, list(THROUGHPUT), repeats=9)
    rows.append(("v1.0 E1 P-proxy（4 库 × 9 配置）", sum(h.values())))
    v10 = sum(v for _, v in rows)

    # ---- v1.1 的分层网格 ----
    tier1 = ["CTIS", "music_genre", "pianos", "bel_canto", "CNPM"]
    tier2 = ["GZ_IsoTech", "chest_falsetto", "erhu_playing_tech"]
    v11 = []
    # T1：3 骨干 × {orig,source} × 2 种子 = 12 配置
    h = per_config_hours(tier1, list(THROUGHPUT), repeats=12)
    v11.append(("T1 五个主力库：3 骨干 × 2 协议 × 2 种子（12 配置）", sum(h.values())))
    # T2：2 骨干（alexnet, swin_t）× 2 协议 × 2 种子 = 8 配置
    h = per_config_hours(tier2, ["alexnet", "swin_t"], repeats=8)
    v11.append(("T2 三个薄库：2 骨干 × 2 协议 × 2 种子（8 配置）", sum(h.values())))
    # 特征对照：2 骨干 × 2 协议 × 1 种子 = 4 配置，CQT+chroma 各一轮
    h = per_config_hours(["CTIS", "music_genre", "CNPM"], ["alexnet", "swin_t"], repeats=4)
    v11.append(("T3 特征对照 CQT（CTIS/music_genre/CNPM，4 配置）", 2 * sum(h.values())))
    # P-proxy：2 骨干 × 3 种子 = 6 配置
    h = per_config_hours(proxy, ["alexnet", "swin_t"], repeats=6)
    v11.append(("T4 P-proxy（4 库 × 6 配置）", sum(h.values())))
    # 等量对照：2 骨干 × 2 协议 × 2 种子 = 8 配置（只对 Δ 最大的 3 个库做）
    h = per_config_hours(["CTIS", "music_genre", "bel_canto"], ["alexnet", "swin_t"], repeats=8)
    v11.append(("T5 同数据量对照（3 库，8 配置）", sum(h.values())))
    v11_total = sum(v for _, v in v11)

    out = {
        "throughput": THROUGHPUT,
        "slices": SLICES,
        "v1_0": {"items": [{"name": n, "gpu_h": round(v, 1)} for n, v in rows], "total": round(v10, 1)},
        "v1_1": {"items": [{"name": n, "gpu_h": round(v, 1)} for n, v in v11], "total": round(v11_total, 1)},
    }
    return out


def _mde_points(p, k, n_sources, rho=0.5, seed_pt=0.4, n_seeds=2,
                alpha=0.05, power=0.80):
    """配对比较的最小可检测效应（百分点），同时给出"伪独立"CI 的收窄倍数。

    三项误差源：
      1) 测试集抽样（按源聚簇）：n_sources 个源，每个源 k 片，源内正确率相关 ρ；
      2) 训练随机性：不同种子之间的指标标准差 seed_pt（百分点，经验取 0.4）；
      3) 配对比较本身消掉了各臂共享的那部分方差。
    返回 (MDE_配对, CI_源簇半宽, CI_伪独立半宽)。
    """
    z = 1.959964 + 0.8416212
    var_p = p * (1 - p)
    # 准确率 CI：按源聚簇（ρ 为源内相关）vs 假装每片独立
    ci_src = 1.959964 * math.sqrt(2 * var_p * (1 + (k - 1) * rho) / n_sources / k)
    ci_naive = 1.959964 * math.sqrt(2 * var_p / (n_sources * k))
    # 配对 Δ：源内共享部分被抵消，只剩逐源二项噪声 + 种子噪声
    se_pair = math.sqrt(2 * var_p / k / n_sources + (seed_pt / 100.0) ** 2 / n_seeds)
    return z * se_pair * 100.0, ci_src * 100.0, ci_naive * 100.0


def _mcnemar_mde(n, p_disc=0.20, alpha=0.05, power=0.80):
    """配对二分类（McNemar）：给定不一致率与样本量，能检出的最小差（百分点）。"""
    se = math.sqrt(p_disc / n)
    return (1.959964 + 0.8416212) * se * 100.0


def mde():
    csv_path = os.path.join(ROOT, "data", "splits", "_audit_summary.csv")
    base_acc = {
        "chest_falsetto": 0.994, "erhu_playing_tech": 0.994, "CTIS": 0.980,
        "bel_canto": 0.948, "CNPM": 0.859, "Guzheng_Tech99": 0.862, "GZ_IsoTech": 0.855,
        "pianos": 0.90, "music_genre": 0.80,
    }
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            try:
                n_src = int(r["sources"])
                slices = int(r["slices"])
                k = float(r["K"]) if r.get("K") else slices / max(n_src, 1)
            except (KeyError, ValueError):
                continue
            p = base_acc.get(ds, 0.90)
            mde_pair, ci_src, ci_naive = _mde_points(p, k, n_src)
            rows.append({
                "dataset": ds, "slices": slices, "sources": n_src, "K": round(k, 2),
                "baseline_acc": p,
                "mde_paired_pt": round(mde_pair, 2),
                "ci_src_pt": round(ci_src, 2),
                "ci_naive_pt": round(ci_naive, 2),
                "ci_inflation": round(ci_src / ci_naive, 2) if ci_naive else None,
            })
    paired = {
        "E2_test_496_real": round(_mcnemar_mde(496), 2),
        "E2_cross_session_5cls_n160": round(_mcnemar_mde(160), 2),
        "E3_287_songs_5fold": round(_mcnemar_mde(287), 2),
        "E4_few_group_230_slices": round(_mcnemar_mde(230), 2),
        "E1_chest_1280": round(_mcnemar_mde(1280), 2),
    }
    return {"per_dataset": rows, "paired_mcnemar": paired}


def class_histograms():
    """CTIS 类频次 + 长尾分组（用于检验 E4 的 few 组假设）。"""
    base = os.path.join(DATA, "CTIS", "data", "audio")
    counts = {}
    if os.path.isdir(base):
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if os.path.isdir(p):
                counts[d] = sum(1 for f in os.listdir(p) if not f.startswith("."))
    vals = sorted(counts.values())
    groups = {
        "many_ge50": sum(1 for v in vals if v >= 50),
        "medium_10_49": sum(1 for v in vals if 10 <= v < 50),
        "few_lt10": sum(1 for v in vals if v < 10),
        "lt5": sum(1 for v in vals if v < 5),
        "eq1": sum(1 for v in vals if v == 1),
    }
    return {
        "n_classes": len(counts), "n_files": sum(vals),
        "min": vals[0] if vals else None, "median": vals[len(vals) // 2] if vals else None,
        "max": vals[-1] if vals else None, "groups": groups,
        "few_share_of_classes_pct": round(100 * groups["few_lt10"] / max(len(counts), 1), 1),
        "few_share_of_files_pct": round(100 * sum(v for v in vals if v < 10) / max(sum(vals), 1), 1),
        "counts": counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "budget", "mde", "hist"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}
    if args.what in ("all", "budget"):
        out["budget"] = budget()
        print("== 机时 ==")
        for k in ("v1_0", "v1_1"):
            print(f"[{k}] 合计 {out['budget'][k]['total']} GPU·h")
            for it in out["budget"][k]["items"]:
                print(f"   {it['gpu_h']:6.1f}  {it['name']}")
    if args.what in ("all", "mde"):
        out["mde"] = mde()
        print("\n== 最小可检测效应（百分点）==")
        for r in out["mde"]["per_dataset"]:
            print(f"   {r['dataset']:18s} n_src={r['sources']:5d} K={r['K']:6.2f} "
                  f"配对MDE={r['mde_paired_pt']:5.2f}  CI: 源簇±{r['ci_src_pt']:4.2f} vs 伪独立±{r['ci_naive_pt']:4.2f} "
                  f"({r['ci_inflation']}×)")
        for k, v in out["mde"]["paired_mcnemar"].items():
            print(f"   {k:34s} McNemar MDE={v} pt")
    if args.what in ("all", "hist"):
        out["ctis_hist"] = class_histograms()
        h = out["ctis_hist"]
        print(f"\n== CTIS 类频次 == {h['n_classes']} 类 / {h['n_files']} 条，"
              f"median={h['median']} min={h['min']} max={h['max']}，分组 {h['groups']}")
    with open(os.path.join(OUT, "design_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if "ctis_hist" in out:
        with open(os.path.join(OUT, "ctis_class_counts.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_dir", "n_files"])
            for k, v in sorted(out["ctis_hist"]["counts"].items(), key=lambda x: -x[1]):
                w.writerow([k, v])
    if "mde" in out:
        with open(os.path.join(OUT, "mde.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out["mde"]["per_dataset"][0]))
            w.writeheader()
            w.writerows(out["mde"]["per_dataset"])
    print(f"\n写出 {os.path.join(OUT, 'design_audit.json')}")


if __name__ == "__main__":
    main()
