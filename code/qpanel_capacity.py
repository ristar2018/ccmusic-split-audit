# -*- coding: utf-8 -*-
"""Q 面板容量可行性实测（v2.3 设计依据）

回答外审致命项 1：按 §3.1 的 Q/R/C 划分，A1 的"50% 切片来自 Q_pool"在 5 个主线库上
是否可行；若不可行，最大的可行暴露比例 α_max 与对应的训练量 v 是多少。

口径（与 v2.2 §3.1/§3.2 对齐）
- 类内录音随机分 Q（20%，下限 5）/ R（15%）/ C（其余）
- 每个 Q 录音的切片按内容哈希排序后偶数位进 Q_test、奇数位进 Q_pool；
  只要 1 片的录音整片进 Q_test，不进 Q_pool（floor(k/2)=0 自动实现）
- m_c = 每类训练切片数 = v × S_c（v=0.6 对应 §4 的"A0/A1 取 60% 训练量"）
- A1 的暴露比例 α 指 m_c 中来自 Q_pool 的切片占比（预设 50%）
  → 可行性条件：α × v × S_c ≤ Q_pool_c

只读 D:\\VSCode\\Datasets；结果写 data/splits/qpanel_capacity.json。
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = r"D:\VSCode\Datasets"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "splits", "qpanel_capacity.json")

TRAIL = re.compile(r"_\d+$")


def stem(name):
    return name.rsplit(".", 1)[0]


def src_of(name):
    """去扩展名后再去尾部 _<n>；与 code/ccmusic_splits.py 的 source_of 同口径"""
    return TRAIL.sub("", stem(name))


def cnpm_pattern_map():
    """CNPM 的类在 data/label.csv 的「样式/Pattern」列；文件名做键（GBK）"""
    import csv
    import io
    path = os.path.join(ROOT, "CNPM", "data", "label.csv")
    m = {}
    with io.open(path, encoding="gbk") as f:
        for row in csv.DictReader(f):
            fn = (row.get("文件名/File Name") or "").strip()
            pat = (row.get("样式/Pattern") or "").strip()
            if fn:
                m[fn] = "pattern_" + (pat or "?")
    return m


def walk_pairs(lib):
    """返回 {(class, source): slice_count}"""
    base = os.path.join(ROOT, lib, "data", "eval", "mel")
    if not os.path.isdir(base):          # pianos 的特征图直接放 data/eval/<品牌>/
        base = os.path.join(ROOT, lib, "data", "eval")
    cnpm_map = cnpm_pattern_map() if lib == "CNPM" else None
    pairs = Counter()
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if not fn.lower().endswith((".jpg", ".png")):
                continue
            rel = os.path.relpath(dirpath, base).replace("\\", "/")
            parts = [] if rel == "." else rel.split("/")
            s = src_of(fn)
            if lib == "CTIS":
                cls = parts[0]                       # C0090
            elif lib == "music_genre":
                cls = parts[-1] if parts else "?"     # 中层 9 类
            elif lib == "pianos":
                cls = parts[0] if parts else "?"      # 品牌
            elif lib == "bel_canto":
                cls = parts[0] if parts else "?"
            elif lib == "CNPM":
                cls = cnpm_map.get(s, "pattern_?") if cnpm_map else "?"
            else:
                raise SystemExit("unknown lib " + lib)
            pairs[(cls, s)] += 1
    return pairs


def class_tables(lib):
    pairs = walk_pairs(lib)
    per_class_sources = defaultdict(dict)
    for (cls, s), k in pairs.items():
        per_class_sources[cls][s] = k
    return per_class_sources


def sim_qpool(ks, q):
    """给一个类的录制切片数列表与 Q 录音数，返回 Q_pool 切片数的 (mean, p5, p95)"""
    n = len(ks)
    q = max(1, min(q, n))
    trials = []
    rng = random.Random(20261104)
    for _ in range(400):
        pick = rng.sample(range(n), q)
        trials.append(sum(ks[i] // 2 for i in pick))
    trials.sort()
    return sum(trials) / len(trials), trials[int(0.05 * len(trials))], trials[int(0.95 * len(trials))]


def analyse(lib, v=0.6, alphas=(0.5, 0.3)):
    per_class = class_tables(lib)
    rows = []
    for cls in sorted(per_class):
        ks = list(per_class[cls].values())
        n_c = len(ks)
        S_c = sum(ks)
        q_c = max(5, round(0.2 * n_c))
        q_c = min(q_c, max(1, n_c - 1))
        mean_qp, p5, p95 = sim_qpool(ks, q_c)
        qp = mean_qp
        m_c = round(v * S_c)
        cov = sum(1 for k in ks if k >= 2) / n_c      # 有兄弟切片的录音占比（=Q_test 覆盖率上界）
        row = dict(cls=cls, n_c=n_c, S_c=S_c, K_c=round(S_c / n_c, 2),
                   q_rec=q_c, q_slices_expected=qp, qp_share=round(qp / S_c, 4),
                   coverage=round(cov, 4), k1_recordings=n_c - sum(1 for k in ks if k >= 2),
                   m_c=m_c, alpha_max=round(qp / m_c, 4) if m_c else 0.0,
                   qp_p5=p5, qp_p95=p95)
        for a in alphas:
            row["feasible_%.1f" % a] = bool(a * m_c <= qp)
            row["need_slices_%.1f" % a] = int(round(a * m_c))
        rows.append(row)

    def agg(rows):
        n = sum(r["n_c"] for r in rows)
        S = sum(r["S_c"] for r in rows)
        qp = sum(r["q_slices_expected"] for r in rows)
        m = sum(r["m_c"] for r in rows)
        feas = {}
        for a in alphas:
            feas["%.1f" % a] = sum(1 for r in rows if r["feasible_%.1f" % a])
        zero = [r for r in rows if r["alpha_max"] <= 0.0]
        al = sorted(r["alpha_max"] for r in rows)
        q = lambda p: al[min(len(al) - 1, int(p * len(al)))]
        return dict(classes=len(rows), n_c=n, S_c=S, K=round(S / n, 2),
                    qp=round(qp, 1), qp_share=round(qp / S, 4), m_c=m,
                    alpha_max=round(qp / m, 4), feasible_classes=feas,
                    v_required_for_alpha_0_5=round(qp / (0.5 * S), 4),
                    coverage_measured=round(sum(r["coverage"] * r["n_c"] for r in rows) / n, 4),
                    alpha_p5=round(q(0.05), 3), alpha_p50=round(q(0.5), 3), alpha_p95=round(q(0.95), 3),
                    zero_exposure_classes=len(zero),
                    zero_share_of_slices=round(sum(r["S_c"] for r in zero) / S, 4))
    return rows, agg(rows)


def main():
    out = {"_meta": {
        "date": "2026-11-04",
        "source": "D:/VSCode/Datasets/<lib>/data/eval/mel 盘面重建 class↔source",
        "rule": "类内录音 20%（下限 5）作 Q；Q 录音 floor(k/2) 切片进 Q_pool；m_c=v*S_c，v=0.6",
        "note": "Q_pool 为 400 次随机抽样的均值；1 片录音不进 Q_pool",
    }}
    for lib in ("CTIS", "music_genre", "pianos", "bel_canto", "CNPM"):
        rows, agg = analyse(lib)
        small = [r["cls"] for r in rows if r["n_c"] < 10]
        out[lib] = {"all": agg, "classes": rows, "small_classes": small}
        print("== %s ==" % lib)
        print("  classes=%d  n_c=%d  S_c=%d  K=%.2f  Q_pool=%d (%.1f%% of S)"
              % (agg["classes"], agg["n_c"], agg["S_c"], agg["K"], agg["qp"], 100 * agg["qp_share"]))
        print("  m_c(v=0.6)=%d  alpha_max=%.3f (p5/50/95 = %.3f/%.3f/%.3f)  coverage=%.1f%%"
              % (agg["m_c"], agg["alpha_max"], agg["alpha_p5"], agg["alpha_p50"], agg["alpha_p95"],
                 100 * agg["coverage_measured"]))
        print("  feasible@0.5=%d/%d  feasible@0.3=%d/%d  v needed for a=0.5: %.3f  zero-exposure classes=%d (%.1f%% of slices)"
              % (agg["feasible_classes"]["0.5"], agg["classes"],
                 agg["feasible_classes"]["0.3"], agg["classes"], agg["v_required_for_alpha_0_5"],
                 agg["zero_exposure_classes"], 100 * agg["zero_share_of_slices"]))
        worst = sorted(rows, key=lambda r: r["alpha_max"])[:3]
        for r in worst:
            print("   worst %-28s n=%3d S=%5d K=%5.2f q=%3d QP=%6.0f alpha_max=%.3f"
                  % (r["cls"], r["n_c"], r["S_c"], r["K_c"], r["q_rec"], r["q_slices_expected"], r["alpha_max"]))
        if small:
            print("   classes with n_c<10:", len(small), small[:6])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
