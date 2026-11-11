# -*- coding: utf-8 -*-
"""neg_qpanel.py — 三个负对照库的容量表（schema 与 data/splits/qpanel_capacity.json 一致）

为什么需要：`code/splits_v2.py` 的 Q 抽签与 α 硬判要读 qpanel（每类 n_c / S_c / q_slices_expected /
qp_p5 / qp_p95 / m_c）。主样本五库的 qpanel 由 `code/qpanel_capacity.py` 生成；负对照库不在那份
事实源里，所以在这里按**同一套公式**重算并单独落 `data/splits/qpanel_neg.json`，
不改动 qpanel_capacity.json（它是主样本的口径真源）。

公式（与 code/qpanel_capacity.py 逐条对齐）
  q_c   = min(max(5, round(0.20·n_c)), max(1, n_c − 1))
  qpool = 400 次随机抽 q_c 条录音的 Σ floor(k_i/2)（random.Random(20261104)）：均值 / p5 / p95
  m_c   = round(0.60·S_c)
  alpha_max = qpool_mean / m_c

K̄≈1 的库上 qpool_mean 恒为 0（每条录音只有 1 片，floor(1/2)=0），这正是负对照的立论：
Q_pool = 0 ⇒ α_c = 0 ⇒ A1 与 A0 数学等价，只跑 A0 vs A2。
"""
import collections
import json
import os
import random
import statistics

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(PROJ, "data", "manifest")
OUT = os.path.join(PROJ, "data", "splits", "qpanel_neg.json")
LIBS = ["chest_falsetto", "erhu_playing_tech", "GZ_IsoTech"]
SEED, DRAWS, V = 20261104, 400, 0.60


def read_csv(p):
    import csv
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def q_of(n):
    q = max(5, int(round(0.20 * n)))
    return min(q, max(1, n - 1))


def sim_pool(ks, q):
    rng = random.Random(SEED)
    vals = []
    for _ in range(DRAWS):
        pick = rng.sample(ks, min(q, len(ks)))
        vals.append(sum(k // 2 for k in pick))
    vals.sort()
    return (statistics.fmean(vals), vals[int(0.05 * (len(vals) - 1))],
            vals[int(0.95 * (len(vals) - 1))])


def main():
    by_lib = {}
    for lib in LIBS:
        rows = read_csv(os.path.join(MAN, lib + ".csv"))
        per = collections.defaultdict(list)
        for r in rows:
            per[r["label"]].append(r["recording_uid"])
        classes = []
        for cls in sorted(per):
            cnt = collections.Counter(per[cls])
            ks = sorted(cnt.values())
            n_c, S_c = len(ks), sum(ks)
            q = q_of(n_c)
            mean, p5, p95 = sim_pool(ks, q)
            m_c = int(round(V * S_c))
            k2 = sum(1 for k in ks if k >= 2)
            classes.append({
                "cls": cls, "n_c": n_c, "S_c": S_c, "K_c": round(S_c / n_c, 4),
                "q_rec": q, "q_slices_expected": round(mean, 4),
                "qp_share": round(mean / S_c, 4) if S_c else 0.0,
                "coverage": round(k2 / n_c, 4), "k1_recordings": n_c - k2,
                "m_c": m_c, "alpha_max": round(mean / m_c, 4) if m_c else 0.0,
                "qp_p5": p5, "qp_p95": p95,
                "feasible_0.5": bool(0.5 * m_c <= mean),
                "need_slices_0.5": int(0.5 * m_c),
                "feasible_0.3": bool(0.3 * m_c <= mean),
                "need_slices_0.3": int(0.3 * m_c),
            })
        n_all = sum(c["n_c"] for c in classes)
        s_all = sum(c["S_c"] for c in classes)
        m_all = sum(c["m_c"] for c in classes)
        qp_all = sum(c["q_slices_expected"] for c in classes)
        by_lib[lib] = {
            "library": lib, "v": V, "seed": SEED, "draws": DRAWS,
            "source": "data/manifest/%s.csv（负对照，schema 对齐 code/qpanel_capacity.py）" % lib,
            "classes": classes,
            "all": {"classes": len(classes), "n_c": n_all, "S_c": s_all,
                    "K": round(s_all / n_all, 4) if n_all else None,
                    "qp": round(qp_all, 1), "qp_share": round(qp_all / s_all, 4) if s_all else 0.0,
                    "m_c": m_all,
                    "alpha_max": round(qp_all / m_all, 4) if m_all else 0.0,
                    "zero_exposure_classes": sum(1 for c in classes if c["q_slices_expected"] < 0.5)},
        }
        a = by_lib[lib]["all"]
        print("[%s] %d 类 / %d 录音 / %d 片 / K̄ %.4f / qpool 期望 %.2f 片 / α_max %.4f"
              % (lib, a["classes"], a["n_c"], a["S_c"], a["K"], a["qp"], a["alpha_max"]), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(by_lib, f, ensure_ascii=False, indent=1)
    print("→", OUT)


if __name__ == "__main__":
    main()
