# -*- coding: utf-8 -*-
"""E1a 审计补齐：为 4 个非主线库（instrument_timbre / timbre_range / acapella / song_structure）
用**盘面实文件**重建"源（录音）分组 → 切片随机 8:1:1"的同源泄漏估计。

与 ccmusic_splits.py 的 audit 口径一致：
  - 源 ID 规则按 v2.1 §4.1（本脚本内明写，一库一条）；
  - 切片随机划分模拟 20 次（seed 0..19），统计"测试切片在训练集里能找到同源兄弟"的比例；
  - 另做一次"按源分组 8:1:1"，验证该口径下泄漏率恒为 0（作为口径自检）。

只读数据集，写：
  data/splits/<dataset>_audit_disk.json
  data/splits/_audit_summary.csv   （保留前 9 行不动，替换/追加这 4 行）
  data/splits/_audit_summary.prev.csv（改前备份）

用法：
    python code/audit_extra4.py              # 全部 4 库
    python code/audit_extra4.py acapella     # 单库
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\VSCode\Datasets"
OUT = os.path.join(ROOT, "data", "splits")
SEEDS = 20
RATIOS = (0.8, 0.1, 0.1)


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def _parent(p):
    return os.path.basename(os.path.dirname(p))


# ---- 每库的切片清单与源 ID 规则（v2.1 §4.1） ----
def slices_instrument_timbre():
    """61 张 mel 图，一乐器一张；源 = 乐器本身（图路径即源），类 = Chinese/Western。"""
    rows = []
    for p in glob.glob(os.path.join(DATA, "instrument_timbre", "data", "mel", "*", "*.jpg")):
        rows.append((p, f"instrument::{_parent(p)}::{_stem(p)}", _parent(p)))
    return rows


def slices_timbre_range():
    """两类子任务共用乐器 id：range/*_n.jpg（源 = 下划线前 id）、timbre/{base,short,split}/*-n.jpg（源 = 连字符前 id）。"""
    rows = []
    base = os.path.join(DATA, "timbre_range", "data")
    for p in glob.glob(os.path.join(base, "range", "images", "*.jpg")):
        sid = _stem(p).split("_")[0]
        rows.append((p, f"range::{sid}", "range"))
    for sub in ("base", "short", "split"):
        for p in glob.glob(os.path.join(base, "timbre", "images", sub, "*.jpg")):
            sid = _stem(p).split("-")[0]
            rows.append((p, f"timbre::{sid}", "timbre"))
    return rows


def slices_acapella():
    """132 张 mel 图 = 6 首 × 22 片段；源 = song{1..6}（曲目）。"""
    rows = []
    for p in glob.glob(os.path.join(DATA, "acapella", "data", "mel", "*", "*.jpg")):
        rows.append((p, f"song::{_parent(p)}", "acapella"))
    return rows


def slices_song_structure():
    """300 张 mel 图，一首歌一张；源 = 曲目名（文件名去扩展名）。"""
    rows = []
    for p in glob.glob(os.path.join(DATA, "song_structure", "data", "mel", "*.jpg")):
        rows.append((p, f"track::{_stem(p)}", "song_structure"))
    return rows


RULES = {
    "instrument_timbre": slices_instrument_timbre,
    "timbre_range": slices_timbre_range,
    "acapella": slices_acapella,
    "song_structure": slices_song_structure,
}


def mc_leak(sources: list[str], seeds: int = SEEDS) -> dict:
    """切片随机 8:1:1：测试切片在训练集里存在同源兄弟的比例。"""
    n = len(sources)
    by_src = defaultdict(list)
    for i, s in enumerate(sources):
        by_src[s].append(i)
    rates = []
    for seed in range(seeds):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_tr = int(round(n * RATIOS[0]))
        n_va = int(round(n * RATIOS[1]))
        train, test = set(idx[:n_tr]), set(idx[n_tr + n_va:])
        tr_src = {sources[i] for i in train}
        with_sib = sum(1 for i in test if sources[i] in tr_src)
        rates.append(with_sib / max(len(test), 1))
    rates.sort()
    return {
        "leak_rate_mean": round(sum(rates) / len(rates), 4),
        "leak_rate_min": round(rates[0], 4),
        "leak_rate_max": round(rates[-1], 4),
        "seeds": seeds,
    }


def grouped_split_leak(sources: list[str], ratio=RATIOS[0], seed=0) -> dict:
    """按源分组 8:1:1（源不跨集）：泄漏率必须为 0，作为口径自检。"""
    by_src = defaultdict(list)
    for i, s in enumerate(sources):
        by_src[s].append(i)
    srcs = sorted(by_src)
    random.Random(seed).shuffle(srcs)
    n_tr = int(round(len(srcs) * ratio))
    train_src = set(srcs[:n_tr])
    train = {i for s in train_src for i in by_src[s]}
    test = [i for s in srcs if s not in train_src for i in by_src[s]]
    with_sib = sum(1 for i in test if sources[i] in train_src)
    return {"test_slices": len(test), "leak_slices": with_sib, "leak_rate": round(with_sib / max(len(test), 1), 4)}


def audit_one(name: str) -> dict:
    rows = RULES[name]()
    paths = [r[0] for r in rows]
    sources = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    by_src = Counter(sources)
    k = len(paths) / max(len(by_src), 1)
    res = {
        "dataset": name,
        "slices": len(paths),
        "sources": len(by_src),
        "K": round(k, 2),
        "source_size_hist": dict(sorted(Counter(by_src.values()).items())),
        "subgroups": dict(Counter(groups)),
        "mode": "slice_random_mc20_disk",
        "leak": mc_leak(sources),
        "grouped_check": grouped_split_leak(sources),
    }
    # 类可用性：只有 instrument_timbre 的目录名就是二分类标签
    if name == "instrument_timbre":
        res["classes"] = dict(Counter(groups))
    return res


def update_summary(results: list[dict]):
    path = os.path.join(OUT, "_audit_summary.csv")
    old = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            old = list(csv.DictReader(f))
        with open(os.path.join(OUT, "_audit_summary.prev.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(old[0].keys()))
            w.writeheader()
            w.writerows(old)
    names = {r["dataset"] for r in results}
    kept = [r for r in old if r["dataset"] not in names]
    new_rows = kept + [{
        "dataset": r["dataset"],
        "slices": r["slices"],
        "sources": r["sources"],
        "K": r["K"],
        "mode": r["mode"],
        "leak_rate": r["leak"]["leak_rate_mean"],
        "n_eff": r["sources"],
    } for r in results]
    order = ["CTIS", "music_genre", "pianos", "bel_canto", "CNPM", "chest_falsetto",
             "erhu_playing_tech", "Guzheng_Tech99", "GZ_IsoTech",
             "instrument_timbre", "timbre_range", "acapella", "song_structure"]
    new_rows.sort(key=lambda r: order.index(r["dataset"]) if r["dataset"] in order else 99)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "slices", "sources", "K", "mode", "leak_rate", "n_eff"])
        w.writeheader()
        w.writerows(new_rows)
    return new_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="all",
                    choices=["all", *RULES.keys()])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    targets = list(RULES) if args.dataset == "all" else [args.dataset]
    results = []
    for name in targets:
        r = audit_one(name)
        results.append(r)
        with open(os.path.join(OUT, f"{name}_audit_disk.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"{name:18s} slices={r['slices']:5d} sources={r['sources']:4d} K={r['K']:6.2f} "
              f"leak={r['leak']['leak_rate_mean']:.3f} "
              f"[{r['leak']['leak_rate_min']:.3f},{r['leak']['leak_rate_max']:.3f}] "
              f"grouped_leak={r['grouped_check']['leak_rate']:.3f}")
        if "classes" in r:
            print(f"{'':18s} classes={r['classes']}")
    rows = update_summary(results)
    print(f"\n_audit_summary.csv -> {len(rows)} 行")


if __name__ == "__main__":
    main()
