"""CCMusic 泄漏审计 + 按源（组）重新划分。

两种输入：
    --index data/index/<dataset>__data_eval.zip.json   （推荐：先跑 ccmusic_zipindex.py，几 MB 即可）
    --zip  <本地 zip>                                   （已下载完整包时用）

两种模式：
    audit    复现官方"切片 → random.shuffle → 8:1:1"过程，估计测试切片的同源泄漏率；
             若路径里自带 train/test 目录（如 GZ_IsoTech），则直接按真实划分统计。
    regroup  按"源"分组、按类分层，生成确定性 8:1:1 划分（可复现，任何源不跨集）。

用法：
    python code/ccmusic_splits.py --project p1 --mode audit
    python code/ccmusic_splits.py --dataset music_genre --index data/index/music_genre__data_eval.zip.json --mode regroup
    python code/ccmusic_splits.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "splits")
IMG_EXT = (".jpg", ".jpeg", ".png")
FEATURES = ("mel", "cqt", "chroma")
SPLIT_DIRS = {"train": "train", "training": "train",
              "valid": "validation", "validation": "validation", "val": "validation",
              "test": "test", "testing": "test"}


# ---------------------------------------------------------------- 解析
def parts_of(member: str) -> list[str]:
    return [p for p in member.replace("\\", "/").split("/") if p]


def source_of(member: str) -> str:
    """源 ID：同一段录音/片段的所有切片必须得到同一个值。"""
    stem = os.path.splitext(os.path.basename(member))[0]
    for f in FEATURES:                      # 去掉特征后缀
        if stem.endswith("_" + f):
            stem = stem[: -(len(f) + 1)]
            break
    if "__" in stem:                        # erhu: detache__detache_01
        return stem.split("__", 1)[1]
    m = re.match(r"^([0-9a-f]{32})", stem)  # music_genre: md5 前缀
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)_", stem)          # chest_falsetto: 0001_m_chest_0
    if m:
        return m.group(1)
    return re.sub(r"_(\d+)$", "", stem)     # GZ_IsoTech / pianos: xxx_12 → xxx


def class_of(member: str) -> str:
    stem = os.path.splitext(os.path.basename(member))[0]
    for p in reversed(parts_of(member)[:-1]):
        if p.lower() in (*FEATURES, "eval", "audio", *SPLIT_DIRS):
            continue
        return p
    if "__" in stem:                        # erhu 的类名写在文件名里
        return stem.split("__", 1)[0]
    return "unknown"


def split_in_path(member: str) -> str | None:
    for p in parts_of(member)[:-1]:
        if p.lower() in SPLIT_DIRS:
            return SPLIT_DIRS[p.lower()]
    return None


def keys_of(member: str) -> tuple[str, str, str | None]:
    """返回 (source_key, class, split_in_path)。source_key 带类前缀，避免不同类下同号相撞。"""
    return f"{class_of(member)}::{source_of(member)}", class_of(member), split_in_path(member)


def names_from_index(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    names = data.get("names")
    if not names:
        raise SystemExit(f"{path} 里没有 names，请用 ccmusic_zipindex.py --full 重新生成")
    return names


def names_from_zip(path: str) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(IMG_EXT)]


# ---------------------------------------------------------------- audit
def audit(names: list[str], seeds: int = 20, ratios=(0.8, 0.1, 0.1)) -> dict:
    keys = [keys_of(n)[0] for n in names]
    classes = [keys_of(n)[1] for n in names]
    real = [keys_of(n)[2] for n in names]
    n, n_src = len(keys), len(set(keys))
    k = n / max(n_src, 1)
    by_key = defaultdict(list)
    for i, key in enumerate(keys):
        by_key[key].append(i)

    out = {
        "slices": n, "sources": n_src, "slices_per_source": round(k, 2),
        "classes": len(set(classes)),
        "source_slice_hist": dict(sorted(Counter(len(v) for v in by_key.values()).items())),
    }

    if all(real):
        # 情况 A：zip 里就有官方 train/test 目录 → 直接统计
        tr = {keys[i] for i, s in enumerate(real) if s == "train"}
        te = [i for i, s in enumerate(real) if s == "test"]
        va = [i for i, s in enumerate(real) if s == "validation"]
        leak = sum(1 for i in te if keys[i] in tr)
        out.update({
            "mode": "official_dirs",
            "split_sizes": {"train": sum(1 for s in real if s == "train"),
                            "validation": len(va), "test": len(te)},
            "leak_rate": round(leak / max(len(te), 1), 4),
            "leak_rate_mean": round(leak / max(len(te), 1), 4),
        })
    else:
        # 情况 B：官方在加载时 random.shuffle（无种子）→ 复现过程、多种子统计
        rates, sizes = [], []
        for seed in range(seeds):
            idx = list(range(n))
            random.Random(seed).shuffle(idx)
            p80, p90 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
            train, test = idx[:p80], idx[p90:]
            tr = {keys[i] for i in train}
            rates.append(sum(1 for i in test if keys[i] in tr) / max(len(test), 1))
            sizes.append({"train": len(train), "validation": p90 - p80, "test": len(test)})
        out.update({
            "mode": "simulated_random",
            "seeds": seeds,
            "leak_rate_mean": round(statistics.mean(rates), 4),
            "leak_rate_min": round(min(rates), 4),
            "leak_rate_max": round(max(rates), 4),
            "theoretical": round(1 - (1 - ratios[0]) ** max(k - 1, 0), 4) if k >= 1 else 0.0,
            "split_sizes": sizes[0],
        })

    out["n_eff"] = n_src
    out["effective_reduction"] = round(n / max(n_src, 1), 2)
    out["note"] = "leak_rate=测试切片中同源切片也出现在训练集的比例；n_eff=独立源数量"
    return out


# ---------------------------------------------------------------- regroup
def regroup(names: list[str], ratios=(0.8, 0.1, 0.1)) -> dict:
    keys = [keys_of(n)[0] for n in names]
    cls_of_key = {keys_of(n)[0]: keys_of(n)[1] for n in names}
    members = defaultdict(list)
    for n, k in zip(names, keys):
        members[k].append(n)

    def h(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    by_class = defaultdict(list)
    for k, c in cls_of_key.items():
        by_class[c].append(k)

    assign: dict[str, str] = {}
    for c in sorted(by_class):
        ks = sorted(by_class[c], key=h)
        n = len(ks)
        if n < 3:                       # 类太小：全部进 train，如实标记
            for k in ks:
                assign[k] = "train"
            continue
        n_test = max(1, round(n * ratios[2]))
        n_val = max(1, round(n * ratios[1]))
        for i, k in enumerate(ks):
            assign[k] = "train" if i < n - n_val - n_test else ("validation" if i < n - n_test else "test")

    splits = defaultdict(list)
    for k, ms in members.items():
        splits[assign[k]].extend(sorted(ms))
    src_splits = defaultdict(set)
    for k in members:
        src_splits[k].add(assign[k])
    assert all(len(v) == 1 for v in src_splits.values()), "存在跨集源"

    return {
        "splits": {k: splits.get(k, []) for k in ("train", "validation", "test")},
        "stats": {
            "slices": {k: len(splits.get(k, [])) for k in ("train", "validation", "test")},
            "sources": {k: len({keys_of(m)[0] for m in splits.get(k, [])})
                        for k in ("train", "validation", "test")},
            "classes": len(by_class),
            "slices_per_source": round(len(names) / max(len(members), 1), 2),
        },
    }


# ---------------------------------------------------------------- 入口
P1 = ["pianos", "music_genre", "chest_falsetto", "CNPM", "bel_canto",
      "erhu_playing_tech", "CTIS", "GZ_IsoTech", "Guzheng_Tech99"]


def index_path(dataset: str) -> str | None:
    hits = glob.glob(os.path.join(ROOT, "data", "index", f"{dataset}__*.json"))
    return hits[0] if hits else None


def selftest() -> int:
    names = []
    for c in ("1_Classic", "2_Non_classic"):
        for s in range(6):
            for f in range(5):
                names.append(f"eval/mel/{c}/{hashlib.md5(str(s).encode()).hexdigest()}_270.0_{f}.jpg")
    a, g = audit(names, seeds=5), regroup(names)
    print(json.dumps(a, ensure_ascii=False, indent=1))
    print(json.dumps(g["stats"], ensure_ascii=False, indent=1))
    assert a["leak_rate_mean"] > 0.95 and g["stats"]["sources"]["test"] >= 1
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CCMusic 泄漏审计 / 按源重划分")
    ap.add_argument("--dataset")
    ap.add_argument("--index")
    ap.add_argument("--zip", nargs="+")
    ap.add_argument("--project", choices=["p1"])
    ap.add_argument("--mode", choices=["audit", "regroup"], default="audit")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(d, index_path(d), None) for d in P1] if args.project else [(args.dataset, args.index, args.zip)]
    summary = []
    for dataset, idx, zips in jobs:
        if idx:
            names = names_from_index(idx)
        elif zips:
            names = [n for z in zips for n in names_from_zip(z)]
        else:
            print(f"[跳过] {dataset}: 没有 index 或 zip")
            continue
        res = audit(names, seeds=args.seeds) if args.mode == "audit" else regroup(names)
        res["dataset"] = dataset
        path = os.path.join(args.out, f"{dataset}_{args.mode}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        if args.mode == "audit":
            summary.append({"dataset": dataset, "slices": res["slices"], "sources": res["sources"],
                            "K": res["slices_per_source"], "mode": res.get("mode"),
                            "leak_rate": res.get("leak_rate_mean"), "n_eff": res["n_eff"]})
            print(f"{dataset:18s} 切片 {res['slices']:6d} 源 {res['sources']:5d} K={res['slices_per_source']:6.2f} "
                  f"泄漏率 {res.get('leak_rate_mean')} ({res.get('mode')})")
        else:
            print(f"{dataset:18s} 重划分 {res['stats']['slices']} 源 {res['stats']['sources']}")
    if summary:
        with open(os.path.join(args.out, "_audit_summary.csv"), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print("汇总:", os.path.join(args.out, "_audit_summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
