"""CCMusic leakage audit + re-split by source (group).

Two input modes:
    --index data/index/<dataset>__data_eval.zip.json   (recommended: run ccmusic_zipindex.py first, a few MB is enough)
    --zip  <local zip>                                 (when the full package is already downloaded)

Two modes:
    audit    replay the official "slice -> random.shuffle -> 8:1:1" procedure and estimate the same-source leakage rate of the test slices;
             if the paths already carry train/test directories (e.g. GZ_IsoTech), measure the true split directly.
    regroup  group by "source", stratify by class, and produce a deterministic 8:1:1 split (reproducible, no source crosses sets).

Usage:
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


# ---------------------------------------------------------------- parsing
def parts_of(member: str) -> list[str]:
    return [p for p in member.replace("\\", "/").split("/") if p]


def source_of(member: str) -> str:
    """Source ID: every slice of the same recording/segment must map to the same value."""
    stem = os.path.splitext(os.path.basename(member))[0]
    for f in FEATURES:                      # strip the feature suffix
        if stem.endswith("_" + f):
            stem = stem[: -(len(f) + 1)]
            break
    if "__" in stem:                        # erhu: detache__detache_01
        return stem.split("__", 1)[1]
    m = re.match(r"^([0-9a-f]{32})", stem)  # music_genre: md5 prefix
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
    if "__" in stem:                        # erhu puts the class name inside the file name
        return stem.split("__", 1)[0]
    return "unknown"


def split_in_path(member: str) -> str | None:
    for p in parts_of(member)[:-1]:
        if p.lower() in SPLIT_DIRS:
            return SPLIT_DIRS[p.lower()]
    return None


def keys_of(member: str) -> tuple[str, str, str | None]:
    """Returns (source_key, class, split_in_path). source_key carries the class prefix so identical numbers in different classes do not collide."""
    return f"{class_of(member)}::{source_of(member)}", class_of(member), split_in_path(member)


def names_from_index(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    names = data.get("names")
    if not names:
        raise SystemExit(f"{path} has no names; regenerate it with ccmusic_zipindex.py --full")
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
        # Case A: the zip already carries official train/test directories -> measure directly
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
        # Case B: the official code shuffles at load time (no seed) -> replay the process and average over seeds
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
    out["note"] = "leak_rate=share of test slices whose same-source siblings also appear in train; n_eff=number of independent sources"
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
        if n < 3:                       # class too small: everything goes to train, flagged as such
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
    assert all(len(v) == 1 for v in src_splits.values()), "a source spans more than one split"

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


# ---------------------------------------------------------------- entry point
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
    ap = argparse.ArgumentParser(description="CCMusic leakage audit / re-split by source")
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
            print(f"[skip] {dataset}: no index or zip")
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
            print(f"{dataset:18s} slices {res['slices']:6d} sources {res['sources']:5d} K={res['slices_per_source']:6.2f} "
                  f"leak rate {res.get('leak_rate_mean')} ({res.get('mode')})")
        else:
            print(f"{dataset:18s} re-split {res['stats']['slices']} sources {res['stats']['sources']}")
    if summary:
        with open(os.path.join(args.out, "_audit_summary.csv"), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print("summary:", os.path.join(args.out, "_audit_summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
