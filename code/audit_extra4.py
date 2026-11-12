# -*- coding: utf-8 -*-
"""E1a audit extension: for the four non-mainline libraries (instrument_timbre / timbre_range / acapella / song_structure)
rebuild the same-source leakage estimate for "source (recording) grouping -> slice-random 8:1:1" from **the files on disk**.

Same caliber as the audit in ccmusic_splits.py:
  - source-ID rules follow v2.1 §4.1 (spelled out here, one rule per library);
  - slice-random splitting is simulated 20 times (seed 0..19) and we count the share of "test slices whose
    same-source siblings are also in the training set";
  - one further run splits 8:1:1 by source group, where the leakage rate must be 0 (caliber self-check).

Reads the datasets, writes:
  data/splits/<dataset>_audit_disk.json
  data/splits/_audit_summary.csv   (the first 9 rows are kept, these 4 rows are replaced/appended)
  data/splits/_audit_summary.prev.csv (backup before the change)

Usage:
    python code/audit_extra4.py              # all four libraries
    python code/audit_extra4.py acapella     # a single library
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


# ---- slice inventory and source-ID rules per library (v2.1 §4.1) ----
def slices_instrument_timbre():
    """61 mel images, one per instrument; source = the instrument itself (the image path is the source), class = Chinese/Western."""
    rows = []
    for p in glob.glob(os.path.join(DATA, "instrument_timbre", "data", "mel", "*", "*.jpg")):
        rows.append((p, f"instrument::{_parent(p)}::{_stem(p)}", _parent(p)))
    return rows


def slices_timbre_range():
    """The two sub-tasks share instrument ids: range/*_n.jpg (source = the id before the underscore), timbre/{base,short,split}/*-n.jpg (source = the id before the hyphen)."""
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
    """132 mel images = 6 songs x 22 segments; source = song{1..6} (the track)."""
    rows = []
    for p in glob.glob(os.path.join(DATA, "acapella", "data", "mel", "*", "*.jpg")):
        rows.append((p, f"song::{_parent(p)}", "acapella"))
    return rows


def slices_song_structure():
    """300 mel images, one per song; source = the track name (file name without extension)."""
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
    """Slice-random 8:1:1: share of test slices whose same-source siblings sit in the training set."""
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
    """8:1:1 grouped by source (no source crosses splits): leakage must be 0; this is the caliber self-check."""
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
    # class availability: only in instrument_timbre is the directory name itself the binary label
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
    print(f"\n_audit_summary.csv -> {len(rows)} rows")


if __name__ == "__main__":
    main()
