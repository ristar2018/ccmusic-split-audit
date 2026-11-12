"""P3 third baseline (B1): recording-grouped folds with the instrument present on both sides.

External review item M7 noted that A0→A1 in P3 changes two things at once:
  1) homologous leakage (sibling slices of a test recording enter training) — present in A0, absent in A1;
  2) instrument novelty (the training side of A1 never contains the instrument used at test) — present only in A1.
So A0−A1 can only be read as an upper bound on the sum of the two. This script adds the third baseline A1r:

  * folds are grouped by **recording** (all slices of one recording share a fold, so homologous leakage is cut exactly as in A0);
  * each recording enters one fold only → the class distribution is stratified per fold so every class keeps ≥1 recording in each fold (where feasible);
  * an instrument appears **on both sides** (whenever it has ≥2 recordings, spread them over different folds where possible) → instrument novelty is no longer a variable.

Hence:
  A0 (slice-random folds)      vs A1r (recording-grouped folds) → changes homologous leakage only
  A1r (recording-grouped folds) vs A1 (instrument-grouped folds) → changes instrument novelty only

Leakage assertions (any failure exits 2, nothing is written):
  L1 every recording sits in exactly one fold, no slice crosses folds
  L2 all recordings sharing one audio_hash (recording_canonical) sit in the same fold
  L3 recordings per (class, fold) cell: classes with ≥3 recordings must have ≥1 in all three folds
  L4 instrument coverage: instruments with ≥3 recordings must appear in at least 2 folds (actual ratio and threshold are reported)
  L5 spread of slice counts across the three folds ≤ 15% (actual value is reported)

Artifacts (data/p3/):
  p3b_splits.csv           sample_id, recording_uid, recording_canonical, technique,
                           instrument, instrument_canon, fold_b1
  p3b_fold_summary.csv     per fold: slices / recordings / classes / instruments
  p3b_instrument_folds.csv per instrument: recordings / slices / folds present / on both sides
  p3b_folds.json           assertions + split_hash + parameters

Usage:
  python code/p3_split_b1.py            # write artifacts, exit 0 means every assertion passed
  python code/p3_split_b1.py --check    # run the assertions and statistics only, write nothing
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import random
import sys

import p3_split
import p3_universe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "p3")
N_FOLDS = 3
SEED = 20261105
MIN_INSTR_RECS = 3      # threshold for "instrument on both sides" (required only when it has ≥3 recordings)
MIN_CLASS_RECS = 3      # threshold for "≥1 recording per fold" per class
MAX_LOAD_SPREAD = 0.15  # upper bound on the spread of slice counts across folds (share of the largest fold)


def build_slices():
    slices, classes = p3_split.build()
    return slices, classes


def group_recordings(slices):
    """recording_canonical -> the slices and recordings of that group. Several published names of one audio enter the fold together."""
    by_grp = collections.defaultdict(list)
    for s in slices:
        by_grp[s["recording_canonical"]].append(s)
    return by_grp


def assign_folds(slices, classes, seed=SEED):
    """Recording-level three-fold stratified by class + load balance across folds (recordings are moved within a class only)."""
    rng = random.Random(seed)
    grp = group_recordings(slices)
    # per canon group: main class (the class holding most slices, ties broken lexicographically), slice count, recording set
    groups = {}
    for g, rows in grp.items():
        tech = collections.Counter(r["technique"] for r in rows)
        main_cls = sorted(tech.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        groups[g] = {
            "rows": rows,
            "cls": main_cls,
            "n": len(rows),
            "recs": {r["recording_uid"] for r in rows},
            "insts": {r["instrument_canon"] for r in rows},
        }

    by_cls = collections.defaultdict(list)
    for g, meta in groups.items():
        by_cls[meta["cls"]].append(g)

    fold_of = {}
    load = [0] * N_FOLDS
    for cls in sorted(by_cls):
        gs = sorted(by_cls[cls], key=lambda g: (-groups[g]["n"], g))
        start = rng.randrange(N_FOLDS)           # per-class rotation start, avoids a systematic fold bias
        for i, g in enumerate(gs):
            f = (start + i) % N_FOLDS
            fold_of[g] = f + 1
            load[f] += groups[g]["n"]

    # local search: swap folds inside one class only, objective = load spread (keeps the class stratification)
    def spread(l):
        return max(l) - min(l)

    best = spread(load)
    for _ in range(4000):
        cls = rng.choice(sorted(by_cls))
        gs = by_cls[cls]
        a, b = rng.choice(gs), rng.choice(gs)
        fa, fb = fold_of[a], fold_of[b]
        if fa == fb:
            continue
        na, nb = groups[a]["n"], groups[b]["n"]
        trial = list(load)
        trial[fa - 1] += nb - na
        trial[fb - 1] += na - nb
        if spread(trial) < best:
            fold_of[a], fold_of[b] = fb, fa
            load = trial
            best = spread(load)
    return fold_of, groups, load


def assert_and_summarise(slices, groups, fold_of, classes):
    fold = {}
    for g, meta in groups.items():
        for r in meta["rows"]:
            fold[r["sample_id"]] = fold_of[g]

    # L1: a recording must not cross folds
    rec_folds = collections.defaultdict(set)
    for s in slices:
        rec_folds[s["recording_uid"]].add(fold[s["sample_id"]])
    l1_bad = {k: sorted(v) for k, v in rec_folds.items() if len(v) > 1}

    # L2: all recordings with one recording_canonical share a fold
    canon_folds = collections.defaultdict(set)
    for s in slices:
        canon_folds[s["recording_canonical"]].add(fold[s["sample_id"]])
    l2_bad = {k: sorted(v) for k, v in canon_folds.items() if len(v) > 1}

    # L3: class × fold support
    cls_recs = collections.defaultdict(lambda: collections.defaultdict(set))
    for s in slices:
        cls_recs[s["technique"]][fold[s["sample_id"]]].add(s["recording_uid"])
    cls_total = {c: len({s["recording_uid"] for s in slices if s["technique"] == c}) for c in classes}
    l3_missing = []
    for c in classes:
        for f in range(1, N_FOLDS + 1):
            if len(cls_recs[c][f]) == 0:
                l3_missing.append({"class": c, "fold": f, "class_recordings": cls_total[c]})
    l3_hard = [d for d in l3_missing if d["class_recordings"] >= MIN_CLASS_RECS]

    # L4: instrument coverage
    inst_recs = collections.defaultdict(set)
    inst_folds = collections.defaultdict(set)
    inst_slices = collections.Counter()
    for s in slices:
        inst_recs[s["instrument_canon"]].add(s["recording_uid"])
        inst_folds[s["instrument_canon"]].add(fold[s["sample_id"]])
        inst_slices[s["instrument_canon"]] += 1
    multi = [i for i in inst_recs if len(inst_recs[i]) >= MIN_INSTR_RECS]
    both = [i for i in multi if len(inst_folds[i]) >= 2]
    inst_rows = [
        {
            "instrument_canon": i,
            "recordings": len(inst_recs[i]),
            "slices": inst_slices[i],
            "folds": "".join(str(f) for f in sorted(inst_folds[i])),
            "in_two_folds": int(len(inst_folds[i]) >= 2),
        }
        for i in sorted(inst_recs, key=lambda x: (-inst_slices[x], x))
    ]
    l4_ratio = len(both) / len(multi) if multi else 1.0
    one_fold_multi = [i for i in multi if len(inst_folds[i]) == 1]

    # L5: fold load
    load = collections.Counter(fold[s["sample_id"]] for s in slices)
    loads = [load[f] for f in range(1, N_FOLDS + 1)]
    spread_ratio = (max(loads) - min(loads)) / max(loads)

    ok = {
        "L1_recording_not_split": not l1_bad,
        "L2_audio_duplicate_same_fold": not l2_bad,
        "L3_class_support_all_folds": not l3_hard,
        "L4_instrument_in_two_folds": l4_ratio >= 0.80,
        "L5_load_spread_le_15pct": spread_ratio <= MAX_LOAD_SPREAD,
    }
    detail = {
        "n_slices": len(slices),
        "n_recordings": len({s["recording_uid"] for s in slices}),
        "n_canon_groups": len(groups),
        "n_classes": len(classes),
        "n_instruments": len(inst_recs),
        "fold_load": {str(f): loads[f - 1] for f in range(1, N_FOLDS + 1)},
        "load_spread_ratio": round(spread_ratio, 4),
        "L1_violations": len(l1_bad),
        "L2_violations": len(l2_bad),
        "L3_missing_cells": len(l3_missing),
        "L3_hard_violations": len(l3_hard),
        "L3_missing_detail": l3_missing,
        "instruments_multi": len(multi),
        "instruments_in_two_folds": len(both),
        "L4_ratio": round(l4_ratio, 4),
        "instruments_multi_single_fold": one_fold_multi,
    }
    return fold, ok, detail, inst_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the assertions and statistics only, write nothing")
    a = ap.parse_args()

    slices, classes = build_slices()
    fold_of, groups, _ = assign_folds(slices, classes)
    fold, ok, detail, inst_rows = assert_and_summarise(slices, groups, fold_of, classes)

    print("== B1 (recording-grouped folds + instrument on both sides)")
    print("   slices %d / recordings %d / audio groups %d / classes %d / instruments %d"
          % (detail["n_slices"], detail["n_recordings"], detail["n_canon_groups"],
             detail["n_classes"], detail["n_instruments"]))
    print("   fold load %s (spread %.1f%%)"
          % (detail["fold_load"], 100 * detail["load_spread_ratio"]))
    print("   L1 recordings crossing folds %d; L2 audio groups crossing folds %d; L3 missing cells %d (hard failures %d)"
          % (detail["L1_violations"], detail["L2_violations"],
             detail["L3_missing_cells"], detail["L3_hard_violations"]))
    print("   L4 instruments with ≥%d recordings: %d, of which %d land in ≥2 folds (%.1f%%)"
          % (MIN_INSTR_RECS, detail["instruments_multi"],
             detail["instruments_in_two_folds"], 100 * detail["L4_ratio"]))
    if detail["instruments_multi_single_fold"]:
        print("   multi-recording instruments confined to a single fold: %s" % detail["instruments_multi_single_fold"][:12])
    for k, v in ok.items():
        print("   [%s] %s" % ("PASS" if v else "FAIL", k))

    all_pass = all(ok.values())
    if not all_pass:
        print("!! not all assertions passed, nothing written (exit 2)")
        for k, v in ok.items():
            if not v:
                print("   - %s" % k)
        return 2
    if a.check:
        print("--check: nothing written")
        return 0

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in slices:
        rows.append({
            "sample_id": s["sample_id"],
            "recording_uid": s["recording_uid"],
            "recording_canonical": s["recording_canonical"],
            "technique": s["technique"],
            "instrument": s["instrument"],
            "instrument_canon": s["instrument_canon"],
            "fold_b1": fold[s["sample_id"]],
        })
    rows.sort(key=lambda r: (int(r["fold_b1"]), r["technique"], r["recording_uid"], r["sample_id"]))
    p1 = os.path.join(OUT, "p3b_splits.csv")
    with io.open(p1, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for fr in range(1, N_FOLDS + 1):
        sel = [r for r in rows if r["fold_b1"] == fr]
        summary.append({
            "fold": fr,
            "slices": len(sel),
            "recordings": len({r["recording_uid"] for r in sel}),
            "classes": len({r["technique"] for r in sel}),
            "instruments": len({r["instrument_canon"] for r in sel}),
            "train_recordings": len({r["recording_uid"] for r in rows if r["fold_b1"] != fr}),
        })
    p2 = os.path.join(OUT, "p3b_fold_summary.csv")
    with io.open(p2, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    p3 = os.path.join(OUT, "p3b_instrument_folds.csv")
    with io.open(p3, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inst_rows[0].keys()))
        w.writeheader()
        w.writerows(inst_rows)

    h = hashlib.sha1()
    for r in rows:
        h.update(("%s|%s\n" % (r["sample_id"], r["fold_b1"])).encode("utf-8"))
    meta = {
        "scheme": "B1: recording-grouped folds with instruments on both sides",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "min_instr_recordings": MIN_INSTR_RECS,
        "min_class_recordings": MIN_CLASS_RECS,
        "max_load_spread": MAX_LOAD_SPREAD,
        "split_hash": h.hexdigest(),
        "detail": detail,
        "assertions": ok,
        "all_pass": all_pass,
        "per_fold": summary,
        "rationale": (
            "A0 vs A1r isolates homologous leakage (only the fold unit changes); "
            "A1r vs A1 isolates instrument novelty (only the grouping key changes). "
            "Recording-level folds keep sibling slices on one side, exactly as A0/A1 do."
        ),
    }
    p4 = os.path.join(OUT, "p3b_folds.json")
    with io.open(p4, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("artifacts: %s / %s / %s / %s" % (p1, p2, p3, p4))
    print("split_hash %s" % meta["split_hash"])
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
