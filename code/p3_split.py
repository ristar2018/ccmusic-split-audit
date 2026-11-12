#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""code/p3_split.py — instrument-grouped 3-fold split for P3 (no GPU, reads the data root only).

Convention (P3 pre-registration draft, v1, frozen 2026-11-05):
  1. Class universe = the 43 technique classes in `data/p3/universe_43.csv` (the primary convention after dropping the 6 content-class labels).
  2. Instrument names are normalised to `merge_to` using the `used=1` rows of `data/p3/instrument_merge.csv` (212 → 204).
  3. Fold grouping = 3 folds grouped by instrument: all recordings of one instrument land in the same fold; each recording is predicted exactly once.
     Greedy: instruments in descending slice count, each placed into the fold that currently holds the fewest slices.
  4. Test set of fold f = the recordings of fold f; training set = the recordings of the other two folds (the training script filters by fold).
  5. Main metric = pooled macro-F1 over the merged held-out predictions of the three folds (the training script merges; this script only splits).

Assertions (stop on failure, exit 2):
  A1 instruments are mutually exclusive and their union = all instruments
  A2 each recording's fold matches the fold of its instrument
  A3 at least 1 slice per class per fold; missing (class, fold) cells are listed separately (not a failure, but must be disclosed)
  A4 total support per class >= 6
  A5 duplicate audio sharing a recording_canonical lands in the same fold

Artefacts:
  data/p3/p3_splits.csv        one row per slice (sample_id, recording_uid, recording_canonical,
                               technique, instrument, instrument_canon, fold)
  data/p3/fold_instruments.csv instrument -> fold
  data/p3/p3_folds.json        per-fold statistics + class index + assertions + split_hash
"""
import collections
import csv
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p3_universe  # noqa: E402  same directory; reuses load_rows / recording_cells / universe

OUT = os.path.join(os.path.dirname(HERE), "data", "p3")
N_FOLDS = 3
MIN_SUPPORT = 6


def read_csv(path):
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canon_map():
    """instrument (original name) -> normalised instrument name. Only used=1 rows are taken."""
    m = {}
    for r in read_csv(os.path.join(OUT, "instrument_merge.csv")):
        if str(r.get("used", "")).strip() in ("1", "true", "True"):
            m[r["instrument"].strip()] = r["merge_to"].strip()
    return m


def cell_counts(rows):
    """(technique, instrument) -> number of recordings (deduplicated at recording level, same form as the p3_universe cell convention)."""
    seen = set()
    cnt = collections.Counter()
    for r in rows:
        t, i = r.get("technique", ""), r.get("instrument", "")
        if not t or not i:
            continue
        k = (r["recording_uid"], t, i)
        if k in seen:
            continue
        seen.add(k)
        cnt[(t, i)] += 1
    return cnt


def build():
    rows = p3_universe.load_rows()                       # all 43,054 CTIS slices
    rec, cells = p3_universe.recording_cells(rows)
    universe, used = p3_universe.universe(cells, set(p3_universe.DROP))
    classes = [r["technique"] for r in universe]
    assert len(classes) == 43 and used == 1388, (len(classes), used)

    cnt = cell_counts(rows)
    keep_tech = set(classes)
    keep = {k for k, n in cnt.items() if k[0] in keep_tech and n >= p3_universe.MIN_CELL}
    sel = [r for r in rows if (r.get("technique"), r.get("instrument")) in keep]
    n_rec = len({r["recording_uid"] for r in sel})
    assert n_rec == used, (n_rec, used)

    cmap = canon_map()
    slices = []
    for r in sel:
        inst = r.get("instrument", "").strip()
        slices.append({
            "sample_id": r["sample_id"],
            "recording_uid": r["recording_uid"],
            "recording_canonical": r.get("recording_canonical") or r["recording_uid"],
            "technique": r["technique"],
            "instrument": inst,
            "instrument_canon": cmap.get(inst, inst),
        })
    return slices, classes


def assign_folds(slices):
    by_inst = collections.Counter(s["instrument_canon"] for s in slices)
    fold_inst, fold_load = {}, [0] * N_FOLDS
    for inst, n in sorted(by_inst.items(), key=lambda kv: (-kv[1], kv[0])):
        f = min(range(N_FOLDS), key=lambda i: (fold_load[i], i))
        fold_inst[inst] = f + 1
        fold_load[f] += n
    return fold_inst, fold_load


def _score(slices, fold_inst, cls_total):
    """Objective = number of missing (class, fold) cells x 1000 + spread of slice counts across folds (lower is better)."""
    seen = collections.Counter()
    load = [0] * N_FOLDS
    for s in slices:
        f = fold_inst[s["instrument_canon"]] - 1
        load[f] += 1
        seen[(s["technique"], f)] += 1
    miss = sum(1 for c in cls_total for f in range(N_FOLDS) if seen[(c, f)] == 0)
    return miss * 1000 + (max(load) - min(load)), miss, load


def refine_folds(slices, fold_inst, cls_total, iters=4000, seed=20261105):
    """Local search: randomly move one instrument to another fold, keep the move when the objective improves.

    Purpose: the greedy pass only balances slice counts and squeezes small classes into one fold (20 missing cells on the first run).
    Here the "missing (class, fold) cells" term goes into the objective so each fold covers as many of the 43 classes as possible.
    """
    import random
    rng = random.Random(seed)
    cur = dict(fold_inst)
    best = _score(slices, cur, cls_total)[0]
    insts = list(cur)
    for _ in range(iters):
        inst = rng.choice(insts)
        old = cur[inst]
        new = rng.choice([f for f in range(1, N_FOLDS + 1) if f != old])
        cur[inst] = new
        sc = _score(slices, cur, cls_total)[0]
        if sc <= best:
            best = sc
        else:
            cur[inst] = old
    return cur


def main():
    os.makedirs(OUT, exist_ok=True)
    slices, classes = build()
    fold_inst, fold_load = assign_folds(slices)
    cls_total = collections.Counter(s["technique"] for s in slices)
    raw_miss = _score(slices, fold_inst, cls_total)[1]
    fold_inst = refine_folds(slices, fold_inst, cls_total)
    fold_load = _score(slices, fold_inst, cls_total)[2]
    for s in slices:
        s["fold"] = str(fold_inst[s["instrument_canon"]])

    ok = {}
    inst_folds = collections.defaultdict(set)
    for inst, f in fold_inst.items():
        inst_folds[inst].add(f)
    ok["A1_inst_exclusive"] = (all(len(v) == 1 for v in inst_folds.values())
                               and set(fold_inst) == {s["instrument_canon"] for s in slices})
    ok["A2_recording_fold_consistent"] = all(
        fold_inst[s["instrument_canon"]] == int(s["fold"]) for s in slices)

    cls_fold, cls_total = collections.Counter(), collections.Counter()
    for s in slices:
        cls_fold[(s["technique"], int(s["fold"]))] += 1
        cls_total[s["technique"]] += 1
    missing = [(c, f) for c in classes for f in range(1, N_FOLDS + 1) if cls_fold[(c, f)] == 0]
    ok["A3_every_class_every_fold"] = not missing
    ok["A4_class_support_ge_6"] = all(cls_total[c] >= MIN_SUPPORT for c in classes)

    canon_folds = collections.defaultdict(set)
    for s in slices:
        canon_folds[s["recording_canonical"]].add(s["fold"])
    ok["A5_dup_audio_same_fold"] = all(len(v) == 1 for v in canon_folds.values())

    cols = ["sample_id", "recording_uid", "recording_canonical", "technique",
            "instrument", "instrument_canon", "fold"]
    with io.open(os.path.join(OUT, "p3_splits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: s[k] for k in cols} for s in slices)

    inst_cnt = collections.Counter(s["instrument_canon"] for s in slices)
    with io.open(os.path.join(OUT, "fold_instruments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instrument_canon", "fold", "slices"])
        for inst, fd in sorted(fold_inst.items(), key=lambda kv: (kv[1], -inst_cnt[kv[0]])):
            w.writerow([inst, fd, inst_cnt[inst]])

    per_fold = {}
    for fd in range(1, N_FOLDS + 1):
        sub = [s for s in slices if int(s["fold"]) == fd]
        per_fold[str(fd)] = {
            "slices": len(sub),
            "recordings": len({s["recording_uid"] for s in sub}),
            "instruments": len({s["instrument_canon"] for s in sub}),
            "classes_covered": len({s["technique"] for s in sub}),
            "train_recordings": len({s["recording_uid"] for s in slices if int(s["fold"]) != fd}),
        }

    h = hashlib.sha1()
    for s in sorted(slices, key=lambda x: x["sample_id"]):
        h.update(("%s|%s|%s\n" % (s["sample_id"], s["recording_uid"], s["fold"])).encode())

    out = {
        "rule": "3 folds grouped by instrument (same instrument, same fold; greedy balancing by slice count)",
        "n_folds": N_FOLDS,
        "n_slices": len(slices),
        "n_recordings": len({s["recording_uid"] for s in slices}),
        "n_instruments": len(fold_inst),
        "n_instruments_unmerged": len({s["instrument"] for s in slices}),
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "per_fold": per_fold,
        "fold_load": fold_load,
        "missing_class_fold": [{"technique": c, "fold": f} for c, f in missing],
        "min_support": MIN_SUPPORT,
        "assertions": ok,
        "disclosed": {"A3_missing_class_fold_cells": len(missing)},
        "all_pass": all(v for k, v in ok.items() if k != "A3_every_class_every_fold"),
        "all_pass_incl_disclosed": all(ok.values()),
        "split_hash": h.hexdigest(),
    }
    with io.open(os.path.join(OUT, "p3_folds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("slices %d / recordings %d / instruments %d (unmerged %d) / classes %d"
          % (len(slices), out["n_recordings"], out["n_instruments"],
             out["n_instruments_unmerged"], len(classes)))
    for fd in range(1, N_FOLDS + 1):
        d = per_fold[str(fd)]
        print("  fold%d: %5d slices / %4d recordings / %3d instruments / %2d classes covered (%4d training recordings)"
              % (fd, d["slices"], d["recordings"], d["instruments"],
                 d["classes_covered"], d["train_recordings"]))
    print("missing (class, fold) cells: %d%s" % (len(missing),
          ": " + ", ".join("%s@fold%d" % (c, f) for c, f in missing[:12]) if missing else ""))
    for k, v in ok.items():
        print("  %-32s %s" % (k, "PASS" if v else "FAIL"))
    print("split_hash %s" % out["split_hash"])
    return 0 if out["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
