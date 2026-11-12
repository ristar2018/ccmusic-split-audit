#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 pre-work (zero GPU): freeze the 43-class attribute universe + candidate instrument-name normalisation.

Design basis
  design note v2.8:635-636 — two data jobs come before writing code/p3_split.py:
    1) instrument-name normalisation (merge simplified/traditional and alias forms among the 212 instrument names, keep a merge table);
    2) freeze the 43-class universe after dropping the 6 content-class labels (re-run the ctis_capacity baseline).
   same file §7.2 — main reading = (cell ≥2 and covering ≥3 instruments) + drop the 6 content-class labels
     (音阶/空弦音/筒音/单音/乐曲片段/特有演奏技法) → 43 classes / 1,388 recordings (median 10).

Reading (aligned cell by cell with the sweep in code/ctis_capacity.py, cross-checked)
  - a cell = the number of **recordings** of one (technique, instrument) pair (de-duplicated by recording_uid; not the slice count —
    the slice count lifts small classes with a large K̄ over the threshold, giving 63 classes / 36,369 slices in practice, which contradicts the 49 classes of the design note).
  - inclusion rule: a technique is included when the number of instruments carrying at least MIN_CELL recordings under it is ≥ MIN_INSTR.
  - counting rule: only recordings inside cells that reach MIN_CELL are counted (`used` in ctis_capacity.py:40
    is `sum(cells[(t,i)] for t in uni for i in ok[t])`), recordings outside those cells are not counted.
    ← counting "every recording once the technique is included" gave 4,129 / 1,455 earlier, 92 / 67 more than the design note; that is exactly this difference.

Artifacts
  data/p3/universe_43.csv        the frozen 43-class list (recordings / instruments covered / cell median per class)
  data/p3/universe_before_drop.csv  the 49 classes before dropping (kept so the P3 pre-registration can cite the drop rule)
  data/p3/instrument_merge.csv   mechanical merge candidates for instrument names (need human confirmation, data is never changed automatically)
  data/p3/P3_universe.json       summary and assertions

Usage
  python code/p3_universe.py            # reads the manifest only, never touches the data root
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "data", "manifest", "CTIS.csv")
OUT = os.path.join(HERE, "data", "p3")

MIN_CELL = 2      # lower bound on recordings per (technique, instrument) cell
MIN_INSTR = 3     # lower bound on the number of instruments a technique covers
DROP = ["音阶", "空弦音", "筒音", "单音", "乐曲片段", "特有演奏技法"]   # content-class labels (not performance techniques)

# book values of design note §7.2, used for cross-checking
EXPECT = {"before": (49, 4037), "after": (43, 1388)}


def load_rows():
    with open(MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recording_cells(rows):
    """recording_uid -> (instrument, technique), first occurrence wins (both columns are constant within a recording)."""
    rec = {}
    for r in rows:
        if r["instrument"] and r["technique"]:
            rec.setdefault(r["recording_uid"], (r["instrument"], r["technique"]))
    cells = collections.Counter((t, i) for i, t in rec.values())
    return rec, cells


def universe(cells, drop):
    ok = collections.defaultdict(set)
    for (t, i), c in cells.items():
        if c >= MIN_CELL:
            ok[t].add(i)
    uni = [t for t in ok if len(ok[t]) >= MIN_INSTR and t not in drop]
    uni.sort(key=lambda t: (-sum(cells[(t, i)] for i in ok[t]), t))
    used = sum(cells[(t, i)] for t in uni for i in ok[t])
    rows = []
    for t in uni:
        insts = ok[t]
        n = sum(cells[(t, i)] for i in insts)
        per = [cells[(t, i)] for i in insts]
        rows.append({"technique": t, "recordings": n, "instruments": len(insts),
                     "cell_median": statistics.median(per), "cell_min": min(per),
                     "cell_max": max(per)})
    return rows, used


SUFFIX_KEY = re.compile(r"[\s\-_]*(?:[A-Ga-g]调|D调|F调|G调)\s*$")
PAREN = re.compile(r"[（(][^）)]*[）)]")


def norm_name(name):
    """Mechanical normalisation key: strip parentheticals, strip the key-name suffix, normalise ideographic spaces. Only for **finding candidates**, never for changing data."""
    s = name.replace("\u3000", " ").strip()
    s = PAREN.sub("", s)
    s = SUFFIX_KEY.sub("", s)
    return s.strip()


# Ruling on instrument-name normalisation (final, 2026-11-05): all 6 groups / 14 names merged, 212 -> 204.
# Reason: fold grouping exists to hold instruments out; if different builds, key names or spellings of one instrument
# land in two different groups, a fold pair can show same-timbre leakage such as "trained on 扬剧主胡 in D, tested on 扬剧主胡 in F",
# which makes the cross-instrument generalisation look more optimistic. Merging is the conservative reading; the unmerged version (212) is a sensitivity note only.
RATIONALE = {
    "扬剧主胡": "D key / F key / xiaoxipi-vocal writing of one instrument, same timbre source",
    "伽倻琴": "traditional and improved builds, same family and timbre",
    "奚琴": "traditional and improved builds, same family and timbre",
    "盖板": "traditional and new build (D key), same family and timbre",
    "短箫": "same instrument name with and without the traditional-build suffix",
    "陇剧陇胡": "traditional and improved builds (D key), same family and timbre",
}


def merge_candidates(rows):
    inst = collections.Counter(r["instrument"] for r in rows if r["instrument"])
    groups = collections.defaultdict(list)
    for name in inst:
        groups[norm_name(name)].append(name)
    out = []
    for key, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        names.sort()
        canon = min(names, key=lambda s: (len(s.strip()), s.strip())).strip()
        for nm in names:
            out.append({"norm_key": key, "instrument": nm.strip(), "recordings_slices": inst[nm],
                        "merge_to": canon, "auto_rule": "same name after stripping parentheticals and the key-name suffix",
                        "used": 1,
                        "decision": "merge: different builds / key names / spellings of one instrument",
                        "rationale": RATIONALE.get(key, "same family, merged as the conservative reading")})
    return out, inst


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows()
    rec, cells = recording_cells(rows)
    print("manifest %d slices / parseable recordings %d / cells %d" % (len(rows), len(rec), len(cells)))

    before, used_before = universe(cells, set())
    after, used_after = universe(cells, set(DROP))
    miss = [t for t in DROP if t not in [r["technique"] for r in before]]
    print("before dropping: %d classes / %d recordings (book %d / %d)" % (len(before), used_before, *EXPECT["before"]))
    print("after dropping: %d classes / %d recordings (book %d / %d) median %s min %d"
          % (len(after), used_after, *EXPECT["after"],
             statistics.median([r["recordings"] for r in after]),
             min(r["recordings"] for r in after)))
    print("of the 6 content-class labels, these meet the threshold: %s; none met: %s"
          % ([t for t in DROP if t not in miss], miss))
    print("Top10: " + ", ".join("%s %d" % (r["technique"], r["recordings"]) for r in after[:10]))

    for fn, data in (("universe_43.csv", after), ("universe_before_drop.csv", before)):
        with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    cand, inst = merge_candidates(rows)
    with open(os.path.join(OUT, "instrument_merge.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["norm_key", "instrument", "recordings_slices", "merge_to", "auto_rule", "used", "decision", "rationale"])
        w.writeheader()
        w.writerows(cand)
    print("instrument names %d -> mechanical merge candidates %d names / %d groups (removes %d names)"
          % (len(inst), len(cand), len({c["norm_key"] for c in cand}), len(cand) - len({c["norm_key"] for c in cand})))

    ok = {"before_classes": len(before) == EXPECT["before"][0],
          "before_recordings": used_before == EXPECT["before"][1],
          "after_classes": len(after) == EXPECT["after"][0],
          "after_recordings": used_after == EXPECT["after"][1]}
    summary = {"generated_by": "code/p3_universe.py",
               "design_ref": ["design note v2.8:635-636", "same file §7.2"],
               "rule": {"min_cell": MIN_CELL, "min_instr": MIN_INSTR, "unit": "recording_uid",
                        "drop_labels": DROP},
               "before": {"classes": len(before), "recordings": used_before},
               "after": {"classes": len(after), "recordings": used_after,
                         "median": statistics.median([r["recordings"] for r in after]),
                         "min": min(r["recordings"] for r in after)},
               "instrument_names": len(inst),
               "merge_candidates": len(cand),
               "checks": ok,
               "all_pass": all(ok.values())}
    with open(os.path.join(OUT, "P3_universe.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("assertions: %s -> %s" % (ok, "all pass" if summary["all_pass"] else "!! deviation"))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
