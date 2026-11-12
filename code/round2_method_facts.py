#!/usr/bin/env python3
"""round2_method_facts.py -- read-only accounting that supplies recomputable numbers for method_disclosure.md.

Covers four kinds of fact (all from artifacts already on disk; no training or bootstrap is re-run):
  A) per library, the audio/recording/slice counts available to the identity check (data/manifest/*.json + datasets_inventory.json)
  B) how the two arms are allocated: qtest/qpool recording overlap, identical hashes across sides, group sizes,
     identical hashes across classes (data/splits/v2/<lib>_{qtest,qpool,r,c}.csv)
  C) class sets and classes missing from the bootstrap: whether n_classes in e1_metrics.csv matches the classes present in Q_test;
     recordings per class -> the probability that a class is entirely absent from one recording-cluster resample
     (closed form, no sampling)
  D) the scale of the number-verification script (data/results/paper_verify.json)

Read-only: writes nothing under data/. Prints by default; a JSON file is written to
data/results/human_review/round2/ only when --json gives a path (that directory is writable).

Usage (subprocess is blocked by the sandbox under fapn python on Windows, so this script avoids it):
    python code/round2_method_facts.py --root .
    python3 code/round2_method_facts.py --root .   # also works under WSL
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MAIN_LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]
SIDES = ["qtest", "qpool", "r", "c"]


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def side_facts(root, lib):
    """B: recording overlap across the two sides, identical hashes across sides, group sizes, groups crossing recordings/classes."""
    rows = {s: load_csv(root / "data" / "splits" / "v2" / ("%s_%s.csv" % (lib, s)))
            for s in SIDES}
    rec = {s: {r["recording_canonical"] for r in rows[s]} for s in SIDES}

    qtest_pool_overlap = sorted(rec["qtest"] & rec["qpool"])
    overlap_slices = {"qtest": 0, "qpool": 0}
    for s in ("qtest", "qpool"):
        for r in rows[s]:
            if r["recording_canonical"] in (rec["qtest"] & rec["qpool"]):
                overlap_slices[s] += 1

    leak = sorted(rec["qtest"] & (rec["r"] | rec["c"]))

    hash_sides = defaultdict(set)
    for s in SIDES:
        for r in rows[s]:
            if r["content_hash"]:
                hash_sides[r["content_hash"]].add(s)
    hash_multi = {h: sorted(v) for h, v in hash_sides.items() if len(v) > 1}

    # rebuild the groups as in splits_v2.py:258-288: inside Q each recording is ordered by slice_index,
    # adjacent pairs (an even position and the next slice) are unioned; then identical content_hash within one label is unioned.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    q_rows = []
    for s in ("qtest", "qpool"):
        for r in rows[s]:
            r = dict(r)
            r["_side"] = s
            q_rows.append(r)

    by_rec = defaultdict(list)
    for r in q_rows:
        by_rec[(r["label"], r["recording_canonical"])].append(r)
    pair_split = 0
    for rs in by_rec.values():
        rs.sort(key=lambda r: int(r["slice_index"]))
        for i, r in enumerate(rs):
            find(r["sample_id"])
            if i % 2 == 1:
                union(rs[i - 1]["sample_id"], r["sample_id"])
                if rs[i - 1]["_side"] != r["_side"]:
                    pair_split += 1
    by_hash = defaultdict(list)
    for r in q_rows:
        if r["content_hash"]:
            by_hash[(r["label"], r["content_hash"])].append(r["sample_id"])
    for ids in by_hash.values():
        for x in ids[1:]:
            union(ids[0], x)

    comp = defaultdict(list)
    for r in q_rows:
        comp[find(r["sample_id"])].append(r)
    sizes = Counter(len(v) for v in comp.values())
    cross_rec = sum(1 for v in comp.values()
                    if len({r["recording_canonical"] for r in v}) > 1)
    cross_lab = sum(1 for v in comp.values() if len({r["label"] for r in v}) > 1)
    side_split = sum(1 for v in comp.values() if len({r["_side"] for r in v}) > 1)

    # total slices per recording inside Q (qtest+qpool is every Q slice of that recording)
    per_rec_slices = Counter(r["recording_canonical"]
                             for s in ("qtest", "qpool") for r in rows[s])
    singleton = {k for k, v in per_rec_slices.items() if v == 1}

    return {
        "slices": {s: len(rows[s]) for s in SIDES},
        "recordings": {s: len(rec[s]) for s in SIDES},
        "classes": {s: len({r["label"] for r in rows[s]}) for s in SIDES},
        "qtest_qpool_recording_overlap": len(qtest_pool_overlap),
        "qtest_qpool_overlap_slices": overlap_slices,
        "qtest_r_c_recording_overlap": len(leak),
        "hash_multi_side": len(hash_multi),
        "groups": len(comp),
        "group_size_hist": dict(sorted(sizes.items())),
        "pairs_split_across_sides": pair_split,
        "groups_split_across_sides": side_split,
        "groups_cross_recording": cross_rec,
        "groups_cross_class": cross_lab,
        "singleton_recordings": len(singleton),
        "singleton_all_in_qtest": all(any(r["recording_canonical"] == k for r in rows["qtest"])
                                      for k in singleton),
    }


def class_facts(root, lib, qtest_rows, n_classes_csv):
    """C: the class set of Q_test, recordings per class, probability that a class is missing from a resample."""
    labels = [r["label"] for r in qtest_rows]
    uniq = sorted(set(labels))
    per_cls = Counter()
    seen = defaultdict(set)
    for r in qtest_rows:
        seen[r["label"]].add(r["recording_canonical"])
    for lab, recs in seen.items():
        per_cls[lab] = len(recs)
    n_rec = len({r["recording_canonical"] for r in qtest_rows})
    p_miss = {lab: (1.0 - n / n_rec) ** n_rec for lab, n in per_cls.items()}
    exp_missing = sum(p_miss.values())
    p_any = 1.0 - math.prod(1.0 - p for p in p_miss.values())
    return {
        "n_classes_in_qtest": len(uniq),
        "n_classes_in_e1_metrics": n_classes_csv,
        "classes_match": (n_classes_csv is None or n_classes_csv == len(uniq)),
        "n_recordings": n_rec,
        "min_recs_per_class": min(per_cls.values()) if per_cls else 0,
        "max_recs_per_class": max(per_cls.values()) if per_cls else 0,
        "classes_with_lt5_recordings": sum(1 for n in per_cls.values() if n < 5),
        "expected_classes_missing_per_resample": round(exp_missing, 3),
        "p_at_least_one_class_missing": round(p_any, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--json", default="", help="optional: path for the result JSON (must be inside the round2 directory)")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    summary = load_json(root / "data" / "manifest" / "summary.json")["libraries"]
    pairs = {}
    for name in ("pairs_summary.json", "ctis_pairs.json"):
        p = root / "data" / "manifest" / name
        if p.exists():
            pairs = load_json(p)
            break
    inv = load_json(root / "data" / "datasets_inventory.json")
    inv_by = {d["dataset"]: d for d in inv["datasets"]}

    out = {"root": str(root), "A": {}, "B": {}, "C": {}, "D": {}}

    # ---------- A
    slices_sum = 0
    for lib in MAIN_LIBS:
        st = summary[lib]
        slices_sum += st["slices"]
        out["A"][lib] = {
            "classes": st.get("classes"),
            "recordings": st.get("recordings_in_manifest") or st.get("recordings"),
            "slices": st["slices"],
            "audio_files_on_disk": inv_by.get(lib, {}).get("audio"),
            "audio_hash_done": st.get("audio_hash_done"),
            "duplicate_audio_recordings": st.get("duplicate_audio_recordings", 0),
        }
    out["A"]["_slices_total"] = slices_sum
    out["A"]["_pairs"] = pairs.get("pairs", []) if isinstance(pairs, dict) else pairs

    # ---------- B / C
    metrics = load_csv(root / "data" / "results" / "e1_metrics.csv")
    ncls = defaultdict(set)
    for r in metrics:
        ncls[r["library"]].add(int(r["n_classes"]))

    for lib in MAIN_LIBS:
        out["B"][lib] = side_facts(root, lib)
        qtest = load_csv(root / "data" / "splits" / "v2" / ("%s_qtest.csv" % lib))
        nc = sorted(ncls[lib])[0] if ncls.get(lib) else None
        out["C"][lib] = class_facts(root, lib, qtest, nc)
        out["C"][lib]["n_classes_consistency"] = sorted(ncls.get(lib, []))

    # ---------- D
    pv = load_json(root / "data" / "results" / "paper_verify.json")
    out["D"] = {"passes": pv.get("passes"), "fails": pv.get("fails"),
                "panels": pv.get("panels"), "totals": pv.get("totals")}

    # ---------- printing
    print("== A surface available to the identity check ==")
    print("lib | classes | recordings | slices | audio_on_disk | audio_hash_done | dup_audio_recs")
    for lib in MAIN_LIBS:
        d = out["A"][lib]
        print("%s | %s | %s | %s | %s | %s | %s" % (
            lib, d["classes"], d["recordings"], d["slices"],
            d["audio_files_on_disk"], d["audio_hash_done"], d["duplicate_audio_recordings"]))
    print("slices_total = %d" % slices_sum)

    print("\n== B allocation of the two arms (qtest/qpool/r/c) ==")
    for lib in MAIN_LIBS:
        b = out["B"][lib]
        print("[%s] slices=%s recs=%s classes=%s" % (lib, b["slices"], b["recordings"], b["classes"]))
        print("    qtest∩qpool recordings=%d (qtest slices %d / qpool slices %d); qtest∩(r∪c) recordings=%d" % (
            b["qtest_qpool_recording_overlap"], b["qtest_qpool_overlap_slices"]["qtest"],
            b["qtest_qpool_overlap_slices"]["qpool"], b["qtest_r_c_recording_overlap"]))
        print("    same hash across sides=%d; groups=%d size hist=%s; groups crossing recordings=%d; groups crossing classes=%d; groups split across sides=%d; adjacent pairs split=%d; singleton recordings=%d all in qtest=%s" % (
            b["hash_multi_side"], b["groups"], b["group_size_hist"],
            b["groups_cross_recording"], b["groups_cross_class"],
            b["groups_split_across_sides"], b["pairs_split_across_sides"],
            b["singleton_recordings"], b["singleton_all_in_qtest"]))

    print("\n== C class sets and classes missing from a resample ==")
    for lib in MAIN_LIBS:
        c = out["C"][lib]
        print("[%s] n_classes(qtest)=%d n_classes(e1_metrics)=%s match=%s | recs=%d per class %d-%d recordings | classes with <5 recordings=%d | E[classes missing per resample]=%.3f P(>=1 class missing)=%.4f" % (
            lib, c["n_classes_in_qtest"], c["n_classes_in_e1_metrics"], c["classes_match"],
            c["n_recordings"], c["min_recs_per_class"], c["max_recs_per_class"],
            c["classes_with_lt5_recordings"], c["expected_classes_missing_per_resample"],
            c["p_at_least_one_class_missing"]))

    print("\n== D verification script ==")
    print("paper_verify.json: passes=%s fails=%s" % (out["D"]["passes"], out["D"]["fails"]))
    print("panels totals=%s" % json.dumps(out["D"]["totals"], ensure_ascii=False))

    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n→ %s" % p)


if __name__ == "__main__":
    main()
