"""C1: injection control with unrelated recordings (the attribution gap raised by external review M1).

Problem (M1): going from A0 to A1 changes three things at once — exposure (sibling slices of the test recordings), recording diversity,
and the slice distribution, so Δ1 cannot be attributed to same-source leakage alone. Splitting them apart needs an injection arm with
**the same volume and the same label distribution, but content unrelated to the test recordings**:

    A0 : m_c slices, all from C (no Q_pool)
    C1 : drop |Q_pool_c| slices per class from the A0 training set, then add back the same number per class —
         the added slices come only from **non-test recordings** (the C remainder + the R partition)
    A1 : drop |Q_pool_c| slices per class from the A0 training set, then add Q_pool (sibling slices of the test recordings)

So C1 and A1 differ only in where the injected content comes from (non-test vs test recordings), with identical volume and class balance;
C1 and A0 differ only by "same-class non-test content swapped in", expected ≈0 — an internal negative control.

Leakage assertions (any failure exits 2, nothing is written):
  K1 injected slices share no sample_id with Q_test
  K2 injected slices share no recording_uid with Q_test (genuinely non-test recordings)
  K3 injected slices share no content_hash with Q_test
  K4 injected slices are disjoint from the A0 training set; sample_ids inside the C1 training set are unique
  K5 per-class training counts equal A0 class by class (volume and class balance both preserved)
  K6 total training count within 0.5% of A1

Known cost (recorded here so nobody misreads it later): when the injection source is mostly the R partition, the R-side validation of the C1 arm is contaminated,
so that arm reports the sealed Q_test only; no arm performs any model selection (fixed 10 epochs, last-epoch weights).

Usage:
  python code/c1_inject.py            # writes data/splits/v2/CTIS_c1_train.csv + c1_audit.json
  python code/c1_inject.py --check    # run the assertions and statistics only
"""

import argparse
import collections
import csv
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPLITS = os.path.join(PROJ, "data", "splits", "v2")
LIB = "CTIS"
SEED = 20261105
MAX_TOTAL_GAP = 0.005   # upper bound on the total-volume gap against A1


def read_csv(path):
    with io.open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load():
    d = {}
    for name in ("a0_train", "a1_train", "qpool", "qtest", "c", "r"):
        p = os.path.join(SPLITS, "%s_%s.csv" % (LIB, name))
        d[name] = read_csv(p)
    return d


def build(d, seed=SEED):
    rng = random.Random(seed)
    a0 = d["a0_train"]
    qpool = d["qpool"]
    qtest = d["qtest"]
    test_sids = {r["sample_id"] for r in qtest}
    test_recs = {r["recording_uid"] for r in qtest}
    test_hashes = {r["content_hash"] for r in qtest if r.get("content_hash")}

    # slices to inject per class = per-class counts in Q_pool
    need = collections.Counter(r["label"] for r in qpool)

    # candidate injection pool: slices in C and R that are not in the A0 training set and whose recording is not a test recording
    a0_sids = {r["sample_id"] for r in a0}
    cand = []
    for name in ("c", "r"):
        for r in d[name]:
            if r["sample_id"] in a0_sids:
                continue
            if r["recording_uid"] in test_recs:
                continue
            if r.get("content_hash") and r["content_hash"] in test_hashes:
                continue
            cand.append(dict(r, _src=name))
    by_cls = collections.defaultdict(list)
    for r in cand:
        by_cls[r["label"]].append(r)

    # round-robin over recordings per class, so the injected slices do not concentrate in a few recordings
    injected, shortfall = [], {}
    for cls, k in sorted(need.items()):
        pool = by_cls.get(cls, [])
        if not pool:
            shortfall[cls] = k
            continue
        by_rec = collections.defaultdict(list)
        for r in pool:
            by_rec[r["recording_uid"]].append(r)
        recs = sorted(by_rec)
        rng.shuffle(recs)
        for rec in recs:
            rng.shuffle(by_rec[rec])
        take, i = [], 0
        while len(take) < k and recs:
            progressed = False
            for rec in list(recs):
                if len(take) >= k:
                    break
                if by_rec[rec]:
                    take.append(by_rec[rec].pop())
                    progressed = True
                if not by_rec[rec]:
                    recs.remove(rec)
            if not progressed:
                break
        injected.extend(take)
        if len(take) < k:
            shortfall[cls] = k - len(take)

    # drop the same number per class from A0: prefer recordings with many slices, avoid emptying a recording
    inj_cnt = collections.Counter(r["label"] for r in injected)
    a0_by_cls = collections.defaultdict(list)
    for r in a0:
        a0_by_cls[r["label"]].append(r)
    drop = []
    for cls, k in sorted(inj_cnt.items()):
        rows = sorted(a0_by_cls.get(cls, []), key=lambda r: r["sample_id"])
        by_rec = collections.defaultdict(list)
        for r in rows:
            by_rec[r["recording_uid"]].append(r)
        recs = sorted(by_rec, key=lambda x: (-len(by_rec[x]), x))
        taken = 0
        # one slice per recording per round, so each recording keeps at least 1 slice
        while taken < k:
            progressed = False
            for rec in recs:
                if taken >= k:
                    break
                if len(by_rec[rec]) > 1:
                    drop.append(by_rec[rec].pop())
                    taken += 1
                    progressed = True
            if not progressed:
                break
        if taken < k:      # fallback: emptying a recording is allowed
            for rec in recs:
                while by_rec[rec] and taken < k:
                    drop.append(by_rec[rec].pop())
                    taken += 1
                if taken >= k:
                    break

    drop_sids = {r["sample_id"] for r in drop}
    c1 = [r for r in a0 if r["sample_id"] not in drop_sids] + injected
    # output uses the same columns as the other splits
    cols = ["sample_id", "label", "recording_uid", "recording_canonical",
            "content_hash", "candidate_group", "slice_index"]
    rows = [{c: r.get(c, "") for c in cols} for r in c1]
    rows.sort(key=lambda r: (r["label"], r["recording_uid"], r["sample_id"]))
    return rows, injected, drop, need, shortfall


def audit(rows, injected, drop, need, d):
    test_sids = {r["sample_id"] for r in d["qtest"]}
    test_recs = {r["recording_uid"] for r in d["qtest"]}
    test_hashes = {r["content_hash"] for r in d["qtest"] if r.get("content_hash")}
    inj_sids = {r["sample_id"] for r in injected}
    inj_recs = {r["recording_uid"] for r in injected}
    inj_hashes = {r["content_hash"] for r in injected if r.get("content_hash")}
    a0_sids = {r["sample_id"] for r in d["a0_train"]}

    per_cls_c1 = collections.Counter(r["label"] for r in rows)
    per_cls_a0 = collections.Counter(r["label"] for r in d["a0_train"])
    cls_equal = all(per_cls_c1[c] == per_cls_a0[c] for c in per_cls_a0)
    n_a1 = len(d["a1_train"])
    gap = abs(len(rows) - n_a1) / n_a1

    ok = {
        "K1_no_test_sample_ids": not (inj_sids & test_sids),
        "K2_no_test_recordings": not (inj_recs & test_recs),
        "K3_no_test_content_hash": not (inj_hashes & test_hashes),
        "K4_injection_disjoint_from_a0": not (inj_sids & a0_sids),
        "K4b_train_ids_unique": len({r["sample_id"] for r in rows}) == len(rows),
        "K5_per_class_equal_to_a0": cls_equal,
        "K6_total_within_0.5pct_of_a1": gap <= MAX_TOTAL_GAP,
    }
    detail = {
        "n_train": len(rows),
        "n_a0_train": len(d["a0_train"]),
        "n_a1_train": n_a1,
        "gap_vs_a1": round(gap, 5),
        "n_injected": len(injected),
        "n_dropped": len(drop),
        "injected_from": dict(collections.Counter(r.get("_src", "?") for r in injected)),
        "injected_recordings": len(inj_recs),
        "injected_classes": len({r["label"] for r in injected}),
        "class_need_total": sum(need.values()),
        "class_counts_match": cls_equal,
        "n_classes": len(per_cls_a0),
    }
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = load()
    rows, injected, drop, need, shortfall = build(d)
    ok, detail = audit(rows, injected, drop, need, d)

    print("== C1 (injection control with unrelated recordings)")
    print("   A0 %d slices -> C1 %d slices (injected %d / dropped %d), A1 %d slices"
          % (detail["n_a0_train"], detail["n_train"], detail["n_injected"],
             detail["n_dropped"], detail["n_a1_train"]))
    print("   injection source %s, from %d non-test recordings, %d classes"
          % (detail["injected_from"], detail["injected_recordings"], detail["injected_classes"]))
    if shortfall:
        print("   !! %d classes under-filled (%d slices short in total)" % (len(shortfall), sum(shortfall.values())))
    for k, v in ok.items():
        print("   [%s] %s" % ("PASS" if v else "FAIL", k))
    all_pass = all(ok.values())
    if not all_pass:
        print("!! not all assertions passed, nothing written (exit 2)")
        return 2
    if a.check:
        print("--check: nothing written")
        return 0

    p = os.path.join(SPLITS, "%s_c1_train.csv" % LIB)
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label", "recording_uid",
                                          "recording_canonical", "content_hash",
                                          "candidate_group", "slice_index"])
        w.writeheader()
        w.writerows(rows)
    meta = {
        "lib": LIB,
        "arm": "C1",
        "scheme": "equal-volume injection from non-test recordings",
        "seed": SEED,
        "assertions": ok,
        "all_pass": all_pass,
        "detail": detail,
        "class_shortfall": shortfall,
        "note": ("Injected slices come from C remainder and R; for this arm only the "
                 "R partition is partially used as a source, so C1's validation "
                 "numbers are not interpretable. No arm performs model selection."),
    }
    with io.open(os.path.join(SPLITS, "c1_audit.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("output: %s" % p)
    print("        %s" % os.path.join(SPLITS, "c1_audit.json"))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
