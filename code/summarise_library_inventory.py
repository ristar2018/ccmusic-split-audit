"""Summarise the library inventory, threshold sensitivity and final-epoch convergence.

Reads only released artifacts: the MEL manifests under data/manifest, the frozen
split CSVs under data/splits/v2, the per-class capacity artifact, the MEL zip
index of the excluded library, and the per-run training logs under runs/.

Writes data/results/library_inventory_and_convergence.json. Read-only otherwise;
exit code 1 means a required artifact is missing.

Usage:  python code/summarise_library_inventory.py
"""

from __future__ import annotations

import collections
import csv
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MANIFEST = os.path.join(PROJ, "data", "manifest")
SPLITS = os.path.join(PROJ, "data", "splits", "v2")
RESULTS = os.path.join(PROJ, "data", "results")

AUDITED = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto",
           "chest_falsetto", "erhu_playing_tech", "GZ_IsoTech"]
PLACEHOLDER = ["D0051", "D0058", "D0060", "D0061", "D0062"]
THRESHOLDS = [2, 3, 5, 10]
ARMS = [("A0", "a0"), ("A1", "a1"), ("A2", "a2"), ("C1", "c1")]


def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def inventory():
    out = {}
    for lib in AUDITED:
        path = os.path.join(MANIFEST, "%s.csv" % lib)
        if not os.path.exists(path):
            raise SystemExit("missing manifest: %s" % path)
        rows = read_csv(path)
        recs = {r.get("recording_canonical") or r.get("recording_uid") for r in rows}
        out[lib] = {"slices": len(rows), "recordings": len(recs),
                    "classes": len({r["label"] for r in rows})}
    index = os.path.join(PROJ, "data", "index", "Guzheng_Tech99__data_mel.zip.json")
    if os.path.exists(index):
        d = json.load(io.open(index, encoding="utf-8"))
        out["Guzheng_Tech99"] = {"slices": d["slices"], "recordings": d["sources"],
                                 "classes": None,
                                 "note": "MEL release has no per-recording class directory"}
    return out


def arm_sizes():
    out = {}
    for label, tag in ARMS:
        rows = read_csv(os.path.join(SPLITS, "CTIS_%s_train.csv" % tag))
        out[label] = {"slices": len(rows),
                      "recordings": len({r.get("recording_canonical") or r["recording_uid"]
                                         for r in rows})}
    return out


def thresholds():
    cap = json.load(io.open(os.path.join(PROJ, "data", "splits", "qpanel_capacity.json"),
                            encoding="utf-8"))["CTIS"]["classes"]
    out = {}
    for t in THRESHOLDS:
        keep = [c for c in cap if c["n_c"] >= t and c["cls"] not in PLACEHOLDER]
        out[str(t)] = {"classes": len(keep),
                       "slices": sum(c["S_c"] for c in keep),
                       "q_recordings": sum(c["q_rec"] for c in keep)}
    rule = [c for c in cap if c["n_c"] < 5]
    placeholder = [c for c in cap if c["cls"] in PLACEHOLDER]
    kept = [c for c in cap if c["n_c"] >= 5 and c["cls"] not in PLACEHOLDER]
    total = sum(c["S_c"] for c in cap)
    out["excluded"] = {
        "rule_classes": len(rule),
        "rule_slices": sum(c["S_c"] for c in rule),
        "placeholder_classes": len(placeholder),
        "placeholder_slices": sum(c["S_c"] for c in placeholder),
        "all_classes": len(rule) + len(placeholder),
        "all_slices": sum(c["S_c"] for c in rule) + sum(c["S_c"] for c in placeholder),
        "total_slices": total,
        "rule_median_recordings": median([c["n_c"] for c in rule]),
        "rule_median_slices": median([c["S_c"] for c in rule]),
        "rule_median_k": round(median([c["K_c"] for c in rule]), 2),
        "kept_median_recordings": median([c["n_c"] for c in kept]),
        "kept_median_slices": median([c["S_c"] for c in kept]),
        "kept_median_k": round(median([c["K_c"] for c in kept]), 2),
        "kept_classes": len(kept),
        "kept_recordings": sum(c["n_c"] for c in kept),
        "low_resource_kept_4to6": sum(1 for c in kept if c["n_c"] <= 6),
        "feasible_50": sum(1 for c in kept if c["feasible_0.5"]),
        "feasible_30": sum(1 for c in kept if c["feasible_0.3"]),
    }
    return out


def convergence():
    groups = collections.defaultdict(list)
    for path in glob.glob(os.path.join(PROJ, "runs", "*", "*", "*", "*", "train_log.csv")):
        parts = path.replace("\\", "/").split("/")
        lib, arm, backbone, seed = parts[-5], parts[-4], parts[-3], parts[-2]
        rows = read_csv(path)
        if not rows:
            continue
        last = rows[-1]
        groups[(lib, arm, backbone)].append(
            (float(last["train_loss"]), float(last["train_acc"]), int(last["step"])))
    out = {}
    for (lib, arm, backbone), vals in sorted(groups.items()):
        out["%s|%s|%s" % (lib, arm, backbone)] = {
            "seeds": len(vals),
            "final_train_loss_median": round(median([v[0] for v in vals]), 4),
            "final_train_acc_median": round(median([v[1] for v in vals]), 4),
            "steps": vals[0][2],
        }
    return out


def main():
    summary = {
        "generated_by": "code/summarise_library_inventory.py",
        "note": "Inventory from the released MEL manifests; thresholds from the per-class "
                "capacity artifact with the five placeholder classes removed; convergence "
                "from the final row of each run's train_log.csv, medians over seeds.",
        "libraries": inventory(),
        "arm_sizes_ctis": arm_sizes(),
        "threshold_sensitivity": thresholds(),
        "convergence": convergence(),
    }
    path = os.path.join(RESULTS, "library_inventory_and_convergence.json")
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=1))
    print("libraries: %d" % len(summary["libraries"]))
    print("thresholds: %s" % json.dumps(summary["threshold_sensitivity"], ensure_ascii=False)[:300])
    print("convergence groups: %d" % len(summary["convergence"]))
    print("report: %s" % os.path.relpath(path, PROJ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
