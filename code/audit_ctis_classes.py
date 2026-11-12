"""CTIS class-universe audit: canonical recording counts before/after placeholder removal.

Read-only statistics; no re-splitting. Reproduces the two rules of splits_v2.py:
  1) a class with fewer than 5 recordings cannot be split at recording level (paper: 219 -> 206);
  2) any recording_uid containing the "没音频" placeholder is dropped entirely (paper: 5 more classes -> 201).

Writes data/results/human_review/ctis_class_drops.csv (per class) and the same-name .json (summary).
"""
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANI = os.path.join(ROOT, "data", "manifest")
OUT = os.path.join(ROOT, "data", "results", "human_review")
MIN_REC = 5
PLACEHOLDER = "没音频"


def main():
    path = os.path.join(MANI, "CTIS.csv")
    if not os.path.isfile(path):
        sys.exit("missing %s" % path)

    recs = defaultdict(set)          # label -> set(recording_canonical)
    placeholders = defaultdict(set)  # label -> set(canonical) with placeholder uid
    ph_uids = defaultdict(set)
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            lab = r["label"]
            recs[lab].add(r["recording_canonical"])
            if PLACEHOLDER in r["recording_uid"]:
                placeholders[lab].add(r["recording_canonical"])
                ph_uids[lab].add(r["recording_uid"])

    rows = []
    for lab in sorted(recs):
        n_src = len(recs[lab])
        n_ph = len(placeholders[lab])
        n_clean = n_src - n_ph
        if n_src < MIN_REC:
            reason = "min5_before_placeholder"
        elif n_clean < MIN_REC:
            reason = "placeholder_push_below5"
        else:
            reason = ""
        rows.append({
            "label": lab,
            "n_rec_source": n_src,
            "n_placeholder_rec": n_ph,
            "n_rec_after_placeholder": n_clean,
            "kept": not reason,
            "dropped_reason": reason,
            "placeholder_uids": ";".join(sorted(ph_uids[lab])),
        })

    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "ctis_class_drops.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    drop_min = [r for r in rows if r["dropped_reason"] == "min5_before_placeholder"]
    drop_ph = [r for r in rows if r["dropped_reason"] == "placeholder_push_below5"]
    kept = [r for r in rows if r["kept"]]
    summary = {
        "manifest": "data/manifest/CTIS.csv",
        "manifest_rows": n_rows,
        "classes_source": len(rows),
        "dropped_min5": len(drop_min),
        "dropped_placeholder_push": len(drop_ph),
        "classes_kept": len(kept),
        "placeholder_recordings_total": sum(r["n_placeholder_rec"] for r in rows),
        "placeholder_classes": [r["label"] for r in rows if r["n_placeholder_rec"]],
        "dropped_min5_detail": [
            {"label": r["label"], "n_rec_source": r["n_rec_source"],
             "n_placeholder_rec": r["n_placeholder_rec"]} for r in drop_min],
        "dropped_placeholder_detail": [
            {"label": r["label"], "n_rec_source": r["n_rec_source"],
             "n_placeholder_rec": r["n_placeholder_rec"],
             "n_rec_after": r["n_rec_after_placeholder"],
             "uids": r["placeholder_uids"]} for r in drop_ph],
        "check_219_minus_13": len(rows) - len(drop_min),
        "check_206_minus_5": len(rows) - len(drop_min) - len(drop_ph),
    }
    json_path = os.path.join(OUT, "ctis_class_drops.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("dropped_min5_detail", "dropped_placeholder_detail")},
                     ensure_ascii=False, indent=1))
    print("\n-- dropped by the min-5 rule (before) --")
    for r in drop_min:
        print("  %s n_rec=%d ph=%d" % (r["label"], r["n_rec_source"], r["n_placeholder_rec"]))
    print("\n-- dropped by the placeholder rule (after) --")
    for r in drop_ph:
        print("  %s n_rec=%d -> %d  ph=%d  %s"
              % (r["label"], r["n_rec_source"], r["n_rec_after_placeholder"],
                 r["n_placeholder_rec"], r["placeholder_uids"]))
    print("\nwrote %s\nwrote %s" % (csv_path, json_path))


if __name__ == "__main__":
    main()
