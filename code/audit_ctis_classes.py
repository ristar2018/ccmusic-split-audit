"""CTIS class-universe audit: canonical recording counts before/after placeholder removal.

只读统计，不重划分。复现 splits_v2.py 的两条口径：
  1) 每类录音数 < 5 不能做录音级切分（论文 219 -> 206）；
  2) recording_uid 含 "没音频" 的 placeholder 整条丢弃（论文再掉 5 类 -> 201）。

输出：data/results/human_review/ctis_class_drops.csv（逐类）与同名 .json（汇总）。
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
    print("\n-- 掉于 min5（前）--")
    for r in drop_min:
        print("  %s n_rec=%d ph=%d" % (r["label"], r["n_rec_source"], r["n_placeholder_rec"]))
    print("\n-- 掉于 placeholder（后）--")
    for r in drop_ph:
        print("  %s n_rec=%d -> %d  ph=%d  %s"
              % (r["label"], r["n_rec_source"], r["n_rec_after_placeholder"],
                 r["n_placeholder_rec"], r["placeholder_uids"]))
    print("\nwrote %s\nwrote %s" % (csv_path, json_path))


if __name__ == "__main__":
    main()
