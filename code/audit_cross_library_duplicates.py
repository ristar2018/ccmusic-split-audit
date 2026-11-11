"""Cross-library duplication check for the released MEL slices.

Reads the per-library manifests and asks whether the same byte-identical
feature slice is registered under two different libraries.  Only exact byte
identity is tested: re-encoded or re-cropped copies of the same audio would
not be caught, and the manuscript says so.

Usage
    python code/audit_cross_library_duplicates.py
Output
    data/results/cross_library_duplicates.json
"""

from __future__ import annotations

import collections
import csv
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]


def main() -> int:
    owner: dict[str, str] = {}
    cross: dict[str, set[str]] = collections.defaultdict(set)
    per_lib: dict[str, int] = {}
    per_lib_unique: dict[str, int] = {}

    for lib in LIBS:
        path = os.path.join(ROOT, "data", "manifest", f"{lib}.csv")
        n = 0
        seen_here: set[str] = set()
        with io.open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                h = row.get("content_hash")
                if not h:
                    continue
                n += 1
                seen_here.add(h)
                if h in owner and owner[h] != lib:
                    cross[h].add(owner[h])
                    cross[h].add(lib)
                else:
                    owner.setdefault(h, lib)
        per_lib[lib] = n
        per_lib_unique[lib] = len(seen_here)

    total = sum(per_lib.values())
    report = {
        "generated_by": "code/audit_cross_library_duplicates.py",
        "check": "exact byte identity of the feature slice (content_hash)",
        "libraries": LIBS,
        "slices_per_library": per_lib,
        "unique_hashes_per_library": per_lib_unique,
        "slices_total": total,
        "unique_hashes_total": len(owner),
        "cross_library_groups": len(cross),
        "within_library_duplicate_instances": total - len(owner),
        "examples": [{"hash": h[:16], "libraries": sorted(v)} for h, v in list(cross.items())[:10]],
        "limit": (
            "Byte identity only. Re-encoded, re-cropped or differently windowed copies of the "
            "same audio are not detected; audio-level identity is not recoverable for libraries "
            "whose image file names do not encode the evaluated recording."
        ),
    }

    out = os.path.join(ROOT, "data", "results", "cross_library_duplicates.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"slices total: {total}")
    print(f"unique hashes: {len(owner)}")
    print(f"cross-library groups: {len(cross)}")
    print(f"written: {os.path.relpath(out, ROOT)}")
    return 0 if len(cross) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
