import re, json, statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("D:/VSCode/Datasets/CTIS")
OUT = Path("data/splits/ctis_capacity.json")
RE = re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$", re.IGNORECASE)
PREFIXES = {"演奏技法"}

rows, unparsed, raw_labels = [], [], set()
for p in ROOT.rglob("*.wav"):
    if not p.is_file():
        continue
    m = RE.match(p.name)
    if not m:
        unparsed.append(str(p)); continue
    toks = m.group("rest").split()
    label = toks[1] if toks and toks[0] in PREFIXES and len(toks) > 1 else toks[0]
    mods = toks[2:] if toks and toks[0] in PREFIXES else toks[1:]
    raw_labels.add(m.group("rest"))
    rows.append({"inst": m.group("inst"), "label": label, "mods": mods, "path": str(p)})

cells = Counter((r["label"], r["inst"]) for r in rows)
insts = sorted({r["inst"] for r in rows})
labels = sorted({r["label"] for r in rows})
res = {"disk_wav": 4956, "rows": len(rows), "unparsed": len(unparsed),
       "n_inst": len(insts), "n_raw_label_strings": len(raw_labels), "n_labels": len(labels),
       "insts": insts, "cells_total": len(cells),
       "records_used_cellsum": sum(cells.values()), "scan": {}}
for mi in (2, 3):
    for mc in (1, 2, 3, 5):
        keep = [l for l in labels if sum(1 for i in insts if cells[(l, i)] >= mc) >= mi]
        sub = {k: v for k, v in cells.items() if k[0] in keep and v >= mc}
        per = [sum(v for (l, i), v in sub.items() if l == l2) for l2 in keep]
        res["scan"][f"mc{mc}_mi{mi}"] = {
            "classes": len(keep), "recordings": sum(sub.values()),
            "median_cell": statistics.median(per) if per else 0,
            "min_cell": min(per) if per else 0, "cells": len(sub)}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("insts",)}, ensure_ascii=False, indent=1))
print("INST", insts)
print("rows+unparsed=", len(rows) + len(unparsed), "cellsum=", sum(cells.values()))
