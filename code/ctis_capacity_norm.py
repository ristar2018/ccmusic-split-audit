# P3 前置：归一乐器名 → 冻结属性类宇宙（格≥2 且覆盖≥3 乐器，剔除 6 个内容类标签）
import re, json, statistics, sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, "code")
from ctis_normlib import canon, PENDING

ROOT = Path("D:/VSCode/Datasets/CTIS")
OUT = Path("data/splits/ctis_capacity_norm.json")
RE = re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$", re.IGNORECASE)
PREFIXES = {"演奏技法"}
CONTENT = ["音阶", "空弦音", "筒音", "单音", "乐曲片段", "特有演奏技法"]
SIBLINGS = ["跨弦音阶", "单弦音阶", "轻六音阶", "筒音D", "音程"]

rows, unparsed = [], 0
for p in ROOT.rglob("*.wav"):
    if not p.is_file():
        continue
    m = RE.match(p.name)
    if not m:
        unparsed += 1; continue
    toks = m.group("rest").split()
    label = toks[1] if toks and toks[0] in PREFIXES and len(toks) > 1 else toks[0]
    rows.append((label, canon(m.group("inst"))))

insts = sorted({i for _, i in rows}); labels = sorted({l for l, _ in rows})
cells = Counter(rows)
def stat(lbls, mc):
    sub = {k: v for k, v in cells.items() if k[0] in lbls and v >= mc}
    per = [sum(v for (l, i), v in sub.items() if l == x) for x in lbls]
    return {"classes": len(lbls), "recordings": sum(sub.values()),
            "median": statistics.median(per) if per else 0, "min": min(per) if per else 0,
            "cells": len(sub), "per_class": dict(zip(lbls, per))}

scan = {}
for mi in (2, 3):
    for mc in (1, 2, 3, 5):
        keep = [l for l in labels if sum(1 for i in insts if cells[(l, i)] >= mc) >= mi]
        s = stat(keep, mc); s.pop("per_class")
        scan[f"mc{mc}_mi{mi}"] = s

keep0 = [l for l in labels if sum(1 for i in insts if cells[(l, i)] >= 2) >= 3]
keep = [l for l in keep0 if l not in CONTENT]
frozen = stat(keep, 2)
pc = frozen["per_class"]
rows_json = {"disk_wav": 4956, "rows": len(rows), "unparsed": unparsed,
             "n_inst_raw": 212, "n_inst_norm": len(insts), "n_labels": len(labels),
             "cells_total": len(cells), "scan": scan,
             "pre_exclusion": {"classes": len(keep0), "recordings": stat(keep0, 2)["recordings"]},
             "frozen": {k: v for k, v in frozen.items() if k != "per_class"},
             "per_class": sorted(pc.items(), key=lambda x: -x[1]),
             "tail_lt6": [k for k, v in pc.items() if v < 6],
             "excluded_present": {c: sum(1 for (l, i) in cells if l == c) and
                                  sum(v for (l, i), v in cells.items() if l == c) for c in CONTENT},
             "excluded_missing": [c for c in CONTENT if c not in labels],
             "siblings_not_covered": {s: sum(v for (l, i), v in cells.items() if l == s) for s in SIBLINGS},
             "pending": {k: v for k, v in PENDING.items()}}

# 半数乐器留出（按乐器录音数降序交替分 A/B，B 为 held-out 测试侧）
order = sorted(insts, key=lambda i: (-sum(v for (l, j), v in cells.items() if j == i), i))
half = set(order[1::2])
test = {l: sum(v for (x, i), v in cells.items() if x == l and i in half and v >= 2) for l in keep}
tp = [v for v in test.values()]
rows_json["holdout"] = {"n_inst_test": len(half), "test_total": sum(tp),
                        "median": statistics.median(tp) if tp else 0, "min": min(tp) if tp else 0,
                        "classes_lt2": [k for k, v in test.items() if v < 2]}
OUT.write_text(json.dumps(rows_json, ensure_ascii=False, indent=1), encoding="utf-8")

print("rows+unparsed=", len(rows) + unparsed, "== disk 4956?", len(rows) + unparsed == 4956)
print("cellsum==", sum(cells.values()) == len(rows))
print("insts", len(insts), "labels", len(labels), "cells", len(cells))
print("--- scan ---")
for k, v in scan.items():
    print(f"{k:10s} classes={v['classes']:3d} rec={v['recordings']:5d} median={v['median']:6} min={v['min']:4} cells={v['cells']}")
print("--- frozen (mc2,mi3, minus 6 content) ---")
print("classes", frozen["classes"], "recordings", frozen["recordings"], "median", frozen["median"], "min", frozen["min"], "cells", frozen["cells"])
print("pre_exclusion", rows_json["pre_exclusion"])
print("excluded_present", rows_json["excluded_present"], "missing", rows_json["excluded_missing"])
print("siblings", rows_json["siblings_not_covered"])
print("tail_lt6", len(rows_json["tail_lt6"]), rows_json["tail_lt6"])
print("holdout", rows_json["holdout"])
print("--- per class ---")
for k, v in rows_json["per_class"]:
    print(f"{v:5d}\t{k}")

# ---- 变体 B：把「演奏技巧」一并视为占位类（它不指任何具体技法）----
import statistics as st
pc_b = {k: v for k, v in pc.items() if k != "演奏技巧"}
vals = sorted(pc_b.values())
rows_json["variantB_no_playtech"] = {"classes": len(pc_b), "recordings": sum(vals),
    "median": st.median(vals), "min": min(vals),
    "note": "剔除 6 个内容类 + 「演奏技巧」占位类"}
OUT.write_text(json.dumps(rows_json, ensure_ascii=False, indent=1), encoding="utf-8")
print("variantB", rows_json["variantB_no_playtech"])
