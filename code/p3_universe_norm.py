#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 前置：在 CTIS 样本清单(data/manifest/CTIS.csv)上做乐器名归一 + 属性类宇宙冻结。
只读清单；产出 data/p3/P3_universe_norm.json + data/p3/universe_norm_43.csv。
"""
import csv, json, collections, statistics, os, sys
sys.path.insert(0, "code")
from ctis_normlib import canon, l1, L2, PENDING

MANIFEST = "data/manifest/CTIS.csv"
OUT = "data/p3/P3_universe_norm.json"
MIN_CELL, MIN_INSTR = 2, 3
DROP = ["音阶", "空弦音", "筒音", "单音", "乐曲片段", "特有演奏技法"]

rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
rec = {}
for r in rows:                      # 录音级去重：切片不重复计数
    if r["recording_uid"]:
        rec.setdefault(r["recording_uid"], (r["instrument"], r["technique"]))
n_rec = len(rec)
pairs = [v for v in rec.values() if v[0] and v[1]]

def cells_of(f):
    return collections.Counter((tech, f(inst)) for inst, tech in pairs)

def universe(cells):
    ok = collections.defaultdict(set)
    for (t, i), c in cells.items():
        if c >= MIN_CELL:
            ok[t].add(i)
    uni = {t: sorted(s) for t, s in ok.items() if len(s) >= MIN_INSTR}
    used = sum(cells[(t, i)] for t in uni for i in uni[t])
    per = {t: sum(cells[(t, i)] for i in uni[t]) for t in uni}
    return uni, used, per

res = {"manifest_rows": len(rows), "recordings": n_rec, "recordings_with_attrs": len(pairs),
       "n_inst_raw": len({i for i, _ in pairs}), "n_tech": len({t for _, t in pairs})}

c_raw = cells_of(lambda x: x)
c_nrm = cells_of(canon)
u_raw, used_raw, per_raw = universe(c_raw)
u_nrm, used_nrm, per_nrm = universe(c_nrm)
res["raw_universe"] = {"classes": len(u_raw), "used": used_raw,
                       "median": statistics.median(per_raw.values()), "min": min(per_raw.values())}
res["norm_universe"] = {"classes": len(u_nrm), "used": used_nrm,
                        "median": statistics.median(per_nrm.values()), "min": min(per_nrm.values())}
drop = lambda per: {t: v for t, v in per.items() if t not in DROP}
a, b = drop(per_raw), drop(per_nrm)
res["after_drop_raw_inst"] = {"classes": len(a), "used": sum(a.values()),
                              "median": statistics.median(a.values()), "min": min(a.values())}
res["after_drop_norm_inst"] = {"classes": len(b), "used": sum(b.values()),
                               "median": statistics.median(b.values()), "min": min(b.values()),
                               "cells": sum(len(u_nrm[t]) for t in u_nrm if t not in DROP)}
res["classes_only_in_raw"] = sorted(set(a) - set(b))
res["classes_only_in_norm"] = sorted(set(b) - set(a))
res["per_class_delta"] = {t: [a[t], b[t]] for t in a if a[t] != b.get(t)}
res["dropped_present"] = {d: sum(v for t, v in per_nrm.items() if t == d) for d in DROP}
res["dropped_missing"] = [d for d in DROP if d not in per_nrm]
res["drop_siblings"] = {s: per_nrm.get(s, 0) for s in ["跨弦音阶","单弦音阶","轻六音阶","筒音D","音程"]}
# 乐器归一：命中的原始名与计数
inst_c = collections.Counter(i for i, _ in pairs)
merged = collections.OrderedDict()
for (i, t) in pairs:
    c = canon(i)
    if c != i:
        merged.setdefault(i, {"canon": c, "n": 0, "reason": ""})
for i, c in inst_c.items():
    if canon(i) != i:
        merged[i]["n"] = c
res["merge_rows"] = [{"raw": k, **v} for k, v in merged.items()]
res["n_inst_after"] = len({canon(i) for i, _ in pairs})
res["pending"] = list(PENDING)
res["per_class"] = sorted(b.items(), key=lambda x: -x[1])
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open("data/p3/universe_norm_43.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["technique", "recordings", "n_instruments"])
    for t, v in res["per_class"]:
        w.writerow([t, v, len(u_nrm[t])])
print(json.dumps({k: v for k, v in res.items() if k not in ("per_class", "merge_rows")}, ensure_ascii=False, indent=1))
print("per_class", res["per_class"])

# ---- 阈值扫描 + 留出（补 §7.2 三行对账）----
def scan(mc, mi):
    ok = collections.defaultdict(set)
    for (t, i), c in c_nrm.items():
        if c >= mc:
            ok[t].add(i)
    uni = {t: s for t, s in ok.items() if len(s) >= mi}
    used = sum(c_nrm[(t, i)] for t in uni for i in uni[t])
    per = [sum(c_nrm[(t, i)] for i in uni[t]) for t in uni]
    return {"classes": len(uni), "used": used,
            "median": statistics.median(per) if per else 0, "min": min(per) if per else 0,
            "cells": sum(len(s) for s in uni.values())}
res["scan"] = {f"mc{mc}_mi{mi}": scan(mc, mi) for mi in (2, 3) for mc in (1, 2, 3, 5)}
# 半数乐器留出：按乐器录音数降序交替分半
inst_n = collections.Counter()
for inst, tech in pairs:
    inst_n[canon(inst)] += 1
order = sorted(inst_n, key=lambda x: (-inst_n[x], x))
half = set(order[1::2])
test = {t: sum(c_nrm[(t, i)] for i in u_nrm[t] if i in half) for t in u_nrm if t not in DROP}
res["holdout"] = {"n_inst_test": len(half), "total": sum(test.values()),
                  "median": statistics.median(test.values()), "min": min(test.values()),
                  "lt4": sorted([k for k, v in test.items() if v < 4])}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("scan", json.dumps(res["scan"], ensure_ascii=False))
print("holdout", res["holdout"])
