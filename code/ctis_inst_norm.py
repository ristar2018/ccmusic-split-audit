# 乐器名归一（合并表）：L1 机械规范化 + L2 同物合并；只读 D:/VSCode/Datasets/CTIS
import re, json, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, "code")
from ctis_normlib import canon, canon_reason, l1, PENDING

ROOT = Path("D:/VSCode/Datasets/CTIS")
OUT = Path("data/splits/ctis_inst_norm.json")
RE = re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$", re.IGNORECASE)
inst_c, pair, rows, unparsed = Counter(), Counter(), 0, 0
for p in ROOT.rglob("*.wav"):
    if not p.is_file():
        continue
    m = RE.match(p.name)
    if not m:
        unparsed += 1; continue
    rows += 1
    raw = m.group("inst"); c = canon(raw)
    inst_c[c] += 1
    if c != raw:
        pair[(raw, c)] += 1
res = {"disk_wav": 4956, "rows": rows, "unparsed": unparsed,
       "n_inst_raw": 212, "n_inst_norm": len(inst_c),
       "merged_pairs": len(pair), "merged_rows": sum(pair.values()),
       "inst_counts_norm": inst_c.most_common(),
       "merge_table": [{"raw": r, "canon": c, "n": n, "reason": canon_reason(r)}
                       for (r, c), n in sorted(pair.items(), key=lambda x: -x[1])],
       "pending": [{"raw": k, "n": 1, "note": v} for k, v in PENDING.items()]}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("rows", rows, "unparsed", unparsed, "raw", 212, "norm", len(inst_c),
      "pairs", len(pair), "merged_rows", sum(pair.values()), "check", rows + unparsed)
for m in res["merge_table"]:
    print(f'{m["n"]:4d}  {m["raw"]} -> {m["canon"]}  # {m["reason"]}')
