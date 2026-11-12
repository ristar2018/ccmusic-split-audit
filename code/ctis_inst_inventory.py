# Read-only over D:/VSCode/Datasets/CTIS; emits the raw inventory of instruments/labels (pre-normalization baseline)
import re, json, sys
from pathlib import Path
from collections import Counter

ROOT = Path("D:/VSCode/Datasets/CTIS")
OUT = Path("data/splits/ctis_inst_inventory.json")
RE = re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$", re.IGNORECASE)
PREFIXES = {"演奏技法"}

inst_c, label_c, raw_c = Counter(), Counter(), Counter()
unparsed = []
disk = 0
for p in ROOT.rglob("*.wav"):
    if not p.is_file():
        continue
    disk += 1
    m = RE.match(p.name)
    if not m:
        unparsed.append(str(p)); continue
    toks = m.group("rest").split()
    label = toks[1] if toks and toks[0] in PREFIXES and len(toks) > 1 else toks[0]
    inst_c[m.group("inst")] += 1
    label_c[label] += 1
    raw_c[m.group("rest")] += 1

res = {"disk_wav": disk, "rows": sum(inst_c.values()), "unparsed": len(unparsed),
       "n_inst": len(inst_c), "n_labels": len(label_c), "n_raw_label_strings": len(raw_c),
       "inst_counts": inst_c.most_common(), "label_counts": label_c.most_common(),
       "unparsed_samples": unparsed[:20]}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("disk", disk, "rows", res["rows"], "unparsed", res["unparsed"], "inst", res["n_inst"], "labels", res["n_labels"])
print("--- inst (name,count) ---")
for n, c in res["inst_counts"]:
    print(f"{c}\t{n}")
print("--- labels ---")
for n, c in res["label_counts"]:
    print(f"{c}\t{n}")
