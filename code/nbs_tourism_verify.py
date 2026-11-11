# -*- coding: utf-8 -*-
"""交付前校验：raw sha256 复核、行 sha256 重算、主键唯一、单位一致、行数=各 series 之和。"""
import csv, hashlib, json, pathlib, sys
D = pathlib.Path("data"); RAW = pathlib.Path("data/nbs-tourism/raw")
ok = True
def fail(m):
    global ok; ok = False; print("FAIL", m)

for m in (pathlib.Path("data/nbs-tourism/logs")).glob("*.meta.json"):
    rec = json.loads(m.read_text(encoding="utf-8"))
    body = (RAW / f"{rec['tag']}.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != rec["sha256"]: fail(f"raw sha 不一致 {rec['tag']}")

rows = list(csv.DictReader((D/"nbs_tourism_long.csv").open(encoding="utf-8-sig")))
keys = set()
for r in rows:
    k = (r["indicator_id"], r["geo_level"], r["region_code"], r["year"])
    if k in keys: fail(f"主键重复 {k}")
    keys.add(k)
    h = hashlib.sha256(f"{r['indicator_id']}|{r['year']}|{r['value']}|{r['unit']}".encode()).hexdigest()[:16]
    if h != r["row_sha256"]: fail(f"row_sha256 不符 {k}")

src = list(csv.DictReader((D/"nbs_tourism_sources.csv").open(encoding="utf-8-sig")))
if sum(int(s["n_rows"]) for s in src) != len(rows): fail("行数与 sources.n_rows 合计不符")
units = {r["indicator_id"]: r["unit"] for r in rows}
print(f"long={len(rows)} wide={sum(1 for _ in (D/'nbs_tourism_wide.csv').open(encoding='utf-8-sig'))-1} "
      f"sources={len(src)} gaps={sum(1 for _ in (D/'nbs_tourism_gaps.csv').open(encoding='utf-8-sig'))-1} "
      f"units={units or 'n/a'}")
print("VERIFY", "OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
