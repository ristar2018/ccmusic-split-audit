"""校验 all 预设的 zip：存在性 + 体积 + CRC 全表扫描 + 条目数，输出 JSON/表格。"""
import json, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccmusic_download import PROJECTS, DATASETS_ROOT

dest = sys.argv[1] if len(sys.argv) > 1 else DATASETS_ROOT
rows = []
for ds, paths in PROJECTS["all"]["files"].items():
    for rel in paths:
        if not rel.endswith(".zip"):
            continue
        p = os.path.join(dest, ds, rel.replace("/", os.sep))
        if not os.path.exists(p):
            rows.append((ds, rel, "MISSING", 0, 0, "文件不存在")); continue
        size = os.path.getsize(p)
        try:
            with zipfile.ZipFile(p) as z:
                n, bad, total = len(z.namelist()), 0, 0
                for info in z.infolist():
                    total += info.file_size
                bad = z.testzip()
            st = "OK" if bad is None else f"BAD:{bad}"
            rows.append((ds, rel, st, size, n, "" if bad is None else f"CRC 损坏 {bad}"))
        except Exception as e:
            rows.append((ds, rel, "BAD:unzip", size, 0, str(e)[:80]))

for r in rows:
    print(f"{r[2]:<14}{r[0]:<18}{r[1]:<20}{r[3]/1048576:>10.1f}MB  {r[4]:>6} 条  {r[5]}")
ok = sum(1 for r in rows if r[2] == "OK")
print(f"\n合计 {len(rows)} 个 zip：完整 {ok}，异常 {len(rows)-ok}")
json.dump([{"dataset": r[0], "path": r[1], "status": r[2], "size": r[3], "entries": r[4], "note": r[5]} for r in rows],
          open(os.path.join(dest, "_zipcheck.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
