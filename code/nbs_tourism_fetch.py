# -*- coding: utf-8 -*-
"""National Bureau of Statistics easyquery harvesting: annual national (hgnd) + annual by province (fsnd). Raw responses only, no data edits."""
import json, hashlib, time, sys, pathlib, datetime
import requests
BASE = "https://data.stats.gov.cn/easyquery.htm"
UA = {"User-Agent": "Mozilla/5.0 (research data audit; contact: local)"}
RAW = pathlib.Path("data/nbs-tourism/raw"); RAW.mkdir(parents=True, exist_ok=True)
LOG = pathlib.Path("data/nbs-tourism/logs"); LOG.mkdir(parents=True, exist_ok=True)

def tree(dbcode, tid):
    r = requests.post(BASE, params={"m": "getTree", "dbcode": dbcode, "id": tid, "wdcode": "zb"},
                      headers=UA, timeout=20)
    return r

def query(dbcode, zb, sj="LAST30"):
    dfwds = json.dumps([{"wdcode": "zb", "valuecode": zb}, {"wdcode": "sj", "valuecode": sj}],
                       ensure_ascii=False)
    r = requests.post(BASE, params={"m": "QueryData", "dbcode": dbcode, "rowcode": "zb",
                                    "colcode": "sj", "dfwds": dfwds, "k1": int(time.time()*1000)},
                      headers=UA, timeout=20)
    return r

def save(tag, r):
    body = r.content
    sha = hashlib.sha256(body).hexdigest()
    p = RAW / f"{tag}.json"; p.write_bytes(body)
    rec = {"tag": tag, "http": r.status_code, "bytes": len(body), "sha256": sha,
           "url": r.url, "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (LOG / f"{tag}.meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec

if __name__ == "__main__":
    for db, tid in (("hgnd", "A0R"), ("fsnd", "E0105")):   # A0R=tourism chapter, E0105=annual by province
        for i in range(3):
            try:
                r = tree(db, tid); print(db, r.status_code, len(r.content)); save(f"tree_{db}", r); break
            except Exception as e:
                print(db, "ERR", e); time.sleep(2)
