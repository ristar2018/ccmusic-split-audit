# -*- coding: utf-8 -*-
"""Build the four delivery tables plus the request plan. Where the data could not be fetched, write only the header, the source and the gap; never fill in a placeholder number."""
import csv, json, pathlib, hashlib
D = pathlib.Path("data"); L = pathlib.Path("data/nbs-tourism/logs")

LONG = ["indicator_id","indicator_cn","caliber_note","geo_level","region_code","region_name",
        "year","value","unit","status","gap_code","source_url","api_url","raw_sha256","row_sha256","fetched_at"]
SRC  = ["series_key","indicator_cn","geo_level","dbcode","zb_code","colcode","years_requested",
        "n_rows","http_status","retcode","raw_file","raw_bytes","raw_sha256","fetched_at","note"]
GAP  = ["indicator_cn","geo_level","year_range","gap_code","evidence","evidence_level","note"]
PLAN = ["series_key","indicator_cn","geo_level","dbcode","zb_code","api_url","page_url"]

def w(name, header, rows):
    p = D / name
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)
    print(name, "rows=", len(rows))

w("nbs_tourism_long.csv", LONG, [])
w("nbs_tourism_wide.csv",
  ["geo_level","region_code","region_name","year","domestic_visitors","domestic_revenue",
   "inbound_overnight","fx_earnings","units","source_urls","row_sha256"], [])

src = []
for tag, db, zb in (("tree_hgnd","hgnd","A0R"), ("tree_fsnd","fsnd","E0105")):
    m = json.loads((L / f"{tag}.meta.json").read_text(encoding="utf-8"))
    src.append([f"{db}:{zb}:tree", "旅游章节树", "national" if db=="hgnd" else "province", db, zb, "zb",
                "—", 0, m["http"], "n/a", f"data/nbs-tourism/raw/{tag}.json", m["bytes"], m["sha256"],
                m["fetched_at"], "站点 WAF 返回 403 UrlACL（HTML），非数据响应；见 probe.log"])
w("nbs_tourism_sources.csv", SRC, src)

gap = [
 ["国内游客/国内旅游人数","national","2020–2022","G3","NBS 年度旅游序列疫情期断更，页面留空","doc","起始年待联网核"],
 ["国内游客/国内旅游人数","province","2020–2022","G3","分省旅游序列同期停发","doc",""],
 ["国内旅游收入","national","2023–","G4","改由文旅部按“国内游客出游总花费”发布，单位亿元→万亿元，不可与旧序列直接拼接","doc","需另立 indicator"],
 ["国内旅游收入","province","2023–","G1","NBS 分省库无新口径分省值","doc",""],
 ["入境过夜游客","national","2020–2022","G3","疫情期口岸数据极低/停发","doc",""],
 ["入境过夜游客","province","2020–","G3","多数省份不再逐年发布","doc",""],
 ["国际旅游外汇收入","national","2020–","G1","万美元口径年度值基本停发","doc",""],
 ["国际旅游外汇收入","province","2020–","G1","分省外汇收入停发","doc",""],
 ["全部指标","province","全部年份","G2","分省库为 31 省区市（码 11–65），不含港澳台；全国口径含港澳台入境","doc",""],
 ["全部指标","national+province","全部年份","G0","本次采集被站点 WAF 拦截（IPv6 出口 403 UrlACL；IPv4 443 读超时），0 行实测","verified","logs/probe.log, raw/*.json"],
]
w("nbs_tourism_gaps.csv", GAP, gap)

plan = []
for ind, zb in (("国内游客","<A0R..>"), ("国内旅游收入","<A0R..>"), ("入境过夜游客","<A0R..>"), ("国际旅游外汇收入","<A0R..>")):
    for lvl, db, cn in (("national","hgnd","C01"), ("province","fsnd","E0105")):
        api = ("https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=%s&rowcode=zb&colcode=sj"
               "&dfwds=[{\"wdcode\":\"zb\",\"valuecode\":\"%s\"},{\"wdcode\":\"sj\",\"valuecode\":\"LAST30\"}]" % (db, zb))
        plan.append([f"{db}:{zb}", ind, lvl, db, zb, api, f"https://data.stats.gov.cn/easyquery.htm?cn={cn}&zb={zb}"])
with (L/"plan.csv").open("w", newline="", encoding="utf-8-sig") as f:
    cw=csv.writer(f); cw.writerow(PLAN); cw.writerows(plan)
print("plan.csv rows=", len(plan), "| indicator codes to be filled in after the tree is parsed")
