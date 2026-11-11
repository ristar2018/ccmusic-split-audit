import re,json,statistics
from pathlib import Path
from collections import Counter
RE=re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$",re.I)
P={"演奏技法"}
rows=[];bad=[]
for p in Path("D:/VSCode/Datasets/CTIS").rglob("*.wav"):
    m=RE.match(p.name)
    if not m: bad.append(p.name); continue
    t=m.group("rest").split(); lab=t[1] if t[0] in P and len(t)>1 else t[0]
    rows.append((lab,m.group("inst").strip()))
cells=Counter(rows);insts=sorted({i for _,i in rows});labels=sorted({l for l,_ in rows})
EX=["音阶","空弦音","筒音","单音","乐曲片段","演奏技巧","特有演奏技法"]
def build(mc,mi,excl=()):
    keep=[l for l in labels if l not in excl and sum(1 for i in insts if cells[(l,i)]>=mc)>=mi]
    sub={k:v for k,v in cells.items() if k[0] in keep and v>=mc}
    return keep,sub
out={"disk":4956,"rows":len(rows),"unparsed":len(bad),"n_inst":len(insts),"n_labels":len(labels),
 "cellcount_median":statistics.median(list(cells.values())),"cellcount_mean":round(sum(cells.values())/len(cells),2)}
for tag,ex in [("EX6",EX[:6]),("EX7",EX)]:
    for mc,mi in [(2,2),(2,3)]:
        k,s=build(mc,mi,ex)
        per=[sum(v for (l,_),v in s.items() if l==x) for x in k]
        test=set(insts[0::2]); ts={kk:v for kk,v in s.items() if kk[1] in test}
        tp=[sum(v for (l,_),v in ts.items() if l==x) for x in k]
        out[f"{tag}_mc{mc}_mi{mi}"]={"classes":len(k),"rec":sum(s.values()),"cells":len(s),
          "med":statistics.median(per),"min":min(per),"test_rec":sum(ts.values()),
          "test_med":statistics.median(tp),"test_min":min(tp),"test_short":sum(1 for x in tp if x<mc)}
json.dump(out,open("data/splits/ctis_capacity.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
print("assert", len(rows)+len(bad)==4956, sum(cells.values())==len(rows))
