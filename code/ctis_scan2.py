import re,statistics
from pathlib import Path
from collections import Counter
RE=re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$",re.I)
P={"演奏技法"}
rows=[]
for p in Path("D:/VSCode/Datasets/CTIS").rglob("*.wav"):
    m=RE.match(p.name)
    if not m: continue
    t=m.group("rest").split(); lab=t[1] if t[0] in P and len(t)>1 else t[0]
    rows.append((lab,m.group("inst").strip()))
cells=Counter(rows); insts=sorted({i for _,i in rows}); labels=sorted({l for l,_ in rows})
EX=["音阶","空弦音","筒音","单音","乐曲片段","演奏技巧"]
print("present",[e for e in EX if e in labels],"missing",[e for e in EX if e not in labels])
def build(mc,mi,excl=()):
    keep=[l for l in labels if l not in excl and sum(1 for i in insts if cells[(l,i)]>=mc)>=mi]
    sub={k:v for k,v in cells.items() if k[0] in keep and v>=mc}
    per={l:sum(v for (l2,_),v in sub.items() if l2==l) for l in keep}
    return keep,sub,per
for mc,mi in [(2,2),(2,3)]:
    k0,s0,p0=build(mc,mi); k1,s1,p1=build(mc,mi,EX)
    print(f"\n=== mc{mc} mi{mi} raw {len(k0)}cls {sum(s0.values())}rec cells{len(s0)} med{statistics.median(p0.values())} min{min(p0.values())}")
    print("  raw:",sorted(k0))
    print(f"  excl {len(k1)}cls {sum(s1.values())}rec cells{len(s1)} med{statistics.median(p1.values())} min{min(p1.values())}")
    print("  kept:",sorted(k1))
    test=set(insts[0::2])
    ts={k:v for k,v in s1.items() if k[1] in test}
    tp=[sum(v for (l,i),v in ts.items() if l==x) for x in k1]
    print(f"  holdout inst {len(test)}/{len(insts)} testrec {sum(ts.values())} med{statistics.median(tp)} min{min(tp)} short{sum(1 for x in tp if x<mc)}")
