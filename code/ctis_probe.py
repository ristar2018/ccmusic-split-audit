import re
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path("D:/VSCode/Datasets/CTIS")
RE=re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$",re.IGNORECASE)
P={"演奏技法"}
bad=[];rows=[]
for p in ROOT.rglob("*.wav"):
    m=RE.match(p.name)
    if not m: bad.append(p.name); continue
    t=m.group("rest").split()
    lab=t[1] if t[0] in P and len(t)>1 else t[0]
    rows.append((m.group("inst").strip(),lab))
cells=Counter(rows)
print("UNPARSED",len(bad))
for b in bad: print(" ",b)
lab_rec=Counter(l for i,l in rows); lab_inst=defaultdict(set)
for (i,l),c in cells.items():
    if c>=2: lab_inst[l].add(i)
print("\nLABELS",len(lab_rec))
for l,c in lab_rec.most_common(50):
    print(l,"\t",c,"\t",len(lab_inst[l]))
ic=Counter(i for i,_ in rows).values()
s=sorted(ic)
print("\ninst n=",len(s),"median",s[len(s)//2],"min",s[0],"max",s[-1])
print("cells dist",Counter(cells.values()).most_common(8))
