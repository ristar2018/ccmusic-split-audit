import os,re,struct,statistics,json,collections
ROOT=r"D:/VSCode/Datasets/GZ_IsoTech/data"
def bases(d):
    out=collections.Counter(); paths={}
    for dp,_,fs in os.walk(d):
        for f in fs:
            stem,ext=os.path.splitext(f)
            b=re.sub(r"_\d+$","",stem)
            out[b]+=1; paths.setdefault(b,[]).append(os.path.join(dp,f))
    return out,paths
res={}
for split in ("train","test"):
    mb,_=bases(f"{ROOT}/eval/mel/{split}")
    ab,_=bases(f"{ROOT}/audio/{split}")
    K=collections.Counter(mb.values())
    res[split]=dict(mel_segs=sum(mb.values()),mel_bases=len(mb),audio=sum(ab.values()),
        K_dist={str(k):v for k,v in sorted(K.items())},extra=sum(mb.values())-len(mb),
        multi=sum(1 for v in mb.values() if v>1),maxK=max(mb.values()))
# 三个表征是否同 base 集
sets={}
for rep in ("mel","chroma","cqt"):
    s=set()
    for split in ("train","test"):
        b,_=bases(f"{ROOT}/eval/{rep}/{split}"); s|=set(b)
    sets[rep]=s
res["rep_base_sets"]={r:len(s) for r,s in sets.items()}
res["rep_sets_equal"]=len(set(map(frozenset,sets.values())))==1
# 时长：纯文件头解析
def dur(p):
    with open(p,"rb") as f:
        h=f.read(12)
        if len(h)<12 or h[8:12]!=b"WAVE": return None
        sr=ch=bits=None; n=0
        while True:
            ch_=f.read(8)
            if len(ch_)<8: break
            cid,sz=ch_[:4],struct.unpack("<I",ch_[4:])[0]
            if cid==b"fmt ":
                d=f.read(sz); fmt,ch,sr=struct.unpack("<HHI",d[:8])
                bits=struct.unpack("<H",d[14:16])[0] if len(d)>=16 else 16
            elif cid==b"data":
                n=sz; break
            else: f.seek(sz,1)
        if not sr: return None
        return n/(sr*ch*bits/8)
tot=[];bad=0;per=collections.defaultdict(list)
for split in ("test",):
    for dp,_,fs in os.walk(f"{ROOT}/audio/{split}"):
        cls=os.path.basename(dp)
        for f in fs:
            d=dur(os.path.join(dp,f))
            if d is None or d<=0: bad+=1; continue
            tot.append(d); per[cls].append(d)
res["test_dur"]=dict(n=len(tot),bad=bad,total_h=round(sum(tot)/3600,4),
    median_s=round(statistics.median(tot),3),max_s=round(max(tot),1),min_s=round(min(tot),3))
res["test_dur_per_class"]={k:dict(n=len(v),h=round(sum(v)/3600,3),median_s=round(statistics.median(v),2)) for k,v in sorted(per.items())}
json.dump(res,open("data/splits/gz_isotech.json","w"),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))


# 多段/单段录音时长对照（验证「长录音切段」机制）
K=collections.Counter()
for split in ("train","test"):
    for dp,_,fs in os.walk(f"{ROOT}/eval/mel/{split}"):
        for f in fs: K[re.sub(r"_\d+$","",os.path.splitext(f)[0])]+=1
idx={}
for split in ("train","test"):
    for dp,_,fs in os.walk(f"{ROOT}/audio/{split}"):
        for f in fs: idx[os.path.splitext(f)[0]]=os.path.join(dp,f)
md=[dur(p) for b,p in idx.items() if K[b]>1]; sd=[dur(p) for b,p in idx.items() if K[b]==1]
md=[x for x in md if x]; sd=[x for x in sd if x]
res["seg_mech"]=dict(multi_n=len(md),multi_median_s=round(statistics.median(md),2),
    single_n=len(sd),single_median_s=round(statistics.median(sd),2),
    K_mult=dict(collections.Counter(K[b] for b in idx if K[b]>1)))
# 三套表征 base 集是否逐一对应、录制会话前缀
sess=collections.Counter()
for dp,_,fs in os.walk(f"{ROOT}/audio/test"):
    for f in fs:
        m=re.match(r"(record\d*)_",f); sess[m.group(1) if m else "other"]+=1
res["test_sessions"]=dict(sess)
res["metadata_files"]=0
json.dump(res,open("data/splits/gz_isotech.json","w"),ensure_ascii=False,indent=1)
print({k:res[k] for k in ("seg_mech","test_sessions","test_dur")})
