"""Arithmetic-only v1.2 audit; no training; preserve historical outputs."""
from pathlib import Path
import csv,json,math,hashlib
from itertools import product
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/design_v1_2';OUT.mkdir(parents=True,exist_ok=True)
S={'CTIS':43054,'music_genre':36375,'pianos':18352,'bel_canto':9910,'CNPM':2724,'GZ_IsoTech':2899,'chest_falsetto':1280,'erhu_playing_tech':1253,'Guzheng_Tech99':99}
Q={'alexnet':339.,'swin_t':341.,'vgg19_bn':167.}
def hours(ds,bs,repeats):
    return sum(S[d]*10/Q[b]/3600*repeats for d,b in product(ds,bs))
main=['CTIS','music_genre','pianos','bel_canto','CNPM'];thin=['GZ_IsoTech','chest_falsetto','erhu_playing_tech'];features=['CTIS','music_genre','CNPM'];proxy=thin+['Guzheng_Tech99']
v11=[('T1',60,hours(main,list(Q),4)),('T2',24,hours(thin,['alexnet','swin_t'],4)),('T3',48,hours(features,['alexnet','swin_t'],4)),('T4',24,hours(proxy,['alexnet','swin_t'],3)),('T5',24,hours(['CTIS','music_genre','bel_canto'],['alexnet','swin_t'],4))]
rows=[]
for ds,b,p,seed in product(main,['alexnet','swin_t'],['clean','exposed'],[0,1,2]):
    rows.append({'run_id':f'e1b_{ds}_{b}_{p}_{seed}','dataset':ds,'backbone':b,'protocol':p,'feature':'provided','seed':seed,'epochs':10,'full_inventory_upper_gpu_h':S[ds]*10/Q[b]/3600})
with (OUT/'e1b_run_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert len(rows)==len({r['run_id'] for r in rows})==60
v12={'core_runs':60,'core_full_inventory_upper_gpu_h':sum(r['full_inventory_upper_gpu_h'] for r in rows),'optional_capacity_runs':30,'optional_capacity_gpu_h':hours(main,['vgg19_bn'],6),'optional_feature_runs':24,'optional_feature_gpu_h':hours(['CTIS','CNPM'],['alexnet','swin_t'],6)}
sensitivity=[{'n_test':n,'discordance_assumed':q,'normal_approx_mde_acc_pt':100*(1.959963984540054+0.8416212335729143)*math.sqrt(q/n)} for n,q in product([50,100,287,496],[.05,.2,.4])]
stats=json.loads((ROOT/'data/actual_stats.json').read_text(encoding='utf-8'))
assets={d:{k:stats.get(d,{}).get(k) for k in ['audio_files','duration_h','bytes','ext']} for d in list(S)+['instrument_timbre']}
out={'v1_1_corrected_tiers':[{'tier':t,'actual_runs':n,'gpu_h':h} for t,n,h in v11],'v1_1_corrected_e1_gpu_h':sum(h for _,_,h in v11),'v1_2':v12,'power_sensitivity_NOT_validated_power':sensitivity,'assets':assets,'sources_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['report/实验设计方案v1.1.md','code/design_audit.py','code/ccmusic_splits.py','data/actual_stats.json']}}
(OUT/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'v11':out['v1_1_corrected_tiers'],'v11_e1_gpu_h':out['v1_1_corrected_e1_gpu_h'],'v12':v12},ensure_ascii=False,indent=2))
