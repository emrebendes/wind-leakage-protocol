# -*- coding: utf-8 -*-
"""Ek C sayıları. Koşum yeri truba/ kökü. Çıktılar paper/appendix_c/veriler/ekC_*.json.
C.1 origin bazlı rejimde kazancın eşli aralığı (8 istasyon x 3 tohum = 24 replika, %95, t=2,069)
C.2 teslim edilen - kalıcılık eşli aralığı (iki rejim) ve tam-seri için ufuk bazlı DM (Newey-West, bant h-1)
C.3 Tablo 8'in MAE karşılığı (aynı tanım: yapılandırma kazancı tohum ortalamasından, yedi mimari medyanı)
Bölüm bazlı rejimin kontrolü analyze.py ile aynı, origin bazlı rejimin ayrıştırılmamış modeli."""
import json, glob, collections, statistics as st, sys
import numpy as np
from math import erfc, sqrt
R='results_causal'; H=['H1','H8','H16','H24']; T24=2.069
meths=['dwt','vmd','emd','eemd','ceemdan']
def load(arm):
    out={}
    for f in glob.glob(f'{R}/{arm}/*/*/W*/seed*/*.metrics.json'):
        m=f.split('/'); out.setdefault((m[2],m[3]),{})[m[5]]=json.load(open(f))
    return out
arms={a:load(a) for a in ('causal','leaky','partition')}
mods=sorted({m for (_,m) in arms['causal']})
def ctrl(arm): return 'causal' if arm=='partition' else arm
def ci(d):
    d=np.asarray(d); n=len(d); assert n==24; se=d.std(ddof=1)/np.sqrt(n)
    return dict(n=n, mean=float(d.mean()), lo=float(d.mean()-T24*se), hi=float(d.mean()+T24*se), pos=int((d>0).sum()))
# ---- C.1 ----
c1={}
for arm in ('causal','partition','leaky'):
    for meth in meths:
        for mod in mods:
            a=arms[arm].get((meth,mod)); b=arms[ctrl(arm)][('none',mod)]
            if a is None: continue
            for h in H:
                g=[]
                for s in sorted(a):
                    for i in range(8):
                        ra=a[s]['per_station'][f'station_{i}']['per_horizon'][h]['RMSE']; rb=b[s]['per_station'][f'station_{i}']['per_horizon'][h]['RMSE']
                        g.append(100*(rb-ra)/rb)
                c1[f'{arm}|{meth}|{mod}|{h}']=ci(g)
oz1={}
for arm in ('causal','partition','leaky'):
    for meth in meths:
        for h in H:
            rows=[c1[f'{arm}|{meth}|{m}|{h}'] for m in mods]
            oz1[f'{arm}|{meth}|{h}']=dict(median_gain=st.median(r['mean'] for r in rows), ci_below0=sum(r['hi']<0 for r in rows), ci_above0=sum(r['lo']>0 for r in rows), n_models=len(rows))
json.dump(dict(tanim=__doc__, per_config=c1, ozet=oz1), open('paper/appendix_c/veriler/ekC_C1_kazanc_aralik.json','w'), indent=1, ensure_ascii=False)
# ---- C.2a eşli aralık ----
c2={}; oz2={}
for store,rej in (('deployment_gap','tam-seri'),('deployment_gap_partition','bölüm bazlı')):
    per={}
    for f in glob.glob(f'{R}/{store}/*/*/W*/seed*/*.metrics.json'):
        m=f.split('/'); per.setdefault((m[2],m[3]),{})[m[5]]=json.load(open(f))
    assert len(per)==35
    for h in H:
        worse=better=span=0
        for (meth,mod),v in per.items():
            d=[]
            for s in sorted(v):
                for i in range(8):
                    ph=v[s]['per_station'][f'station_{i}']['per_horizon'][h]; d.append(ph['RMSE']-ph['persistence_RMSE'])
            r=ci(d); c2[f'{rej}|{meth}|{mod}|{h}']=r
            if r['lo']>0: worse+=1
            elif r['hi']<0: better+=1
            else: span+=1
        oz2[f'{rej}|{h}']=dict(ci_above0_teslim_kotu=worse, ci_below0_teslim_iyi=better, sifiri_kapsar=span)
# ---- C.2b DM tam-seri ----
dm=json.load(open('paper/appendix_c/veriler/ekC_dm_tam_seri_ham.json'))
ozdm={}
for h in H:
    per=collections.defaultdict(list)
    for k,v in dm.items(): meth,mod,seed=k.split('|'); per[(meth,mod)].append(v[h])
    ozdm[h]=dict(uc_tohumda_kalicilik_iyi=sum(all(r['DM']<0 and r['p']<0.05 for r in v) for v in per.values()),
                 uc_tohumda_model_iyi=sum(all(r['DM']>0 and r['p']<0.05 for r in v) for v in per.values()),
                 dm_min=min(r['DM'] for v in per.values() for r in v), dm_max=max(r['DM'] for v in per.values() for r in v),
                 p_max=max(r['p'] for v in per.values() for r in v), n_origin=7583*8)
json.dump(dict(tanim='teslim edilen RMSE - kalıcılık RMSE, istasyon x tohum, %95 eşli aralık; DM tam-seri rejim, kaydedilmiş tahminlerden (npz), kayıp kare hata, Newey-West bant h-1, istasyon başına ilk origin düşer (x_t kayıtta yok)', aralik=c2, ozet_aralik=oz2, dm_tam_seri_ozet=ozdm, dm_tam_seri=dm), open('paper/appendix_c/veriler/ekC_C2_teslim_kalicilik.json','w'), indent=1, ensure_ascii=False)
# ---- C.3 MAE ----
c3={}
for key in ('RMSE','MAE','MAPE','nRMSE','R2','DA'):   # R2 ve DA için puan farkı, diğerleri yüzde azalma
    for h in H:
        for meth in meths:
            for arm in ('leaky','partition','causal'):
                gs=[]
                for mod in mods:
                    a=arms[arm].get((meth,mod)); b=arms[ctrl(arm)][('none',mod)]
                    if a is None: continue
                    ma=np.mean([a[s]['test']['per_horizon'][h][key] for s in a]); mb=np.mean([b[s]['test']['per_horizon'][h][key] for s in b])
                    gs.append((ma-mb) if key in ('R2','DA') else 100*(mb-ma)/mb)
                c3[f'{key}|{h}|{meth}|{arm}']=dict(median_gain=st.median(gs), gains=gs)
json.dump(dict(tanim='Tablo 8 tanımı: yapılandırma kazancı tohum ortalamasından, yedi mimari medyanı. RMSE/MAE/MAPE/nRMSE yüzde azalma, R2/DA kontrole göre puan farkı (yöntem - kontrol)', ozet=c3), open('paper/appendix_c/veriler/ekC_C3_mae.json','w'), indent=1, ensure_ascii=False)
# ---- istasyon işaret ----
isr={}
for meth in meths:
    for mod in mods:
        a=arms['causal'][(meth,mod)]; b=arms['causal'][('none',mod)]
        isr[f'{meth}|{mod}']=c1[f'causal|{meth}|{mod}|H1']['pos']
json.dump(dict(tanim='origin bazlı rejim, h=1, 24 replikanın kaçında yöntem kontrolden iyi', pos24=isr), open('paper/appendix_c/veriler/ekC_istasyon_isaret.json','w'), indent=1, ensure_ascii=False)
print('C1', {k:v for k,v in oz1.items() if k.startswith('causal')})
print('C2', oz2); print('DM', ozdm)
print('C3 MAE H1', {k:round(v['median_gain'],1) for k,v in c3.items() if k.startswith('MAE|H1')})
print('isaret', isr)
