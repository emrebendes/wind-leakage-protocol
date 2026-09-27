import numpy as np, glob, json, sys, collections
from math import erfc, sqrt
store=sys.argv[1]
NS=8; NO=7584
def dm_nw(d, lag):
    """DM with Newey-West variance, bandwidth = lag (h-1). d: (stations, T)"""
    T=d.shape[1]; n=d.size; m=d.mean()
    dc=d-m
    g0=(dc*dc).sum()/n
    var=g0
    for k in range(1,lag+1):
        gk=(dc[:,k:]*dc[:,:-k]).sum()/n
        var+=2*(1-k/(lag+1))*gk
    dm=m/np.sqrt(var/n); p=erfc(abs(dm)/sqrt(2))
    return float(dm), float(p)
out={}
for f in sorted(glob.glob(f'results_causal/{store}/*/*/W*/seed*/*.npz')):
    m=f.split('/'); meth,mod,seed=m[2],m[3],m[5]
    if meth=='none': continue
    z=np.load(f); yt=z['y_true'].reshape(NS,NO,24); yd=z['y_pred_deployed'].reshape(NS,NO,24)
    last=yt[:,:-1,0]            # x_t for origins 1..NO-1
    yt=yt[:,1:,:]; yd=yd[:,1:,:]
    res={}
    for h in (1,8,16,24):
        e1=yt[:,:,h-1]-last; e2=yt[:,:,h-1]-yd[:,:,h-1]
        d=e1**2-e2**2
        dm,p=dm_nw(d,h-1)
        res[f'H{h}']=dict(DM=dm,p=p,rmse_pers=float(np.sqrt((e1**2).mean())),rmse_del=float(np.sqrt((e2**2).mean())))
    out[f'{meth}|{mod}|{seed}']=res
json.dump(out,open(f'{store}_dm_h.json'.replace('/','_'),'w'),indent=1)
print(store,len(out))
