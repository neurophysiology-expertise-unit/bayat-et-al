import numpy as np, os
rng=np.random.default_rng(20260809); B=20000

def fwhm(al,c):
    kp=int(np.argmax(c)); half=c[kp]/2.0; Lc=Rc=None
    for i in range(kp,0,-1):
        if c[i-1]<half: Lc=al[i-1]+(al[i]-al[i-1])*(half-c[i-1])/(c[i]-c[i-1]); break
    for i in range(kp,len(c)-1):
        if c[i+1]<half: Rc=al[i]+(al[i+1]-al[i])*(c[i]-half)/(c[i]-c[i+1]); break
    return (Rc-Lc) if (Lc is not None and Rc is not None) else None

def analyze(npz, key):
    z=np.load(npz,allow_pickle=False)
    if key not in z.files: return None
    A=z[key]; al=z['alphas']; ns=A.shape[0]; it=len(al)-1
    H=np.empty(B); PT=np.empty(B); W=[]; nres=0
    for b in range(B):
        idx=rng.integers(0,ns,ns); m=A[idx].mean(0); kp=int(np.argmax(m))
        H[b]=m[kp]; PT[b]=m[kp]/m[it] if m[it]>1e-12 else np.nan
        w=fwhm(al,m)
        if w is not None: W.append(w); nres+=1
    W=np.array(W)
    return dict(ns=ns,H=np.median(H),H_ci=np.percentile(H,[2.5,97.5]),
        PT=np.nanmedian(PT),PT_ci=np.nanpercentile(PT,[2.5,97.5]),
        fres=nres/B,W=(np.median(W) if len(W) else None),
        Wci=(np.percentile(W,[2.5,97.5]) if len(W) else None))

for key,label in [('chi_true','chi_TRUE = N*Var_t(Cbar)'),('chi','chi_EXT = N*Var_t(max-min)'),
                  ('R','R (Golomb-Rinzel synchrony)'),('Sc','S_C (spatial heterogeneity)')]:
    print(f"\n===== {label} =====")
    print(f"{'L':>4} {'ns':>3} | {'height':>16} | {'FWHM':>22} | {'peak/tail':>16}")
    res={}
    for L in (10,32,64):
        f=f'processed_data/sharp_L{L}.npz'
        if not os.path.exists(f): continue
        r=analyze(f,key)
        if r is None: print(f"{L:>4}  (no {key})"); continue
        res[L]=r
        hs=f"{r['H']:6.2f}[{r['H_ci'][0]:5.2f},{r['H_ci'][1]:5.2f}]"
        ws=(f"{r['W']:.3f}[{r['Wci'][0]:.3f},{r['Wci'][1]:.3f}]" if r['W'] is not None and r['fres']>0.9
            else (f"~{r['W']:.3f}({r['fres']*100:.0f}%res)" if r['W'] is not None else "NO RES WIDTH"))
        ps=f"{r['PT']:5.2f}[{r['PT_ci'][0]:4.2f},{r['PT_ci'][1]:4.2f}]"
        print(f"{L:>4} {r['ns']:>3} | {hs:>16} | {ws:>22} | {ps:>16}")
    Ls=sorted(res)
    for i in range(len(Ls)-1):
        a,b=Ls[i],Ls[i+1]
        for stat in ['H_ci','PT_ci']:
            ca,cb=res[a][stat],res[b][stat]; ov=not(ca[1]<cb[0] or cb[1]<ca[0])
            print(f"    {stat[:-3]:>4} L{a} vs L{b}: {'OVERLAP' if ov else 'DISJOINT'}")
