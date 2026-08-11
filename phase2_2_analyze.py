import numpy as np, os, sys
rng=np.random.default_rng(20260810); B=20000; L=32
def load(tag):
    f=f'processed_data/phase2_2_L{L}_{tag}.npz'
    return (np.load(f,allow_pickle=False)['active'], np.load(f,allow_pickle=False)['alphas']) if os.path.exists(f) else None
actA,al=load('A_full'); mA=actA.mean(0)
def pct_inside(X):
    ns,na=X.shape
    idx=rng.integers(0,ns,(B,ns))
    means=X[idx].mean(1)               # (B, na) vectorised bootstrap
    lo=np.percentile(means,2.5,axis=0); hi=np.percentile(means,97.5,axis=0)
    return 100.0*np.mean((lo<=mA)&(mA<=hi))
CH=["gamma","sigma","Deff","theta","I0","tauh"]
tags=[f"LOO_no{c}" for c in CH]+[f"OAT_only{c}" for c in CH]
print(f"Phase 2.2 (L={L}, 40 seeds). %% alpha where Sweep A active mean INSIDE condition band.",flush=True)
print(f">=90 reproduces A; <=50 does not. Pattern2 (multi-channel): no OAT reproduces A AND",flush=True)
print(f"LOO_no_(gamma|sigma|tauh) each fail.\n\n{'condition':>16} | {'A inside %':>10} | verdict",flush=True)
res={}
for tag in tags:
    r=load(tag)
    if r is None: print(f"{tag:>16} | MISSING",flush=True); continue
    p=pct_inside(r[0]); res[tag]=p
    v="REPRODUCES-A" if p>=90 else ("does-NOT" if p<=50 else "AMBIGUOUS")
    print(f"{tag:>16} | {p:9.1f}% | {v}",flush=True)
print("\nraw seed-mean active(alpha):",flush=True)
print("  alpha :"," ".join("%.2f"%a for a in al),flush=True)
print("  A_full:"," ".join("%.2f"%x for x in mA),flush=True)
for tag in tags:
    r=load(tag)
    if r: print(f"  {tag:12}:"," ".join("%.2f"%x for x in r[0].mean(0)),flush=True)
