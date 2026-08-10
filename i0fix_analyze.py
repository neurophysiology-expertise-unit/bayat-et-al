import numpy as np, os
rng=np.random.default_rng(20260810); B=20000
def boot(A, al):
    ns=A.shape[0]; it=len(al)-1
    pk=np.empty(B); tl=np.empty(B); pt=np.empty(B); a0=np.empty(B)
    for b in range(B):
        m=A[rng.integers(0,ns,ns)].mean(0); k=int(np.argmax(m))
        pk[b]=m[k]; tl[b]=m[it]; pt[b]=m[k]/m[it]; a0[b]=al[k]
    q=lambda x:np.percentile(x,[2.5,97.5])
    return dict(pk=np.median(pk),pk_ci=q(pk),tl=np.median(tl),tl_ci=q(tl),
                pt=np.median(pt),pt_ci=q(pt),a0=np.median(a0),a0_ci=q(a0))
for key in ('chi_true','R','chi'):
    print(f"\n===== I0-CONTROLLED model — {key} =====")
    print(f"{'L':>4} | {'alpha0 (CI)':>20} {'loc':>9} | {'peak (CI)':>20} | {'tail(0.35)':>16} | {'peak/tail (CI)':>18}")
    res={}
    for L in (10,32,64):
        f=f'processed_data/sharp_i0fix_L{L}.npz'
        if not os.path.exists(f): print(f"{L:>4} | pending"); continue
        z=np.load(f,allow_pickle=False); al=z['alphas']; A=z[key]
        r=boot(A,al); res[L]=r
        loc = "BOUNDARY!" if abs(r['a0']-al[0])<1e-6 else "interior"
        print(f"{L:>4} | {r['a0']:.3f}[{r['a0_ci'][0]:.3f},{r['a0_ci'][1]:.3f}] {loc:>9} | "
              f"{r['pk']:6.2f}[{r['pk_ci'][0]:5.2f},{r['pk_ci'][1]:6.2f}] | "
              f"{r['tl']:5.2f}[{r['tl_ci'][0]:.2f},{r['tl_ci'][1]:.2f}] | "
              f"{r['pt']:5.2f}[{r['pt_ci'][0]:4.2f},{r['pt_ci'][1]:4.2f}]")
    Ls=sorted(res)
    for i in range(len(Ls)-1):
        a,b=Ls[i],Ls[i+1]
        for lab,st in [('peak','pk_ci'),('peak/tail','pt_ci')]:
            ca,cb=res[a][st],res[b][st]; ov=not(ca[1]<cb[0] or cb[1]<ca[0])
            print(f"     {lab:9} L{a} vs L{b}: {'OVERLAP' if ov else 'DISJOINT'}")
