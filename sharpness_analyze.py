import numpy as np, os
rng=np.random.default_rng(20260809); B=20000

def half_crossings(al, c):
    """FWHM from half-max crossings around the argmax. Returns width or None if the
    curve does not fall to half-max on BOTH sides within the grid (too flat)."""
    kp=int(np.argmax(c)); half=c[kp]/2.0
    # left
    L=None
    for i in range(kp,0,-1):
        if c[i-1] < half:
            L=al[i-1]+(al[i]-al[i-1])*(half-c[i-1])/(c[i]-c[i-1]); break
    # right
    Rr=None
    for i in range(kp, len(c)-1):
        if c[i+1] < half:
            Rr=al[i]+(al[i+1]-al[i])*(c[i]-half)/(c[i]-c[i+1]); break
    return (Rr-L) if (L is not None and Rr is not None) else None

def analyze(npz):
    z=np.load(npz,allow_pickle=False); chi=z['chi']; al=z['alphas']; ns=chi.shape[0]
    itail=len(al)-1                      # alpha=0.35 tail reference
    H=np.empty(B); PT=np.empty(B); W=[]; nres=0
    for b in range(B):
        idx=rng.integers(0,ns,ns); m=chi[idx].mean(0)
        kp=int(np.argmax(m)); H[b]=m[kp]; PT[b]=m[kp]/m[itail]
        w=half_crossings(al,m)
        if w is not None: W.append(w); nres+=1
    W=np.array(W)
    return dict(ns=ns,
        H=np.median(H), H_ci=np.percentile(H,[2.5,97.5]),
        PT=np.median(PT), PT_ci=np.percentile(PT,[2.5,97.5]),
        frac_res=nres/B,
        W=(np.median(W) if len(W) else None),
        W_ci=(np.percentile(W,[2.5,97.5]) if len(W) else None))

print(f"Sharpness across L (T=200, grid [0.02,0.35], inside-resample bootstrap B={B})\n")
print(f"{'L':>4} {'ns':>3} | {'height':>18} | {'FWHM':>26} | {'peak/tail(a=0.35)':>22}")
print("-"*80)
res={}
for L in (10,32,64,128):
    f=f'processed_data/sharp_L{L}.npz'
    if not os.path.exists(f): continue
    r=analyze(f); res[L]=r
    hs=f"{r['H']:6.1f} [{r['H_ci'][0]:5.1f},{r['H_ci'][1]:5.1f}]"
    if r['W'] is not None and r['frac_res']>0.9:
        ws=f"{r['W']:.3f} [{r['W_ci'][0]:.3f},{r['W_ci'][1]:.3f}]"
    elif r['W'] is not None:
        ws=f"~{r['W']:.3f} (only {r['frac_res']*100:.0f}% resolvable)"
    else:
        ws="NO RESOLVABLE WIDTH"
    ps=f"{r['PT']:5.2f} [{r['PT_ci'][0]:4.2f},{r['PT_ci'][1]:4.2f}]"
    print(f"{L:>4} {r['ns']:>3} | {hs:>18} | {ws:>26} | {ps:>22}")
# peak-to-tail monotonic + disjoint?
print()
Ls=sorted(res)
for i in range(len(Ls)-1):
    a,b=Ls[i],Ls[i+1]; ca,cb=res[a]['PT_ci'],res[b]['PT_ci']
    ov=not(ca[1]<cb[0] or cb[1]<ca[0])
    print(f"  peak/tail L={a}[{ca[0]:.2f},{ca[1]:.2f}] vs L={b}[{cb[0]:.2f},{cb[1]:.2f}] -> {'OVERLAP' if ov else 'DISJOINT'}")
