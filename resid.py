import numpy as np
from scipy.optimize import curve_fit
rng=np.random.default_rng(1); B=4000
def pl(a,A,p,c): return A*a**(-p)+c
def analyze(al, curve):
    lo = al < 0.08                                   # transition contributes nothing here
    try:
        popt,_=curve_fit(pl, al[lo], curve[lo], p0=[curve[0]*al[0]**0.5,0.5,curve[-1]], maxfev=20000)
    except Exception: return None
    resid = curve - pl(al,*popt)
    interior = (al>=0.098)&(al<=0.31)
    ai=al[interior]; ri=resid[interior]
    kloc=int(np.argmax(ri))
    return dict(popt=popt, resid=resid, a_max=ai[kloc], r_max=ri[kloc],
                interior=interior, prominence=ri[kloc]-np.median(ri))
print("Residual after subtracting low-alpha power-law trend  (a*alpha^-p + c fit to alpha<0.08)\n")
for key in ('chi_true','R'):
    print(f"===== {key} =====")
    for L in (10,32,64):
        z=np.load(f'processed_data/sharp_ext_L{L}.npz',allow_pickle=False); al=z['alphas']; A=z[key]
        m=A.mean(0)
        r=analyze(al,m)
        if r is None: print(f"  L={L}: powerlaw fit failed"); continue
        # bootstrap prominence
        proms=[]
        for b in range(B):
            mm=A[rng.integers(0,A.shape[0],A.shape[0])].mean(0)
            rr=analyze(al,mm)
            if rr: proms.append(rr['prominence'])
        proms=np.array(proms); lo,hi=np.percentile(proms,[2.5,97.5])
        A_,p_,c_=r['popt']
        print(f"  L={L:2d}: powerlaw p={p_:.2f} c={c_:.2f} | interior residual max at alpha={r['a_max']:.3f} "
              f"prominence={r['prominence']:.3f} CI[{lo:.3f},{hi:.3f}] {'(>0)' if lo>0 else '(spans 0)'}")
    print()
# second derivative of log chi_true wrt log alpha (shoulder finder), L=64
z=np.load('processed_data/sharp_ext_L64.npz',allow_pickle=False); al=z['alphas']; m=z['chi_true'].mean(0)
la=np.log(al); lc=np.log(m); d1=np.gradient(lc,la); d2=np.gradient(d1,la)
print("d^2(log chi_true)/d(log alpha)^2  L=64 (power-law => flat ~0; shoulder => bump):")
for a,x in zip(al,d2):
    if a>=0.02: print(f"    alpha={a:.3f}  d2={x:+.2f}")
