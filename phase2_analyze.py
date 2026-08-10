"""Phase 2 collapse test, both observables x both controls x both alpha_ref.
Coordination = COVARIANCE-NORMALIZED synchrony = (N*R^2-1)/(N-1) from the saved R.
This is EXACT (validated vs brute-force to 5e-13), NOT the mean correlation coefficient:
it deviates from the true coefficient by <=0.03 (L=10 check), roughly A/B-symmetric, so the
A-vs-B collapse verdict is robust unless it lands near the 50%/90% threshold (then recompute
the true coefficient at O(N^2) for that comparison).
The dissociation claim needs BOTH: activity SEPARATES (low % inside) while coordination
does NOT (high % inside). If A separates from B but not from B', theta carries it and the
editor's objection survives - stated plainly."""
import numpy as np, os
rng=np.random.default_rng(20260810); B=20000; L=64; N=L*L
def rho(Rarr): return (N*Rarr**2 - 1.0)/(N-1.0)
def load(f):
    z=np.load(f,allow_pickle=False); return z['active'], rho(z['R']), z['alphas']
def band(X):
    ns,na=X.shape; lo=np.zeros(na); hi=np.zeros(na)
    for j in range(na):
        bs=np.array([X[rng.integers(0,ns,ns),j].mean() for _ in range(B)])
        lo[j],hi[j]=np.percentile(bs,[2.5,97.5])
    return lo,hi
def pct_inside(mA, X): lo,hi=band(X); return 100.0*np.mean((mA>=lo)&(mA<=hi))

fA=f'processed_data/phase2_L{L}_A_full.npz'
if not os.path.exists(fA): print("Sweep A not present yet"); raise SystemExit
actA,rhoA,al=load(fA); mActA=actA.mean(0); mRhoA=rhoA.mean(0)
print(f"Phase 2 collapse test (L={L}). %% of D_eff-matched points where Sweep A's mean lies")
print(f"inside the control's 95%% bootstrap band. Pre-reg: >=90%% COLLAPSE, <=50%% SEPARATION.\n")
print(f"{'alpha_ref':>9} {'control':>22} | {'active %inside':>14} | {'coord(synchrony) %inside':>24}")
print('-'*66)
rows={}
for aref in (0.10,0.90):
    for tag,lab in [(f'B_coupling_aref{aref:.2f}','B (coupling only)'),
                    (f'Bprime_coupθ_aref{aref:.2f}',"B' (coupling+theta)")]:
        f=f'processed_data/phase2_L{L}_{tag}.npz'
        if not os.path.exists(f): print(f"{aref:>9} {lab:>22} | pending"); continue
        actB,rhoB,_=load(f)
        pa=pct_inside(mActA, actB); pc=pct_inside(mRhoA, rhoB)
        rows[(aref,lab)]=(pa,pc)
        va="SEP" if pa<=50 else ("COLL" if pa>=90 else "AMB")
        vc="SEP" if pc<=50 else ("COLL" if pc>=90 else "AMB")
        print(f"{aref:>9} {lab:>22} | {pa:6.1f}% [{va:4}] | {pc:6.1f}% [{vc:3}]")
print("\nDISSOCIATION requires: active SEPARATES (low %) AND coordination COLLAPSES (high %).")
print("If active separates from B but NOT from B', theta carries the effect -> editor's objection survives.")
