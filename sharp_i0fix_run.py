"""I0-FIXED recompute (CORRECTED model, not the submitted one): I0 held at alpha=0.175
so the 1/sqrt(alpha) low-alpha divergence is removed. Extended grid to 0.005.
Checks: (a) is chi_true max interior? (b) peak and tail reported separately."""
import sys, time; sys.path.insert(0,'/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from phase1_finite_size import sweep_ensemble, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path
GRID=np.concatenate([np.geomspace(0.005,0.05,12), np.linspace(0.062,0.35,9)])
I0REF=0.175
L=int(sys.argv[1]); NS=int(sys.argv[2]) if len(sys.argv)>2 else 40
T=200.0; steps=int(T/DT); seeds=np.arange(11,11+NS); sig=SIGMA_EM_PREDICTED
t0=time.time()
Sc,chi_ext,chi_true,R=sweep_ensemble(seeds,False,GRID,steps,L,L,L*L,sig,I0REF,False)
p=save_result(Path("processed_data")/f"sharp_i0fix_L{L}.npz",
   {"L":L,"n_seeds":NS,"T":T,"steps":steps,"sigma":sig,"i0_ref":I0REF,
    "grid":"geom[0.005,0.05]+lin[0.062,0.35]","model":"I0-FIXED (corrected)","alphas":GRID.tolist()},
   alphas=GRID,chi=chi_ext,chi_true=chi_true,R=R,Sc=Sc)
m=chi_true.mean(0); k=int(m.argmax())
print(f"L={L}: chi_true alpha0={GRID[k]:.4f} {'BOUNDARY(!)' if k==0 else 'INTERIOR'} "
      f"peak={m[k]:.3g} tail(0.35)={m[-1]:.3g} ({time.time()-t0:.0f}s)",flush=True)
