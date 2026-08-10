"""Extended-grid rerun: dense at low alpha to locate the interior peak (or show it
keeps hitting the boundary = I0=1/sqrt(alpha) divergence, not a transition)."""
import sys, time; sys.path.insert(0,'/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from phase1_finite_size import sweep_ensemble, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path
# dense geometric low-alpha coverage from 0.005, then linear to the 0.35 tail
GRID = np.concatenate([np.geomspace(0.005, 0.05, 12), np.linspace(0.062, 0.35, 9)])
L=int(sys.argv[1]); NS=int(sys.argv[2]) if len(sys.argv)>2 else 40
T=200.0; steps=int(T/DT); seeds=np.arange(11,11+NS); sig=SIGMA_EM_PREDICTED
t0=time.time()
Sc,chi_ext,chi_true,R=sweep_ensemble(seeds,False,GRID,steps,L,L,L*L,sig,-1.0,False)
p=save_result(Path("processed_data")/f"sharp_ext_L{L}.npz",
              {"L":L,"n_seeds":NS,"T":T,"steps":steps,"sigma":sig,
               "grid":"geom[0.005,0.05]+lin[0.062,0.35]","alphas":GRID.tolist()},
              alphas=GRID,chi=chi_ext,chi_true=chi_true,R=R,Sc=Sc)
k=int(chi_true.mean(0).argmax())
print(f"L={L}: alpha0(chi_true)={GRID[k]:.4f} {'BOUNDARY' if k==0 else 'interior'} "
      f"({time.time()-t0:.0f}s) -> {p.name}",flush=True)
