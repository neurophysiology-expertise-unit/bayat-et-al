import sys, time; sys.path.insert(0,'/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from phase1_finite_size import sweep_ensemble, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path
WIDE = np.linspace(0.02, 0.35, 17)          # widened grid, no far tail; 0.35 is the tail ref
L = int(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv)>2 else 40
T = 200.0; steps = int(T/DT); seeds = np.arange(11, 11+NS); sig = SIGMA_EM_PREDICTED
t0=time.time()
_, chi, R = sweep_ensemble(seeds, False, WIDE, steps, L, L, L*L, sig, -1.0, False)
params={"L":L,"n_seeds":NS,"T":T,"steps":steps,"sigma":sig,"grid":"wide[0.02,0.35]",
        "alphas":WIDE.tolist()}
p=save_result(Path("processed_data")/f"sharp_L{L}.npz", params, alphas=WIDE, chi=chi, R=R)
print(f"L={L} ns={NS}: chi shape={chi.shape} ({time.time()-t0:.0f}s) -> {p.name}", flush=True)
