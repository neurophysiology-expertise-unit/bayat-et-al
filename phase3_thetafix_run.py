"""Phase 3.1 readout test: const-I0 Sweep A with theta FROZEN at its alpha=0 value (0.5),
everything else following alpha. Same seeds as phase2v3_const A_full. If the high-alpha
A_act saturation vanishes here (A_act tracks f_det linearly to 1.11), the saturation was a
readout artifact of the alpha-dependent threshold theta=0.5+0.7*alpha, not the dynamics.
Run: python phase3_thetafix_run.py"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from phase2_coupling import ensemble_p2, ALPHAS, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path
L = 32; ns = 40; T = 200.0; steps = int(T/DT); seeds = np.arange(11, 11+ns); sig = SIGMA_EM_PREDICTED
t0 = time.time()
# mode 0 (Sweep A, all channels follow alpha), i0_form=3 (const), theta_ovr=0.0 (theta=0.5 fixed)
A, R, D = ensemble_p2(seeds, 0, ALPHAS, 0.10, steps, L, L, L*L, sig, 3, 0.0)
p = save_result(Path("processed_data")/"phase3_thetafix_const_A.npz",
                {"L": L, "mode": 0, "i0_form": 3, "theta_ovr": 0.0, "n_seeds": ns, "T": T,
                 "steps": steps, "sigma": sig, "alphas": ALPHAS.tolist()},
                alphas=ALPHAS, active=A, R=R, Deff=D)
print(f"theta-fixed const Sweep A: active max {A.mean(0).max():.4f} @a={ALPHAS[A.mean(0).argmax()]:.3f} "
      f"({time.time()-t0:.0f}s) -> {p.name}", flush=True)
