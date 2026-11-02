"""Option 3 (new_plan 2026-08-11): can a DEFENSIBLE model put activity where coupling lives?
bayat-I0 Sweep A (mode 0, all channels follow alpha), L=32, 40 seeds, testing whether the
wave-killing disjointness is due to theta's (unjustified) alpha-dependence rather than the I0 fix.

  cond1 theta=0.5 fixed, D_eff still alpha-suppressed (kappa_scale=1.0)
  cond2 theta=0.5 fixed AND kappa_scale=0.25 so mean D_eff stays >0.1 across the window
  (baseline = existing phase2v1_bayat_L32_A_full.npz, both alpha-dependent)

Run: python phase3_opt3_run.py
"""
import time
import numpy as np
from phase2_coupling import ensemble_p2, ALPHAS, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path

L = 32; ns = 40; T = 200.0; steps = int(T / DT); seeds = np.arange(11, 11 + ns); sig = SIGMA_EM_PREDICTED
jobs = [("theta05", 0.0, 1.0), ("theta05_kap025", 0.0, 0.25)]
for tag, tov, ks in jobs:
    t0 = time.time()
    A, R, D = ensemble_p2(seeds, 0, ALPHAS, 0.10, steps, L, L, L * L, sig, 1, tov, ks)
    p = save_result(Path("processed_data") / f"phase3_opt3_bayat_{tag}.npz",
                    {"L": L, "mode": 0, "i0_form": 1, "theta_ovr": tov, "kappa_scale": ks,
                     "n_seeds": ns, "T": T, "steps": steps, "sigma": sig, "alphas": ALPHAS.tolist()},
                    alphas=ALPHAS, active=A, R=R, Deff=D)
    print(f"{tag}: act max {A.mean(0).max():.4f}  Deff@hi {D.mean(0)[-1]:.3f}  "
          f"({time.time()-t0:.0f}s) -> {p.name}", flush=True)
