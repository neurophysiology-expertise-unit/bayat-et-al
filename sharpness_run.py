"""Run the healthy alpha-sweep at one L and save per-seed curves for ALL observables:
   chi_ext = N*Var_t(max-min)  (range statistic, discredited as non-intensive)
   chi_true= N*Var_t(Cbar)     (proper finite-size-scaling susceptibility)
   R       = Golomb-Rinzel synchrony ;  S_C = spatial heterogeneity
Usage: python sharpness_run.py <L> [n_seeds]"""
import sys, time; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np
from phase1_finite_size import sweep_ensemble, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path
WIDE = np.linspace(0.02, 0.35, 17)
L = int(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
T = 200.0; steps = int(T/DT); seeds = np.arange(11, 11+NS); sig = SIGMA_EM_PREDICTED
t0 = time.time()
Sc, chi_ext, chi_true, R = sweep_ensemble(seeds, False, WIDE, steps, L, L, L*L, sig, -1.0, False)
params = {"L": L, "n_seeds": NS, "T": T, "steps": steps, "sigma": sig,
          "grid": "wide[0.02,0.35]", "alphas": WIDE.tolist()}
p = save_result(Path("processed_data")/f"sharp_L{L}.npz", params,
                alphas=WIDE, chi=chi_ext, chi_true=chi_true, R=R, Sc=Sc)
print(f"L={L} ns={NS}: shapes chi_ext{chi_ext.shape} chi_true{chi_true.shape} "
      f"R{R.shape} Sc{Sc.shape} ({time.time()-t0:.0f}s) -> {p.name}", flush=True)
