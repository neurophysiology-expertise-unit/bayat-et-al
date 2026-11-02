"""Baseline-excitability test (2026-08-12): does raising the ATP-independent I0 baseline recover
waves? Bayat-I0 alpha-dependence (I0 = baseline + alpha*U(0.1,0.5)) with the baseline constant
swept. Full alpha-sweep, Sweep A (full model, mode 0), L=32, 40 seeds. Report active & rho-bar(alpha).
Hypothesis: a baseline placing cells near-but-below threshold -> low alpha (strong D_eff, noise-
triggered firing) gives coordinated propagation; high alpha (drive-dominated, collapsed D_eff)
fragments. Run: python phase3_baseline_run.py"""
import time
import numpy as np
from phase2_coupling import ensemble_p2, ALPHAS, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path

L = 32; ns = 40; T = 200.0; steps = int(T / DT); seeds = np.arange(11, 11 + ns); sig = SIGMA_EM_PREDICTED
for base in (0.05, 0.15, 0.25, 0.35, 0.45):
    t0 = time.time()
    A, R, D = ensemble_p2(seeds, 0, ALPHAS, 0.10, steps, L, L, L * L, sig, 1, -1.0, 1.0, base)
    p = save_result(Path("processed_data") / f"phase3_baseline_b{base:.2f}.npz",
                    {"L": L, "mode": 0, "i0_form": 1, "i0_baseline": base, "n_seeds": ns, "T": T,
                     "steps": steps, "sigma": sig, "alphas": ALPHAS.tolist()},
                    alphas=ALPHAS, active=A, R=R, Deff=D)
    print(f"baseline={base:.2f}: active max {A.mean(0).max():.3f} ({time.time()-t0:.0f}s) -> {p.name}", flush=True)
