"""Per-unit-resolved const-I0 Sweep A, for the two mechanism tests (new_plan 2026-08-11):
 (test 1) partition activity by each unit's own Hopf status (past-onset vs sub-threshold)
 (test 2) confound-breaker: gamma_scale=-1 makes drive (and f_det) FALL with alpha while
          sigma_eff=sigma(1+4a) still RISES -> does activity follow f_det or the noise?

Returns per-unit time-averaged C_active (na x N) plus the per-unit params (gamma_eff, i0c,
tau_base) needed to compute each unit's single-unit Hopf onset. Draw order and inner loop are
bit-matched to phase2_coupling.sweep_p2 (const path), so unit-mean activity == the aggregate A_act.
Run: python phase3_perunit_run.py
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit, prange
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from phase2_coupling import ALPHAS
from pathlib import Path


@njit(fastmath=True, cache=False)
def sweep_perunit(seed, alpha_values, steps, nx, ny, sigma, gamma_scale):
    np.random.seed(seed)
    na = alpha_values.shape[0]; n = nx * ny
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))      # const path uses this range
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    t_start = int(0.3 * steps); sqrt_dt = DT ** 0.5
    act_field = np.zeros((na, n))
    for idx in range(na):
        a = alpha_values[idx]
        I0 = 0.05 + I0_base                                 # const, ATP-independent
        tau_h = 10.0 / ((1.0 + 0.8 * a) * tau_base)
        Deff = D0_base / (1.0 + (kappa_base * a) ** 4)
        theta = THETA_BASE + 0.7 * a
        sigma_eff = sigma * (1.0 + 4.0 * a)
        gdrive = gamma_base * gamma_scale * a
        cact = np.zeros((nx, ny)); cnt = 0
        for t in range(steps):
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
            C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
            diff = Deff * laplacian(C_active)
            dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff
            dh = (C + A_FHN - B_FHN * h) / tau_h
            C = C + DT * dC + sqrt_dt * noise
            h = h + DT * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)
            if t >= t_start:
                cact += C_active; cnt += 1
        act_field[idx, :] = (cact / cnt).reshape(n)
    return act_field, (gamma_base * gamma_scale).reshape(n), (0.05 + I0_base).reshape(n), tau_base.reshape(n)


@njit(parallel=True, fastmath=True, cache=False)
def ensemble_perunit(seeds, alpha_values, steps, nx, ny, sigma, gamma_scale):
    ns = seeds.shape[0]; na = alpha_values.shape[0]; n = nx * ny
    AF = np.zeros((ns, na, n)); GE = np.zeros((ns, n)); I0c = np.zeros((ns, n)); TB = np.zeros((ns, n))
    for i in prange(ns):
        af, ge, i0c, tb = sweep_perunit(seeds[i], alpha_values, steps, nx, ny, sigma, gamma_scale)
        AF[i] = af; GE[i] = ge; I0c[i] = i0c; TB[i] = tb
    return AF, GE, I0c, TB


def main():
    L = 32; ns = 40; T = 200.0; steps = int(T / DT); seeds = np.arange(11, 11 + ns)
    sig = SIGMA_EM_PREDICTED
    for tag, gscale in [("baseline", 1.0), ("gammaflip", -1.0)]:
        t0 = time.time()
        AF, GE, I0c, TB = ensemble_perunit(seeds, ALPHAS, steps, L, L, sig, gscale)
        p = save_result(Path("processed_data") / f"phase3_perunit_{tag}.npz",
                        {"L": L, "n_seeds": ns, "T": T, "steps": steps, "sigma": sig,
                         "gamma_scale": gscale, "alphas": ALPHAS.tolist()},
                        alphas=ALPHAS, act_field=AF, gamma_eff=GE, i0c=I0c, tau_base=TB)
        print(f"{tag} (gamma_scale={gscale}): A_act max {AF.mean((0,2)).max():.4f} "
              f"({time.time()-t0:.0f}s) -> {p.name}", flush=True)


if __name__ == "__main__":
    main()
