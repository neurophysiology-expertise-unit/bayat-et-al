"""
phase2_2_decomp.py — Phase 2.2 channel decomposition (RUN ONLY AFTER 2.1 lands).

Answers the anticipated referee objection to a separation result — "of course adding
drive and noise raises activity" — by showing WHICH channels raise activity and whether
any single channel reproduces the distinct (active-but-fragmented) regime, or whether it
needs the combination.

General sweep with a per-channel active mask over the six ATP channels:
  0 gamma  (excitability, gamma*alpha)   3 theta  (transmission threshold)
  1 sigma  (noise, sigma_eff)            4 I0     (baseline, 1/sqrt(alpha))
  2 Deff   (coupling)                    5 tau_h  (recovery)
mask[c]=1 -> channel c follows alpha; mask[c]=0 -> held at alpha_ref.

  leave-one-out (LOO_no<c>): all channels follow alpha except c (c held at alpha_ref)
  one-at-a-time (OAT_only<c>): only channel c follows alpha; the other five held at alpha_ref

All at L=64, alpha_ref=0.10, 40 seeds, T=200, submitted model (I0=1/sqrt(alpha)).
Per-alpha observables: active fraction, R (coordination), mean D_eff.

Run (only after 2.1):  python phase2_2_decomp.py
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from numba import njit, prange
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED

CHANNELS = ["gamma", "sigma", "Deff", "theta", "I0", "tauh"]
ALPHAS = np.linspace(0.01, 1.11, 21)


@njit(fastmath=True, cache=True)
def sweep_mask(seed, mask, alpha_values, alpha_ref, steps, nx, ny, n, sigma):
    np.random.seed(seed)
    na = alpha_values.shape[0]
    act = np.zeros(na); Rs = np.zeros(na); dm = np.zeros(na)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    t_start = int(0.3 * steps); sqrt_dt = DT ** 0.5
    for idx in range(na):
        a = alpha_values[idx]
        aG = a if mask[0] else alpha_ref
        aS = a if mask[1] else alpha_ref
        aD = a if mask[2] else alpha_ref
        aT = a if mask[3] else alpha_ref
        aI = a if mask[4] else alpha_ref
        aH = a if mask[5] else alpha_ref
        I0 = 0.05 + (1.0 / np.sqrt(aI)) * I0_base
        tau_h = 10.0 / ((1.0 + 0.8 * aH) * tau_base)
        Deff = D0_base / (1.0 + (kappa_base * aD) ** 4)
        theta = THETA_BASE + 0.7 * aT
        sigma_eff = sigma * (1.0 + 4.0 * aS)
        gdrive = gamma_base * aG
        asum = 0.0; msum = 0.0; msum2 = 0.0; cnt = 0
        csum = np.zeros(n); csum2 = np.zeros(n)
        for t in range(steps):
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
            Ca = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
            diff = Deff * laplacian(Ca)
            dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff
            dh = (C + A_FHN - B_FHN * h) / tau_h
            C = C + DT * dC + sqrt_dt * noise
            h = h + DT * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)
            if t >= t_start:
                asum += np.mean(Ca)
                Cf = C.reshape(n); mm = np.mean(Cf)
                csum += Cf; csum2 += Cf * Cf; msum += mm; msum2 += mm * mm; cnt += 1
        act[idx] = asum / cnt
        var_m = msum2 / cnt - (msum / cnt) ** 2
        mean_var_i = np.mean(csum2 / cnt - (csum / cnt) ** 2)
        r2 = var_m / (mean_var_i + 1e-12)
        Rs[idx] = np.sqrt(r2) if r2 > 0.0 else 0.0
        dm[idx] = np.mean(Deff)
    return act, Rs, dm


@njit(parallel=True, fastmath=True, cache=True)
def ensemble_mask(seeds, mask, alpha_values, alpha_ref, steps, nx, ny, n, sigma):
    ns = seeds.shape[0]; na = alpha_values.shape[0]
    A = np.zeros((ns, na)); R = np.zeros((ns, na)); D = np.zeros((ns, na))
    for i in prange(ns):
        a, r, d = sweep_mask(seeds[i], mask, alpha_values, alpha_ref, steps, nx, ny, n, sigma)
        A[i, :] = a; R[i, :] = r; D[i, :] = d
    return A, R, D


def main():
    from core.provenance import save_result
    L = 32; aref = 0.10; ns = 40   # L=32 first (separation was stark at L=64); 40 seeds kept
    T = 200.0; steps = int(T / DT); seeds = np.arange(11, 11 + ns); sig = SIGMA_EM_PREDICTED
    jobs = [("A_full", np.ones(6, np.int8))]                                    # reference: all follow alpha
    for c in range(6):
        m = np.ones(6, np.int8); m[c] = 0; jobs.append((f"LOO_no{CHANNELS[c]}", m))     # leave-one-out
    for c in range(6):
        m = np.zeros(6, np.int8); m[c] = 1; jobs.append((f"OAT_only{CHANNELS[c]}", m))   # one-at-a-time
    for tag, mask in jobs:
        t0 = time.time()
        A, R, D = ensemble_mask(seeds, mask, ALPHAS, aref, steps, L, L, L * L, sig)
        p = save_result(Path("processed_data") / f"phase2_2_L{L}_{tag}.npz",
                        {"L": L, "mask": mask.tolist(), "channels": CHANNELS, "alpha_ref": aref,
                         "n_seeds": ns, "T": T, "steps": steps, "sigma": sig,
                         "alphas": ALPHAS.tolist()},
                        alphas=ALPHAS, active=A, R=R, Deff=D)
        print(f"{tag}: active max {A.mean(0).max():.3f} ({time.time()-t0:.0f}s) -> {p.name}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
