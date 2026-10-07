"""
phase4_shared.py — test 6: does a SHARED fluctuating drive reproduce slice/in-vivo synchrony, with and without coupling?

Data constraints: drive-induced slice synchrony survives gap-junction block (test 5, LY379268 + CBX: coordination up in
9/9 slices); in vivo, astrocytes synchronize with running (a global noradrenergic input), 26/26 sessions. The published
model treats ATP as a uniform but STATIC level with independent noise per cell, and never synchronizes at its coupling.

Extension (a_c = 0 is exactly the published model): every ATP-dependent channel follows A(t) = max(0, A0 + a_c*xi(t)),
xi an Ornstein-Uhlenbeck process (tau 10 s, unit SD) SHARED by all cells, i.e. a uniform level that fluctuates in time.
Runs: a_c in {0, 0.05, 0.15} x coupling {model (d_mult 1), removed (d_mult 0, the CBX analogue)}, 21 levels of A0, 10 seeds.

Written BEFORE the runs. Prediction: rho-bar rises with a_c at the model coupling AND stays raised with coupling removed;
a_c = 0 reproduces the published decorrelation (rho-bar 0.056 -> ~0) and gives ~0 without coupling.
Run: python phase4_shared.py
"""
from __future__ import annotations
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
from numba import njit, prange
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result
from phase3_rhobar import rho_exact

ALPHAS = np.linspace(0.01, 1.11, 21)
STRIDE = 10
TAU_C = 10.0


@njit(fastmath=True, cache=True)
def sweep_s(seed, alpha_values, a_c, d_mult, steps, nx, ny, n, sigma, i0_baseline, stride):
    np.random.seed(seed)
    na = alpha_values.shape[0]
    act = np.zeros(na); rho = np.zeros(na)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    t_start = int(0.3 * steps)
    sqrt_dt = DT ** 0.5
    nsamp = (steps - t_start + stride - 1) // stride
    buf = np.zeros((nsamp, n))
    xi = 0.0
    ou_k = DT / TAU_C; ou_s = (2.0 * DT / TAU_C) ** 0.5
    for idx in range(na):
        a0 = alpha_values[idx]
        asum = 0.0; cnt = 0; k = 0
        for t in range(steps):
            xi += -xi * ou_k + ou_s * np.random.standard_normal()
            alpha = max(0.0, a0 + a_c * xi)
            I0 = i0_baseline + alpha * I0_base
            tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
            Deff = d_mult * D0_base / (1.0 + (kappa_base * alpha) ** 4)
            theta = THETA_BASE + 0.7 * alpha
            sigma_eff = sigma * (1.0 + 4.0 * alpha)
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
            C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
            diff = Deff * laplacian(C_active)
            dC = C - (C ** 3) / 3.0 - h + I0 + gamma_base * alpha + diff
            dh = (C + A_FHN - B_FHN * h) / tau_h
            C = C + DT * dC + sqrt_dt * noise
            h = h + DT * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)
            if t >= t_start:
                asum += np.mean(C_active); cnt += 1
                if (t - t_start) % stride == 0:
                    buf[k, :] = C.reshape(n); k += 1
        act[idx] = asum / cnt
        rho[idx] = rho_exact(buf[:k])
    return act, rho


@njit(parallel=True, cache=True)
def ensemble_s(seeds, alpha_values, a_c, d_mult, steps, nx, ny, n, sigma, i0_baseline, stride):
    ns = seeds.shape[0]; na = alpha_values.shape[0]
    A = np.zeros((ns, na)); P = np.zeros((ns, na))
    for i in prange(ns):
        a, p = sweep_s(seeds[i], alpha_values, a_c, d_mult, steps, nx, ny, n, sigma, i0_baseline, stride)
        A[i, :] = a; P[i, :] = p
    return A, P


def main(L=32, T=200.0, ns=10):
    steps = int(T / DT); seeds = np.arange(11, 11 + ns)
    for d_mult in (1.0, 0.0):
        for a_c in (0.0, 0.05, 0.15):
            t0 = time.time()
            A, P = ensemble_s(seeds, ALPHAS, a_c, d_mult, steps, L, L, L * L, SIGMA_EM_PREDICTED, I0_BASE, STRIDE)
            out = f"shared_ac{a_c:g}_dmult{d_mult:g}_L{L}.npz"
            save_result(Path("processed_data") / out,
                        {"L": L, "T": T, "seeds": seeds.tolist(), "a_c": a_c, "d_mult": d_mult, "tau_c": TAU_C,
                         "i0_baseline": I0_BASE, "stride": STRIDE, "alphas": ALPHAS.tolist()},
                        alphas=ALPHAS, active=A, rhobar=P)
            print(f"a_c={a_c:g} d_mult={d_mult:g} ({time.time()-t0:.0f}s) -> {out}", flush=True)
            for a, r, s_, q in zip(ALPHAS[::4], P.mean(0)[::4], P.std(0)[::4], A.mean(0)[::4]):
                print(f"   A0={a:.3f}  rho={r:.4f}+-{s_:.4f}  active={q:.3f}", flush=True)


if __name__ == "__main__":
    main()
