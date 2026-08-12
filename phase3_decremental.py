"""Decremental (partially-regenerative) release variant — WRITTEN, NOT LAUNCHED (2026-08-12).
Ready to run if the refractory result is positive (synchrony -> propagation); this variant targets
the OTHER problem, unbounded extent.

Reference: MacDonald & Silva partial regeneration — release amplitude decays with each successive
regeneration step rather than being fully restored, so a focally-initiated front attenuates over a
characteristic distance instead of self-sustaining indefinitely.

Implementation (per-cell broadcast gain, keeps native D_eff so SPEED is preserved):
  Each cell broadcasts B = C_active * g, where g in [0,1] is a release gain. Coupling into a cell is
  the usual diff = D_eff * laplacian(B). At firing (rising edge of C_active>0.5) the cell's gain is
  (re)set to inherit a SUB-UNITY fraction of the coupling input that triggered it:
        g_new = min(1, g_floor + GAMMA_REGEN * (coupling_in / SCALE))
  - The focal-initiation patch is seeded with g=1 (full-strength source).
  - Downstream cell n is triggered by upstream broadcast ~ g_{n-1}, so g_n ~ GAMMA_REGEN * g_{n-1}
    => geometric decay with regeneration depth (~distance) => the front dies once g falls below what
    is needed to trigger the next cell. GAMMA_REGEN=1 -> full regeneration (self-sustaining, the
    current over-propagating model); GAMMA_REGEN<1 -> bounded extent.
  - g_floor is a small spontaneous-release term so noise-triggered events can seed local activity
    without indefinitely propagating (its fixed point sits below the firing threshold).

SWEEP: GAMMA_REGEN over [0.3 .. 1.0]; report focal front EXTENT (cells / um) and SPEED (um/s)
simultaneously at alpha=0.01, L=64, deterministic + noisy. TARGET: extent 2-5 cells (100-250 um)
WHILE speed stays near 13.3 um/s. If no GAMMA_REGEN gives both, report that (a finding).
Optionally combine with the refractory clamp (REF_ON) once the refractory result is in.

Run (when ready):  python phase3_decremental.py
"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from numba import njit
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED
from phase3_refractory import lag_vs_distance, focal_stats   # reuse discriminator + extent/speed

C_DOWN = -1.2
SCALE = 0.15          # normalizes coupling_in so a fully-driven cell reaches g~1 (expose/tune)
G_FLOOR = 0.02        # small spontaneous release; fixed point stays sub-threshold


@njit(fastmath=True, cache=False)
def run_decr(seed, alpha, baseline, gamma_regen, noise_on, do_focal, steps, nx, ny, sigma, stride, patch):
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    I0 = baseline + alpha * I0_base
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
    Deff = D0_base / (1.0 + (kappa_base * alpha) ** 4)
    theta = THETA_BASE + 0.7 * alpha
    sigma_eff = sigma * (1.0 + 4.0 * alpha)
    gdrive = gamma_base * alpha
    g = np.zeros((nx, ny))                                   # per-cell broadcast gain
    if do_focal:
        C = np.full((nx, ny), C_DOWN); h = (C + A_FHN) / B_FHN
        c0 = nx // 2; c1 = ny // 2
        for i in range(c0 - patch, c0 + patch + 1):
            for j in range(c1 - patch, c1 + patch + 1):
                C[i, j] = 1.5; g[i, j] = 1.0                 # seed source at full gain
    else:
        C = np.random.uniform(-0.1, 0.3, (nx, ny)); h = np.random.uniform(0.4, 1.2, (nx, ny))
    prev = np.zeros((nx, ny))
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    sqrt_dt = DT ** 0.5; fi = 0
    for t in range(steps):
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        broadcast = C_active * g                             # decremental broadcast
        coupling_in = Deff * laplacian(broadcast)
        active = (C_active > 0.5).astype(np.float64)
        rose = active * (1.0 - prev)                         # rising edge = firing
        # at firing, set gain from the coupling input that triggered it (sub-unity regeneration)
        g_new = np.minimum(1.0, G_FLOOR + gamma_regen * (coupling_in / SCALE))
        g = np.where(rose > 0.5, np.maximum(g_new, 0.0), g)
        if noise_on:
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        else:
            noise = np.zeros((nx, ny))
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        prev = active
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld


def main():
    L = 64; sig = SIGMA_EM_PREDICTED; baseline = 0.45; stride = 5; patch = 2; um = 50.0
    dtf = stride * DT; alpha = 0.01
    print(f"Decremental release, alpha={alpha}, L={L}. Target: extent 2-5 cells (100-250um) & speed ~13.3um/s.")
    print(f"  {'gamma_regen':>11} | focal-det: extent/speed/frac | focal-noisy: extent/speed/frac")
    for gr in (0.3, 0.5, 0.7, 0.85, 1.0):
        fd = run_decr(11, alpha, baseline, gr, False, True, int(100.0 / DT), L, L, sig, stride, patch)
        fn = run_decr(11, alpha, baseline, gr, True, True, int(100.0 / DT), L, L, sig, stride, patch)
        fr_d, r_d, v_d = focal_stats(fd, dtf, um, L)
        fr_n, r_n, v_n = focal_stats(fn, dtf, um, L)
        print(f"  {gr:11.2f} | {r_d:4.1f}cells {r_d*um:4.0f}um {v_d:5.1f}um/s {fr_d*100:3.0f}% "
              f"| {r_n:4.1f}cells {v_n:6.1f}um/s {fr_n*100:3.0f}%", flush=True)
    print("  window exists only if some gamma_regen gives extent 2-5 cells AND speed ~13.3um/s together.")


if __name__ == "__main__":
    main()
