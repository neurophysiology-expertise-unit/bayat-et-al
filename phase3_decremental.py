"""Decremental (partially-regenerative) release test (2026-08-12) — the extent mechanism.
Refractory (tau_ref=15s, the value that best suppressed spurious nucleation) + decremental broadcast
gain, on FOCAL initiation at alpha=0.01, baseline=I0_BASE (core/model.py), L=64. Sweep the decay parameter GAMMA_REGEN.

Mechanism (MacDonald-Silva partial regeneration): broadcast B = C_active * g * (not refractory),
g in [0,1] a per-cell release gain reset at firing to inherit a SUB-UNITY fraction of the coupling
input that triggered it: g_new = min(1, G_FLOOR + GAMMA_REGEN * coupling_in/SCALE). Downstream gain
decays geometrically with regeneration depth (~distance) => front attenuates over a characteristic
length. GAMMA_REGEN=1 -> full regeneration (over-propagates); <1 -> bounded extent. Native D_eff kept
so SPEED is preserved. Report per GAMMA_REGEN: extent (cells/um), speed (um/s), linearity R^2 of
t_act vs radius. TARGET: extent 2-5 cells (100-250um) AND speed ~13 um/s AND linear, simultaneously.
Report the full extent-vs-decay curve (monotone = mechanism works even if the window is narrow).
Run: python phase3_decremental.py
"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from numba import njit
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)

C_DOWN = -1.2; K_REF = 20.0; SCALE = 0.15; G_FLOOR = 0.02; UM = 50.0


@njit(fastmath=True, cache=False)
def run_dr(seed, alpha, baseline, gamma_regen, tau_ref, noise_on, steps, nx, ny, sigma, stride, patch):
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
    C = np.full((nx, ny), C_DOWN); h = (C + A_FHN) / B_FHN
    g = np.zeros((nx, ny)); reft = np.zeros((nx, ny)); prev = np.zeros((nx, ny))
    c0 = nx // 2; c1 = ny // 2
    for i in range(c0 - patch, c0 + patch + 1):
        for j in range(c1 - patch, c1 + patch + 1):
            C[i, j] = 1.5; g[i, j] = 1.0
    # seed the seed patch as already-active so t=0 is NOT a rising edge (keeps its g=1 source gain)
    prev = (0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5).astype(np.float64)
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    sqrt_dt = DT ** 0.5; fi = 0
    for t in range(steps):
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        active = (C_active > 0.5).astype(np.float64)
        inref = (reft > 0.0).astype(np.float64)
        broadcast = C_active * g * (1.0 - inref)             # decremental + refractory gate
        coupling_in = Deff * laplacian(broadcast)
        rose = active * (1.0 - prev)
        fell = prev * (1.0 - active)
        g_new = np.minimum(1.0, G_FLOOR + gamma_regen * (coupling_in / SCALE))
        g = np.where(rose > 0.5, np.maximum(g_new, 0.0), g)  # set gain at firing (sub-unity inherit)
        reft = np.where(fell > 0.5, tau_ref, reft)           # enter refractory when spike ends
        clamp = K_REF * inref * (C - C_DOWN)
        if noise_on:
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        else:
            noise = np.zeros((nx, ny))
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - DT, 0.0)
        prev = active
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld


def focal_lin(fld, dtf, L, patch):
    """extent(cells), speed(um/s), linearity R^2 of t_act vs radius, excluding the seed patch."""
    T = fld.shape[0]; c = L // 2
    act = fld > 0.5
    tact = np.where(act.any(0), act.argmax(0), -1)
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c)); dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    reached = (tact >= 0) & (r > patch)                      # exclude seeded patch
    if reached.sum() < 6:
        return float(patch), np.nan, np.nan, reached.mean()
    extent = float(r[reached].max())
    rr = r[reached]; tt = tact[reached].astype(float) * dtf
    band = rr <= extent                                       # whole propagated region
    speed = np.nan; r2 = np.nan
    if band.sum() > 6 and np.ptp(tt[band]) > 1e-6 and np.ptp(rr[band]) > 0.5:
        m, b0 = np.polyfit(rr[band], tt[band], 1)             # t_act vs radius
        yh = m * rr[band] + b0
        r2 = 1.0 - np.sum((tt[band] - yh) ** 2) / (np.sum((tt[band] - tt[band].mean()) ** 2) + 1e-12)
        if m > 1e-6:
            speed = (1.0 / m) * UM
    return extent, speed, r2, reached.mean()


def main():
    L = 64; sig = SIGMA_EM_PREDICTED; baseline = I0_BASE; stride = 5; patch = 2; dtf = stride * DT
    alpha = 0.01; tau_ref = 15.0; steps = int(120.0 / DT)
    GAMMAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.20, 0.15, 0.10, 0.05]
    print(f"Decremental release + refractory(tau=15s), focal alpha={alpha}, baseline={baseline}, L={L}.")
    print(f"Target: extent 2-5 cells (100-250um) AND speed ~13um/s AND linear, together.\n")
    print(f"  {'gamma_regen':>11} | {'extent(cells)':>13} {'extent(um)':>10} {'speed(um/s)':>11} "
          f"{'linear R^2':>10} {'frac':>6} | {'noisy extent/speed':>19}")
    for gr in GAMMAS:
        t0 = time.time()
        fd = run_dr(11, alpha, baseline, gr, tau_ref, False, steps, L, L, sig, stride, patch)
        fn = run_dr(11, alpha, baseline, gr, tau_ref, True, steps, L, L, sig, stride, patch)
        ext, spd, r2, fr = focal_lin(fd, dtf, L, patch)
        extn, spdn, r2n, frn = focal_lin(fn, dtf, L, patch)
        intarget = (2 <= ext <= 5) and (spd == spd) and (8 <= spd <= 20) and (r2 == r2 and r2 > 0.9)
        flag = "  <-- TARGET" if intarget else ""
        print(f"  {gr:11.2f} | {ext:13.1f} {ext*UM:10.0f} {spd:11.1f} {r2:10.3f} {fr*100:5.0f}% "
              f"| {extn:5.1f}c {spdn:7.1f} ({time.time()-t0:.0f}s){flag}", flush=True)
    print("\n  Report: is there a gamma_regen with extent 2-5 cells AND speed ~13 AND R^2>0.9 together?")
    print("  Extent-vs-gamma monotone => mechanism works as intended (even if physiological window narrow).")


if __name__ == "__main__":
    main()
