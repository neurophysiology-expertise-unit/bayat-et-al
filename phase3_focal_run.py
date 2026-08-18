"""Focal-initiation wave test (the experimental protocol: stimulate one spot, watch it spread).
The spontaneous-field test showed zero-lag synchrony, but noise nucleates everywhere at once so it
cannot reveal a front even if the medium supports one. Here we initialize the lattice at rest,
activate a central patch at t=0, and measure activation-time vs radius:
  front propagates -> t_act rises ~linearly with distance (speed = dr/dt);
  no propagation    -> only the patch (+ noise) activates, t_act flat/undefined beyond it.
Deterministic (noise off) for a clean speed, plus noisy for realism. baseline=I0_BASE, L=64,
alpha in {0.01, 0.12}. 1 cell=50um, 1 model-t=1s; measured wave speed 15-28 um/s (0.3-0.56 cells/s).
Run: python phase3_focal_run.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)


@njit(fastmath=True, cache=False)
def focal(seed, alpha, baseline, steps, nx, ny, sigma, noise_on, patch, stride):
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
    # initialize at DOWN state (below threshold), uniform
    C = np.full((nx, ny), -1.2)
    h = (C + A_FHN) / B_FHN
    c0 = nx // 2; c1 = ny // 2                              # activate central patch
    for i in range(c0 - patch, c0 + patch + 1):
        for j in range(c1 - patch, c1 + patch + 1):
            C[i, j] = 1.5
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    sqrt_dt = DT ** 0.5; fi = 0
    for t in range(steps):
        if noise_on:
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        else:
            noise = np.zeros((nx, ny))
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        diff = Deff * laplacian(C_active)
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld


def activation_analysis(fld, dt_frame, um, tag):
    T, L, _ = fld.shape
    c = L // 2
    # first frame each cell exceeds 0.5 activation
    act = fld > 0.5
    tact = np.full((L, L), -1)
    for i in range(L):
        for j in range(L):
            w = np.where(act[:, i, j])[0]
            if len(w):
                tact[i, j] = w[0]
    # distance from center (respect periodicity)
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c))
    dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    reached = tact >= 0
    frac = reached.mean()
    print(f"  [{tag}] fraction of lattice ever activated: {frac*100:.1f}%")
    # activation time vs radius (binned), for cells that activated
    print(f"    {'radius(cells)':>13} {'mean t_act(frames)':>18} {'t_act(model-t=s)':>16} {'n':>5}")
    prev_t = None; speeds = []
    for rlo in range(0, L // 2, 3):
        m = reached & (r >= rlo) & (r < rlo + 3)
        if m.sum() < 3:
            continue
        mt = tact[m].mean()
        rc = rlo + 1.5
        print(f"    {rc:13.1f} {mt:18.1f} {mt*dt_frame:16.3f} {int(m.sum()):5d}")
    # front speed: linear fit t_act(frames) vs radius over the propagating band
    rr = r[reached]; tt = tact[reached].astype(float)
    band = (rr >= 2) & (rr <= L // 2 - 2)
    if band.sum() > 20 and np.ptp(tt[band]) > 0:
        slope = np.polyfit(rr[band], tt[band] * dt_frame, 1)[0]   # model-t per cell
        if slope > 1e-6:
            v_cells = 1.0 / slope                                  # cells per model-t
            print(f"    front speed = {v_cells:.3f} cells/s = {v_cells*um:.1f} um/s "
                  f"(measured 15-28){'  IN RANGE' if 15 <= v_cells*um <= 28 else '  OUT OF RANGE'}")
        else:
            print("    t_act ~ flat vs radius => no coherent front (near-instant/global activation)")
    else:
        print("    front did not propagate beyond the patch (activation stayed local or died)")


def main():
    L = 64; T = 100.0; steps = int(T / DT); sig = SIGMA_EM_PREDICTED
    baseline = I0_BASE; stride = 5; dt_frame = stride * DT; um = 50.0; patch = 2
    for alpha in (0.01, 0.12):
        for noise_on, ntag in ((False, "deterministic"), (True, "noisy")):
            fld = focal(11, alpha, baseline, steps, L, L, sig, noise_on, patch, stride)
            activation_analysis(fld, dt_frame, um, f"a={alpha:.2f} {ntag}")
        print()


if __name__ == "__main__":
    main()
