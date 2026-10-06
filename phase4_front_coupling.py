"""
phase4_front_coupling.py — does the evoked front still match Bowser at the stronger coupling that synchrony needs?

phase4_constrain (map) showed drive-induced synchrony only at 2-4x the model's coupling. Coupling also sets the
front, so the focal front with refractory state + decremental release (phase3_decremental.run_dr, tau_ref 15 s,
A = 0.01, deterministic, L = 64, 120 s) is rerun at d_mult = 1, 2, 4 over gamma_regen, 10 seeds each.
In the decremental scheme the per-step gain is gamma_regen*D_eff/D_scale, so the gamma_regen interval that
bounds the front is expected to shift down ~1/d_mult; the question is whether SOME gamma_regen still gives
extent 100-250 um (4-10 cells at 25 um) AND a stopping time on the ~15 s scale (Bowser & Khakh 2007).
d_mult = 1 must reproduce termination_time.npz (gamma 0.10: 10.1 +- 1.7 s, 4.4 cells).
Run: python phase4_front_coupling.py
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)
from core.provenance import save_result
from phase3_decremental import C_DOWN, K_REF, SCALE, G_FLOOR, focal_lin

L, STRIDE, PATCH, UM, ALPHA, TAU_REF, T_WIN = 64, 5, 2, 25.0, 0.01, 15.0, 120.0
MULTS = [1.0, 2.0, 4.0]
GAMMAS = [1.0, 0.5, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1, 0.075, 0.05, 0.04, 0.025]
SEEDS = list(range(11, 21))
dtf = STRIDE * DT


@njit(fastmath=True, cache=False)
def run_drm(seed, alpha, baseline, gamma_regen, tau_ref, noise_on, steps, nx, ny, sigma, stride, patch, d_mult):
    # phase3_decremental.run_dr verbatim, plus d_mult on D_eff
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    I0 = baseline + alpha * I0_base
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
    Deff = d_mult * D0_base / (1.0 + (kappa_base * alpha) ** 4)
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




def one(args):
    m, gr, seed = args
    f = run_drm(seed, ALPHA, I0_BASE, gr, TAU_REF, False, int(T_WIN / DT), L, L, SIGMA_EM_PREDICTED,
                STRIDE, PATCH, m)
    c = L // 2; yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(abs(xx - c), L - abs(xx - c)); dy = np.minimum(abs(yy - c), L - abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    cum = np.zeros((L, L), bool); ext = []
    for t in range(f.shape[0]):
        cum |= f[t] > 0.5; mm = cum & (r > PATCH); ext.append(r[mm].max() if mm.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    t99 = float(np.argmax(ext >= 0.99 * final) * dtf)
    e2, sp, r2, frac = focal_lin(f, dtf, L, PATCH)
    return m, gr, seed, final, t99, sp * UM / 50.0 if np.isfinite(sp) else np.nan, r2


def main():
    jobs = [(m, gr, s) for m in MULTS for gr in GAMMAS for s in SEEDS]
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        t0 = time.time(); print(one((1.0, 0.10, 11)), f"{time.time()-t0:.0f}s"); return
    with Pool(12) as p:
        res = p.map(one, jobs, chunksize=1)
    E = np.zeros((len(MULTS), len(GAMMAS), len(SEEDS))); T = np.zeros_like(E); S = np.zeros_like(E); R = np.zeros_like(E)
    for m, gr, s, e, t, sp, r2 in res:
        i, j, k = MULTS.index(m), GAMMAS.index(gr), SEEDS.index(s)
        E[i, j, k], T[i, j, k], S[i, j, k], R[i, j, k] = e, t, sp, r2
    for i, m in enumerate(MULTS):
        print(f"d_mult = {m:g}")
        for j, gr in enumerate(GAMMAS):
            print(f"   gamma {gr:5.3f}: extent {E[i,j].mean():5.1f}+-{E[i,j].std():4.1f} cells ({E[i,j].mean()*UM:4.0f} um)  "
                  f"stop {T[i,j].mean():5.1f}+-{T[i,j].std():4.1f} s  speed {np.nanmean(S[i,j]):5.1f} um/s")
    p = save_result(Path("processed_data") / "front_vs_coupling.npz",
                    {"L": L, "alpha": ALPHA, "tau_ref": TAU_REF, "T_window": T_WIN, "um_per_cell": UM,
                     "mults": MULTS, "gammas": GAMMAS, "seeds": SEEDS, "criterion": "99% of final extent"},
                    mults=np.array(MULTS), gammas=np.array(GAMMAS), extent=E, stop_time=T, speed=S, r2=R)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
