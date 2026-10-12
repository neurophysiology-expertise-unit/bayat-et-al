"""
paperC_model.py — the model code that paper C uses, in one movable module.

Paper C (Aydın & Özdemir; not part of paper A) uses the paper-A lattice with refractory state and decremental
release. This module holds VERBATIM copies of the integrators it runs, so the paper-C scripts import only this
module and core/model.py (laplacian + constants) and can move to their own repo unchanged:
  run_drm   = phase4_front_coupling.run_drm (heterogeneous lattice, optional noise), verbatim
  draw      = paperC_mechanism.draw (run_drm's parameter draws, same order), verbatim
  run_gen   = paperC_mechanism.run_gen (run_drm's loop with per-cell arrays, noise off), verbatim
  hom_params= paperC_mechanism.hom_params (every per-cell parameter at its ensemble mean), verbatim
The decremental constants are copied as literals from phase3_decremental (C_DOWN, K_REF, SCALE, G_FLOOR).

numba fastmath copies can diverge from the original (vault memory 'numba fastmath copy diverges'), so
`check()` asserts that these copies reproduce the stored results bit for bit:
  - run_drm  vs processed_data/paperC_phase_map.npz (extent and stop_time, 3 points x 2 seeds)
  - run_gen  vs processed_data/paperC_mechanism.npz hom2d (homogeneous lattice, 2 points)
  - run_gen(draw(...)) vs run_drm field, exactly (as M0 in paperC_mechanism)
Run: python -s paperC_model.py      (prints 'paperC_model reproduces stored results')
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)

# decremental-release constants (phase3_decremental, paper A)
C_DOWN = -1.2; K_REF = 20.0; SCALE = 0.15; G_FLOOR = 0.02
# focal protocol (phase4_front_coupling, paper A)
L, STRIDE, PATCH, UM, ALPHA, TAU_REF, T_WIN = 64, 5, 2, 25.0, 0.01, 15.0, 120.0
SEEDS = list(range(11, 21))
dtf = STRIDE * DT
STEPS = int(T_WIN / DT)
THETA = THETA_BASE + 0.7 * ALPHA
HERE = pathlib.Path(__file__).resolve().parent


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


@njit(fastmath=True, cache=False)
def draw(seed, alpha, baseline, nx, ny, d_mult):
    # the parameter draws of phase4_front_coupling.run_drm, verbatim, same order
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
    gdrive = gamma_base * alpha
    return I0, tau_h, Deff, gdrive, D0_base


@njit(fastmath=True, cache=False)
def run_gen(I0, tau_h, Deff, gdrive, theta, seedmask, gamma_regen, tau_ref, steps, stride):
    # run_drm's loop (noise off) with per-cell arrays passed in and a per-cell record of the first firing
    nx, ny = I0.shape
    C = np.full((nx, ny), C_DOWN); h = (C + A_FHN) / B_FHN
    g = np.zeros((nx, ny)); reft = np.zeros((nx, ny)); prev = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if seedmask[i, j]:
                C[i, j] = 1.5; g[i, j] = 1.0
    prev = (0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5).astype(np.float64)
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    t_rise = np.full((nx, ny), -1.0); g_rise = np.zeros((nx, ny)); cin_rise = np.zeros((nx, ny))
    peak = np.full((nx, ny), -9.0); episode = np.zeros((nx, ny))
    fi = 0
    for t in range(steps):
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        active = (C_active > 0.5).astype(np.float64)
        inref = (reft > 0.0).astype(np.float64)
        broadcast = C_active * g * (1.0 - inref)
        coupling_in = Deff * laplacian(broadcast)
        rose = active * (1.0 - prev)
        fell = prev * (1.0 - active)
        g_new = np.minimum(1.0, G_FLOOR + gamma_regen * (coupling_in / SCALE))
        g = np.where(rose > 0.5, np.maximum(g_new, 0.0), g)
        reft = np.where(fell > 0.5, tau_ref, reft)
        clamp = K_REF * inref * (C - C_DOWN)
        noise = np.zeros((nx, ny))
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + np.sqrt(DT) * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - DT, 0.0)
        prev = active
        # bookkeeping only (does not feed back)
        for i in range(nx):
            for j in range(ny):
                if rose[i, j] > 0.5 and t_rise[i, j] < 0.0:
                    t_rise[i, j] = t * DT; g_rise[i, j] = g[i, j]; cin_rise[i, j] = coupling_in[i, j]
                    episode[i, j] = 1.0
                if episode[i, j] > 0.5:
                    if C[i, j] > peak[i, j]:
                        peak[i, j] = C[i, j]
                    if fell[i, j] > 0.5:
                        episode[i, j] = 0.0
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld, t_rise, g_rise, cin_rise, peak


def hom_params(shape, d_mult):
    """every per-cell parameter replaced by its ensemble mean (numerical mean over 2e6 draws, fixed seed)."""
    rng = np.random.default_rng(0); n = 2_000_000
    gb = rng.uniform(0.05, 0.34, n) * (1 + 2 * rng.standard_normal(n))
    tb = rng.uniform(0.5, 1.1, n); D0 = rng.uniform(0.05, 0.5, n); kb = rng.uniform(1.0, 4.0, n)
    I0 = I0_BASE + ALPHA * 0.3
    tau_h = np.mean(10.0 / ((1.0 + 0.8 * ALPHA) * tb))
    Deff = d_mult * np.mean(D0 / (1.0 + (kb * ALPHA) ** 4))
    gd = ALPHA * gb.mean()
    f = lambda v: np.full(shape, v, dtype=np.float64)
    return f(I0), f(tau_h), f(Deff), f(gd)


def extent_t99(fld, n, patch=PATCH):
    """phase4_front_coupling.one()'s extent and t99, for an n x n lattice (cells from the patch centre)."""
    c = n // 2; yy, xx = np.mgrid[0:n, 0:n]
    dx = np.minimum(abs(xx - c), n - abs(xx - c)); dy = np.minimum(abs(yy - c), n - abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    cum = np.zeros((n, n), bool); ext = []
    for t in range(fld.shape[0]):
        cum |= fld[t] > 0.5; mm = cum & (r > patch); ext.append(r[mm].max() if mm.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    return final, float(np.argmax(ext >= 0.99 * final) * dtf)


def check():
    pm = np.load(HERE / "processed_data" / "paperC_phase_map.npz")
    dm, gm, sd = list(pm["d_mults"]), list(pm["gammas"]), list(pm["seeds"])
    for d, gr in [(4.0, 0.075), (2.0, 1.0), (6.0, 0.05)]:
        for s in (11, 14):
            f = run_drm(s, ALPHA, I0_BASE, gr, TAU_REF, False, STEPS, L, L, SIGMA_EM_PREDICTED, STRIDE, PATCH, d)
            e, t = extent_t99(f, L)
            i, j, k = dm.index(d), gm.index(gr), sd.index(s)
            assert e == pm["extent"][i, j, k] and t == pm["stop_time"][i, j, k], ("run_drm", d, gr, s, e, t)
            if s == 11:
                I0, th, De, gd, _ = draw(s, ALPHA, I0_BASE, L, L, d)
                sm = np.zeros((L, L), np.bool_); c = L // 2; sm[c - PATCH:c + PATCH + 1, c - PATCH:c + PATCH + 1] = True
                assert np.array_equal(f, run_gen(I0, th, De, gd, THETA, sm, gr, TAU_REF, STEPS, STRIDE)[0]), "M0"
    mc = np.load(HERE / "processed_data" / "paperC_mechanism.npz")
    dh, gh = list(mc["d_hom"]), list(mc["gammas"])
    for d, gr in [(4.0, 1.0), (5.0, 0.1)]:
        sm = np.zeros((L, L), np.bool_); c = L // 2; sm[c - PATCH:c + PATCH + 1, c - PATCH:c + PATCH + 1] = True
        I0, th, De, gd = hom_params((L, L), d)
        f = run_gen(I0, th, De, gd, THETA, sm, gr, TAU_REF, STEPS, STRIDE)[0]
        e, t = extent_t99(f, L)   # centre-based radius = paperC_mechanism.front_stats for a centred square patch
        ref = mc["hom2d"][dh.index(d), gh.index(gr)]
        assert e == ref[0] and t == ref[1], ("run_gen hom", d, gr, e, t, ref)
    print("paperC_model reproduces stored results (phase_map 6 runs, M0 field, hom2d 2 runs)", flush=True)


if __name__ == "__main__":
    check()
