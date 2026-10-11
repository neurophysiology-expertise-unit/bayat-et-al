"""
paperC_mechanism.py — FEASIBILITY for a short paper C (pure excitable-media dynamics; not part of paper A).
Why does the focal front at x4 coupling stop by itself at full regeneration, and travel farther when regeneration
is weakened? Candidate mechanism: SOURCE-SINK failure through the broadcast Laplacian.

In run_drm the coupling input of cell i is D_i * lap(b)_i with b = sigma(C) * g * (not refractory). A freshly
fired cell therefore loses 2*D*g (chain) / ~2*D*g (planar 2-D front: 4 minus 2 lateral partners) of its own
broadcast while receiving D*g_up from upstream; at the gain fixed point g = g_up its NET input is -D*g. Two
quasi-static FHN conditions (fast variable C, slow h frozen at rest h_r, I0 = cell drive) follow, with no fitted
parameter:
   recruit:  D*g_up > I_rec = 2/3 + h_r - I0          (the resting cell loses its lower branch; ~0.03)
   sustain:  D*g    < M_sus = min_{C in [theta,1]} [f(C) - h_r + I0 + D g (1 - 2 sigma(C)) ] solved for D g
             (the fired cell can still climb to its excited branch against its own drain; ~1.1-1.3)
with g the fixed point of the inheritance map g' = min(1, G_FLOOR + gamma*D*g/SCALE):
   g* = 1 if gamma*D/SCALE >= 1 - G_FLOOR, else G_FLOOR/(1 - gamma*D/SCALE).
A front propagates indefinitely iff I_rec < D g* < M_sus. Too much regeneration -> D g* above M_sus (sink
failure); too little -> D g* below I_rec (decremental failure). Between them: a propagation window, which at
strong D sits at INTERMEDIATE gamma -> non-monotonic extent.
With cell-to-cell disorder (D0_i ~ U(0.05,0.5)) each site is 'passable' or not; the front is a percolation
problem on the passable sites (square-lattice site threshold p_c = 0.5927).

Experiments (deterministic, noise off, alpha = 0.01, tau_ref = 15, as phase4_front_coupling):
 M0  validation: run_gen fed with run_drm's own parameter draws reproduces run_drm's field EXACTLY (asserted).
 M1  homogeneous 2-D lattice (every per-cell parameter = its ensemble mean), L = 64: d_mult x gamma grid.
 M2  homogeneous 1-D chain (N = 256, same parameters), same grid.
 M3  heterogeneous lattice, per-cell record at first firing: D_i, g_i, peak C of the first episode
     ('full' = peak C > 1). d_mult {2,3,4,5,6} x gamma {1, 0.3, 0.1, 0.075, 0.05} x seeds 11-20.
 M4  intervention: heterogeneous lattice at gamma = 1, d_mult {4, 5, 6}, with every site whose d_mult*D0_i
     exceeds M_sus clipped to 0.95*M_sus/d_mult (all other draws unchanged), seeds 11-20.

PRE-REGISTERED PREDICTIONS (written 2026-10-11 before any run of this script):
 Q1. M1 and M2 show the same two failure modes as the quasi-static boundary: at fixed d_mult above the sink
     line, propagation only in an intermediate-gamma window; the measured window edges lie within a factor 2
     (in gamma) of the quasi-static ones. Offsets are reported, nothing is fitted.
 Q2. Disorder is required for the x4 stop at gamma = 1: the homogeneous lattice (M1) still propagates at
     d_mult = 4, gamma = 1 (quasi-static sink line at d_mult ~ M_sus/<D0> ~ 4-4.7).
 Q3. M3: fraction of 'full' first episodes vs x = D_i*g_i falls from > 0.8 to < 0.2 across x within
     [0.8, 1.6]; the 50% point M_emp lies within 25% of the quasi-static M_sus.
 Q4. M4: clipping the high-D sites restores propagation at x4 (>= 5/10 seeds propagate, t99 >= 0.9*T_WIN).
     Percolation estimate of the gamma = 1 sink coupling: d* = M_emp / (0.05 + 0.45*0.5927), to be compared
     with P3 of paperC_phase_map.py.
Failed predictions are reported as failed; criteria are not revised after the data land.

Output: processed_data/paperC_mechanism.npz (provenance-stamped). Checkpoints in /tmp/paperC_ckpt.
Run: python -s paperC_mechanism.py
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from multiprocessing import Pool
from pathlib import Path
from scipy.optimize import brentq
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, I0_BASE)
from core.provenance import save_result
from phase3_decremental import C_DOWN, K_REF, SCALE, G_FLOOR
from phase4_front_coupling import run_drm, one, L, ALPHA, TAU_REF, T_WIN, UM, PATCH, SEEDS, STRIDE

dtf = STRIDE * DT
STEPS = int(T_WIN / DT)
CKPT = Path("/tmp/paperC_ckpt")
P_C = 0.592746                     # square-lattice site percolation threshold
D_HOM = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
GAMMAS = [1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1, 0.09, 0.075, 0.06, 0.05, 0.04, 0.03,
          0.025, 0.015]
M3_D = [2.0, 3.0, 4.0, 5.0, 6.0]
M3_G = [1.0, 0.3, 0.1, 0.075, 0.05]
M4_D = [4.0, 5.0, 6.0]
N_CHAIN = 256


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


def front_stats(fld, centre_mask):
    """final extent (cells beyond the seed, from the patch centre, periodic) and t99, as phase4 one()."""
    nx, ny = fld.shape[1:]
    ci, cj = np.argwhere(centre_mask).mean(0).round().astype(int)
    ii, jj = np.mgrid[0:nx, 0:ny]
    di = np.minimum(abs(ii - ci), nx - abs(ii - ci)); dj = np.minimum(abs(jj - cj), ny - abs(jj - cj))
    r = np.sqrt(di ** 2 + dj ** 2)
    cum = np.zeros((nx, ny), bool); ext = []
    for t in range(fld.shape[0]):
        cum |= fld[t] > 0.5; mm = cum & ~centre_mask; ext.append(r[mm].max() if mm.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    return final, float(np.argmax(ext >= 0.99 * final) * dtf)


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


def seedmask(nx, ny):
    m = np.zeros((nx, ny), np.bool_)
    if ny == 1:
        m[nx // 2 - PATCH: nx // 2 + PATCH + 1, 0] = True
    else:
        m[nx // 2 - PATCH: nx // 2 + PATCH + 1, ny // 2 - PATCH: ny // 2 + PATCH + 1] = True
    return m


THETA = THETA_BASE + 0.7 * ALPHA


def job_hom(args):
    geom, d, gr = args
    nx, ny = (L, L) if geom == "2d" else (N_CHAIN, 1)
    I0, th, De, gd = hom_params((nx, ny), d)
    sm = seedmask(nx, ny)
    fld, *_ = run_gen(I0, th, De, gd, THETA, sm, gr, TAU_REF, STEPS, STRIDE)
    return front_stats(fld, sm)


def job_cell(args):
    d, gr, seed = args
    I0, th, De, gd, D0 = draw(seed, ALPHA, I0_BASE, L, L, d)
    sm = seedmask(L, L)
    fld, tr, g, cin, pk = run_gen(I0, th, De, gd, THETA, sm, gr, TAU_REF, STEPS, STRIDE)
    fired = (tr >= 0) & ~sm
    return De[fired], g[fired], pk[fired], cin[fired], front_stats(fld, sm)


def job_clip(args):
    d, seed, m_sus = args
    I0, th, De, gd, D0 = draw(seed, ALPHA, I0_BASE, L, L, d)
    De = np.where(De > m_sus, 0.95 * m_sus, De)
    sm = seedmask(L, L)
    fld, *_ = run_gen(I0, th, De, gd, THETA, sm, 1.0, TAU_REF, STEPS, STRIDE)
    return front_stats(fld, sm) + ((De >= 0.95 * m_sus - 1e-12).mean(),)


def quasistatic(I0):
    """h_r, I_rec, M_sus for a cell with drive I0 (theta = THETA)."""
    Cr = brentq(lambda C: C - C ** 3 / 3 - (C + A_FHN) / B_FHN + I0, -3, 0); hr = (Cr + A_FHN) / B_FHN
    I_rec = 2 / 3 + hr - I0
    Cs = np.linspace(THETA - 0.6, 1.0, 4001)
    f = Cs - Cs ** 3 / 3; sg = 0.5 * (1 + np.tanh(ETA * (Cs - THETA)))
    # largest x = D g (with g_up = g) such that the fresh cell's drive stays positive on its way to C = 1
    ok = lambda x: np.all((f - hr + I0 + x * (1 - 2 * sg))[Cs >= THETA] > 0)
    M_sus = brentq(lambda x: 1.0 if ok(x) else -1.0, 0.01, 5.0, xtol=1e-6)
    return hr, I_rec, M_sus


def objarr(lst):
    a = np.empty(len(lst), dtype=object); a[:] = lst; return a


def cached(name, fn):
    ck = CKPT / name
    if ck.exists():
        return np.load(ck, allow_pickle=True)
    out = fn(); np.save(ck, out, allow_pickle=True); return np.load(ck, allow_pickle=True)


def main():
    CKPT.mkdir(exist_ok=True)
    I0h = I0_BASE + ALPHA * 0.3
    hr, I_rec, M_sus = quasistatic(I0h)
    Dbar = float(hom_params((1, 1), 1.0)[2][0, 0])
    print(f"quasi-static: h_r {hr:.4f}  I_rec {I_rec:.4f}  M_sus {M_sus:.4f}  <D0> {Dbar:.4f}"
          f"  sink line at gamma=1: d = {M_sus/Dbar:.2f}", flush=True)

    # M0: exact reproduction of run_drm through draw + run_gen
    for d, gr, s in [(4.0, 1.0, 11), (4.0, 0.075, 11), (1.0, 0.1, 12)]:
        ref = run_drm(s, ALPHA, I0_BASE, gr, TAU_REF, False, STEPS, L, L, 0.0, STRIDE, PATCH, d)
        I0, th, De, gd, _ = draw(s, ALPHA, I0_BASE, L, L, d)
        got = run_gen(I0, th, De, gd, THETA, seedmask(L, L), gr, TAU_REF, STEPS, STRIDE)[0]
        assert np.array_equal(ref, got), ("M0 failed", d, gr, s)
    print("M0: run_gen reproduces run_drm exactly", flush=True)

    with Pool(11) as p:
        res = {}
        for geom in ("2d", "chain"):
            jobs = [(geom, d, gr) for d in D_HOM for gr in GAMMAS]
            r = cached(f"hom_{geom}.npy", lambda: np.array(p.map(job_hom, jobs, chunksize=1)))
            res[geom] = r.reshape(len(D_HOM), len(GAMMAS), 2)
            print(f"M1/M2 {geom} done", flush=True)
        jobs3 = [(d, gr, s) for d in M3_D for gr in M3_G for s in SEEDS]
        r3 = cached("cells.npy", lambda: objarr(p.map(job_cell, jobs3, chunksize=1)))
        print("M3 done", flush=True)
        jobs4 = [(d, s, M_sus) for d in M4_D for s in SEEDS]
        r4 = cached("clip.npy", lambda: np.array(p.map(job_clip, jobs4, chunksize=1)))
        print("M4 done", flush=True)

    # M3: per-cell 'full' fraction vs x = D g
    x = np.concatenate([a[0] * a[1] for a in r3]); full = np.concatenate([a[2] > 1.0 for a in r3])
    edges = np.linspace(0, 3, 31); ctr = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(x, edges) - 1
    frac = np.array([full[idx == k].mean() if (idx == k).sum() >= 20 else np.nan for k in range(len(ctr))])
    cnt = np.array([(idx == k).sum() for k in range(len(ctr))])
    v = np.where(np.isfinite(frac))[0]
    cross = [k for k in v[:-1] if frac[k] >= 0.5 > frac[k + 1]] if len(v) > 1 else []
    M_emp = float(np.interp(0.5, [frac[cross[0] + 1], frac[cross[0]]], [ctr[cross[0] + 1], ctr[cross[0]]])) \
        if cross else np.nan
    d_star = M_emp / (0.05 + 0.45 * P_C) / np.mean(1.0 / (1.0 + (np.random.default_rng(0).uniform(1, 4, 10**6)
                                                                  * ALPHA) ** 4))
    print("M3 full fraction vs D*g:", " ".join(f"{c:.1f}:{f:.2f}" for c, f in zip(ctr, frac) if np.isfinite(f)))
    print(f"M3: M_emp {M_emp:.3f} (quasi-static M_sus {M_sus:.3f});  percolation d* (gamma=1) = {d_star:.2f}")
    ext4 = np.array([[a[4][0], a[4][1]] for a in r3]).reshape(len(M3_D), len(M3_G), len(SEEDS), 2)
    print("M4 clipped (gamma=1):", [(d, float(r4[i*10:(i+1)*10, 0].mean()),
                                     int((r4[i*10:(i+1)*10, 1] >= 0.9 * T_WIN).sum()), float(r4[i*10, 2]))
                                    for i, d in enumerate(M4_D)])
    for geom in ("2d", "chain"):
        print(f"{geom}: extent (cells), * = propagates")
        for i, d in enumerate(D_HOM):
            print(f"  d {d:4.1f}: " + " ".join(f"{res[geom][i,j,0]:5.1f}{'*' if res[geom][i,j,1] >= 0.9*T_WIN else ' '}"
                                               for j in range(len(GAMMAS))))
    out = save_result(Path("processed_data") / "paperC_mechanism.npz",
                      {"model": "run_gen == phase4_front_coupling.run_drm loop (M0 asserted), noise off",
                       "L": L, "N_chain": N_CHAIN, "alpha": ALPHA, "tau_ref": TAU_REF, "T_window": T_WIN,
                       "patch": PATCH, "d_hom": D_HOM, "gammas": GAMMAS, "m3_d": M3_D, "m3_g": M3_G, "m4_d": M4_D,
                       "seeds": SEEDS, "full": "peak C of first episode > 1.0", "p_c": P_C,
                       "propagates": "t99 >= 0.9*T_window", "hom_I0": I0h},
                      d_hom=np.array(D_HOM), gammas=np.array(GAMMAS),
                      hom2d=res["2d"], chain=res["chain"],
                      quasistatic=np.array([hr, I_rec, M_sus, Dbar]),
                      m3_x=x, m3_full=full, m3_bins=ctr, m3_frac=frac, m3_count=cnt, m3_front=ext4,
                      M_emp=np.array(M_emp), d_star=np.array(d_star),
                      m4=r4, m4_d=np.array(M4_D))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
