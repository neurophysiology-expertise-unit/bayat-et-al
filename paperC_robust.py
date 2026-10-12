"""
paperC_robust.py — paper C, NEXT ACTION 3: do the two key feasibility results survive noise and a larger lattice?
(Not part of paper A.)

Key results (paperC_phase_map.npz; heterogeneous lattice, noise off, L = 64, seeds 11-20):
  RIDGE    for d >= 2.5 the mean extent peaks at gamma_regen*d = 0.30 (0.27-0.31);
  FAILURE  at gamma_regen = 1 the 5x5 seed stops propagating between d = 2 and d = 2.5.
MODEL: paperC_model.run_drm (verbatim copy of phase4_front_coupling.run_drm; reproduction asserted by
paperC_model.check()), alpha 0.01, tau_ref 15 s, 120 s, 5x5 seed, seeds 11-20.
Points: RIDGE d in {3, 4, 6} x gamma in G_RIDGE (12 values, 0.025..0.2 and 1); FAILURE gamma = 1 x
d in {1.5, 2, 2.25, 2.5, 3}.
Conditions:
  L128   noise off, L = 128 (new disorder realisations: the draws depend on the lattice shape).
  NOISE  noise on at the paper-A level (noise_on = True, sigma = SIGMA_EM_PREDICTED, as phase3_decremental /
         the paper-A noisy focal runs), L = 64. Each seeded run is paired with an UNSEEDED control (patch = -1:
         same seed, so the same parameter draws and the same noise stream). A cell is EVOKED if its first
         activation in the seeded run precedes its first activation in the control (never = infinity); the
         front is the 8-connected component of evoked + seed cells that contains the seed. Extent and t99 as
         phase4_front_coupling.one() but over front cells only.
         Calibration (reported, not a criterion): noise OFF, d = 4, gamma 0.075 and 1, the evoked measure vs
         phase4 one()'s extent on the same runs.
Propagates ('advancing', as the feasibility figure): t99 >= 0.9*T_window or extent >= 0.95 * (L/2)*sqrt(2).

PRE-REGISTERED PREDICTIONS (written 2026-10-12 before any run of this script), in EACH condition (L128, NOISE):
 R1. Ridge location: at each d in {3, 4, 6} the arg-max gamma of mean extent satisfies gamma_peak*d in
     [0.24, 0.36] (one grid step either side of 0.30).
 R2. Ridge height: at each d, mean extent at gamma_peak exceeds mean extent at gamma = 1 by more than 2 SD
     (SD over seeds at gamma = 1).
 R3. Failure coupling: at gamma = 1 the fraction of seeds advancing is >= 0.5 at d = 2 and < 0.5 at d = 2.5
     (d* in (2, 2.5], as at L = 64 noise off).
 R4. (L128 only) finite size: mean extent at the ridge peak within 25 % of the L = 64 value at the same
     (d, gamma) (stored paperC_phase_map.npz), since the L = 64 ridge extents (13-18 cells) are far from the edge.
Failed predictions are reported as failed; criteria are not revised after the data land.

Checkpoints: /tmp/paperC_ckpt/rb_*.npy (outside the repo). Output: processed_data/paperC_robust.npz.
Run: python -s paperC_robust.py     (about 1.5 h on 11 cores)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from scipy.ndimage import label
from core.provenance import save_result
from paperC_model import (run_drm, extent_t99, check, ALPHA, I0_BASE, TAU_REF, STEPS, STRIDE, PATCH, SEEDS,
                          T_WIN, SIGMA_EM_PREDICTED, dtf)

D_RIDGE = [3.0, 4.0, 6.0]
G_RIDGE = [1.0, 0.2, 0.15, 0.125, 0.1, 0.09, 0.075, 0.06, 0.05, 0.04, 0.03, 0.025]
D_FAIL = [1.5, 2.0, 2.25, 2.5, 3.0]
POINTS = [(d, g) for d in D_RIDGE for g in G_RIDGE] + [(d, 1.0) for d in D_FAIL if d not in D_RIDGE]
CKPT = Path("/tmp/paperC_ckpt")
HERE = pathlib.Path(__file__).resolve().parent


def rgrid(n):
    c = n // 2; yy, xx = np.mgrid[0:n, 0:n]
    dx = np.minimum(abs(xx - c), n - abs(xx - c)); dy = np.minimum(abs(yy - c), n - abs(yy - c))
    return np.sqrt(dx ** 2 + dy ** 2)


def first_act(fld):
    act = fld > 0.5
    return np.where(act.any(0), act.argmax(0), np.iinfo(np.int64).max).astype(np.int64)


def evoked_extent(f_seed, f_ctrl, n):
    r = rgrid(n); ts, tc = first_act(f_seed), first_act(f_ctrl)
    big = np.iinfo(np.int64).max
    c = n // 2; seedm = np.zeros((n, n), bool); seedm[c - PATCH:c + PATCH + 1, c - PATCH:c + PATCH + 1] = True
    ev = ((ts < tc) & (ts < big)) | seedm
    lab, _ = label(ev, structure=np.ones((3, 3)))
    front = (lab == lab[c, c]) & ~seedm & (r > PATCH)
    if not front.any():
        return 0.0, 0.0, float(((tc < big)).mean())
    order = np.argsort(ts[front]); tt = ts[front][order]; rr = np.maximum.accumulate(r[front][order])
    idx = np.searchsorted(tt, np.arange(f_seed.shape[0]), side="right") - 1
    ext_t = np.where(idx >= 0, rr[np.maximum(idx, 0)], 0.0)
    final = ext_t[-1]
    return float(final), float(np.argmax(ext_t >= 0.99 * final) * dtf), float((tc < big).mean())


def job_l128(args):
    d, g, s = args
    f = run_drm(s, ALPHA, I0_BASE, g, TAU_REF, False, STEPS, 128, 128, SIGMA_EM_PREDICTED, STRIDE, PATCH, d)
    return extent_t99(f, 128) + (0.0,)


def job_noise(args):
    d, g, s, noise = args
    fs = run_drm(s, ALPHA, I0_BASE, g, TAU_REF, noise, STEPS, 64, 64, SIGMA_EM_PREDICTED, STRIDE, PATCH, d)
    fc = run_drm(s, ALPHA, I0_BASE, g, TAU_REF, noise, STEPS, 64, 64, SIGMA_EM_PREDICTED, STRIDE, -1, d)
    out = evoked_extent(fs, fc, 64)
    if not noise:
        out = out + extent_t99(fs, 64)
    return out


def run(name, fn, jobs, p):
    ck = CKPT / f"rb_{name}.npy"
    if ck.exists():
        return np.load(ck)
    r = np.array(p.map(fn, jobs, chunksize=1)); np.save(ck, r); print(f"{name} done", flush=True); return r


def assess(E, T, n, tag):
    rmax = (n // 2) * np.sqrt(2)
    adv = (T >= 0.9 * T_WIN) | (E >= 0.95 * rmax)
    res = {}
    print(f"--- {tag}: mean extent (cells) per point, '*' = >= half the seeds advancing")
    for d in D_RIDGE:
        ii = [POINTS.index((d, g)) for g in G_RIDGE]
        m = E[ii].mean(1)
        kpk = int(np.argmax(m)); gpk = G_RIDGE[kpk]
        r1 = 0.24 <= gpk * d <= 0.36
        r2 = m[kpk] > m[0] + 2 * E[ii[0]].std()
        res[d] = dict(gamma_peak=gpk, gd=gpk * d, ext_peak=float(m[kpk]), ext_g1=float(m[0]),
                      sd_g1=float(E[ii[0]].std()), R1=bool(r1), R2=bool(r2))
        print(f"  d {d:3.1f}: " + " ".join(f"{g}:{v:5.1f}{'*' if adv[i].mean() >= 0.5 else ''}"
                                            for g, v, i in zip(G_RIDGE, m, ii))
              + f"  -> gamma_peak {gpk} (gamma*d {gpk*d:.3f})  R1 {r1}  R2 {r2}", flush=True)
    fr = {d: float(adv[POINTS.index((d, 1.0))].mean()) for d in D_FAIL}
    r3 = fr[2.0] >= 0.5 and fr[2.5] < 0.5
    print(f"  gamma=1 fraction advancing: {fr}  R3 {r3}", flush=True)
    return res, fr, bool(r3), adv


def main():
    CKPT.mkdir(exist_ok=True)
    check()
    pm = np.load(HERE / "processed_data" / "paperC_phase_map.npz")
    with Pool(11) as p:
        cal = run("cal", job_noise, [(4.0, g, s, False) for g in (0.075, 1.0) for s in SEEDS], p)
        print("calibration (noise off, d 4): evoked extent vs one() extent, per seed:")
        for k, g in enumerate((0.075, 1.0)):
            c = cal[k * 10:(k + 1) * 10]
            print(f"  gamma {g}: evoked {np.round(c[:,0],1).tolist()}\n            one()  {np.round(c[:,3],1).tolist()}"
                  f"  equal {int((c[:,0] == c[:,3]).sum())}/10", flush=True)
        jn = [(d, g, s, True) for d, g in POINTS for s in SEEDS]
        rn = run("noise", job_noise, jn, p).reshape(len(POINTS), len(SEEDS), 3)
        jl = [(d, g, s) for d, g in POINTS for s in SEEDS]
        rl = run("l128", job_l128, jl, p).reshape(len(POINTS), len(SEEDS), 3)
    resN, frN, r3N, advN = assess(rn[..., 0], rn[..., 1], 64, "NOISE (L 64, evoked front)")
    print(f"  background: fraction of cells active in the unseeded control {rn[...,2].mean():.3f}")
    resL, frL, r3L, advL = assess(rl[..., 0], rl[..., 1], 128, "L128 (noise off)")
    r4 = {}
    for d in D_RIDGE:
        g = resL[d]["gamma_peak"]
        e64 = float(pm["extent"][list(pm["d_mults"]).index(d), list(pm["gammas"]).index(g)].mean())
        r4[d] = dict(L128=resL[d]["ext_peak"], L64=e64, R4=bool(abs(resL[d]["ext_peak"] - e64) <= 0.25 * e64))
    print(f"R4 (L128 ridge extent within 25% of L64): {r4}")
    summ = {"NOISE": {"ridge": resN, "frac_adv_g1": frN, "R3": r3N},
            "L128": {"ridge": resL, "frac_adv_g1": frL, "R3": r3L, "R4": r4}}
    out = save_result(Path("processed_data") / "paperC_robust.npz",
                      {"model": "paperC_model.run_drm (verbatim)", "alpha": ALPHA, "tau_ref": TAU_REF,
                       "T_window": T_WIN, "seeds": SEEDS, "points": POINTS, "noise_sigma": SIGMA_EM_PREDICTED,
                       "evoked": "first activation earlier than in the unseeded control; 8-connected to the seed",
                       "advancing": "t99 >= 0.9 T or extent >= 0.95 (L/2) sqrt 2", "summary": summ},
                      points=np.array(POINTS), noise=rn, l128=rl, cal=cal, adv_noise=advN, adv_l128=advL)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
