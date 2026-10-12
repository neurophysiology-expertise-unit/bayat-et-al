"""
paperC_critical_radius.py — paper C, NEXT ACTION 1: an analytic boundary for the critical seed radius R_c(d)
and for how the regeneration factor gamma_regen enters it.  (Not part of paper A.)

Background (feasibility, committed): on the HOMOGENEOUS lattice at gamma_regen = 1 a focal front from a 5x5 seed
stops for d >= 4, a 9x9 seed for d >= 4.5, a 17x17 seed only at d = 8, and a planar stripe propagates at every
d tested (<= 8) (paperC_geometry.npz). With gamma_regen < 1 a window of unbounded propagation opens between the
n = 2 and n = 1 gain-saturation lines n*gamma*D/SCALE = 1 - G_FLOOR (paperC_mechanism.npz; lines drawn post hoc).

MODEL (paperC_model.run_gen, verbatim copy, reproduction asserted by paperC_model.check()): homogeneous lattice
(hom_params), deterministic, alpha 0.01, tau_ref 15 s, 120 s, L = 96 (periodic). D = d * Dbar, Dbar = 0.27494.
Seeds: a DISK {r <= R0} (r = Euclidean distance from the centre, cells), C = 1.5 and g = 1 inside, and a planar
STRIPE (5 columns, full height).
GRID: d in D_MULTS (16, 2..24) x R0 in R0S (17, 1..16) x gamma in {1, 0.2, 0.1, 0.05}; stripes at every (d, gamma).
MEASURES: beyond-seed extent = max over activated non-seed cells of the distance to the nearest seed cell
(torus, as paperC_geometry). PROPAGATES := beyond-seed extent >= 20 cells.  R_c(d, gamma) = midpoint between the
largest failing R0 and the smallest propagating R0 above it ('bracket'); R_c <= 1 if all propagate, > 16 if none;
outcomes that are not monotone in R0 are flagged and reported.  Planar speed c0 (cells/s) = 1/slope of mean
first-activation time vs distance from the stripe edge over distances 5..25.  Front gain: g_near = mean g at
first firing over non-seed cells within 3 cells of the disk; g_far (stripe) = mean over distances 15..25.

THEORY AND PRE-REGISTERED PREDICTIONS (written 2026-10-12 before any run of this script):
The coupling transmits b = g*Phi(C) through a Laplacian, so a front cell of gain g loses 4*D*g*Phi(C) and gains
D*sum(b_neighbours): the relevant coupling is the transmitted product x = D*g (at gamma = 1, g = 1 and x = D).
 H1 (eikonal, nonlinear-diffusion form). With radial symmetry the Laplacian of b adds (D g/R) Phi'(C) C_r to the
    planar travelling-wave equation; projecting onto the front gives c(R) = c0(x) - phi * x / R, with phi a
    profile-weighted mean of Phi'(C), 0 < phi <= ETA/2 = 4. Critical radius R_c = phi * x / c0(x), c0 measured
    from the stripes at the same d (no other input). PASS iff (H1a) one fitted phi puts every resolved R_c
    (1 < R_c < 16) at gamma = 1 within +-30 % or inside its bracket, AND (H1b) 0 < phi <= 4.
    H1 also predicts that the stripe propagates at every d (c0 > 0).
 H2 (sink balance, critical nucleus). A fresh front cell drains x*(4 - n_eff) with n_eff = n_p - beta/R active
    partners; it survives iff x*(a + beta/R) < M, so R_c = x / (A - B x), diverging at x* = A/B = M/a, beyond
    which even a planar front fails. PASS iff (H2a) a two-parameter fit (A, B > 0) puts every resolved R_c at
    gamma = 1 within +-30 % or inside its bracket, AND (H2b) the stripe fails at some d <= 24 and the fitted x*
    lies within one d-grid step (in x) of the observed planar failure coupling.
    (Expectation from feasibility numbers only, stated so it can be checked: from 5x5/9x9/17x17 squares at
    d = 4..8, R_c grows faster than D/c0 would allow with the stripe times of paperC_geometry; we expect H1a to
    FAIL and the steep growth to favour H2. The test decides; the criteria above are not changed afterwards.)
 H3 (gamma enters only through the gain g, i.e. through x = D g). Inheritance map at the rising edge:
    g' = min(1, G_FLOOR + gamma * D * sum_up(g_j Phi_j) / SCALE), with n upstream already-fired neighbours.
    A planar front cell has n = 1 (its lateral partners rise in the same step with old gain 0); an off-axis
    convex-front cell has n = 2.
    (H3a) Above the n = 1 line (gamma*D/SCALE >= 1 - G_FLOOR) every front cell saturates at g = 1: outcomes equal
          gamma = 1's, R_c within one R0-grid step of R_c(d, 1), at every such (d, gamma).
    (H3b) Planar gain, analytic: below the n = 1 line the stripe's gain decays to g* = G_FLOOR/(1 - gamma*D/SCALE),
          above it g = 1; measured g_far within 20 % of this g* at every (d, gamma) whose stripe reaches 25 cells.
    (H3c) Collapse: in the window between the n = 2 and n = 1 lines, R_c(d, gamma) equals the gamma = 1 curve
          R_c(x) (log-linear interpolation in x of the gamma = 1 data) evaluated at the measured x_f = D*g_near,
          within one R0-grid step, for >= 80 % of window points (x_f below the smallest gamma = 1 x with a
          resolved R_c -> predicted R_c <= 1 i.e. every seed propagates).
Consistency with the heterogeneous ridge (gamma_regen*d ~ 0.30, paperC_phase_map) is a reading, not a test:
the n = 2 and n = 1 lines sit at gamma*d = 0.267 and 0.535 with Dbar.
Failed predictions are reported as failed; criteria are not revised after the data land.

Checkpoints: /tmp/paperC_ckpt/cr_*.npy (outside the repo).  Output: processed_data/paperC_critical_radius.npz.
Run: python -s paperC_critical_radius.py    (about 1.5 h on 11 cores)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from core.provenance import save_result
from paperC_model import (run_gen, hom_params, check, THETA, TAU_REF, STEPS, STRIDE, T_WIN, DT, SCALE, G_FLOOR,
                          ETA)

LC = 96
D_MULTS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
R0S = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0]
GAMMAS = [1.0, 0.2, 0.1, 0.05]
PROP = 20.0
CKPT = Path("/tmp/paperC_ckpt")


def radius():
    c = LC // 2; ii, jj = np.mgrid[0:LC, 0:LC]
    di = np.minimum(abs(ii - c), LC - abs(ii - c)); dj = np.minimum(abs(jj - c), LC - abs(jj - c))
    return np.sqrt(di ** 2 + dj ** 2)


def mask(R0):
    if R0 == "stripe":
        m = np.zeros((LC, LC), np.bool_); c = LC // 2; m[:, c - 2:c + 3] = True; return m
    return radius() <= R0


def job(args):
    d, gr, R0 = args
    sm = mask(R0)
    I0, th, De, gd = hom_params((LC, LC), d)
    fld, tr, g, cin, pk = run_gen(I0, th, De, gd, THETA, sm, gr, TAU_REF, STEPS, STRIDE)
    dist = distance_transform_edt(~np.tile(sm, (3, 3)))[LC:2 * LC, LC:2 * LC]
    fired = (tr >= 0) & ~sm
    ext = float(dist[fired].max()) if fired.any() else 0.0
    near = fired & (dist <= 3); g_near = float(g[near].mean()) if near.any() else np.nan
    c0 = np.nan; g_far = np.nan
    if R0 == "stripe":
        ds = np.arange(5, 26); tm = np.array([tr[fired & (np.round(dist) == k)].mean() if (fired & (np.round(dist) == k)).any()
                                              else np.nan for k in ds])
        ok = np.isfinite(tm)
        if ok.sum() >= 10:
            sl = np.polyfit(ds[ok], tm[ok], 1)[0]; c0 = 1.0 / sl if sl > 0 else np.nan
        far = fired & (dist >= 15) & (dist <= 25)
        if far.any() and ext >= 25:
            g_far = float(g[far].mean())
    return ext, g_near, c0, g_far, float(fired.mean())


def rc_of(prop, r0s):
    """R_c, bracket (lo, hi), monotone flag."""
    p = np.asarray(prop, bool)
    if p.all():
        return 1.0, (0.0, r0s[0]), True
    if not p.any():
        return np.inf, (r0s[-1], np.inf), True
    first = int(np.argmax(p)); mono = bool(p[first:].all())
    lo, hi = r0s[first - 1] if first > 0 else 0.0, r0s[first]
    if first == 0:
        return 1.0, (0.0, r0s[0]), mono
    return 0.5 * (lo + hi), (lo, hi), mono


def within(pred, rc, br):
    return (abs(pred - rc) <= 0.3 * rc) or (br[0] <= pred <= br[1])


def main():
    CKPT.mkdir(exist_ok=True)
    check()
    Dbar = float(hom_params((1, 1), 1.0)[2][0, 0])
    jobs = [(d, gr, R0) for gr in GAMMAS for d in D_MULTS for R0 in R0S + ["stripe"]]
    ck = CKPT / "cr_runs.npy"
    if ck.exists():
        res = np.load(ck)
    else:
        with Pool(11) as p:
            res = np.array(p.map(job, jobs, chunksize=1))
        np.save(ck, res)
    nG, nD, nR = len(GAMMAS), len(D_MULTS), len(R0S) + 1
    res = res.reshape(nG, nD, nR, 5)
    ext, g_near, c0, g_far = res[..., 0], res[..., 1], res[:, :, -1, 2], res[:, :, -1, 3]
    prop = ext >= PROP
    x = np.array(D_MULTS) * Dbar
    RC = np.full((nG, nD), np.nan); BR = np.full((nG, nD, 2), np.nan); MONO = np.ones((nG, nD), bool)
    for a in range(nG):
        print(f"gamma {GAMMAS[a]}: R0 grid {R0S}; '+' propagates; last column stripe (c0 cells/s)")
        for b, d in enumerate(D_MULTS):
            rc, br, mono = rc_of(prop[a, b, :-1], R0S); RC[a, b] = rc; BR[a, b] = br; MONO[a, b] = mono
            print(f"  d {d:5.1f}: " + "".join("+" if v else "." for v in prop[a, b, :-1])
                  + f"  | {'+' if prop[a,b,-1] else '.'} c0 {c0[a,b]:5.2f}  R_c {rc:5.2f} {'' if mono else 'NON-MONOTONE'}"
                  + f"  g_near(R0=2) {g_near[a,b,R0S.index(2.0)]:.3f}", flush=True)
    # ---- H1, H2 on gamma = 1
    res_ = (RC[0] > 1.0) & np.isfinite(RC[0])
    xs, rcs, brs = x[res_], RC[0][res_], BR[0][res_]
    h1 = h2 = {}
    if res_.sum() >= 2:
        q = xs / c0[0][res_]
        okq = np.isfinite(q)
        phi = float(np.exp(np.mean(np.log(rcs[okq] / q[okq]))))
        pred1 = phi * q
        h1a = all(within(pr, rc, br) for pr, rc, br, o in zip(pred1, rcs, brs, okq) if o) and okq.all()
        h1 = dict(phi=phi, pred=pred1, H1a=bool(h1a), H1b=bool(0 < phi <= ETA / 2),
                  stripe_always=bool(prop[0, :, -1].all()))
        fit = least_squares(lambda p: np.log(xs / np.maximum(p[0] - p[1] * xs, 1e-9)) - np.log(rcs),
                            [2 * xs.max(), 0.1], bounds=([1e-6, 1e-6], [1e3, 1e3]))
        A, B = fit.x; pred2 = xs / np.maximum(A - B * xs, 1e-9); xstar = A / B
        h2a = all(within(pr, rc, br) for pr, rc, br in zip(pred2, rcs, brs))
        sf = np.where(~prop[0, :, -1])[0]
        if len(sf):
            k = sf[0]; lo = x[k - 1] if k > 0 else 0.0; hi = x[k]
            steps = np.diff(x); step_lo = steps[max(k - 2, 0)]; step_hi = steps[min(k, len(steps) - 1)]
            h2b = bool(lo - step_lo <= xstar <= hi + step_hi); planar_fail = (lo, hi)
        else:
            h2b = False; planar_fail = (np.nan, np.nan)
        h2 = dict(A=A, B=B, xstar=xstar, pred=pred2, H2a=bool(h2a), H2b=h2b, planar_fail_x=planar_fail)
        print(f"H1: phi {phi:.3f}  H1a {h1['H1a']}  H1b {h1['H1b']}  stripe propagates at every d: {h1['stripe_always']}")
        print("    resolved x, R_c, bracket, H1 pred, H2 pred:")
        for xx, rc, br, p1, p2 in zip(xs, rcs, brs, pred1, pred2):
            print(f"    x {xx:5.2f}  R_c {rc:5.2f}  [{br[0]:4.1f},{br[1]:4.1f}]  H1 {p1:6.2f}  H2 {p2:6.2f}")
        print(f"H2: A {A:.3f} B {B:.3f} x* {xstar:.3f} (d* {xstar/Dbar:.2f})  H2a {h2['H2a']}  H2b {h2['H2b']}"
              f"  observed planar failure x in {planar_fail}")
    else:
        print("fewer than 2 resolved R_c at gamma = 1: H1/H2 cannot be fitted (reported as FAIL)")
    # ---- H3
    lam1 = np.array(GAMMAS)[:, None] * x[None, :] / SCALE          # gamma D / SCALE
    above1 = lam1 >= 1 - G_FLOOR; window = (2 * lam1 >= 1 - G_FLOOR) & ~above1
    r0a = np.array(R0S)
    def grid_idx(v):
        return np.searchsorted(r0a, v) if np.isfinite(v) else len(r0a)
    h3a = [abs(grid_idx(RC[a, b]) - grid_idx(RC[0, b])) <= 1 for a in range(1, nG) for b in range(nD) if above1[a, b]]
    gstar = np.where(above1, 1.0, G_FLOOR / np.maximum(1 - lam1, 1e-9))
    h3b_pts = [(GAMMAS[a], D_MULTS[b], g_far[a, b], gstar[a, b]) for a in range(nG) for b in range(nD)
               if np.isfinite(g_far[a, b])]
    h3b = [abs(gf - gs) <= 0.2 * gs for _, _, gf, gs in h3b_pts]
    xr = x[res_]; rr = RC[0][res_]
    xf = np.array(D_MULTS)[None, :] * Dbar * g_near[:, :, R0S.index(2.0)]
    h3c = []
    for a in range(1, nG):
        for b in range(nD):
            if window[a, b]:
                xv = xf[a, b]
                if len(xr) == 0 or not np.isfinite(xv):
                    pr = np.nan
                elif xv < xr.min():
                    pr = 1.0
                elif xv > xr.max():
                    pr = np.inf
                else:
                    pr = float(np.exp(np.interp(xv, xr, np.log(rr))))
                h3c.append((GAMMAS[a], D_MULTS[b], xv, pr, RC[a, b],
                            abs(grid_idx(pr) - grid_idx(RC[a, b])) <= 1 if np.isfinite(pr) or np.isinf(pr) else False))
    print(f"H3a (above n=1 line, same as gamma=1 within one R0 step): {sum(h3a)}/{len(h3a)}")
    print(f"H3b (stripe gain within 20% of analytic g*): {sum(h3b)}/{len(h3b)}")
    for gg, dd, gf, gs in h3b_pts:
        print(f"    gamma {gg:5.3f} d {dd:5.1f}: g_far {gf:.3f}  g* {gs:.3f}")
    print(f"H3c (window collapse on x = D g_near(R0=2)): {sum(t[-1] for t in h3c)}/{len(h3c)}")
    for t in h3c:
        print(f"    gamma {t[0]:5.3f} d {t[1]:5.1f}: x_f {t[2]:.3f}  pred R_c {t[3]:6.2f}  obs {t[4]:6.2f}  {t[5]}")
    out = save_result(Path("processed_data") / "paperC_critical_radius.npz",
                      {"model": "paperC_model.run_gen (verbatim), homogeneous, noise off", "L": LC,
                       "T_window": T_WIN, "tau_ref": TAU_REF, "d_mults": D_MULTS, "R0s": R0S, "gammas": GAMMAS,
                       "propagates": f"beyond-seed extent >= {PROP} cells", "Dbar": Dbar,
                       "H1": {k: v for k, v in h1.items() if k != "pred"},
                       "H2": {k: v for k, v in h2.items() if k != "pred"},
                       "H3a": [int(sum(h3a)), len(h3a)], "H3b": [int(sum(h3b)), len(h3b)],
                       "H3c": [int(sum(t[-1] for t in h3c)), len(h3c)]},
                      d_mults=np.array(D_MULTS), R0s=np.array(R0S), gammas=np.array(GAMMAS), x=x,
                      extent=ext, propagates=prop, g_near=g_near, c0=c0, g_far=g_far, frac_fired=res[..., 4],
                      R_c=RC, bracket=BR, monotone=MONO, g_star=gstar, x_f=xf,
                      h1_phi=np.array(h1.get("phi", np.nan)), h2_AB=np.array([h2.get("A", np.nan), h2.get("B", np.nan)]),
                      h3c=np.array([t[:5] for t in h3c], dtype=float) if h3c else np.zeros((0, 5)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
