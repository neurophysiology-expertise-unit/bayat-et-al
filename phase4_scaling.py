"""
phase4_scaling.py — a closed-form law for the extent of the decremental front (PRE-REGISTERED 2026-10-17).

Analysis only: reads two stored npz files, runs no simulation.
  fit/predict : processed_data/front_vs_coupling.npz  (extent[mult, gamma, seed]; d_mult 1, 2, 4; tau_ref 15 s, L 64)
  replication : processed_data/fig4_mechanisms_ens.npz (extent_vs_gamma, d_mult 1, independent runs)

Derivation (phase3_decremental.run_dr). A cell fires when its coupling input u reaches a threshold u_th, and at firing
its release gain is set to g = G_FLOOR + gamma*u/SCALE. Treat the front as a chain of rings n = 1, 2, ...: the input to
ring n+1 is proportional to the gain of ring n, u_{n+1} = m*k*g_n (m = d_mult, k an effective geometric constant), and
ring 1 is driven by the seed patch (g = 1). In units of the threshold, v = u/u_th, this is a linear map
    v_1 = m*p,     v_{n+1} = m*p*G_FLOOR + b*v_n,     b = m*q*gamma,     p = k/u_th, q = k/SCALE,
so v_n = v* + (v_1 - v*) b^(n-1) with fixed point v* = m*p*G_FLOOR/(1-b). The front keeps going while v_n >= 1:
    n*(gamma, m) = 1 + ln[(1 - v*)/(v_1 - v*)] / ln b          (bounded: b < 1 and v* < 1; n* = 0 if v_1 < 1)
    extent r = PATCH + s*n*                                      (s: cells advanced per ring)
and propagation is unbounded above the critical regeneration factor
    gamma_c(m) = (1 - m*p*G_FLOOR) / (m*q).
Far below gamma_c the law reduces to r ~ ln(1/v_1)/ln(b): extent grows as 1/ln(1/gamma). Three free constants
(p, q, s), fitted at d_mult = 1 only; d_mult enters with no new parameter.

Saturation rule (fixed now): at each d_mult a point is SATURATED if its seed-mean extent is >= 0.9 x the gamma = 1
mean at that d_mult (window/lattice-limited, not a bounded front). Saturated points are excluded from fits and errors
and used only for Q3.

Predictions / pass criteria (fixed before running):
  Q1  d_mult = 1 fit (seed means of non-saturated gammas, unweighted least squares on r): R^2 >= 0.95, and every
      residual within max(2 x seed SD, 1 cell).
  Q2  ZERO-parameter prediction of d_mult = 2 and 4 from the d_mult = 1 constants: median |error| <= 15% of the
      measured seed-mean extent over the non-saturated points of each d_mult (both must pass).
  Q3  predicted gamma_c(m) lies between the largest non-saturated and the smallest saturated sampled gamma, at each m.
  Q4  replication: the d_mult = 1 constants predict fig4_mechanisms_ens extent_vs_gamma (non-saturated) with median
      |error| <= 15%.
  Reported, no pass/fail: the one-term form r = r0 + A/ln(1/gamma) fitted at d_mult = 1 (AIC vs the full law).
A failed Q is reported as failed; the law is then not claimed in the paper.

Disclosure: the d_mult = 2, 4 extents were plotted in Fig. coupling C and seen by the authors before this test;
the functional form above was derived from the update rule, not fitted to them. phase4_front_coupling.py's docstring
already expected the bounding gamma interval to shift ~1/d_mult (qualitative); this test makes it quantitative.
POST-HOC NOTE (2026-10-17, after running; does not change any computed value): Q4 is NOT an independent replication.
fig4_mechanisms_ens.npz and front_vs_coupling.npz at d_mult = 1 are deterministic runs with the same seeds and code,
so every shared gamma (1.0, 0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05) reproduces identical extents (max abs difference 0).
The only gamma in fig4 that is new to the fit and not saturated is 0.4: the law predicts unbounded propagation there
(gamma_c = 0.333), measured 27.6 +- 4.4 cells (below the 0.9 x reference saturation cut of 30.7). Q4's "PASS" should be
read as a consistency check, not an out-of-sample test; the evidence for the law is the d_mult = 1 fit (Q1) and
gamma_c falling in the measured collapse (0.3-0.4), which the clip argument in the manuscript places at 0.30 a priori.
Run: python phase4_scaling.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from core.provenance import save_result
from phase3_decremental import G_FLOOR

PATCH, SAT, TOL = 2, 0.9, 0.15


def law(gamma, m, p, q, s):
    gamma = np.asarray(gamma, float)
    b = m * q * gamma
    v1 = m * p
    with np.errstate(divide="ignore", invalid="ignore"):
        vs = m * p * G_FLOOR / (1.0 - b)
        n = 1.0 + np.log((1.0 - vs) / (v1 - vs)) / np.log(b)
    n = np.where(v1 < 1.0, 0.0, n)
    n = np.where((b >= 1.0) | (vs >= 1.0), np.inf, n)
    return PATCH + s * n


def gamma_c(m, p, q):
    return (1.0 - m * p * G_FLOOR) / (m * q)


def bounded(mean, ref):
    return mean < SAT * ref


def main():
    d = np.load("processed_data/front_vs_coupling.npz")
    mults, gammas, ext = d["mults"], d["gammas"], d["extent"]          # ext[mult, gamma, seed]
    mean, sd = ext.mean(-1), ext.std(-1, ddof=1)
    ref = mean[:, np.argmin(np.abs(gammas - 1.0))]
    ok = np.array([bounded(mean[i], ref[i]) for i in range(len(mults))])

    i1 = int(np.argmin(np.abs(mults - 1.0)))
    g1, r1, sd1 = gammas[ok[i1]], mean[i1, ok[i1]], sd[i1, ok[i1]]

    def res(x):
        return law(g1, 1.0, *x) - r1
    best = None
    for p0 in (1.5, 3.0, 6.0):
        for q0 in (0.5, 1.0, 2.0):
            try:
                f = least_squares(res, [p0, q0, 1.0], bounds=([1.0, 1e-3, 0.1], [1e3, 1e2, 5.0]))
            except ValueError:
                continue
            if np.all(np.isfinite(f.fun)) and (best is None or f.cost < best.cost):
                best = f
    p, q, s = best.x
    pred1 = law(g1, 1.0, p, q, s)
    r2_1 = 1 - np.sum((r1 - pred1) ** 2) / np.sum((r1 - r1.mean()) ** 2)
    q1 = bool(r2_1 >= 0.95 and np.all(np.abs(r1 - pred1) <= np.maximum(2 * sd1, 1.0)))

    out = {"p": p, "q": q, "s": s, "R2_m1": r2_1, "Q1": q1}
    q2, q3 = [], []
    pred_all = np.full_like(mean, np.nan)
    for i, m in enumerate(mults):
        pred_all[i] = law(gammas, m, p, q, s)
        rel = np.abs(pred_all[i, ok[i]] - mean[i, ok[i]]) / mean[i, ok[i]]
        gc = gamma_c(m, p, q)
        lo = gammas[ok[i]].max() if ok[i].any() else 0.0
        hi = gammas[~ok[i]].min() if (~ok[i]).any() else np.inf
        out[f"m{m:g}_median_relerr"] = float(np.median(rel))
        out[f"m{m:g}_gamma_c"] = float(gc)
        out[f"m{m:g}_bracket"] = [float(lo), float(hi)]
        if m != 1.0:
            q2.append(bool(np.median(rel) <= TOL))
        q3.append(bool(lo < gc < hi))
        print(f"d_mult {m:g}: n_bounded {ok[i].sum()}, median |err| {np.median(rel):.3f}, "
              f"gamma_c {gc:.3f} vs bracket ({lo}, {hi})")
        for g, mm, ss, pp, b in zip(gammas, mean[i], sd[i], pred_all[i], ok[i]):
            print(f"   gamma {g:6.3f}  measured {mm:6.2f} +- {ss:4.2f}  law {pp:7.2f}  {'' if b else 'SATURATED'}")
    out["Q2"], out["Q3"] = bool(all(q2)), bool(all(q3))

    f4 = np.load("processed_data/fig4_mechanisms_ens.npz")
    g4, m4 = f4["gammas"], f4["extent_vs_gamma_mean"]
    ok4 = bounded(m4, m4[np.argmin(np.abs(g4 - 1.0))])
    rel4 = np.abs(law(g4[ok4], 1.0, p, q, s) - m4[ok4]) / m4[ok4]
    out["fig4_median_relerr"] = float(np.median(rel4))
    out["Q4"] = bool(np.median(rel4) <= TOL)

    # reported only: one-term form r0 + A/ln(1/gamma), AIC vs the full law (Gaussian, n points, k params)
    x1 = 1.0 / np.log(1.0 / g1)
    A_, r0_ = np.polyfit(x1, r1, 1)
    rss_1t, rss_law, n = np.sum((r1 - (r0_ + A_ * x1)) ** 2), np.sum((r1 - pred1) ** 2), len(r1)
    out["onet_r0"], out["onet_A"] = float(r0_), float(A_)
    out["AIC_onet"] = float(n * np.log(rss_1t / n) + 2 * 2)
    out["AIC_law"] = float(n * np.log(rss_law / n) + 2 * 3)
    print(json.dumps(out, indent=1, default=float))

    save_result(Path("processed_data") / "extent_scaling_law.npz",
                {"source": ["front_vs_coupling.npz", "fig4_mechanisms_ens.npz"], "G_FLOOR": G_FLOOR,
                 "PATCH": PATCH, "saturation": f">= {SAT} x gamma=1 mean", "tol": TOL,
                 "preregistered": "phase4_scaling.py docstring", "summary": out},
                mults=mults, gammas=gammas, measured_mean=mean, measured_sd=sd, bounded=ok, law=pred_all,
                fit_params=np.array([p, q, s]), gamma_c=np.array([gamma_c(m, p, q) for m in mults]),
                fig4_gammas=g4, fig4_mean=m4, fig4_law=law(g4, 1.0, p, q, s))


if __name__ == "__main__":
    main()
