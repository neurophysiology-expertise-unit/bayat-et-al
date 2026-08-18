"""Fig 2 panel data under the ATP-dependent affine-current formulation.
Recomputes the deterministic single-unit bifurcation analysis and the heterogeneous-ensemble
Hopf-onset fraction, and persists ONE provenance-stamped npz so no panel is built from a log or
from an unstamped array (phase3_bifurcation.npz / phase3_hopf_fraction.npz predate save_result).
The simulated activity curve is read from the corresponding provenance-stamped Phase-2 sweep.
For this model I0(A)=I0_BASE+s_I A (core/model.py), so the deterministic drive entering
the single-cell equations is I0_BASE+(gamma+s_I)A.
Run: python figdata_fig2.py
"""
import sys; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr, spearmanr
from core.model import I0_BASE
from core.provenance import save_result
from phase3_bifurcation import fixed_point, hopf_alpha, limit_cycle_extrema, jac_eigs
from phase3_onset_duty import onset

# representative units: the population-mean unit, and a positive-gamma tail unit that does bifurcate
G_MEAN, I0_SLOPE_MEAN, I0_INTERCEPT, TB_MEAN = 0.195, 0.30, I0_BASE, 0.8
G_TAIL, I0_SLOPE_TAIL, TB_TAIL = 0.80, 0.30, 0.8
AWINDOW = 1.11
M = 40000
SEED = 11
# The earlier phase2v1 sweeps ran at the sweep_p2 default baseline of 0.2, which was recorded
# nowhere, so pairing them with a recruited fraction computed at I0_BASE spanned a baseline
# mismatch. This file is the same sweep re-run with i0_baseline stamped at I0_BASE.
SIM_NPZ = "processed_data/phase2v1_bayat_b0.42_L32_A_full.npz"


def main():
    ext = np.linspace(0.0, 3.0, 301)
    adia = np.linspace(0.0, 2.0, 161)

    # (1) mean unit — the substantive negative result: it never bifurcates in-window
    ge_mean = G_MEAN + I0_SLOPE_MEAN
    ge_tail = G_TAIL + I0_SLOPE_TAIL
    ha_mean, _ = hopf_alpha(ge_mean, I0_INTERCEPT, TB_MEAN, ext)
    print("mean unit (gamma=0.195, I0 slope=0.30, tau_base=0.8): "
          f"Hopf alpha = {'NONE (never bifurcates on [0,3])' if ha_mean is None else f'{ha_mean:.3f}'}")

    # (2) tail unit — bifurcation diagram + leading eigenvalue
    hd, red = hopf_alpha(ge_tail, I0_INTERCEPT, TB_TAIL, adia)
    Cstar = np.array([fixed_point(a, ge_tail, I0_INTERCEPT) for a in adia])
    stab = red < 0
    cmin = Cstar.copy(); cmax = Cstar.copy(); per = np.full(len(adia), np.nan)
    for i, a in enumerate(adia):
        if not stab[i]:
            cmin[i], cmax[i], per[i] = limit_cycle_extrema(a, ge_tail, I0_INTERCEPT, TB_TAIL)
    print(f"tail unit (gamma=0.80, I0 slope=0.30): Hopf alpha = {hd:.3f}")

    # (3) Hopf vs SNIC — period just past onset (finite and ~flat => Hopf; diverging => SNIC)
    dists = np.array([0.02, 0.05, 0.1, 0.3, 0.6])
    pers = np.array([limit_cycle_extrema(hd + d, ge_tail, I0_INTERCEPT, TB_TAIL)[2] for d in dists])
    print("  period past onset: " + "  ".join(f"+{d:.2f}:{p:.2f}" for d, p in zip(dists, pers)))

    # (4) heterogeneous ensemble: deterministic Hopf fraction vs simulated activity
    rng = np.random.default_rng(SEED)
    g = rng.uniform(0.05, 0.34, M) * (1.0 + 2.0 * rng.standard_normal(M))
    i0_slope = rng.uniform(0.10, 0.50, M)
    tb = rng.uniform(0.5, 1.1, M)
    agrid = np.linspace(0.0, 3.0, 601)
    ons = onset(g + i0_slope, np.full(M, I0_INTERCEPT), tb, agrid)

    sim = np.load(SIM_NPZ, allow_pickle=True)
    als = sim["alphas"]; act = sim["active"]           # (seeds, alphas)
    aA = act.mean(0); aS = act.std(0)
    f_det = np.array([np.mean(ons <= a) for a in als])
    pr, _ = pearsonr(f_det, aA); sr, _ = spearmanr(f_det, aA)
    slope, icept = np.polyfit(f_det, aA, 1)
    print(f"ensemble: Pearson r = {pr:.4f}  Spearman = {sr:.4f}  "
          f"A_act = {slope:.3f}*f_det {icept:+.3f}")
    print(f"  recruited inside [0,{AWINDOW}]: {100*np.mean(ons <= AWINDOW):.2f}%   "
          f"never on [0,3]: {100*np.mean(~np.isfinite(ons)):.1f}%")
    noise_reg = (f_det == 0.0) & (aA > 0)
    if noise_reg.any():
        print(f"  sub-threshold (noise-driven) region: alpha <= {als[noise_reg].max():.3f}, "
              f"A_act there up to {aA[noise_reg].max():.3f} with f_det = 0")

    p = save_result(Path("processed_data") / "fig2_excitability.npz",
                    {"model": f"ATP-dependent affine current: I0(A)={I0_INTERCEPT}+s_I*A",
                     "gamma_mean_unit": G_MEAN, "i0_slope_mean_unit": I0_SLOPE_MEAN,
                     "i0_intercept": I0_INTERCEPT, "tau_mean_unit": TB_MEAN,
                     "gamma_tail_unit": G_TAIL, "i0_slope_tail_unit": I0_SLOPE_TAIL,
                     "tau_tail_unit": TB_TAIL,
                     "M": M, "seed": SEED, "atp_window": AWINDOW, "sim_source": SIM_NPZ,
                     "mean_unit_hopf": ("none" if ha_mean is None else ha_mean),
                     "tail_unit_hopf": hd, "pearson": float(pr), "spearman": float(sr),
                     "reg_slope": float(slope), "reg_intercept": float(icept)},
                    adia=adia, Cstar=Cstar, stab=stab, cmin=cmin, cmax=cmax, period=per, re=red,
                    hopf_tail=np.array([hd]),
                    mean_hopf=np.array([np.inf if ha_mean is None else ha_mean]),
                    snic_dist=dists, snic_period=pers,
                    alphas=als, f_det=f_det, A_act=aA, A_act_sd=aS, onset=ons,
                    gamma=g, i0_slope=i0_slope, tau_base=tb,
                    stats=np.array([pr, sr, slope, icept]))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
