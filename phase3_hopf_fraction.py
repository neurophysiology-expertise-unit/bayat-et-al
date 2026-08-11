"""phase3_hopf_fraction.py — quantitative Hopf-onset fraction vs simulated activity (const-I0).

For each unit (sampled gamma, I0_base, tau_base) the single-unit Hopf onset is the smallest alpha
where the Jacobian trace crosses zero with det>0:
    trace(alpha) = (1 - C*(alpha)^2) - b/tau_h(alpha),   b=0.8,  tau_h=10/((1+0.8a)*tau_base)
    C*(alpha) = unique real root of C - C^3/3 - (C+a)/b + (I0c + gamma*alpha) = 0  (Cardano, p=0.75>0)
f_det(alpha) = fraction of the population past Hopf at alpha. Overlay on the simulated mean active
fraction A_act(alpha); report Pearson/Spearman correlation and the linear-regression offset
(A_act ~ s*f_det + c). The low-alpha region where A_act>0 but f_det=0 is the noise-driven excitable
contribution (units below Hopf are excitable, not silent), reported separately, not as disagreement.
Run: python phase3_hopf_fraction.py
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

A_FHN, B_FHN, P_CUBIC = 1.0, 0.8, 0.75          # p = 3*(1/b-1) = 0.75

def Cstar(alpha, gamma, i0c):
    """vectorized unique real root of C^3 + 0.75 C + q = 0, q = -3(I - a/b)."""
    I = i0c + gamma * alpha
    q = -3.0 * (I - A_FHN / B_FHN)
    D = (q / 2.0) ** 2 + (P_CUBIC / 3.0) ** 3   # >0 always -> one real root
    s = np.sqrt(D)
    return np.cbrt(-q / 2.0 + s) + np.cbrt(-q / 2.0 - s)

def trace(alpha, gamma, i0c, taub):
    th = 10.0 / ((1.0 + 0.8 * alpha) * taub)
    return (1.0 - Cstar(alpha, gamma, i0c) ** 2) - B_FHN / th

def onset(gamma, i0c, taub, agrid):
    """first alpha on agrid where trace>=0 (Hopf), else inf. Broadcast over units."""
    T = np.stack([trace(a, gamma, i0c, taub) for a in agrid])   # (nA, M)
    past = T >= 0.0
    idx = np.where(past.any(0), past.argmax(0), len(agrid))      # first True, or sentinel
    return np.where(idx < len(agrid), agrid[np.clip(idx, 0, len(agrid) - 1)], np.inf)

def main():
    rng = np.random.default_rng(11)
    agrid = np.linspace(0.0, 3.0, 601)          # fine, extends past the ATP window
    M = 40000
    g = rng.uniform(0.05, 0.34, M) * (1.0 + 2.0 * rng.standard_normal(M))
    i0 = 0.05 + rng.uniform(0.01, 0.15, M)
    tb = rng.uniform(0.5, 1.1, M)
    ons = onset(g, i0, tb, agrid)

    # mean-parameter representative unit
    on_mean = float(onset(np.array([0.195]), np.array([0.13]), np.array([0.8]), agrid)[0])
    print("=== representative (mean) unit ===")
    print(f"  Hopf onset alpha = {on_mean if np.isfinite(on_mean) else 'NONE'}  "
          f"{'INSIDE [0,1.11]' if on_mean<=1.11 else 'OUTSIDE the ATP window [0,1.11]'}")

    sim = np.load("processed_data/phase2v3_const_L32_A_full.npz")
    aA = sim["active"].mean(0); als = sim["alphas"]
    f_det = np.array([np.mean(ons <= a) for a in als])

    print("\n=== deterministic Hopf fraction vs simulated activity (numbers first) ===")
    print(f"  units past Hopf inside [0,1.11]: {100*np.mean(ons<=1.11):5.2f}%   "
          f"(finite-onset median {np.median(ons[np.isfinite(ons)]):.3f}, "
          f"min {np.min(ons):.3f})")
    print(f"  {'alpha':>6} {'f_det':>8} {'A_act(sim)':>11} {'A_act-f_det':>12}")
    for a, fd, ac in zip(als, f_det, aA):
        print(f"  {a:6.3f} {fd:8.4f} {ac:11.4f} {ac-fd:12.4f}")

    # quantitative agreement
    pr, pp = pearsonr(f_det, aA); sr, sp = spearmanr(f_det, aA)
    # regression A_act ~ s*f_det + c  (offset = scale s, intercept c)
    Xok = f_det > 0
    s, c = np.polyfit(f_det, aA, 1)
    print("\n=== agreement (quantitative) ===")
    print(f"  Pearson r(f_det, A_act)  = {pr:.4f}  (p={pp:.2e})")
    print(f"  Spearman rho             = {sr:.4f}  (p={sp:.2e})")
    print(f"  regression A_act = {s:.4f}*f_det + {c:+.4f}   "
          f"(slope = per-oscillating-unit mean activity; intercept = low-alpha offset)")

    # noise-driven excitable gap: A_act>0 where f_det==0
    noise_reg = (f_det == 0.0) & (aA > 0)
    if noise_reg.any():
        amax = als[noise_reg].max(); acontrib = aA[noise_reg].sum()
        print(f"\n=== noise-driven excitable contribution (f_det=0 but A_act>0) ===")
        print(f"  present for alpha in [{als[noise_reg].min():.3f}, {amax:.3f}]  "
              f"(deterministic fraction is exactly 0 there; activity is sub-threshold noise firing)")
        print(f"  e.g. alpha={amax:.3f}: A_act={aA[np.argmin(abs(als-amax))]:.4f}, f_det=0")
    else:
        print("\n  no alpha with A_act>0 while f_det=0 (no separable noise-driven region)")

    np.savez_compressed("processed_data/phase3_hopf_fraction.npz",
                        alphas=als, f_det=f_det, A_act=aA, onset=ons,
                        pearson=pr, spearman=sr, reg_slope=s, reg_intercept=c,
                        mean_unit_onset=on_mean)
    print("\nwrote processed_data/phase3_hopf_fraction.npz")

if __name__ == "__main__":
    main()
