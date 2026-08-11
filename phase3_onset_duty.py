"""Phase 3.1 follow-ups (deterministic, const-I0):
 (A) onset distribution across the sampled ensemble + which heterogeneity (gamma/tau_base/I0_base)
     dominates the spread of Hopf thresholds.
 (B) duty-cycle check: for tail units past Hopf, the fraction of the limit cycle spent above
     theta(alpha) (and the smooth <C_active>) — the mechanistic prediction of the regression slope
     0.503 (mean activity per oscillating unit). Parameter-free confirmation if ~0.5.
Run: python phase3_onset_duty.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

A_FHN, B_FHN, ETA, THETA_BASE = 1.0, 0.8, 8.0, 0.5

def Cstar(alpha, gamma, i0c):
    I = i0c + gamma * alpha
    q = -3.0 * (I - A_FHN / B_FHN)
    D = (q / 2.0) ** 2 + (0.75 / 3.0) ** 3
    s = np.sqrt(D)
    return np.cbrt(-q / 2.0 + s) + np.cbrt(-q / 2.0 - s)

def trace(alpha, gamma, i0c, taub):
    th = 10.0 / ((1.0 + 0.8 * alpha) * taub)
    return (1.0 - Cstar(alpha, gamma, i0c) ** 2) - B_FHN / th

def onset(gamma, i0c, taub, agrid):
    T = np.stack([trace(a, gamma, i0c, taub) for a in agrid])
    past = T >= 0.0
    idx = np.where(past.any(0), past.argmax(0), len(agrid))
    return np.where(idx < len(agrid), agrid[np.clip(idx, 0, len(agrid) - 1)], np.inf)

def duty(alpha, gamma, i0c, taub, T=400.0, burn=0.6):
    """fraction of limit cycle with C>theta(alpha), and smooth <C_active>, over the attractor."""
    theta = THETA_BASE + 0.7 * alpha
    tauh = 10.0 / ((1.0 + 0.8 * alpha) * taub)
    def rhs(t, y):
        C, h = y
        return [C - C**3/3.0 - h + i0c + gamma*alpha, (C + A_FHN - B_FHN*h)/tauh]
    Cs = Cstar(alpha, gamma, i0c)
    sol = solve_ivp(rhs, (0, T), [Cs+0.05, (Cs+A_FHN)/B_FHN], max_step=0.1, rtol=1e-6, atol=1e-9)
    m = sol.t >= burn*T; C = sol.y[0][m]; t = sol.t[m]
    if C.max()-C.min() < 1e-3:
        return np.nan, np.nan
    # time-weighted averages over the (non-uniform) integration grid
    dt = np.diff(t); Cm = 0.5*(C[1:]+C[:-1]); tot = dt.sum()
    hard = float((dt * (Cm > theta)).sum() / tot)
    smooth = float((dt * 0.5*(1.0+np.tanh(ETA*(Cm-theta)))).sum() / tot)
    return hard, smooth

def main():
    rng = np.random.default_rng(11)
    agrid = np.linspace(0.0, 3.0, 601)
    M = 40000
    g = rng.uniform(0.05, 0.34, M) * (1.0 + 2.0 * rng.standard_normal(M))
    i0 = 0.05 + rng.uniform(0.01, 0.15, M)
    tb = rng.uniform(0.5, 1.1, M)
    ons = onset(g, i0, tb, agrid)
    fin = np.isfinite(ons)

    print("=== (A) Hopf-onset distribution across the ensemble (const-I0) ===")
    print(f"  never bifurcates on [0,3]: {100*np.mean(~fin):.1f}%   finite: {100*np.mean(fin):.1f}%")
    print(f"  recruited inside [0,1.11]: {100*np.mean(ons<=1.11):.2f}%")
    qs = [1,5,10,25,50,75,90]
    qv = np.percentile(ons[fin], qs)
    print("  finite-onset percentiles:  " + "  ".join(f"p{p}={v:.2f}" for p,v in zip(qs,qv)))
    print(f"  min onset {ons.min():.3f}   in-window onset density by band:")
    for lo,hi in [(0,0.3),(0.3,0.6),(0.6,0.9),(0.9,1.11)]:
        print(f"    alpha[{lo:.2f},{hi:.2f}): {100*np.mean((ons>lo)&(ons<=hi)):5.2f}% of all units")

    print("\n=== which heterogeneity sets the onset spread? (finite-onset units) ===")
    for name, arr in [("gamma", g), ("tau_base", tb), ("I0_base", i0-0.05)]:
        rho,_ = spearmanr(arr[fin], ons[fin])
        print(f"  Spearman(onset, {name:9s}) = {rho:+.3f}")
    # freeze test: recompute recruited-fraction when each param is frozen at its mean
    base_recruit = np.mean(ons<=1.11)
    for name, arr, mean in [("gamma", g, g.mean()), ("tau_base", tb, tb.mean()), ("I0_base", i0, i0.mean())]:
        gg = g.copy(); ii=i0.copy(); tt=tb.copy()
        if name=="gamma": gg[:] = mean
        elif name=="tau_base": tt[:] = mean
        else: ii[:] = mean
        o2 = onset(gg, ii, tt, agrid)
        print(f"  freeze {name:9s} -> recruited inside window {100*np.mean(o2<=1.11):5.2f}%  "
              f"(vs {100*base_recruit:.2f}% baseline)")

    print("\n=== (B) duty cycle of oscillating tail units vs the 0.503 slope ===")
    print(f"  {'gamma':>6} {'alpha':>6} {'past?':>6} {'hard_frac':>10} {'<C_active>':>11}")
    hs=[]; ss=[]
    for gam in (0.6, 0.8, 1.0):
        on_g = float(onset(np.array([gam]), np.array([0.13]), np.array([0.8]), agrid)[0])
        for a in (on_g+0.1, on_g+0.2, on_g+0.35):
            if a > 1.6: continue
            hard, smooth = duty(a, gam, 0.13, 0.8)
            print(f"  {gam:6.2f} {a:6.3f} {'yes':>6} {hard:10.3f} {smooth:11.3f}")
            if np.isfinite(smooth): hs.append(hard); ss.append(smooth)
    print(f"\n  mean hard-threshold duty = {np.mean(hs):.3f}   mean <C_active> = {np.mean(ss):.3f}")
    print(f"  regression slope to match = 0.503  -> {'CONFIRMED' if abs(np.mean(ss)-0.503)<0.08 else 'CHECK'}")

if __name__ == "__main__":
    main()
