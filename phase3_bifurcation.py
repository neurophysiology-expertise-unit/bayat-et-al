"""phase3_bifurcation.py — Phase 3.1: single-unit bifurcation analysis (const-I0 model).

The manuscript calls ATP a "bifurcation parameter" with no bifurcation analysis. This supplies
it, on the const-I0 model (the cleanest; the Phase-2 separation does not depend on I0).

Deterministic single FitzHugh-Nagumo unit (coupling and noise off), matching core/model.py:
    dC/dt = C - C^3/3 - h + I_total(alpha)     I_total = I0_const + gamma*alpha
    dh/dt = (C + a - b*h)/tau_h(alpha)          a=A_FHN=1.0, b=B_FHN=0.8
    tau_h(alpha) = 10/((1+0.8*alpha)*tau_base)
Fixed point: unique (cubic C - C^3/3 - (C+a)/b + I_total = 0 is monotone decreasing in C, so no
saddle-node / no SNIC possible in the single unit — the only bifurcation available is Hopf).
Jacobian J = [[1-C*^2, -1],[1/tau_h, -b/tau_h]]; trace = (1-C*^2) - b/tau_h.
Hopf (trace=0, det>0):  1 - C*^2 = b/tau_h(alpha)   [implicit in alpha via C*(alpha) and tau_h(alpha)].

Outputs: eigenvalue curves, Hopf alpha (numeric), bifurcation diagram (FP + limit-cycle extrema),
period-vs-alpha (finite at onset => Hopf, not SNIC), and the heterogeneous-ensemble Hopf-onset
distribution compared to the simulated activity curve.
Run: python phase3_bifurcation.py
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

A_FHN, B_FHN = 1.0, 0.8
def I_total(alpha, gamma, i0c): return i0c + gamma * alpha
def tau_h(alpha, taub): return 10.0 / ((1.0 + 0.8 * alpha) * taub)

def fixed_point(alpha, gamma, i0c):
    """Unique real root of C - C^3/3 - (C+a)/b + I = 0  ->  C^3 + pC + q = 0."""
    I = I_total(alpha, gamma, i0c)
    p = 3.0 * (1.0 / B_FHN - 1.0)              # = 0.75 for b=0.8
    q = -3.0 * (I - A_FHN / B_FHN)
    r = np.roots([1.0, 0.0, p, q])
    Cstar = float(r[np.isreal(r)].real[0]) if np.any(np.isreal(r)) else float(r.real[np.argmin(abs(r.imag))])
    return Cstar

def jac_eigs(alpha, gamma, i0c, taub):
    Cs = fixed_point(alpha, gamma, i0c)
    th = tau_h(alpha, taub)
    J = np.array([[1.0 - Cs ** 2, -1.0], [1.0 / th, -B_FHN / th]])
    return np.linalg.eigvals(J), Cs, th

def hopf_alpha(gamma, i0c, taub, agrid):
    """smallest alpha where max Re(eig) crosses 0 upward; None if never unstable on the grid."""
    re = np.array([jac_eigs(a, gamma, i0c, taub)[0].real.max() for a in agrid])
    s = np.where((re[:-1] < 0) & (re[1:] >= 0))[0]
    if len(s) == 0:
        return None, re
    i = s[0]                                    # linear interp of the zero crossing
    a0, a1 = agrid[i], agrid[i + 1]; r0, r1 = re[i], re[i + 1]
    return a0 + (a1 - a0) * (-r0) / (r1 - r0), re

def limit_cycle_extrema(alpha, gamma, i0c, taub, T=600.0, burn=0.5):
    """integrate to steady state; return (Cmin,Cmax,period) of the attractor."""
    def rhs(t, y):
        C, h = y
        return [C - C ** 3 / 3.0 - h + I_total(alpha, gamma, i0c),
                (C + A_FHN - B_FHN * h) / tau_h(alpha, taub)]
    Cs = fixed_point(alpha, gamma, i0c)
    y0 = [Cs + 0.05, (Cs + A_FHN) / B_FHN]      # kick off the fixed point
    sol = solve_ivp(rhs, (0, T), y0, max_step=0.05, rtol=1e-8, atol=1e-10, dense_output=False)
    m = sol.t >= burn * T
    C = sol.y[0][m]; t = sol.t[m]
    Cmin, Cmax = C.min(), C.max()
    period = np.nan
    if Cmax - Cmin > 1e-3:                       # oscillating: period from upward mean-crossings
        mid = 0.5 * (Cmin + Cmax); cr = np.where((C[:-1] < mid) & (C[1:] >= mid))[0]
        if len(cr) >= 2:
            period = float(np.mean(np.diff(t[cr])))
    return Cmin, Cmax, period

def main():
    out = Path("processed_data"); rng = np.random.default_rng(11)
    # representative (mean) unit
    gbar = 0.195      # mean of gamma_base = mean U(0.05,0.34) * mean(1+2N) = 0.195
    i0bar = 0.13      # 0.05 + mean U(0.01,0.15)
    tbar = 0.8        # mean U(0.5,1.1)
    ext = np.linspace(0.0, 3.0, 301)             # extended alpha to reveal the Hopf
    ha, re = hopf_alpha(gbar, i0bar, tbar, ext)
    print("=== 3.1 representative unit (mean params gamma=0.195, I0=0.13, tau_base=0.8) ===")
    print(f"  Hopf alpha (numeric eig crossing) = {ha}  "
          f"{'(OUTSIDE the ATP window [0,1.11])' if (ha is None or ha>1.11) else '(inside window)'}")
    # eigenvalues at a few alpha in-window
    for a in (0.1, 0.5, 1.0, 1.11):
        ev, Cs, th = jac_eigs(a, gbar, i0bar, tbar)
        print(f"    a={a:4.2f}: C*={Cs:+.3f} tau_h={th:5.2f} eig={ev[0]:+.3f},{ev[1]:+.3f} "
              f"{'STABLE' if ev.real.max()<0 else 'UNSTABLE'}")
    # bifurcation diagram over extended alpha
    Cstar = np.array([fixed_point(a, gbar, i0bar) for a in ext])
    stab = np.array([jac_eigs(a, gbar, i0bar, tbar)[0].real.max() < 0 for a in ext])
    lc = np.array([limit_cycle_extrema(a, gbar, i0bar, tbar) for a in ext])
    cmin, cmax, per = lc[:, 0], lc[:, 1], lc[:, 2]

    # period behaviour near onset -> Hopf vs SNIC
    print("\n=== Hopf vs SNIC (period just past onset; finite+~const => Hopf, diverging => SNIC) ===")
    if ha is not None:
        for a in [ha + d for d in (0.02, 0.05, 0.1, 0.3, 0.6)]:
            _, _, P = limit_cycle_extrema(a, gbar, i0bar, tbar)
            print(f"    a={a:5.3f}: period={P:7.3f}")

    # heterogeneous ensemble: distribution of Hopf-onset alpha over the real param distributions
    print("\n=== 3.1 heterogeneity: fraction of units past Hopf vs alpha (const-I0) ===")
    M = 4000
    g = rng.uniform(0.05, 0.34, M) * (1.0 + 2.0 * rng.standard_normal(M))
    i0 = 0.05 + rng.uniform(0.01, 0.15, M)
    tb = rng.uniform(0.5, 1.1, M)
    onset = np.array([(lambda v: np.inf if v is None else v)(hopf_alpha(g[k], i0[k], tb[k], ext)[0])
                      for k in range(M)], dtype=float)
    win = np.linspace(0.0, 1.11, 21)
    frac = np.array([np.mean(onset <= a) for a in win])
    # simulated activity for shape comparison
    sim = np.load("processed_data/phase2v3_const_L32_A_full.npz")
    aA = sim["active"].mean(0); als = sim["alphas"]
    print(f"  units with Hopf-onset inside [0,1.11]: {100*np.mean(onset<=1.11):.1f}%   "
          f"(median finite onset {np.median(onset[np.isfinite(onset)]):.2f})")
    print(f"  {'alpha':>6} {'frac_osc':>9} {'sim_active':>11}")
    for a, fr in zip(win, frac):
        j = int(np.argmin(abs(als - a)))
        print(f"  {a:6.3f} {fr:9.3f} {aA[j]:11.3f}")

    np.savez_compressed(out / "phase3_bifurcation.npz", ext=ext, Cstar=Cstar, stab=stab,
                        cmin=cmin, cmax=cmax, period=per, re=re, hopf_alpha=(ha or np.inf),
                        onset=onset, win=win, frac=frac, sim_alpha=als, sim_active=aA)

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    st = stab
    ax[0].plot(ext[st], Cstar[st], 'b-', lw=2, label='stable FP')
    ax[0].plot(ext[~st], Cstar[~st], 'b--', lw=2, label='unstable FP')
    osc = (cmax - cmin) > 1e-3
    ax[0].plot(ext[osc], cmax[osc], 'r.', ms=3); ax[0].plot(ext[osc], cmin[osc], 'r.', ms=3, label='limit cycle')
    if ha: ax[0].axvline(ha, color='k', ls=':', label=f'Hopf a={ha:.2f}')
    ax[0].axvspan(0, 1.11, color='gray', alpha=0.12, label='ATP window')
    ax[0].set_xlabel('alpha (ATP)'); ax[0].set_ylabel('C'); ax[0].set_title('Bifurcation diagram (mean unit)'); ax[0].legend(fontsize=7)
    ax[1].plot(ext, re, 'k-'); ax[1].axhline(0, color='r', ls='--'); ax[1].axvspan(0,1.11,color='gray',alpha=0.12)
    ax[1].set_xlabel('alpha'); ax[1].set_ylabel('max Re(eig)'); ax[1].set_title('Leading eigenvalue')
    ax[2].plot(win, frac, 'g-o', ms=3, label='frac units past Hopf')
    ax[2].plot(als, aA / aA.max(), 'm-s', ms=3, label='sim active (norm)')
    ax[2].set_xlabel('alpha'); ax[2].set_ylabel('fraction'); ax[2].set_title('Heterogeneity -> macro activity'); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "phase3_bifurcation.png", dpi=110)
    print(f"\nwrote {out/'phase3_bifurcation.npz'} and .png")

if __name__ == "__main__":
    main()
