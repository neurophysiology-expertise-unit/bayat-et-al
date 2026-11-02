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

def limit_cycle_extrema(alpha, gamma, i0c, taub, T=400.0, burn=0.6):
    """integrate to steady state; return (Cmin,Cmax,period) of the attractor."""
    def rhs(t, y):
        C, h = y
        return [C - C ** 3 / 3.0 - h + I_total(alpha, gamma, i0c),
                (C + A_FHN - B_FHN * h) / tau_h(alpha, taub)]
    Cs = fixed_point(alpha, gamma, i0c)
    y0 = [Cs + 0.05, (Cs + A_FHN) / B_FHN]      # kick off the fixed point
    sol = solve_ivp(rhs, (0, T), y0, max_step=0.1, rtol=1e-6, atol=1e-9, dense_output=False)
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
    out = Path("processed_data")
    ext = np.linspace(0.0, 3.0, 301)
    # (1) mean unit — the substantive negative result: it never bifurcates in-window
    gbar, i0bar, tbar = 0.195, 0.13, 0.8
    ha_mean, _ = hopf_alpha(gbar, i0bar, tbar, ext)
    print("=== 3.1a representative MEAN unit (gamma=0.195, I0=0.13, tau_base=0.8) ===")
    print(f"  Hopf alpha = {ha_mean}  "
          f"{'never bifurcates on [0,3]' if ha_mean is None else ('OUTSIDE window' if ha_mean>1.11 else 'inside window')}")
    for a in (0.1, 0.5, 1.0, 1.11):
        ev, Cs, th = jac_eigs(a, gbar, i0bar, tbar)
        print(f"    a={a:4.2f}: C*={Cs:+.3f} tau_h={th:5.2f} maxRe={ev.real.max():+.4f} "
              f"{'STABLE' if ev.real.max()<0 else 'UNSTABLE'}")

    # (2) representative OSCILLATING TAIL unit for the diagram (high positive gamma)
    gdia, i0dia, tdia = 0.80, 0.13, 0.8
    adia = np.linspace(0.0, 2.0, 161)
    hd, red = hopf_alpha(gdia, i0dia, tdia, adia)
    Cstar = np.array([fixed_point(a, gdia, i0dia) for a in adia])
    stab = red < 0
    cmin = Cstar.copy(); cmax = Cstar.copy(); per = np.full(len(adia), np.nan)
    for i, a in enumerate(adia):                 # integrate limit cycle ONLY where FP is unstable
        if not stab[i]:
            cmin[i], cmax[i], per[i] = limit_cycle_extrema(a, gdia, i0dia, tdia)
    print(f"\n=== 3.1b representative TAIL unit (gamma=0.80): Hopf alpha = {hd:.3f} ===")

    # (3) Hopf vs SNIC — period just past onset (finite+~const => Hopf; diverging => SNIC)
    print("\n=== Hopf vs SNIC: period vs distance past onset (tail unit) ===")
    for d in (0.02, 0.05, 0.1, 0.3, 0.6):
        _, _, P = limit_cycle_extrema(hd + d, gdia, i0dia, tdia)
        print(f"    a=onset+{d:4.2f} ({hd+d:5.3f}): period={P:7.3f}")
    print("  (unique fixed point for all alpha => no SNIC structurally; finite period confirms Hopf)")

    # (4) ensemble fraction panel — reuse the quantitative result from phase3_hopf_fraction
    hf = np.load("processed_data/phase3_hopf_fraction.npz")
    als, aA, frac = hf["alphas"], hf["A_act"], hf["f_det"]

    np.savez_compressed(out / "phase3_bifurcation.npz", adia=adia, Cstar=Cstar, stab=stab,
                        cmin=cmin, cmax=cmax, period=per, re=red, hopf_tail=hd,
                        mean_hopf=(np.inf if ha_mean is None else ha_mean),
                        sim_alpha=als, sim_active=aA, frac=frac)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(adia[stab], Cstar[stab], 'b-', lw=2, label='stable FP')
    ax[0].plot(adia[~stab], Cstar[~stab], 'b--', lw=2, label='unstable FP')
    osc = (cmax - cmin) > 1e-3
    ax[0].plot(adia[osc], cmax[osc], 'r.', ms=3); ax[0].plot(adia[osc], cmin[osc], 'r.', ms=3, label='limit cycle')
    ax[0].axvline(hd, color='k', ls=':', label=f'Hopf a={hd:.2f}')
    ax[0].axvspan(0, 1.11, color='gray', alpha=0.12, label='ATP window')
    ax[0].set_xlabel('alpha (ATP)'); ax[0].set_ylabel('C'); ax[0].set_title('Bifurcation diagram (tail unit, gamma=0.8)'); ax[0].legend(fontsize=7)
    ax[1].plot(adia, red, 'k-'); ax[1].axhline(0, color='r', ls='--'); ax[1].axvspan(0,1.11,color='gray',alpha=0.12)
    ax[1].set_xlabel('alpha'); ax[1].set_ylabel('max Re(eig)'); ax[1].set_title('Leading eigenvalue (tail unit)')
    ax[2].plot(als, frac, 'g-o', ms=3, label='f_det (frac past Hopf)')
    ax[2].plot(als, aA, 'm-s', ms=3, label='sim active fraction')
    ax[2].set_xlabel('alpha'); ax[2].set_ylabel('fraction'); ax[2].set_title('Heterogeneity -> macro activity'); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "phase3_bifurcation.png", dpi=110)
    print(f"\nwrote {out/'phase3_bifurcation.npz'} and .png")

if __name__ == "__main__":
    main()
