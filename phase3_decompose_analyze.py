"""Separate correlation from mechanism for Phase 3.1 (const-I0), numbers first.
 test 1: partition simulated activity by each unit's own Hopf status (past-onset vs sub-threshold).
 test 2: confound-breaker (gamma_scale=-1): does A_act follow f_det (falls) or the noise (rises)?
Run: python phase3_decompose_analyze.py"""
import numpy as np
A_FHN, B_FHN = 1.0, 0.8

def Cstar(a, g, i0c):
    I = i0c + g * a; q = -3.0 * (I - A_FHN / B_FHN)
    D = (q / 2.0) ** 2 + (0.75 / 3.0) ** 3; s = np.sqrt(D)
    return np.cbrt(-q / 2.0 + s) + np.cbrt(-q / 2.0 - s)

def onset(g, i0c, tb, agrid):
    """first alpha where trace = (1-C*^2) - b/tau_h >= 0, else inf. Vectorized over units."""
    out = np.full(g.shape, np.inf)
    for a in agrid:
        th = 10.0 / ((1.0 + 0.8 * a) * tb)
        tr = (1.0 - Cstar(a, g, i0c) ** 2) - B_FHN / th
        hit = (tr >= 0) & ~np.isfinite(out)
        out[hit] = a
    return out

agrid = np.linspace(0.0, 3.0, 601)

# ---- test 1: activity decomposition (baseline) ----
z = np.load("processed_data/phase3_perunit_baseline.npz")
al = z["alphas"]; AF = z["act_field"]; g = z["gamma_eff"]; i0c = z["i0c"]; tb = z["tau_base"]
ns, na, N = AF.shape
ons = np.stack([onset(g[s], i0c[s], tb[s], agrid) for s in range(ns)])   # (ns, N)
print("=== TEST 1: activity decomposition by each unit's own Hopf status (const-I0) ===")
print(f"  unit-mean act_field == aggregate check: {AF.mean((0,2)).max():.4f} (peak A_act)\n")
print(f"  {'alpha':>6} {'A_tot':>7} {'A_pastHopf':>11} {'A_subthr':>9} {'%past':>6} {'%units_past':>11}")
tot_past = 0.0; tot_all = 0.0
for j, a in enumerate(al):
    past = ons <= a                                   # (ns,N) bool
    af = AF[:, j, :]
    A_past = af[past].sum() / (ns * N) if past.any() else 0.0
    A_tot = af.sum() / (ns * N)
    A_sub = A_tot - A_past
    fpast = 100 * A_past / A_tot if A_tot > 1e-9 else 0.0
    upast = 100 * past.mean()
    tot_past += af[past].sum() if past.any() else 0.0; tot_all += af.sum()
    print(f"  {a:6.3f} {A_tot:7.4f} {A_past:11.4f} {A_sub:9.4f} {fpast:5.1f}% {upast:10.2f}%")
print(f"\n  integrated over alpha: {100*tot_past/tot_all:.1f}% of ALL simulated activity comes from "
      f"units past their own Hopf onset")
print(f"  -> {'SUB-THRESHOLD (noise) DOMINATES' if tot_past/tot_all < 0.5 else 'HOPF-RECRUITED DOMINATES'}")

# ---- test 2: confound-breaker ----
z2 = np.load("processed_data/phase3_perunit_gammaflip.npz")
AF2 = z2["act_field"]; g2 = z2["gamma_eff"]; i2 = z2["i0c"]; tb2 = z2["tau_base"]
ons2 = np.stack([onset(g2[s], i2[s], tb2[s], agrid) for s in range(ns)])
sigma_eff = 0.0233238 * (1.0 + 4.0 * al)             # rises with alpha regardless
print("\n=== TEST 2: gamma_scale=-1 (drive & f_det FALL with alpha; sigma_eff RISES) ===")
print(f"  {'alpha':>6} {'f_det':>8} {'A_act':>8} {'sigma_eff':>10}")
fd2 = np.array([np.mean(ons2 <= a) for a in al]); AA2 = AF2.mean((0, 2))
for a, fd, aa, se in zip(al, fd2, AA2, sigma_eff):
    print(f"  {a:6.3f} {fd:8.4f} {aa:8.4f} {se:10.4f}")
d_fdet = fd2[-1] - fd2[0]; d_act = AA2[-1] - AA2[0]
print(f"\n  f_det change 0->1.11: {d_fdet:+.4f}   A_act change: {d_act:+.4f}   "
      f"sigma_eff change: {sigma_eff[-1]-sigma_eff[0]:+.4f}")
print(f"  -> A_act tracks {'NOISE/sigma_eff (f_det was a correlate)' if (d_act>0 and d_fdet<=0) else 'f_det (mechanism real)' if (np.sign(d_act)==np.sign(d_fdet)) else 'AMBIGUOUS'}")
