#!/usr/bin/env python3
"""
Figure 2 -- ATP sets network activity through single-unit excitability.

(A) Single-unit bifurcation diagram for a positive-gamma unit: the fixed point loses
    stability at a Hopf bifurcation and a limit cycle is born.
(B) Leading Jacobian eigenvalue for the same unit; the oscillation period stays finite
    across onset (inset), identifying the transition as Hopf rather than SNIC.
(C) Deterministic recruitment of the heterogeneous population, f_det(A), against the
    simulated mean active fraction (mean +/- SD over 40 seeds).
(D) Distribution of single-unit Hopf onsets over the sampled population.
(E) Leave-one-ATP-channel-out test: recruitment with each ATP-dependent channel held
    at its A=0 value.

All panels read provenance-stamped npz files written by figdata_fig2.py and
figdata_fig2d.py -- no panel is built from a log.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plotstyle import apply_style, clean_spines, panel_label, save_fig

COL_FP = "#1F4E79"      # fixed point / deterministic theory
COL_LC = "#C0392B"      # limit cycle / instability
COL_SIM = "#6E7B8B"     # simulation
COL_HL = "#2E7D5B"      # recruitment highlight
WINDOW = 1.11           # ATP window studied

apply_style()

d2 = np.load("processed_data/fig2_excitability.npz", allow_pickle=True)
d2d = np.load("processed_data/fig2d_gamma_freeze.npz", allow_pickle=True)

adia, Cstar, stab = d2["adia"], d2["Cstar"], d2["stab"].astype(bool)
cmin, cmax, per, re = d2["cmin"], d2["cmax"], d2["period"], d2["re"]
hopf = float(d2["hopf_tail"][0])
als, f_det, A_act, A_sd = d2["alphas"], d2["f_det"], d2["A_act"], d2["A_act_sd"]
pear, spear = float(d2["stats"][0]), float(d2["stats"][1])
snic_d, snic_p = d2["snic_dist"], d2["snic_period"]

onset = d2d["onset"]
fnames = [str(s) for s in d2d["freeze_names"]]
frec = d2d["freeze_recruited"]
base_rec = float(d2d["baseline_recruited"][0])
srho = dict(zip([str(s) for s in d2d["spearman_names"]], d2d["spearman_vals"]))
M = onset.size

fig = plt.figure(figsize=(11.4, 6.9))
gs = GridSpec(2, 6, figure=fig, hspace=0.48, wspace=1.15)

# ---------------------------------------------------------------- (A) bifurcation
axA = fig.add_subplot(gs[0, 0:2])
axA.axvspan(0, WINDOW, color="0.85", alpha=0.55, lw=0, zorder=0)
axA.plot(adia[stab], Cstar[stab], "-", color=COL_FP, lw=2.0, label="stable fixed point")
axA.plot(adia[~stab], Cstar[~stab], "--", color=COL_FP, lw=1.6, label="unstable fixed point")
osc = (cmax - cmin) > 1e-3
axA.plot(adia[osc], cmax[osc], ".", color=COL_LC, ms=2.6)
axA.plot(adia[osc], cmin[osc], ".", color=COL_LC, ms=2.6, label="limit cycle")
axA.axvline(hopf, color="k", ls=":", lw=1.2)
axA.set_ylim(-2.35, 2.45)
axA.annotate(f"Hopf, $A$ = {hopf:.2f}", xy=(hopf, 0.35), xytext=(hopf + 0.16, 0.55),
             fontsize=7.2, color="k", ha="left")
axA.set_xlabel("ATP level $A$"); axA.set_ylabel("calcium $C$")
axA.set_xlim(0, 2.0)
axA.legend(fontsize=6.6, loc="lower left", handlelength=1.5)
axA.text(0.5 * WINDOW, 2.28, "ATP window", fontsize=6.6, color="0.4", ha="center", va="top")
clean_spines(axA); panel_label(axA, "A")

# ---------------------------------------------------------------- (B) eigenvalue
axB = fig.add_subplot(gs[0, 2:4])
axB.axvspan(0, WINDOW, color="0.85", alpha=0.55, lw=0, zorder=0)
axB.plot(adia, re, "-", color=COL_FP, lw=1.8)
axB.axhline(0.0, color=COL_LC, ls="--", lw=1.0)
axB.axvline(hopf, color="k", ls=":", lw=1.2)
axB.set_xlabel("ATP level $A$"); axB.set_ylabel(r"max Re $\lambda$")
axB.set_xlim(0, 2.0); axB.set_ylim(-0.55, 0.95)
axB.text(0.5 * WINDOW, 0.90, "ATP window", fontsize=6.6, color="0.4", ha="center", va="top")
clean_spines(axB); panel_label(axB, "B")
axins = axB.inset_axes([0.52, 0.13, 0.42, 0.28])
axins.plot(snic_d, snic_p, "o-", color=COL_LC, ms=3, lw=1.2)
axins.set_ylim(0, 40)
axins.set_xlabel(r"$A-A_{\rm Hopf}$", fontsize=6.2, labelpad=1)
axins.set_ylabel("period", fontsize=6.2, labelpad=1)
axins.tick_params(labelsize=5.6, length=2.5, pad=1)
for s in ("top", "right"):
    axins.spines[s].set_visible(False)

# ---------------------------------------------------------------- (C) recruitment
axC = fig.add_subplot(gs[0, 4:6])
axC.plot(als, 100 * f_det, "-o", color=COL_HL, ms=3.2, lw=1.6,
         label=r"$f_{\rm det}$ (past Hopf)")
axC.errorbar(als, 100 * A_act, yerr=100 * A_sd, fmt="-s", color=COL_SIM, ecolor=COL_SIM,
             ms=3.0, lw=1.6, elinewidth=0.9, capsize=2, label="simulated active fraction")
axC.set_xlabel("ATP level $A$"); axC.set_ylabel("fraction of cells (%)")
axC.set_xlim(0, WINDOW)
axC.legend(fontsize=6.6, loc="upper left", handlelength=1.5)
axC.text(0.04, 0.60, f"Spearman $\\rho$ = {spear:.3f}\nPearson $r$ = {pear:.3f}",
         transform=axC.transAxes, fontsize=6.8, color="0.25")
clean_spines(axC); panel_label(axC, "C")

# ---------------------------------------------------------------- (D) onset distribution
axD = fig.add_subplot(gs[1, 0:3])
fin = np.isfinite(onset)
axD.axvspan(0, WINDOW, color="0.85", alpha=0.55, lw=0, zorder=0)
axD.hist(onset[fin], bins=np.linspace(0, 3, 121), color=COL_FP, alpha=0.85, lw=0)
axD.set_xlabel(r"single-unit Hopf onset $A_{\rm Hopf}$")
axD.set_ylabel(f"units (of {M:,})")
axD.set_xlim(0, 3.0)
never = 100 * np.mean(~fin)
mean_hopf = float(d2["mean_hopf"][0])
mean_label = ("population-mean unit: never" if not np.isfinite(mean_hopf)
              else f"population-mean onset: {mean_hopf:.2f}")
axD.text(0.36, 0.86, f"{100*base_rec:.1f}% recruited inside the ATP window\n"
                     f"{never:.1f}% never bifurcate on [0, 3]\n"
                     f"{mean_label}",
         transform=axD.transAxes, fontsize=7.0, color="0.2", va="top")
axD.text(0.5 * WINDOW / 3.0, 0.965, "ATP window", transform=axD.transAxes,
         fontsize=6.6, color="0.4", ha="center", va="top")
clean_spines(axD); panel_label(axD, "D", dx=-0.055)

# ---------------------------------------------------------------- (E) freeze test
axE = fig.add_subplot(gs[1, 3:6])
lab = {"gamma": r"freeze $\gamma(A)$", "tau_base": r"freeze $\tau_h(A)$",
       "I0_slope": r"freeze $I_0(A)$"}
names = ["baseline"] + fnames
vals = np.concatenate([[base_rec], frec])
se = np.sqrt(vals * (1 - vals) / M)                    # binomial SE over the sampled population
cols = [COL_SIM] + [COL_LC if n == "gamma" else COL_FP for n in fnames]
xs = np.arange(len(names))
axE.bar(xs, 100 * vals, yerr=100 * se, color=cols, width=0.62, capsize=3,
        error_kw=dict(elinewidth=0.9))
axE.axhline(100 * base_rec, color="0.4", ls=":", lw=1.0)
axE.set_xticks(xs)
axE.set_xticklabels(["baseline"] + [lab[n] for n in fnames], fontsize=7.6)
axE.set_ylabel("recruited inside window (%)")
axE.set_ylim(0, max(80, 115 * vals.max()))
for x, v in zip(xs, vals):
    axE.text(x, 100 * v + 0.6, f"{100*v:.2f}", ha="center", fontsize=6.8, color="0.2")
axE.text(0.02, 0.90, "Spearman(onset, parameter):  "
                     rf"$\gamma$ {srho['gamma']:+.2f},  $\tau_h$ {srho['tau_base']:+.2f},  "
                     rf"$I_0$ slope {srho['I0_slope']:+.2f}",
         transform=axE.transAxes, fontsize=6.8, color="0.25")
clean_spines(axE); panel_label(axE, "E", dx=-0.055)

save_fig(fig, "Figure_2")
print("wrote Figure_2.pdf / Figure_2.png")
