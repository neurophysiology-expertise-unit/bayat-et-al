#!/usr/bin/env python3
"""
Figure_coupling -- what controls coordination and the front when coupling is scaled.

(A) Mean pairwise correlation rho-bar versus drive A with coupling held fixed at its A = 0.01 value
    (excitability-only drive) and multiplied by 0.25 ... 4 (phase4_constrain.py).
(B) The same at x1 coupling with one excitability channel held at its A = 0.01 value
    (noise, gamma spread, threshold) or all three (phase4_ablate.py).
(C) Focal front extent and (D) stopping time versus gamma_regen at x1 / x2 / x4 coupling, refractory
    state on (tau_ref 15 s), deterministic, L = 64, against Bowser & Khakh 2007: 100-250 um, ~15 s
    (phase4_front_coupling.py).

Every panel uses seeds 11-20 (AGENTS.md); the x1 constrain and ablation files hold seeds 11-30 and
only the first ten rows are drawn. Mean +/- SD across seeds. 25 um per cell; 1 model unit ~ 1 s.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotstyle import apply_style, clean_spines, panel_label, save_fig

PD = "processed_data/"
NS = 10                                   # seeds 11-20
COL_HL = "#2E7D5B"
apply_style()


def rho(f):
    d = np.load(PD + f)
    P = d["rhobar"][:NS]
    return d["alphas"], P.mean(0), P.std(0)


fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.0))
fig.subplots_adjust(hspace=0.42, wspace=0.32)
axA, axB, axC, axD = axs.ravel()

# ---------------------------------------------------------------- (A) drive x coupling
mults = [(0.25, "constrain_map_dmult0.25_L32.npz"), (0.5, "constrain_map_dmult0.5_L32.npz"),
         (1, "constrain_excit_dmult1_L32.npz"), (2, "constrain_map_dmult2_L32.npz"),
         (4, "constrain_map_dmult4_L32.npz")]
cm = plt.get_cmap("viridis_r")
for k, (m, f) in enumerate(mults):
    a, mu, sd = rho(f)
    c = cm(0.12 + 0.8 * k / (len(mults) - 1))
    axA.plot(a, mu, "-o", color=c, ms=3.2, lw=1.5, label=f"×{m:g}")
    axA.fill_between(a, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
axA.set_xlabel("drive $A$ (coupling fixed)")
axA.set_ylabel(r"mean pairwise correlation $\bar\rho$")
axA.legend(title="coupling", loc="upper right", handlelength=1.4)
axA.set_ylim(bottom=-0.02)
clean_spines(axA); panel_label(axA, "A")

# ---------------------------------------------------------------- (B) ablations at x1
abl = [("all channels", "constrain_excit_dmult1_L32.npz", "0.15", "-"),
       ("noise held", "ablate_noise_L32.npz", "#C0392B", "-"),
       (r"$\gamma$ uniform", "ablate_gamma_L32.npz", "#E69F00", "-"),
       (r"$\theta$ held", "ablate_theta_L32.npz", "#1F4E79", "-"),
       ("all three held", "ablate_all3_L32.npz", "#6E7B8B", "--")]
for lab, f, c, ls in abl:
    a, mu, sd = rho(f)
    axB.errorbar(a, mu, yerr=sd, fmt=ls, marker="o", color=c, ecolor=c, ms=3.0, lw=1.4,
                 elinewidth=0.6, capsize=1.5, alpha=0.9, label=lab)
axB.set_xlabel("drive $A$ (coupling fixed, ×1)")
axB.set_ylabel(r"$\bar\rho$")
axB.legend(loc="upper right", handlelength=1.8)
clean_spines(axB); panel_label(axB, "B")

# ---------------------------------------------------------------- (C, D) front vs coupling
d = np.load(PD + "front_vs_coupling.npz")
g = d["gammas"]; um = 25.0
assert d["extent"].shape[2] == NS
fcol = {1.0: "0.15", 2.0: "#1F4E79", 4.0: "#C0392B"}
for i, m in enumerate(d["mults"]):
    E = d["extent"][i] * um; T = d["stop_time"][i]
    kw = dict(fmt="-o", color=fcol[m], ecolor=fcol[m], ms=3.6, lw=1.5, elinewidth=0.9, capsize=2.2,
              label=f"×{m:g}")
    axC.errorbar(g, E.mean(1), yerr=E.std(1), **kw)
    axD.errorbar(g, T.mean(1), yerr=T.std(1), **kw)
axC.axhspan(100, 250, color=COL_HL, alpha=0.16, lw=0)
axD.axhline(15, color=COL_HL, lw=1.0, ls="--")
axD.axhline(120, color="0.45", lw=0.8, ls=":")
axD.text(0.33, 16.5, "reported ~15 s", fontsize=7, color=COL_HL)
axD.text(0.026, 128, "end of 120 s window", fontsize=7, color="0.45")
for ax in (axC, axD):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.025, 0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_xticklabels(["0.025", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xlabel(r"release gain $\gamma_{\rm regen}$")
axC.set_yticks([100, 250, 500, 1000]); axC.set_yticklabels(["100", "250", "500", "1000"])
axD.set_yticks([2, 5, 15, 50, 120]); axD.set_yticklabels(["2", "5", "15", "50", "120"])
axD.set_ylim(1.5, 200)
axC.set_ylabel("front extent (µm)")
axC.legend(title="coupling", loc="upper left", handlelength=1.4)
axC.text(1.0, 105, "reported\n100–250 µm", fontsize=7, color=COL_HL, ha="right", va="bottom")
axD.set_ylabel("stopping time (s)")
clean_spines(axC); panel_label(axC, "C")
clean_spines(axD); panel_label(axD, "D")

save_fig(fig, "Figure_coupling")
