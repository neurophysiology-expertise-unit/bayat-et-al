#!/usr/bin/env python3
"""
Figure 4 -- a refractory state suppresses spurious nucleation; decremental release bounds the front.

(A) Spontaneous nucleation rate versus refractory duration.
(B) Focal front extent and the activated fraction under noise, versus refractory duration:
    the refractory state removes the noise-driven ignition but leaves the extent untouched.
(C) Front extent versus the decremental release gain, against the reported wave extent.
(D) Front speed over the same sweep: the decrement bounds the extent without degrading speed.
(E) Linearity control: the same restricted-radius fit applied to a known-good, unbounded front
    scores no better than the bounded front, so R^2 > 0.9 is unreachable at that extent.

Distances use 25 um per cell (Methods); the 50 um values appear in the caption only.
All quantities are mean +/- SD over 10 seeds, read from a provenance-stamped npz.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plotstyle import apply_style, clean_spines, panel_label, save_fig

UM = 25.0
COL_DET = "#1F4E79"
COL_NOISY = "#C0392B"
COL_SIM = "#6E7B8B"
COL_HL = "#2E7D5B"
EXT_LO, EXT_HI = 100.0, 250.0        # reported astrocyte wave extent (um)

apply_style()

d = np.load("processed_data/fig4_mechanisms_ens.npz", allow_pickle=True)
P = json.loads(str(d["__params__"]))
taus, gam, rmax = d["taus"], d["gammas"], d["control_rmax"]
nseed = len(d["seeds"])

fig = plt.figure(figsize=(11.4, 7.0))
gs = GridSpec(2, 6, figure=fig, hspace=0.55, wspace=1.30)

# ---------------------------------------------------------------- (A) nucleation
axA = fig.add_subplot(gs[0, 0:3])
x = np.arange(len(taus))
axA.bar(x, d["nucleation_rate_mean"], yerr=d["nucleation_rate_sd"], width=0.55,
        color=[COL_NOISY] + [COL_DET] * (len(taus) - 1), capsize=3,
        error_kw=dict(elinewidth=0.9))
for xi, m, s in zip(x, d["nucleation_rate_mean"], d["nucleation_rate_sd"]):
    axA.text(xi, m + s + 0.04, f"{m:.2f}", ha="center", fontsize=6.9, color="0.2")
axA.set_xticks(x); axA.set_xticklabels([f"{t:.0f}" for t in taus])
axA.set_xlabel(r"refractory duration $\tau_{\rm ref}$ (s)")
axA.set_ylabel("nucleation events\n" + r"(10$^{3}$ cells)$^{-1}$ s$^{-1}$")
axA.set_ylim(0, max(d["nucleation_rate_mean"]) * 1.30)
clean_spines(axA); panel_label(axA, "A", dx=-0.055)

# ---------------------------------------------------------------- (B) extent + noisy fraction
axB = fig.add_subplot(gs[0, 3:6])
axB.errorbar(taus, d["extent_vs_tau_mean"], yerr=d["extent_vs_tau_sd"], fmt="-o",
             color=COL_DET, ecolor=COL_DET, ms=4.2, lw=1.6, elinewidth=1.0, capsize=3,
             label="front extent (deterministic)")
axB.set_xlabel(r"refractory duration $\tau_{\rm ref}$ (s)")
axB.set_ylabel("front extent (cells)", color=COL_DET)
axB.tick_params(axis="y", colors=COL_DET)
axB.set_ylim(0, 40)
axB2 = axB.twinx()
axB2.errorbar(taus, 100 * d["frac_noisy_vs_tau_mean"], yerr=100 * d["frac_noisy_vs_tau_sd"],
              fmt="--s", color=COL_NOISY, ecolor=COL_NOISY, ms=4.0, lw=1.5, elinewidth=1.0,
              capsize=3, label="activated fraction (noisy)")
axB2.set_ylabel("activated fraction under noise (%)", color=COL_NOISY)
axB2.tick_params(axis="y", colors=COL_NOISY)
axB2.set_ylim(0, 110)
axB2.spines["top"].set_visible(False)
h1, l1 = axB.get_legend_handles_labels(); h2, l2 = axB2.get_legend_handles_labels()
axB.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="center right", handlelength=1.8)
clean_spines(axB); panel_label(axB, "B", dx=-0.06)

# ---------------------------------------------------------------- (C) extent vs gamma_regen
axC = fig.add_subplot(gs[1, 0:2])
axC.axhspan(EXT_LO / UM, EXT_HI / UM, color=COL_HL, alpha=0.16, lw=0)
axC.axhline(EXT_LO / UM, color=COL_HL, lw=0.8); axC.axhline(EXT_HI / UM, color=COL_HL, lw=0.8)
axC.errorbar(gam, d["extent_vs_gamma_mean"], yerr=d["extent_vs_gamma_sd"], fmt="-o",
             color=COL_DET, ecolor=COL_DET, ms=4.0, lw=1.6, elinewidth=1.0, capsize=2.5)
axC.set_xlabel(r"regeneration factor $\gamma_{\rm regen}$")
axC.set_ylabel("front extent (cells)")
axC.set_xlim(0, 1.05)
sec = axC.secondary_yaxis("right", functions=(lambda v: v * UM, lambda v: v / UM))
sec.set_ylabel(r"extent ($\mu$m)", fontsize=8); sec.tick_params(labelsize=7)
axC.text(0.30, 0.16, "reported extent", transform=axC.transAxes, fontsize=6.6, color=COL_HL)
clean_spines(axC); panel_label(axC, "C")

# ---------------------------------------------------------------- (D) speed vs gamma_regen
axD = fig.add_subplot(gs[1, 2:4])
axD.errorbar(gam, d["speed50_vs_gamma_mean"] * UM / 50.0, yerr=d["speed50_vs_gamma_sd"] * UM / 50.0,
             fmt="-o", color=COL_DET, ecolor=COL_DET, ms=4.0, lw=1.6, elinewidth=1.0, capsize=2.5)
axD.set_xlabel(r"regeneration factor $\gamma_{\rm regen}$")
axD.set_ylabel(r"front speed ($\mu$m s$^{-1}$)")
axD.set_xlim(0, 1.05); axD.set_ylim(0, None)
clean_spines(axD); panel_label(axD, "D")

# ---------------------------------------------------------------- (E) linearity control
axE = fig.add_subplot(gs[1, 4:6])
axE.plot(rmax, d["control_r2_mean"], "-o", color=COL_DET, ms=4.0, lw=1.6, label="unbounded front")
axE.fill_between(rmax, d["control_r2_mean"] - d["control_r2_sd"],
                 d["control_r2_mean"] + d["control_r2_sd"], color=COL_DET, alpha=0.18, lw=0)
i10 = int(np.argmin(np.abs(gam - 0.10)))
b_m, b_s = d["r2_vs_gamma_mean"][i10], d["r2_vs_gamma_sd"][i10]
axE.axhline(b_m, color=COL_NOISY, ls="--", lw=1.2,
            label=r"bounded front ($\gamma_{\rm regen}$ = 0.10)")
axE.axhspan(b_m - b_s, b_m + b_s, color=COL_NOISY, alpha=0.14, lw=0)
axE.axhline(0.9, color="0.35", ls=":", lw=1.0)
axE.text(rmax.min() + 0.4, 0.912, "pre-set criterion", fontsize=6.6, color="0.35", ha="left")
axE.set_xlabel("fit restricted to radius $\\leq R$ (cells)")
axE.set_ylabel(r"linear $R^{2}$")
axE.set_ylim(0.5, 1.02)
axE.legend(fontsize=6.6, loc="lower right", handlelength=1.8)
clean_spines(axE); panel_label(axE, "E")

save_fig(fig, "Figure_4")
print(f"{nseed} seeds; gamma=0.10 extent "
      f"{d['extent_vs_gamma_mean'][i10]:.1f}+/-{d['extent_vs_gamma_sd'][i10]:.1f} cells, speed "
      f"{d['speed50_vs_gamma_mean'][i10]*UM/50:.1f}+/-{d['speed50_vs_gamma_sd'][i10]*UM/50:.1f} um/s")
print("wrote Figure_4.pdf / Figure_4.png")
