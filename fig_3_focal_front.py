#!/usr/bin/env python3
"""
Figure 3 -- focal initiation produces a coherent deterministic front; under noise there is no front.

(A) Activation time versus distance from the initiation site, deterministic condition
    (representative seed shown; speed quoted as the 10-seed ensemble mean +/- SD).
(B,C) Activation maps for the same seed, deterministic and noisy: a coherent expanding
    front versus a field that ignites throughout.
(D) Across-seed coefficient of variation of the three measured quantities. Extent and
    activated fraction reproduce tightly in both conditions; only the front-timing fit
    degrades under noise, which is why no noisy front speed is reported.
(E) Spontaneous transient rate at A=0 versus baseline excitability (negative control),
    with the somatic and process rates reported experimentally.

Distances use 25 um per cell (Methods); the 50 um values appear in the caption only.
All panels read provenance-stamped npz files -- no panel is built from a log.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plotstyle import apply_style, clean_spines, panel_label, save_fig

UM = 25.0                # primary lattice spacing (um per cell)
COL_DET = "#1F4E79"
COL_NOISY = "#C0392B"
COL_SIM = "#6E7B8B"
COL_HL = "#2E7D5B"

apply_style()

d3 = np.load("processed_data/fig3_focal.npz", allow_pickle=True)
d2 = np.load("processed_data/i0_fine.npz", allow_pickle=True)
import json
P = json.loads(str(d3["__params__"]))
dtf, patch, L = P["dt_frame"], P["patch"], P["L"]
rep_seed = P["representative_seed"]

r = d3["radius"]
tact_d, tact_n = d3["tact_deterministic"], d3["tact_noisy"]
E = {k: d3[f"extent_{k}_all"] for k in ("deterministic", "noisy")}
F = {k: d3[f"frac_{k}_all"] for k in ("deterministic", "noisy")}
S = {k: d3[f"speed50_{k}_all"] for k in ("deterministic", "noisy")}

fig = plt.figure(figsize=(11.4, 7.0))
gs = GridSpec(2, 6, figure=fig, hspace=0.42, wspace=1.25, height_ratios=[1.0, 1.18])

# ---------------------------------------------------------------- (A) t_act vs distance
axA = fig.add_subplot(gs[0, 0:2])
m = (tact_d >= 0) & (r > patch)
rr, tt = r[m] * UM, tact_d[m].astype(float) * dtf
axA.plot(rr, tt, ".", color=COL_DET, ms=2.2, alpha=0.45, rasterized=True)
sl, ic = np.polyfit(rr, tt, 1)
xs = np.linspace(rr.min(), rr.max(), 50)
axA.plot(xs, sl * xs + ic, "-", color="k", lw=1.3)
v_um = np.nanmean(S["deterministic"]) * UM / 50.0        # stored at 50 um; convert to 25 um
v_sd = np.nanstd(S["deterministic"]) * UM / 50.0
axA.text(0.05, 0.90, f"{v_um:.1f} $\\pm$ {v_sd:.1f} " + r"$\mu$m s$^{-1}$" + "\n(10 seeds)",
         transform=axA.transAxes, fontsize=7.4, color="k", va="top")
axA.set_title("deterministic", fontsize=8.5, color=COL_DET, pad=4)
axA.set_xlabel(r"distance from initiation ($\mu$m)")
axA.set_ylabel("activation time (s)")
axA.text(0.98, 0.04, f"seed {rep_seed}", transform=axA.transAxes, fontsize=6.4,
         color="0.45", ha="right")
clean_spines(axA); panel_label(axA, "A")

# ---------------------------------------------------------------- (B,C) activation maps
tmax = max(tact_d.max(), tact_n.max()) * dtf
for j, (tag, tac, col) in enumerate((("deterministic", tact_d, COL_DET),
                                     ("noisy", tact_n, COL_NOISY))):
    ax = fig.add_subplot(gs[0, 2 + 2 * j: 4 + 2 * j])
    img = np.where(tac >= 0, tac.astype(float) * dtf, np.nan)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("0.88")
    ext = [-L / 2 * UM, L / 2 * UM, -L / 2 * UM, L / 2 * UM]
    h = ax.imshow(img, cmap=cm, vmin=0, vmax=tmax, extent=ext, origin="lower")
    ax.set_title(tag, fontsize=8.5, color=col, pad=4)
    ax.set_xlabel(r"x ($\mu$m)")
    if j == 0:
        ax.set_ylabel(r"y ($\mu$m)")
    cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("activation time (s)", fontsize=6.8); cb.ax.tick_params(labelsize=6)
    ax.tick_params(labelsize=7)
    panel_label(ax, "B" if j == 0 else "C")

# ---------------------------------------------------------------- (D) reproducibility
axD = fig.add_subplot(gs[1, 0:3])
def cv(v):
    return 100.0 * np.nanstd(v) / abs(np.nanmean(v))
groups = ["front extent", "activated fraction", "front-timing fit"]
vals_d = [cv(E["deterministic"]), cv(F["deterministic"]), cv(S["deterministic"])]
vals_n = [cv(E["noisy"]), cv(F["noisy"]), cv(S["noisy"])]
x = np.arange(3); w = 0.36
axD.bar(x - w / 2, vals_d, w, color=COL_DET, label="deterministic")
axD.bar(x + w / 2, vals_n, w, color=COL_NOISY, label="noisy")
for xi, v in zip(x - w / 2, vals_d):
    axD.text(xi, v + 0.6, f"{v:.1f}%", ha="center", fontsize=6.8, color="0.2")
for xi, v in zip(x + w / 2, vals_n):
    axD.text(xi, v + 0.6, f"{v:.1f}%", ha="center", fontsize=6.8, color="0.2")
axD.set_xticks(x); axD.set_xticklabels(groups, fontsize=7.6)
axD.set_ylabel("across-seed CV (%)")
axD.set_ylim(0, 27)
axD.legend(fontsize=7, loc="upper left", handlelength=1.4)
# derived from the same arrays the bars use, so it cannot go stale against the data
axD.text(0.035, 0.63, f"a quantity measured to {vals_d[2]:.1f}%\n"
                      f"in one condition and {vals_n[2]:.1f}%\n"
                      "in the other is not the\nsame quantity",
         transform=axD.transAxes, fontsize=6.8, color="0.3", ha="left", va="top",
         style="italic")
clean_spines(axD); panel_label(axD, "D", dx=-0.055)

# ---------------------------------------------------------------- (E) spontaneous control
axE = fig.add_subplot(gs[1, 3:6])
b, rm, rs = d2["baselines"], d2["rate_mean"], d2["rate_sd"]
axE.axhspan(0.121 - 0.098, 0.121 + 0.098, color=COL_HL, alpha=0.16, lw=0)
axE.axhline(0.121, color=COL_HL, lw=1.0, ls="-")
axE.text(0.492, 0.248, "somatic, in vivo (Hirase et al.)", fontsize=6.6, color=COL_HL,
         ha="right")
axE.axhline(0.65, color="0.45", lw=1.0, ls="--")
axE.text(0.492, 0.672, "astrocytic processes (Stobart et al.)", fontsize=6.6, color="0.45",
         ha="right")
axE.errorbar(b, rm, yerr=rs, fmt="-o", color=COL_SIM, ecolor=COL_SIM, ms=4.2, lw=1.6,
             elinewidth=1.0, capsize=3)
axE.axvline(0.42, color="k", lw=1.0, ls=":")
axE.text(0.422, 0.735, "selected", fontsize=6.6, color="k", ha="left", va="top")
axE.set_xlabel(r"baseline excitability $I_0^{\rm base}$")
axE.set_ylabel(r"transients min$^{-1}$ per cell")
axE.set_xlim(0.34, 0.505); axE.set_ylim(-0.05, 1.06)
# NOT "silent": the zero is the peak detector returning no events, while the detector-free
# occupancy of Phi(C) rises smoothly across this whole range (see Methods, third caveat).
axE.text(0.36, 0.035, "no detected transients", fontsize=7, color="0.3", ha="center",
         clip_on=True)
clean_spines(axE); panel_label(axE, "E", dx=-0.055)

save_fig(fig, "Figure_3")
print(f"deterministic speed {v_um:.2f} +/- {v_sd:.2f} um/s at {UM:.0f} um per cell "
      f"({np.nanmean(S['deterministic']):.1f} +/- {np.nanstd(S['deterministic']):.1f} at 50 um)")
print("CV det :", [f"{v:.2f}%" for v in vals_d])
print("CV nois:", [f"{v:.2f}%" for v in vals_n])
print("wrote Figure_3.pdf / Figure_3.png")
