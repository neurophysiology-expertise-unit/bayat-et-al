"""Figure 5 (PRE version): ring-map range law, Eq. (range), against the measured front extent.

Reads processed_data/extent_scaling_law.npz (fit constants, bounded mask; phase4_scaling.py) and
processed_data/front_vs_coupling.npz (per-seed extents). Mean +- SD across seeds (ddof = 0, the manuscript's
ensemble convention). Colours as in Fig. coupling C (x1 dark grey, x2 blue). x4 is not drawn: the law predicts
unbounded propagation there for every sampled gamma.
Run: python fig_ringmap.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import apply_style, save_fig
from phase4_scaling import law, gamma_c

apply_style()
s = np.load("processed_data/extent_scaling_law.npz")
d = np.load("processed_data/front_vs_coupling.npz")
p, q, sc = s["fit_params"]
g, ext, ok = d["gammas"], d["extent"], s["bounded"]
fcol = {1.0: "0.15", 2.0: "#1F4E79"}
gg = np.logspace(np.log10(0.02), np.log10(1.0), 400)

fig, ax = plt.subplots(figsize=(3.4, 2.7))
for i, m in enumerate(d["mults"]):
    if m not in fcol:
        continue
    c = fcol[m]; mu, sd = ext[i].mean(1), ext[i].std(1)
    ax.errorbar(g[ok[i]], mu[ok[i]], yerr=sd[ok[i]], fmt="o", color=c, ms=3.6, elinewidth=0.9, capsize=2.2,
                label=f"×{m:g} measured")
    ax.errorbar(g[~ok[i]], mu[~ok[i]], yerr=sd[~ok[i]], fmt="o", mfc="white", color=c, ms=3.6, elinewidth=0.9,
                capsize=2.2)
    r = law(gg, m, p, q, sc)
    ax.plot(gg, np.where(np.isfinite(r), r, np.nan), "-" if m == 1.0 else "--", color=c, lw=1.4,
            label="×1 ring-map fit" if m == 1.0 else "×2 prediction (no new constants)")
    ax.axvline(gamma_c(m, p, q), color=c, lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"regeneration factor $\gamma_{\mathrm{regen}}$")
ax.set_ylabel("front extent (cells)")
ax.set_ylim(0, 50)
ax.legend(loc="upper left", fontsize=7)
save_fig(fig, "Figure_ringmap")
print("gamma_c x1 %.3f x2 %.3f" % (gamma_c(1.0, p, q), gamma_c(2.0, p, q)))
