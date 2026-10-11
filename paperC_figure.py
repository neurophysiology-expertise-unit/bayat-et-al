#!/usr/bin/env python3
"""
paperC_figure.py — FEASIBILITY figure for a possible short paper C (not a paper-A figure).

Extent of the focal front (25 um/cell) over coupling multiplier d x regeneration factor gamma_regen.
(A) heterogeneous lattice, paper-A model, mean of seeds 11-20 (paperC_phase_map.npz);
(B) extent vs gamma_regen at d = 1, 2, 4, 6 (mean +- SD over seeds);
(C) homogeneous 2-D lattice (every per-cell parameter at its mean; paperC_mechanism.npz);
(D) homogeneous 1-D chain, same cells.
White dots: front still advancing at 120 s or reached the lattice edge (>= 95% of the largest distance).
White lines (drawn AFTER the data, descriptive, not pre-registered): gain-saturation lines of the inheritance
map for recruitment by n simultaneously active neighbours, n*gamma*d*<D0>/D_scale = 1 - g_floor, n = 1 (solid)
and n = 2 (dashed).
Run: python -s paperC_figure.py  ->  Figure_paperC.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotstyle import apply_style, clean_spines, panel_label, save_fig
from phase3_decremental import SCALE, G_FLOOR

apply_style()
UM = 25.0
pm = np.load("processed_data/paperC_phase_map.npz")
mc = np.load("processed_data/paperC_mechanism.npz")
Dbar = float(mc["quasistatic"][3])
TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
NORM = LogNorm(75, 3200)


def heat(ax, d, g, E, prop, title):
    m = ax.pcolormesh(np.log10(g), d, E * UM, shading="nearest", cmap="viridis", norm=NORM)
    G, D = np.meshgrid(g, d)
    ax.scatter(np.log10(G[prop]), D[prop], s=5, c="w")
    dd = np.linspace(d.min(), d.max(), 200)
    for n, ls in ((1, "-"), (2, "--")):
        ax.plot(np.log10((1 - G_FLOOR) * SCALE / (n * dd * Dbar)), dd, color="w", ls=ls, lw=1.2)
    ax.set_xlim(np.log10(g.min()) - 0.06, 0.06); ax.set_ylim(d.min() - 0.3, d.max() + 0.3)
    ax.set_xticks(np.log10(TICKS)); ax.set_xticklabels([str(v) for v in TICKS])
    ax.set_xlabel(r"regeneration factor $\gamma_{\mathrm{regen}}$"); ax.set_ylabel(r"coupling multiplier $d$")
    ax.set_title(title, fontsize=9)
    return m


def advancing(ext, t99, rmax):
    return (t99 >= 0.9 * 120.0) | (ext >= 0.95 * rmax)


fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
d, g = pm["d_mults"], pm["gammas"]
E, T = pm["extent"], pm["stop_time"]
prop = advancing(E, T, 32 * np.sqrt(2)).mean(-1) >= 0.5
m = heat(axs[0, 0], d, g, E.mean(-1), prop, "heterogeneous lattice (10 seeds)")

ax = axs[0, 1]
for mm, c in zip([1.0, 2.0, 4.0, 6.0], ["#9e9e9e", "#4c72b0", "#c44e52", "#55a868"]):
    i = list(d).index(mm)
    ax.errorbar(g, E[i].mean(-1) * UM, yerr=E[i].std(-1) * UM, marker="o", ms=3, lw=1, capsize=2,
                color=c, label=f"×{mm:g}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"regeneration factor $\gamma_{\mathrm{regen}}$")
ax.set_ylabel("final extent (µm)"); ax.legend(frameon=False, fontsize=8, title="coupling", title_fontsize=8)

dh, gh = mc["d_hom"], mc["gammas"]
H, C = mc["hom2d"], mc["chain"]
heat(axs[1, 0], dh, gh, H[..., 0], advancing(H[..., 0], H[..., 1], 32 * np.sqrt(2)), "homogeneous lattice")
heat(axs[1, 1], dh, gh, C[..., 0], advancing(C[..., 0], C[..., 1], 128.0), "homogeneous chain")
fig.colorbar(m, ax=axs[:, 0], label="front extent (µm)", shrink=0.6)

for k, a in zip("ABCD", axs.flat):
    clean_spines(a); panel_label(a, k)
save_fig(fig, "Figure_paperC")
print("wrote Figure_paperC")
