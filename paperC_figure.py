#!/usr/bin/env python3
"""
paperC_figure.py — figure for short paper C (not a paper-A figure). Every number comes from a committed npz.

(A) heterogeneous lattice (paper-A model), mean final extent of seeds 11-20 over coupling multiplier d x
    regeneration factor gamma_regen (paperC_phase_map.npz). Markers: ridge peak at L = 128 (squares) and with
    noise on (triangles) at d = 3, 4, 6 (paperC_robust.npz).
(B) the ridge in three conditions at d = 4 (L = 64 noise off; L = 128 noise off; L = 64 noise on, evoked front),
    mean +- SD over seeds, against gamma_regen*d (paperC_phase_map.npz, paperC_robust.npz). Grey lines: the
    n = 1 (solid) and n = 2 (dashed) gain-saturation values of gamma*d.
(C) homogeneous 2-D lattice, 5x5 seed (paperC_mechanism.npz).
(D) homogeneous 1-D chain (paperC_mechanism.npz) = the axis-planar front, exactly (paperC_direction D1).
(E) gamma_regen = 1, homogeneous lattice: largest coupling at which each front geometry propagates
    (bar to the last propagating d; whisker and tick to the first failing d). Open diamonds: the exact planar
    reduction's prediction for every non-axis direction, the axis bracket halved (paperC_direction.npz,
    paperC_facet.npz, paperC_critical_radius.npz).
(F) axis facet length: shortest propagating facet of 5-row rectangles (line) and disks by the
    length of their tip row (filled = propagates, open = fails) (paperC_facet.npz). Not a single threshold:
    disks propagate with shorter tips than rectangles (facet test F1/F2 failed as pre-registered).
Lines in A, C, D (analytic gain-saturation lines of the inheritance map, n simultaneously active upstream
neighbours): n*gamma*d*<D0>/SCALE = 1 - G_FLOOR, n = 1 (solid), n = 2 (dashed). Drawn, not fitted.
White dots in A, C, D: front still advancing at 120 s or reached the lattice edge.
Run: python -s paperC_figure.py  ->  Figure_paperC.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotstyle import apply_style, clean_spines, panel_label, save_fig
from paperC_model import SCALE, G_FLOOR, T_WIN

apply_style()
UM = 25.0
P = "processed_data/"
pm = np.load(P + "paperC_phase_map.npz")
mc = np.load(P + "paperC_mechanism.npz")
rb = np.load(P + "paperC_robust.npz")
dr = np.load(P + "paperC_direction.npz")
fc = np.load(P + "paperC_facet.npz")
cr = np.load(P + "paperC_critical_radius.npz")
Dbar = float(mc["quasistatic"][3])
TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
NORM = LogNorm(75, 3200)
COL = {"L64": "#4c72b0", "L128": "#c44e52", "noise": "#55a868"}
INK = "#333333"


def gain_lines(ax, dd, colour="w"):
    for n, ls in ((1, "-"), (2, "--")):
        ax.plot(np.log10((1 - G_FLOOR) * SCALE / (n * dd * Dbar)), dd, color=colour, ls=ls, lw=1.2)


def heat(ax, d, g, E, prop, title):
    m = ax.pcolormesh(np.log10(g), d, E * UM, shading="nearest", cmap="viridis", norm=NORM)
    G, D = np.meshgrid(g, d)
    ax.scatter(np.log10(G[prop]), D[prop], s=5, c="w")
    gain_lines(ax, np.linspace(d.min(), d.max(), 200))
    ax.set_xlim(np.log10(g.min()) - 0.06, 0.06); ax.set_ylim(d.min() - 0.3, d.max() + 0.3)
    ax.set_xticks(np.log10(TICKS)); ax.set_xticklabels([str(v) for v in TICKS])
    ax.set_xlabel(r"regeneration factor $\gamma_{\mathrm{regen}}$"); ax.set_ylabel(r"coupling multiplier $d$")
    ax.set_title(title, fontsize=9)
    return m


def advancing(ext, t99, rmax):
    return (t99 >= 0.9 * T_WIN) | (ext >= 0.95 * rmax)


def bracket(dgrid, prop):
    """(last propagating d, first failing d) along an increasing d grid."""
    prop = np.asarray(prop, bool); f = np.where(~prop)[0]
    if len(f) == 0:
        return dgrid[-1], np.nan
    return (dgrid[f[0] - 1] if f[0] > 0 else 0.0), dgrid[f[0]]


fig, axs = plt.subplots(2, 3, figsize=(10.8, 6.6), constrained_layout=True)

# A — heterogeneous map + robustness peaks
d, g = pm["d_mults"], pm["gammas"]
E, T = pm["extent"], pm["stop_time"]
m = heat(axs[0, 0], d, g, E.mean(-1), advancing(E, T, 32 * np.sqrt(2)).mean(-1) >= 0.5,
         "heterogeneous lattice (10 seeds)")
pts = [tuple(p) for p in rb["points"]]
GR = sorted({p[1] for p in pts if p[0] == 4.0}, reverse=True)
for key, mk, c in (("l128", "s", COL["L128"]), ("noise", "^", COL["noise"])):
    for dd in (3.0, 4.0, 6.0):
        mean = np.array([rb[key][pts.index((dd, gg)), :, 0].mean() for gg in GR])
        axs[0, 0].plot(np.log10(GR[int(np.argmax(mean))]), dd, mk, ms=6, mfc="none", mec=c, mew=1.4)

# B — ridge at d = 4 in three conditions, against gamma*d
ax = axs[0, 1]
dd = 4.0
ip = list(d).index(dd); gl = np.array(GR)
series = [("L = 64", COL["L64"], "-", "o", np.array([E[ip, list(g).index(gg)] for gg in GR])),
          ("L = 128", COL["L128"], "--", "s", np.array([rb["l128"][pts.index((dd, gg)), :, 0] for gg in GR])),
          ("noise on", COL["noise"], ":", "^", np.array([rb["noise"][pts.index((dd, gg)), :, 0] for gg in GR]))]
for lab, c, ls, mk, A in series:
    ax.errorbar(gl * dd, A.mean(1) * UM, yerr=A.std(1) * UM, color=c, ls=ls, marker=mk, ms=4, lw=1.2, capsize=2,
                label=lab)
for n, ls in ((1, "-"), (2, "--")):
    ax.axvline((1 - G_FLOOR) * SCALE / (n * Dbar), color="#9e9e9e", ls=ls, lw=1)
ax.set_xscale("log"); ax.set_xlabel(r"$\gamma_{\mathrm{regen}}\cdot d$"); ax.set_ylabel("final extent (µm)")
ax.set_title(r"ridge at $d = 4$ in three conditions", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper right")

# C, D — homogeneous lattice and chain
dh, gh = mc["d_hom"], mc["gammas"]
H, C = mc["hom2d"], mc["chain"]
heat(axs[1, 0], dh, gh, H[..., 0], advancing(H[..., 0], H[..., 1], 32 * np.sqrt(2)), "homogeneous lattice, 5×5 seed")
heat(axs[1, 1], dh, gh, C[..., 0], advancing(C[..., 0], C[..., 1], 128.0), "homogeneous chain (= axis-planar front)")
fig.colorbar(m, ax=axs[:, 0], label="front extent (µm)", shrink=0.6)

# E — propagation limit by front geometry (gamma = 1, homogeneous)
ax = axs[0, 2]
rows = []
dd_ = dr["d"]; dirs = [tuple(x) for x in dr["dirs"]]
ax_lo, ax_hi = bracket(dd_, dr["propagates"][dirs.index((1, 0))])
rows.append(("planar, axis", ax_lo, ax_hi))
for pq in [(1, 1), (2, 1), (3, 1), (3, 2)]:
    lo, hi = bracket(dd_, dr["propagates"][dirs.index(pq)])
    rows.append((f"planar, ({pq[0]},{pq[1]})", lo, hi))
fd = fc["d"]; tips = fc["tips"]
sq = dr["squares"]; sqd = dr["sq_d"]
for a, w in enumerate(dr["sq_w"]):
    lo, hi = bracket(sqd, sq[a, :, 0] >= 20)
    rows.append((f"square, side {2*w+1}", lo, hi))
for k in (4, 1, 0):
    lo, hi = bracket(fd, fc["disk_prop"][:, k])
    rows.append((f"disk R0 {fc['R0_disk'][k]:g}, tip {tips[k]}", lo, hi))
crd = cr["d_mults"]; R0s = list(cr["R0s"])
lo, hi = bracket(crd, cr["propagates"][0][:, R0s.index(4.5)])
rows.append(("disk R0 4.5, tip 5", lo, hi))
ny = len(rows)
for i, (lab, lo, hi) in enumerate(rows):
    y = ny - 1 - i
    ax.plot([0, lo], [y, y], color=INK, lw=3, solid_capstyle="butt")
    if np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=INK, lw=1)
        ax.plot(hi, y, "|", color=INK, ms=7)
for i in range(1, 5):
    ax.plot([ax_lo / 2, ax_hi / 2], [ny - 1 - i] * 2, "D", ms=4, mfc="none", mec=COL["L128"], mew=1.2)
ax.set_yticks(range(ny)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
ax.set_xlabel(r"coupling multiplier $d$ ($\gamma_{\mathrm{regen}} = 1$)"); ax.set_xlim(0, 17)
ax.set_title("propagates up to (bar), fails by (tick)", fontsize=9)
ax.text(16.8, ny - 3.0, "diamonds: exact reduction,\nnon-axis = axis / 2", color=COL["L128"], fontsize=7, ha="right",
        va="center")

# F — facet threshold
ax = axs[1, 2]
lc = fc["l_c"]; ok = np.isfinite(lc)
ax.step(fd[ok], lc[ok], where="mid", color=INK, lw=1.4, label="5-row rectangle: shortest facet that propagates")
for k, t in enumerate(tips):
    pr = fc["disk_prop"][:, k]
    ax.plot(fd[pr], [t] * int(pr.sum()), "o", ms=5, color=COL["L64"])
    ax.plot(fd[~pr], [t] * int((~pr).sum()), "o", ms=5, mfc="none", mec=COL["L64"], mew=1.2)
ax.plot([], [], "o", color=COL["L64"], label="disk, by tip-row length: propagates")
ax.plot([], [], "o", mfc="none", mec=COL["L64"], label="disk, by tip-row length: fails")
ax.set_xlabel(r"coupling multiplier $d$ ($\gamma_{\mathrm{regen}} = 1$)"); ax.set_ylabel("axis facet length (cells)")
ax.set_title("axis facet length: rectangles vs disk tip rows", fontsize=9)
ax.set_ylim(0, 64)
ax.legend(frameon=False, fontsize=7, loc="upper left")

for k, a in zip("ABCDEF", [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[0, 2], axs[1, 2]]):
    clean_spines(a); panel_label(a, k)
save_fig(fig, "Figure_paperC")
print("wrote Figure_paperC")
