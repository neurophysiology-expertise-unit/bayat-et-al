#!/usr/bin/env python3
"""
Figure 1 -- single-astrocyte dynamics across extracellular ATP.

Left column: representative calcium traces in three ATP regimes (low / intermediate /
high), each with a coupling schematic (a 3-astrocyte motif progressively uncoupled as
ATP rises). Right column: transient rate, mean peak amplitude and resting baseline
versus ATP, mean +/- SD over 10 seeds of a 10 min record.

Transients are prominent peaks (height + prominence thresholds), so a genuine Ca2+
transient is counted regardless of ATP; peak amplitude is therefore comparable across
regimes and the ATP dependence is isolated into the transient rate.

Time is seconds via a fixed conversion from model units (1 unit ~ 1 s); the oscillation
period (~20-30 s) then matches spontaneous astrocytic Ca2+ oscillations. This stylized
single-unit parametrization illustrates the three regimes and is not identical to the
network model of Figs 2-4.

All panels read processed_data/fig1_single_cell.npz (figdata_fig1.py).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from plotstyle import apply_style, clean_spines, save_fig

COND_LABELS = ["Low ATP", "Intermediate ATP", "High ATP"]
COND_SUBTITLE = ["noise-driven excitable", "oscillation onset", "irregular high-frequency"]

# ============================================================
# COUPLING SCHEMATIC
# ============================================================
NODE_C = "#6E7B8B"
BROKEN = "#C0392B"
NODE_R = 0.42
TRI = np.array([(-1.1, 0.75), (1.1, 0.75), (0.0, -1.05)])
NLAB = ["A", "B", "C"]


def draw_nodes(ax, pts):
    for (x, y), lab in zip(pts, NLAB):
        ax.add_patch(Circle((x, y), NODE_R, facecolor=NODE_C,
                            edgecolor="black", lw=1.0, zorder=3))
        ax.text(x, y, lab, ha="center", va="center",
                color="white", fontweight="bold", fontsize=9, zorder=4)


def _trim(p0, p1):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    d = d / np.hypot(*d)
    return p0 + d * NODE_R, p1 - d * NODE_R


def bidir(ax, p0, p1):
    a, b = _trim(p0, p1)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=9,
                                 lw=1.5, color="black", zorder=2))


def unidir(ax, p0, p1):
    a, b = _trim(p0, p1)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.5, color="black", zorder=2))


def broken(ax, p0, p1):
    a, b = _trim(p0, p1)
    a, b = np.array(a), np.array(b)
    mid = (a + b) / 2
    d = (b - a)
    d = d / np.hypot(*d)
    ax.plot([a[0], (mid - d * 0.16)[0]], [a[1], (mid - d * 0.16)[1]],
            color="0.6", lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.plot([(mid + d * 0.16)[0], b[0]], [(mid + d * 0.16)[1], b[1]],
            color="0.6", lw=1.0, ls=(0, (2, 2)), zorder=1)
    s = 0.13
    ax.plot([mid[0] - s, mid[0] + s], [mid[1] - s, mid[1] + s],
            color=BROKEN, lw=1.8, zorder=3)
    ax.plot([mid[0] - s, mid[0] + s], [mid[1] + s, mid[1] - s],
            color=BROKEN, lw=1.8, zorder=3)


def draw_schematic(ax, level):
    """level: 0 low (fully coupled), 1 intermediate, 2 high (fully uncoupled)."""
    pA, pB, pC = [(dx, dy) for dx, dy in TRI]
    if level == 0:
        bidir(ax, pA, pB); bidir(ax, pB, pC); bidir(ax, pA, pC)
    elif level == 1:
        unidir(ax, pA, pB); broken(ax, pB, pC); bidir(ax, pA, pC)
    else:
        broken(ax, pA, pB); broken(ax, pB, pC); broken(ax, pA, pC)
    draw_nodes(ax, [pA, pB, pC])
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.5)
    ax.set_aspect("equal"); ax.axis("off")


# ============================================================
# PLOT
# ============================================================
TRACE_C = "black"


def main():
    apply_style()
    d = np.load("processed_data/fig1_single_cell.npz", allow_pickle=True)
    P = json.loads(str(d["__params__"]))
    t, atp, traces = d["t_disp"], d["atp"], d["traces"]
    nseed = len(d["seeds"])

    fig = plt.figure(figsize=(12, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.28)
    left = outer[0, 0].subgridspec(3, 2, width_ratios=[0.5, 2.4], hspace=0.65, wspace=0.42)
    right = outer[0, 1].subgridspec(3, 1, hspace=0.35)

    # ---- left: schematic + representative trace per condition ----
    for ci in range(3):
        ax_sch = fig.add_subplot(left[ci, 0])
        draw_schematic(ax_sch, ci)

        ax_tr = fig.add_subplot(left[ci, 1])
        ax_tr.plot(t, traces[ci], lw=0.8, color=TRACE_C)
        ax_tr.set_xlim(0, t[-1])
        ax_tr.set_ylabel(r"$C$ (Ca$^{2+}$)")
        ax_tr.set_title(f"{COND_LABELS[ci]}  ({COND_SUBTITLE[ci]})", fontsize=10, loc="left")
        clean_spines(ax_tr)
        if ci < 2:
            ax_tr.spines["bottom"].set_visible(False)
            ax_tr.tick_params(bottom=False, labelbottom=False)
        else:
            ax_tr.set_xlabel("Time (s)")
        if ci == 0:
            ax_sch.text(0.0, 1.0, "A", transform=ax_sch.transAxes,
                        fontsize=13, fontweight="bold", va="bottom", ha="left")
            ax_tr.text(-0.13, 1.08, "B", transform=ax_tr.transAxes,
                       fontsize=13, fontweight="bold", va="bottom", ha="right")

    # ---- right: three quantifications vs ATP (mean +/- SD over seeds) ----
    def ms(x):
        return np.nanmean(x, axis=1), np.nanstd(x, axis=1, ddof=1)

    rate_m, rate_e = ms(d["rate"]); peak_m, peak_e = ms(d["peak"])
    base_m, base_e = ms(d["baseline"])
    panels = [
        (0, rate_m, rate_e, r"Transient rate (min$^{-1}$)", "Transient rate"),
        (1, peak_m, peak_e, r"Peak amplitude $C_{\max}$", "Peak amplitude"),
        (2, base_m, base_e, r"Resting baseline $C$", "Resting baseline"),
    ]
    letters = ["C", "D", "E"]
    for row, m, e, ylab, title in panels:
        ax = fig.add_subplot(right[row, 0])
        ax.errorbar(atp, m, yerr=e, fmt="-", color="black", ecolor="black", capsize=0, lw=1.8)
        ax.set_xlim(0.12, 0.97); ax.set_xticks(atp)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
        if row == 1:                       # peak amplitude: start near 0 so it reads flat
            ax.set_ylim(0, max(2.2, float((m + e).max()) * 1.1))
        clean_spines(ax)
        ax.text(-0.22, 1.08, letters[row], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="right")
        if row == 2:
            ax.set_xlabel(r"ATP level $\alpha$")
        else:
            ax.tick_params(labelbottom=False)

    save_fig(fig, "Figure_1")
    for ci in range(3):
        print(f"  {COND_LABELS[ci]:18s} rate={rate_m[ci]:5.2f}+/-{rate_e[ci]:.2f}/min  "
              f"peak={peak_m[ci]:.2f}+/-{peak_e[ci]:.2f}  baseline={base_m[ci]:+.2f}"
              f"  (rep seed {d['rep_seed'][ci]})")
    print(f"{nseed} seeds; wrote Figure_1.pdf / Figure_1.png")


if __name__ == "__main__":
    main()
