#!/usr/bin/env python3
"""
Figure 1 -- single-astrocyte dynamics across extracellular ATP.

Left column: representative calcium traces in three ATP regimes (low / intermediate
/ high), each accompanied by a coupling schematic (a 3-astrocyte motif that is
progressively uncoupled as ATP rises). Right column: quantification computed over
LONG (10 min) records across multiple stochastic realizations -- calcium-transient
(event) rate, mean peak amplitude, and firing regularity versus ATP, with error
bars over seeds.

Transients are detected as prominent peaks (height + prominence thresholds), so a
genuine Ca2+ transient is counted regardless of ATP; this makes peak amplitude
comparable across regimes and isolates the ATP dependence into the transient rate.

Time is reported in seconds via a fixed conversion from model time units
(SEC_PER_AU); at this scale the oscillation period (~20-30 s) matches the range of
spontaneous astrocytic Ca2+ oscillations. The single-cell parametrization here is a
stylized illustration of the three regimes and is not identical to the network
model used in Figs 2-3.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from numba import njit
from scipy.signal import find_peaks

from plotstyle import apply_style, clean_spines, save_fig

# ============================================================
# TIME  (model units -> seconds)
# ============================================================
dt = 0.0034
SEC_PER_AU = 1.0          # 1 model time unit ~ 1 s (period ~20-30 s, astrocytic Ca2+ range)
T_SEC = 600.0             # 10 min record per realization (for the statistics)
T_AU = T_SEC / SEC_PER_AU
N = int(T_AU / dt)
DISP_SEC = 500.0          # window shown in the trace panels

# transient detection
PK_HEIGHT = 0.5           # a transient peak must exceed this C value
PK_PROM = 1.0             # ... and rise at least this much above its surroundings
PK_DIST_AU = 2.0          # minimum separation between peaks (model units)

# ============================================================
# MODEL  (stylized single-cell FHN-type unit; identical to the legacy Fig 1 model)
# ============================================================
sigma = 0.4

ATP_levels = [0.19, 0.40, 0.9]   # low / intermediate / high (single cell fires from ~0.19; 0.10 is subthreshold)
COND_LABELS = ["Low ATP", "Intermediate ATP", "High ATP"]
COND_SUBTITLE = ["noise-driven excitable", "oscillation onset", "irregular high-frequency"]
N_SEEDS = 10
SEEDS = [11, 57, 22, 19, 29, 28, 53, 16, 55, 49]


@njit(fastmath=True, cache=True)
def simulate(A0, seed, n, dt, sigma):
    np.random.seed(seed)
    C = np.zeros(n)
    h = np.zeros(n)
    C[0] = 0.091
    h[0] = 0.8
    a = 1.0
    b = 0.8
    tau_h_eff = 10.0 / (1.0 + 0.8 * A0)
    gamma = 0.72
    if A0 < 0.2:
        I0 = 0.38
    elif A0 < 0.7:
        I0 = 0.5
    else:
        I0 = 0.62
    sigma_eff = sigma * (1.0 + A0)
    for i in range(n - 1):
        noise = sigma_eff * 3.0 * np.random.standard_normal()
        dC = C[i] - (C[i] ** 3) / 3.0 - h[i] + I0 + gamma * A0 + noise
        dh = (C[i] + a - b * h[i]) / tau_h_eff
        C[i + 1] = min(max(C[i] + dt * dC, -4.0), 4.0)
        h[i + 1] = h[i] + dt * dh
    return C


def event_stats(C):
    """Transient rate (events/min), mean peak amplitude, and resting baseline
    (median C), from prominent peaks of C over the full record."""
    peaks, _ = find_peaks(C, height=PK_HEIGHT, prominence=PK_PROM,
                          distance=int(PK_DIST_AU / dt))
    T_min = (len(C) * dt * SEC_PER_AU) / 60.0
    rate = len(peaks) / T_min
    peak = float(C[peaks].mean()) if len(peaks) else np.nan
    baseline = float(np.median(C))
    return rate, peak, baseline


# ============================================================
# COUPLING SCHEMATIC  (adapted from the user-supplied schematic script)
# ============================================================
NODE_C = "#6E7B8B"
BROKEN = "#C0392B"
NODE_R = 0.42
TRI = np.array([(-1.1, 0.75), (1.1, 0.75), (0.0, -1.05)])
NLAB = ["A", "B", "C"]


def _tri_pts():
    return [(dx, dy) for dx, dy in TRI]


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
    pA, pB, pC = _tri_pts()
    if level == 0:
        bidir(ax, pA, pB); bidir(ax, pB, pC); bidir(ax, pA, pC)
    elif level == 1:
        unidir(ax, pA, pB); broken(ax, pB, pC); bidir(ax, pA, pC)
    else:
        broken(ax, pA, pB); broken(ax, pB, pC); broken(ax, pA, pC)
    draw_nodes(ax, [pA, pB, pC])
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ============================================================
# COMPUTE
# ============================================================
def compute():
    print(f"Simulating single cell: {len(ATP_levels)} conditions x {N_SEEDS} seeds, "
          f"T={T_SEC:g}s ({N} steps)")
    traces = {}
    rate_stack = np.full((len(ATP_levels), N_SEEDS), np.nan)
    peak_stack = np.full((len(ATP_levels), N_SEEDS), np.nan)
    base_stack = np.full((len(ATP_levels), N_SEEDS), np.nan)
    n_disp = int(min(DISP_SEC / SEC_PER_AU, T_AU) / dt)
    dist = int(PK_DIST_AU / dt)
    for ci, A0 in enumerate(ATP_levels):
        # representative trace: the seed whose DISPLAY window holds >=2 transients
        # (pick the fewest >=2, i.e. typical rather than an outlier; fall back to most)
        rep_ge2 = (10 ** 9, None)
        rep_max = (-1, None)
        for si, seed in enumerate(SEEDS):
            C = simulate(A0, seed, N, dt, sigma)
            r, p, base = event_stats(C)
            rate_stack[ci, si] = r
            peak_stack[ci, si] = p
            base_stack[ci, si] = base
            wc = len(find_peaks(C[:n_disp], height=PK_HEIGHT,
                                prominence=PK_PROM, distance=dist)[0])
            if 2 <= wc < rep_ge2[0]:
                rep_ge2 = (wc, C)
            if wc > rep_max[0]:
                rep_max = (wc, C)
        traces[ci] = rep_ge2[1] if rep_ge2[1] is not None else rep_max[1]
    return traces, rate_stack, peak_stack, base_stack


def mean_sd(x):
    return np.nanmean(x, axis=1), np.nanstd(x, axis=1, ddof=1)


# ============================================================
# PLOT
# ============================================================
TRACE_C = "black"


def plot(traces, rate_stack, peak_stack, base_stack, save_stem="Figure_1"):
    apply_style()
    t_sec = np.arange(N) * dt * SEC_PER_AU
    disp = t_sec <= DISP_SEC

    fig = plt.figure(figsize=(12, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.28)
    left = outer[0, 0].subgridspec(3, 2, width_ratios=[0.5, 2.4],
                                   hspace=0.65, wspace=0.42)
    right = outer[0, 1].subgridspec(3, 1, hspace=0.35)

    # ---- left: schematic + trace per condition ----
    for ci in range(3):
        ax_sch = fig.add_subplot(left[ci, 0])
        draw_schematic(ax_sch, ci)

        ax_tr = fig.add_subplot(left[ci, 1])
        ax_tr.plot(t_sec[disp], traces[ci][disp], lw=0.8, color=TRACE_C)
        ax_tr.set_xlim(0, DISP_SEC)
        ax_tr.set_ylabel(r"$C$ (Ca$^{2+}$)")
        ax_tr.set_title(f"{COND_LABELS[ci]}  ({COND_SUBTITLE[ci]})",
                        fontsize=10, loc="left")
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
    atp = np.array(ATP_levels)
    rate_m, rate_e = mean_sd(rate_stack)
    peak_m, peak_e = mean_sd(peak_stack)
    base_m, base_e = mean_sd(base_stack)
    panels = [
        (0, rate_m, rate_e, r"Transient rate (min$^{-1}$)", "Transient rate"),
        (1, peak_m, peak_e, r"Peak amplitude $C_{\max}$", "Peak amplitude"),
        (2, base_m, base_e, r"Resting baseline $C$", "Resting baseline"),
    ]
    letters = ["C", "D", "E"]
    for row, m, e, ylab, title in panels:
        ax = fig.add_subplot(right[row, 0])
        ax.errorbar(atp, m, yerr=e, fmt="-", color="black", ecolor="black",
                    capsize=0, lw=1.8)
        ax.set_xlim(0.12, 0.97)
        ax.set_xticks(atp)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        if row == 1:                       # peak amplitude: start y near 0 so it reads flat
            ax.set_ylim(0, max(2.2, float((m + e).max()) * 1.1))
        clean_spines(ax)
        ax.text(-0.22, 1.08, letters[row], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="right")
        if row == 2:
            ax.set_xlabel(r"ATP level $\alpha$")
        else:
            ax.tick_params(labelbottom=False)

    save_fig(fig, save_stem)
    print(f"Saved -> {save_stem}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    traces, rate_stack, peak_stack, base_stack = compute()
    for ci in range(3):
        print(f"  {COND_LABELS[ci]:18s} rate={np.nanmean(rate_stack[ci]):5.2f}"
              f"+/-{np.nanstd(rate_stack[ci], ddof=1):.2f} /min (mean+/-SD)  "
              f"peak={np.nanmean(peak_stack[ci]):.2f}  baseline={np.nanmean(base_stack[ci]):+.2f}")
    plot(traces, rate_stack, peak_stack, base_stack)
