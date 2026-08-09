#!/usr/bin/env python3
"""
Figure 2 -- network-level calcium activity across extracellular ATP.

Laid out left-to-right like Fig 1: representative calcium traces (left), the
per-cell activity heatmap (middle), and example network metrics versus ATP (right),
for three ATP levels -- low (coordinated waves), intermediate (quiescent), and high
(fragmented). Network activity is non-monotonic in ATP: coordinated and active at
low ATP, suppressed at intermediate ATP, and active but decorrelated at high ATP.

Heatmaps share one colour scale so the three regimes are directly comparable. The
right-hand metrics (mean pairwise correlation = how coordinated the network is;
active fraction = population activity) are computed over an ensemble of seeds
(mean +/- SD). Same network model as Fig 3. Time in seconds (SEC_PER_AU).
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from plotstyle import apply_style, clean_spines, save_fig, panel_label

# ============================================================
# GRID / TIME
# ============================================================
Nx = Ny = 10
N = Nx * Ny
dt = 0.0034
SEC_PER_AU = 1.0
T = 500
STEPS = int(T / dt)
SUB = 20                     # subsample stride for storage / display
sigma = 0.4

ATP = [("Low ATP", 0.10), ("Intermediate ATP", 0.60), ("High ATP", 0.90)]  # mid=0.60: activity present (~0.08) but coordination collapsed (~0.003) => active-but-desynchronized
DISP_SEED = 290
METRIC_SEEDS = [290, 11, 29, 28, 19, 55]
cells = [(1, 1), (2, 7), (5, 5), (7, 2), (8, 8)]


# ============================================================
# NUMBA NETWORK CORE  (same model as fig_3)
# ============================================================
@njit(fastmath=True, cache=True)
def laplacian(Z):
    nx, ny = Z.shape
    L = np.empty_like(Z)
    for i in range(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            L[i, j] = Z[ip, j] + Z[im, j] + Z[i, jp] + Z[i, jm] - 4.0 * Z[i, j]
    return L


@njit(fastmath=True, cache=True)
def sim_net(A0, seed, steps, nx, ny, n, sub):
    """Fixed-ATP network run; returns subsampled activity (n_sub, n)."""
    np.random.seed(seed)
    se = sigma * (1.0 + 4.0 * A0)
    a, b = 1.0, 0.8
    gamma = (np.random.uniform(0.05, 0.34, (nx, ny))
             * (1.0 + 2.0 * A0 * np.random.standard_normal((nx, ny))))
    I0 = 0.05 + (1.0 / np.sqrt(A0)) * np.random.uniform(0.01, 0.15, (nx, ny))
    tau = 10.0 / ((1.0 + 0.8 * A0) * np.random.uniform(0.5, 1.1, (nx, ny)))
    D0 = np.random.uniform(0.05, 0.5, (nx, ny))
    kap = np.random.uniform(1.0, 4.0, (nx, ny))
    Deff = D0 / (1.0 + (kap * A0) ** 4)
    th = 0.5 + 0.7 * A0

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    n_sub = steps // sub
    out = np.zeros((n_sub, n))
    si = 0
    for step in range(steps):
        noise = se * 3.0 * np.random.standard_normal((nx, ny))
        Ca = 0.5 * (1.0 + np.tanh(8.0 * (C - th)))
        diff = Deff * laplacian(Ca)
        C = np.minimum(np.maximum(
            C + dt * (C - C ** 3 / 3.0 - h + I0 + gamma * A0 + diff) + dt ** 0.5 * noise, -4.0), 4.0)
        h = h + dt * (C + a - b * h) / tau
        if step % sub == 0 and si < n_sub:
            out[si] = C.reshape(n)
            si += 1
    return out


# ============================================================
# METRICS
# ============================================================
def metrics(sub_act):
    """active fraction (population activity) and mean pairwise correlation."""
    st = sub_act[len(sub_act) // 3:]                     # drop transient
    active_frac = float((st > 0.0).mean())
    s = st[::5]                                          # thin for corrcoef
    cc = np.corrcoef(s.T)
    mpc = float(cc[np.triu_indices(N, 1)].mean())
    return active_frac, mpc


def compute():
    disp = {}
    corr = np.zeros((len(ATP), len(METRIC_SEEDS)))
    frac = np.zeros((len(ATP), len(METRIC_SEEDS)))
    print(f"Simulating network: {len(ATP)} conditions x {len(METRIC_SEEDS)} seeds, T={T}s")
    for ci, (label, A0) in enumerate(ATP):
        for si, seed in enumerate(METRIC_SEEDS):
            act = sim_net(A0, seed, STEPS, Nx, Ny, N, SUB)
            f, c = metrics(act)
            frac[ci, si], corr[ci, si] = f, c
            if seed == DISP_SEED:
                disp[ci] = act
        print(f"  {label:18s} corr={corr[ci].mean():+.3f}  active_frac={frac[ci].mean():.3f}")
    return disp, corr, frac


def mean_sd(x):
    return x.mean(axis=1), x.std(axis=1, ddof=1)


# ============================================================
# PLOT
# ============================================================
def plot(disp, corr, frac, save_stem="Figure_2"):
    apply_style()
    n_sub = disp[0].shape[0]
    t_sec = np.arange(n_sub) * SUB * dt * SEC_PER_AU
    allC = np.concatenate([disp[ci].ravel() for ci in disp])
    vmin, vmax = np.percentile(allC, 1), np.percentile(allC, 99)

    fig = plt.figure(figsize=(14, 7.5))
    outer = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.35, 0.05, 1.0], wspace=0.4)
    gs_tr = outer[0, 0].subgridspec(3, 1, hspace=0.45)
    gs_hm = outer[0, 1].subgridspec(3, 1, hspace=0.45)
    cax = fig.add_subplot(outer[0, 2])
    gs_me = outer[0, 3].subgridspec(2, 1, hspace=0.4)

    im = None
    for ci, (label, A0) in enumerate(ATP):
        # traces (left)
        ax_tr = fig.add_subplot(gs_tr[ci, 0])
        for k, (i, j) in enumerate(cells):
            ax_tr.plot(t_sec, disp[ci][:, i * Ny + j] + 4 * k, lw=0.6, color="black")
        ax_tr.set_title(label, fontsize=10, loc="center")
        panel_label(ax_tr, "ABC"[ci], dx=-0.04)
        ax_tr.set_xlim(0, T * SEC_PER_AU)
        ax_tr.set_yticks([])
        ax_tr.spines["left"].set_visible(False)
        ax_tr.spines["top"].set_visible(False)
        ax_tr.spines["right"].set_visible(False)
        if ci < 2:
            ax_tr.spines["bottom"].set_visible(False)
            ax_tr.tick_params(bottom=False, labelbottom=False)
        else:
            ax_tr.set_xlabel("Time (s)")

        # heatmap (middle)
        ax_hm = fig.add_subplot(gs_hm[ci, 0])
        im = ax_hm.imshow(disp[ci].T, aspect="auto", cmap="inferno", origin="lower",
                          extent=[0, T * SEC_PER_AU, 0, N], vmin=vmin, vmax=vmax)
        ax_hm.set_ylabel("Cell index")
        panel_label(ax_hm, "DEF"[ci])
        if ci < 2:
            ax_hm.tick_params(labelbottom=False)
        else:
            ax_hm.set_xlabel("Time (s)")

    # short, narrow colorbar centered vertically, nudged close to the heatmaps (D-F)
    hm_pos = im.axes.get_position()
    p = cax.get_position()
    ch = p.height * 0.42
    cax.set_position([hm_pos.x1 + 0.012, p.y0 + (p.height - ch) / 2.0, p.width * 0.8, ch])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Ca$^{2+}$ activity $C$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # metrics (right)
    atp = np.array([A0 for _, A0 in ATP])
    corr_m, corr_e = mean_sd(corr)
    frac_m, frac_e = mean_sd(frac)
    for row, (m, e, ylab, title) in enumerate([
            (corr_m, corr_e, "Mean pairwise corr.", "Coordination"),
            (frac_m, frac_e, "Active fraction", "Population activity")]):
        ax = fig.add_subplot(gs_me[row, 0])
        ax.errorbar(atp, m, yerr=e, fmt="-", color="black", ecolor="black",
                    capsize=0, lw=1.8)
        ax.set_xlim(0.03, 0.97)
        ax.set_xticks(atp)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        panel_label(ax, "GH"[row])
        clean_spines(ax)
        if row == 1:
            ax.set_xlabel(r"ATP level $\alpha$")
        else:
            ax.tick_params(labelbottom=False)

    save_fig(fig, save_stem)
    print(f"Saved -> {save_stem}.pdf / .png  (colour scale {vmin:.2f}..{vmax:.2f})")
    plt.close(fig)


if __name__ == "__main__":
    disp, corr, frac = compute()
    plot(disp, corr, frac)
