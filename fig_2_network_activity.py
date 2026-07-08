#!/usr/bin/env python3
"""
Figure 2 -- network-level calcium activity across extracellular ATP.

An example 10x10 astrocyte network is driven at three ATP levels (low / intermediate
/ high, matching Fig 1). Top row: per-cell activity rasters (cell index vs time),
shown on a SHARED colour scale so the three regimes are directly comparable. Bottom
row: representative calcium traces from five cells. As ATP rises the network moves
from sparse, weakly coordinated activity to fast, spatially fragmented oscillation.

Same network model as Fig 3 (activity-dependent diffusive coupling that ATP
suppresses). Time is reported in seconds (SEC_PER_AU), consistent with Fig 1.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotstyle import apply_style, save_fig

np.random.seed(290)

# ============================================================
# GRID / TIME
# ============================================================
Nx = Ny = 10
N = Nx * Ny
dt = 0.0034
SEC_PER_AU = 1.0
T = 500
steps = int(T / dt)
t = np.arange(steps) * dt

sigma = 0.4

ATP_conditions = [("Low ATP", 0.19), ("Intermediate ATP", 0.27), ("High ATP", 0.9)]

cells = [(1, 1), (2, 7), (5, 5), (7, 2), (8, 8)]


def laplacian(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0)
            + np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4 * Z)


def simulate_network(A0):
    """Network model identical to fig_3 (heterogeneous FHN units, activity-dependent
    diffusive coupling suppressed by ATP)."""
    sigma_eff = sigma * (1 + 4 * A0)
    a, b = 1.0, 0.8
    gamma = (np.random.uniform(0.05, 0.34, (Nx, Ny))
             * (1 + 2.0 * A0 * np.random.randn(Nx, Ny)))
    I0 = 0.05 + (1 / A0 ** 0.5) * np.random.uniform(0.01, 0.15, (Nx, Ny))
    tau_h_eff = 10.0 / ((1 + 0.8 * A0) * np.random.uniform(0.5, 1.1, (Nx, Ny)))
    D0 = np.random.uniform(0.05, 0.5, (Nx, Ny))
    kappa = np.random.uniform(1, 4, (Nx, Ny))
    D_eff = D0 / (1 + (kappa * A0) ** 4)
    theta = 0.5 + 0.7 * A0
    sharpness = 8.0

    C = np.random.uniform(-0.1, 0.3, (Nx, Ny))
    h = np.random.uniform(0.4, 1.2, (Nx, Ny))
    activity = np.zeros((steps, N))
    for step in range(steps):
        noise = sigma_eff * 3 * np.random.randn(Nx, Ny)
        C_active = 0.5 * (1 + np.tanh(sharpness * (C - theta)))
        diff = D_eff * laplacian(C_active)
        dC = C - (C ** 3) / 3 - h + I0 + gamma * A0 + diff + noise
        dh = (C + a - b * h) / tau_h_eff
        C = np.clip(C + dt * dC, -4, 4)
        h = h + dt * dh
        activity[step] = C.flatten()
    return activity


# ============================================================
# RUN
# ============================================================
all_activity = [(label, A0, simulate_network(A0)) for label, A0 in ATP_conditions]

# shared colour scale across all conditions
allC = np.concatenate([a.ravel() for _, _, a in all_activity])
vmin, vmax = np.percentile(allC, 1), np.percentile(allC, 99)

# ============================================================
# FIGURE
# ============================================================
apply_style()
ncol = len(all_activity)
fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 7))

ims = []
for col, (label, A0, activity) in enumerate(all_activity):
    ax_map = axes[0, col]
    im = ax_map.imshow(activity.T, aspect="auto", cmap="inferno", origin="lower",
                       extent=[0, T * SEC_PER_AU, 0, N], vmin=vmin, vmax=vmax)
    ims.append(im)
    ax_map.set_title(label)
    ax_map.set_xlabel("Time (s)")
    if col == 0:
        ax_map.set_ylabel("Cell index")

    ax_tr = axes[1, col]
    for k, (i, j) in enumerate(cells):
        ax_tr.plot(t * SEC_PER_AU, activity[:, i * Ny + j] + 4 * k,
                   lw=0.6, color="black")
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_yticks([])
    if col == 0:
        ax_tr.set_ylabel("C (Ca$^{2+}$)  (offset)")

fig.tight_layout()

# single shared colorbar to the right of the raster row (keeps the 2x3 grid aligned)
pos = axes[0, ncol - 1].get_position()
cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.012, pos.height])
cbar = fig.colorbar(ims[-1], cax=cax)
cbar.set_label(r"Ca$^{2+}$ activity $C$")

save_fig(fig, "Figure_2")
print(f"Saved Figure_2.pdf / .png  (shared colour scale vmin={vmin:.2f}, vmax={vmax:.2f})")
plt.close(fig)
