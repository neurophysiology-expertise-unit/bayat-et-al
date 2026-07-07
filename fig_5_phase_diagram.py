"""
Figure 5: 2D phase diagram of susceptibility chi over (ATP level alpha, noise sigma).

For each point on an (alpha, sigma) grid we simulate the stochastic astrocyte
network and compute the extreme-value fluctuation susceptibility chi (same
definition as fig_3_criticality.py). The resulting heatmap locates the "critical line" --
the ridge of high susceptibility marking the crossover from coherent
wave propagation to fragmented local oscillation -- and shows how it moves with
the background noise amplitude sigma. Healthy and disease-perturbed networks are
shown side by side.

The single-run inner loop is a Numba @njit core (same style as fig_4_lyapunov_robustness.py).
sigma is the *base* noise amplitude; the model's effective noise remains
sigma * (1 + 4*alpha) * 3, exactly as in fig_2_network_activity.py / fig_3_criticality.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

from plotstyle import apply_style, save_fig

PROC_DIR = "processed_data"
CACHE = os.path.join(PROC_DIR, "fig5_phase.npz")
GRID_CSV = os.path.join(PROC_DIR, "fig5_phase_grid.csv")

# ============================================================
# GRID / MODEL PARAMETERS
# ============================================================
Nx, Ny = 10, 10
N = Nx * Ny

dt = 0.0034
eta = 8.0
a = 1.0
b = 0.8
theta_base = 0.5

# phase-diagram axes
ALPHA_MIN, ALPHA_MAX = 0.01, 1.11
SIGMA_MIN, SIGMA_MAX = 0.05, 0.80


@njit(fastmath=True, cache=True)
def laplacian(Z):
    Nx, Ny = Z.shape
    L = np.empty_like(Z)
    for i in range(Nx):
        ip = (i + 1) % Nx
        im = (i - 1) % Nx
        for j in range(Ny):
            jp = (j + 1) % Ny
            jm = (j - 1) % Ny
            L[i, j] = Z[ip, j] + Z[im, j] + Z[i, jp] + Z[i, jm] - 4.0 * Z[i, j]
    return L


@njit(fastmath=True, cache=True)
def chi_one(alpha, sigma_base, disease, seed, steps):
    """Extreme-value fluctuation susceptibility chi for one (alpha, sigma) run."""
    np.random.seed(seed)

    gamma = (np.random.uniform(0.05, 0.34, (Nx, Ny))
             * (1.0 + 2.0 * np.random.standard_normal((Nx, Ny))))
    I0 = 0.05 + (1.0 / np.sqrt(alpha)) * np.random.uniform(0.01, 0.15, (Nx, Ny))
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * np.random.uniform(0.5, 1.1, (Nx, Ny)))
    D0 = np.random.uniform(0.05, 0.5, (Nx, Ny))
    kappa = np.random.uniform(1.0, 4.0, (Nx, Ny))

    if disease:
        gamma = gamma * 2.0
        tau_h = tau_h * 3.0
        D0 = D0 * 0.5
        kappa = kappa * 1.5

    Deff = D0 / (1.0 + (kappa * alpha) ** 4)
    theta = theta_base + 0.7 * alpha
    sigma_eff = sigma_base * (1.0 + 4.0 * alpha)

    C = np.random.uniform(-0.1, 0.3, (Nx, Ny))
    h = np.random.uniform(0.4, 1.2, (Nx, Ny))

    t_start = int(0.3 * steps)
    sumD = 0.0
    sumD2 = 0.0
    cntD = 0

    for t in range(steps):
        noise = sigma_eff * 3.0 * np.random.standard_normal((Nx, Ny))
        C_active = 0.5 * (1.0 + np.tanh(eta * (C - theta)))
        diff = Deff * laplacian(C_active)

        dC = C - (C ** 3) / 3.0 - h + I0 + gamma * alpha + diff + noise
        dh = (C + a - b * h) / tau_h

        C = C + dt * dC
        h = h + dt * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)

        if t >= t_start:
            D_ = np.max(C) - np.min(C)
            sumD += D_
            sumD2 += D_ * D_
            cntD += 1

    meanD = sumD / cntD
    meanD2 = sumD2 / cntD
    return N * (meanD2 - meanD * meanD)


@njit(parallel=True, fastmath=True, cache=True)
def phase_grid_core(disease, alphas, sigmas, n_avg, steps, seed0):
    """chi heatmap over the (alpha, sigma) grid, cells computed in parallel."""
    ns = sigmas.shape[0]
    na = alphas.shape[0]
    grid = np.zeros((ns, na))       # rows = sigma, cols = alpha
    for idx in prange(ns * na):
        si = idx // na
        ai = idx % na
        acc = 0.0
        for k in range(n_avg):
            acc += chi_one(alphas[ai], sigmas[si], disease, seed0 + k, steps)
        grid[si, ai] = acc / n_avg
    return grid


def phase_grid(disease, alphas, sigmas, n_avg, steps, seed0=100):
    grid = phase_grid_core(disease, alphas, sigmas, n_avg, steps, seed0)
    print(f"  {'disease' if disease else 'healthy'} grid done ({sigmas.size}x{alphas.size})")
    return grid


# ============================================================
# COMPUTE  ->  cache to processed_data/
# ============================================================
def compute(n_alpha=24, n_sigma=24, n_avg=3, T=200.0):
    os.makedirs(PROC_DIR, exist_ok=True)
    steps = int(T / dt)
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, n_alpha)
    sigmas = np.linspace(SIGMA_MIN, SIGMA_MAX, n_sigma)
    print(f"Computing phase diagram: {n_alpha} x {n_sigma} grid, n_avg={n_avg}, T={T}, steps={steps}")

    print("Healthy grid...")
    H = phase_grid(False, alphas, sigmas, n_avg, steps)
    print("Disease grid...")
    D = phase_grid(True, alphas, sigmas, n_avg, steps)

    np.savez(CACHE, alphas=alphas, sigmas=sigmas, H=H, D=D,
             n_avg=n_avg, T=T)
    rows = []
    for si, s in enumerate(sigmas):
        for ai, al in enumerate(alphas):
            rows.append({"alpha": al, "sigma": s, "chi_healthy": H[si, ai], "chi_disease": D[si, ai]})
    pd.DataFrame(rows).to_csv(GRID_CSV, index=False)
    print(f"Cached -> {CACHE} and {GRID_CSV}")
    return {"alphas": alphas, "sigmas": sigmas, "H": H, "D": D}


def load():
    d = np.load(CACHE, allow_pickle=True)
    return {"alphas": d["alphas"], "sigmas": d["sigmas"], "H": d["H"], "D": d["D"]}


# ============================================================
# PLOT  (loads from cache)
# ============================================================
def plot(data, save_stem="Figure_5"):
    apply_style()
    H, D = data["H"], data["D"]
    alphas, sigmas = data["alphas"], data["sigmas"]
    vmax = np.percentile(np.concatenate([H.ravel(), D.ravel()]), 99)
    extent = [alphas.min(), alphas.max(), sigmas.min(), sigmas.max()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    im = None
    for ax, grid, title in ((axes[0], H, r"(A) Healthy"), (axes[1], D, r"(B) Disease")):
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent,
                       cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")
        ax.set_title(title + r"  —  susceptibility $\chi$")
        ax.set_xlabel(r"ATP level $\alpha$")
        ax.set_ylabel(r"Noise amplitude $\sigma$")
        ax.tick_params(direction="out", length=4, width=1.0)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label(r"Susceptibility $\chi$")

    save_fig(fig, save_stem)
    print(f"Saved {save_stem}.pdf / .png")
    plt.show()


def main(recompute=False, n_alpha=24, n_sigma=24, n_avg=3, T=200.0, save_stem="Figure_5"):
    if recompute or not os.path.exists(CACHE):
        data = compute(n_alpha=n_alpha, n_sigma=n_sigma, n_avg=n_avg, T=T)
    else:
        print(f"Loading cached results from {CACHE} (use recompute=True to rerun)")
        data = load()
    plot(data, save_stem=save_stem)


if __name__ == "__main__":
    import sys
    main(recompute="--recompute" in sys.argv)
