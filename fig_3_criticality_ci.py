"""
Figure 3 with 95% confidence bands (ensemble version of fig_3_criticality.py).

Same model and observables as fig_3_criticality.py (spatial heterogeneity S_C,
extreme-value fluctuation chi, spatial coherence proxy xi), but the network is
simulated over an ensemble of independent seeds so that panels A-C show the
ensemble mean +/- a 95% confidence interval of the mean rather than a single
realization. The inner time-integration loop is ported to a Numba @njit core
(same structure as fig_4_lyapunov_robustness.py) so that ~20 seeds x 2 conditions is fast.

The spatial-snapshot rows (D/E) are drawn from one representative seed, exactly
as in fig_3_criticality.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

from plotstyle import apply_style, clean_spines, save_fig

PROC_DIR = "processed_data"
CACHE = os.path.join(PROC_DIR, "fig3_ci.npz")
CURVES_CSV = os.path.join(PROC_DIR, "fig3_ci_curves.csv")

# ============================================================
# GRID / TIME / SWEEP  (identical to fig_3_criticality.py)
# ============================================================
Nx, Ny = 10, 10
N = Nx * Ny

dt = 0.0034
T_DEFAULT = 1000.0

alpha_values = np.linspace(0.01, 1.11, 21)

# ============================================================
# MODEL PARAMETERS  (identical to fig_3_criticality.py)
# ============================================================
a = 1.0
b = 0.8
sigma = 0.4
eta = 8.0
theta_base = 0.5

target_snaps = [0.175, 0.395, 1.0]   # low / mid / high ATP snapshot levels

# ============================================================
# ENSEMBLE CONFIG
# ============================================================
N_SEEDS_DEFAULT = 20
ENSEMBLE_SEEDS = list(range(11, 11 + N_SEEDS_DEFAULT))   # 11..30
SNAP_SEED = 29        # representative seed for the spatial snapshots (rows D/E)


# ============================================================
# NUMBA CORES
# ============================================================
@njit(fastmath=True, cache=True)
def laplacian(Z):
    """Periodic (toroidal) 5-point Laplacian."""
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
def run_one_seed_core(seed, disease, alpha_values, steps):
    """
    Run one full quasi-static ATP sweep for a single seed.

    Returns per-alpha arrays:
      Sc     : spatial heterogeneity  (mean over full history of var/|mean|)
      chi    : extreme-value fluctuation  N*(<DeltaC^2> - <DeltaC>^2) over stationary part
      last5  : mean of last 5 stationary frames, flattened  -> used for xi outside numba
      lastf  : final frame, flattened  -> used for snapshots outside numba

    Formulas mirror fig_3_criticality.py exactly; only the loop is JIT-compiled.
    """
    np.random.seed(seed)

    n_alpha = alpha_values.shape[0]
    Sc_out = np.zeros(n_alpha)
    chi_out = np.zeros(n_alpha)
    last5 = np.zeros((n_alpha, N))
    lastf = np.zeros((n_alpha, N))

    # fixed heterogeneity fields for this seed (same draws as fig_3_criticality.py)
    gamma_base = (np.random.uniform(0.05, 0.34, (Nx, Ny))
                  * (1.0 + 2.0 * np.random.standard_normal((Nx, Ny))))
    I0_base = np.random.uniform(0.01, 0.15, (Nx, Ny))
    tau_base = np.random.uniform(0.5, 1.1, (Nx, Ny))
    D0_base = np.random.uniform(0.05, 0.5, (Nx, Ny))
    kappa_base = np.random.uniform(1.0, 4.0, (Nx, Ny))

    # state carried across ATP levels (sequential quasi-static up-sweep)
    C = np.random.uniform(-0.1, 0.3, (Nx, Ny))
    h = np.random.uniform(0.4, 1.2, (Nx, Ny))

    t_start = int(0.3 * steps)

    for idx in range(n_alpha):
        alpha = alpha_values[idx]

        gamma = gamma_base.copy()
        I0 = 0.05 + (1.0 / np.sqrt(alpha)) * I0_base
        tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
        D0 = D0_base.copy()
        kappa = kappa_base.copy()

        if disease:
            gamma = gamma * 2.0
            tau_h = tau_h * 3.0
            D0 = D0 * 0.5
            kappa = kappa * 1.5

        Deff = D0 / (1.0 + (kappa * alpha) ** 4)
        theta = theta_base + 0.7 * alpha
        sigma_eff = sigma * (1.0 + 4.0 * alpha)

        sc_sum = 0.0
        sumD = 0.0
        sumD2 = 0.0
        cntD = 0

        ring = np.zeros((5, N))       # last-5-frame ring buffer
        ring_pos = 0
        ring_filled = 0

        for t in range(steps):
            noise = sigma_eff * 3.0 * np.random.standard_normal((Nx, Ny))
            C_active = 0.5 * (1.0 + np.tanh(eta * (C - theta)))
            diff = Deff * laplacian(C_active)

            dC = C - (C ** 3) / 3.0 - h + I0 + gamma * alpha + diff + noise
            dh = (C + a - b * h) / tau_h

            C = C + dt * dC
            h = h + dt * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)

            # spatial heterogeneity accumulates over the FULL history
            m = np.mean(C)
            v = np.mean(C * C) - m * m
            sc_sum += v / (abs(m) + 1e-9)

            # fluctuation + coherence accumulate over the stationary part
            if t >= t_start:
                D_ = np.max(C) - np.min(C)
                sumD += D_
                sumD2 += D_ * D_
                cntD += 1
                ring[ring_pos, :] = C.reshape(N)
                ring_pos = (ring_pos + 1) % 5
                if ring_filled < 5:
                    ring_filled += 1

        Sc_out[idx] = sc_sum / steps
        meanD = sumD / cntD
        meanD2 = sumD2 / cntD
        chi_out[idx] = N * (meanD2 - meanD * meanD)

        acc = np.zeros(N)
        for k in range(ring_filled):
            acc += ring[k, :]
        last5[idx, :] = acc / ring_filled
        lastf[idx, :] = C.reshape(N)

    return Sc_out, chi_out, last5, lastf


@njit(parallel=True, fastmath=True, cache=True)
def run_ensemble_core(seeds, disease, alpha_values, steps):
    """Run all seeds in parallel (one independent quasi-static sweep each)."""
    ns = seeds.shape[0]
    na = alpha_values.shape[0]
    Sc = np.zeros((ns, na))
    chi = np.zeros((ns, na))
    last5 = np.zeros((ns, na, N))
    for i in prange(ns):
        s_i, c_i, l5_i, _ = run_one_seed_core(seeds[i], disease, alpha_values, steps)
        Sc[i, :] = s_i
        chi[i, :] = c_i
        last5[i, :, :] = l5_i
    return Sc, chi, last5


# ============================================================
# POST-PROCESSING HELPERS (pure python)
# ============================================================
def corr_length_from_frame(frame):
    """Spatial coherence proxy xi from a flattened field (matches fig_3_criticality.py)."""
    x = frame - frame.mean()
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]
    corr = corr / corr[0]
    return np.sum(corr > 0.2)


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode='same')


def ci95(stack):
    """Mean and half-width of the 95% CI of the mean, across axis 0."""
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    half = 1.96 * sd / np.sqrt(n)
    return mean, half


# ============================================================
# ENSEMBLE DRIVER
# ============================================================
def run_ensemble(disease, seeds, steps):
    na = len(alpha_values)
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    Sc, chi, last5 = run_ensemble_core(seeds_arr, disease, alpha_values, steps)
    corr = np.zeros((len(seeds), na))
    for si in range(len(seeds)):
        for j in range(na):
            corr[si, j] = corr_length_from_frame(last5[si, j])
    print(f"  {'disease' if disease else 'healthy'} ensemble done ({len(seeds)} seeds)")
    return Sc, chi, corr


def snapshots_for_seed(disease, seed, steps):
    _, _, _, lastf = run_one_seed_core(seed, disease, alpha_values, steps)
    snaps = {}
    for key, lvl in zip(("low", "mid", "high"), target_snaps):
        idx = int(np.argmin(np.abs(alpha_values - lvl)))
        snaps[key] = lastf[idx].reshape(Nx, Ny)
    return snaps


# ============================================================
# COMPUTE  ->  cache to processed_data/
# ============================================================
OBS = ("Sc", "chi", "corr")
OBS_SMOOTH = {"Sc": 3, "chi": 3, "corr": 5}


def compute(n_seeds=N_SEEDS_DEFAULT, T=T_DEFAULT):
    """Run the ensemble, aggregate to mean/95%-CI, and cache to processed_data/."""
    os.makedirs(PROC_DIR, exist_ok=True)
    steps = int(T / dt)
    seeds = list(range(11, 11 + n_seeds))
    print(f"Computing Fig 3 ensemble: {n_seeds} seeds, T={T}, steps={steps}")

    stacks = {}
    print("Healthy ensemble...")
    stacks["Sc", "H"], stacks["chi", "H"], stacks["corr", "H"] = run_ensemble(False, seeds, steps)
    print("Disease ensemble...")
    stacks["Sc", "D"], stacks["chi", "D"], stacks["corr", "D"] = run_ensemble(True, seeds, steps)

    print(f"Snapshots from seed {SNAP_SEED}...")
    snaps = {"H": snapshots_for_seed(False, SNAP_SEED, steps),
             "D": snapshots_for_seed(True, SNAP_SEED, steps)}

    data = {"alpha": alpha_values, "n_seeds": n_seeds, "T": T, "snap_seed": SNAP_SEED}
    rows = []
    for obs in OBS:
        for cond in ("H", "D"):
            mean, half = ci95(stacks[obs, cond])
            data[f"{cond}_{obs}_mean"] = mean
            data[f"{cond}_{obs}_ci"] = half
            for k, al in enumerate(alpha_values):
                rows.append({"observable": obs, "condition": cond, "alpha": al,
                             "mean": mean[k], "ci_lo": mean[k] - half[k], "ci_hi": mean[k] + half[k]})
    for cond in ("H", "D"):
        for key in ("low", "mid", "high"):
            data[f"{cond}_snap_{key}"] = snaps[cond][key]

    np.savez(CACHE, **data)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    print(f"Cached -> {CACHE} and {CURVES_CSV}")
    return data


def load():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ============================================================
# PLOT  (loads from cache)
# ============================================================
TITLES = {
    "Sc": r"(A) Spatial heterogeneity $S_C$",
    "chi": r"(B) Extreme-value fluctuation $\chi_{ext}$",
    "corr": r"(C) Spatial coherence length $\xi$",
}
COND_STYLE = {"H": ("Healthy", "#0072B2"), "D": ("Disease", "#D55E00")}   # colorblind-safe


def plot(data, save_stem="Figure_3_ci"):
    apply_style()
    alpha = data["alpha"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)

    for col, obs in enumerate(OBS):
        ax = axes[0, col]
        w = OBS_SMOOTH[obs]
        for cond in ("H", "D"):
            label, color = COND_STYLE[cond]
            mean = data[f"{cond}_{obs}_mean"]
            half = data[f"{cond}_{obs}_ci"]
            m = smooth(mean, w)
            lo = smooth(mean - half, w)
            hi = smooth(mean + half, w)
            ax.plot(alpha, m, linewidth=2, label=label, color=color)
            ax.fill_between(alpha, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.set_title(TITLES[obs])
        ax.set_xlabel(r"ATP level $\alpha$")
        ax.set_xlim(alpha.min(), alpha.max())
        clean_spines(ax)
        if col == 0:
            ax.legend(loc="upper right")

    row_titles = {"H": ("D", "Healthy"), "D": ("E", "Disease")}
    keys = ["low", "mid", "high"]
    labels = ["Low ATP", "Mid ATP", "High ATP"]
    im = None
    for r, cond in enumerate(("H", "D"), start=1):
        letter, name = row_titles[cond]
        for i, k in enumerate(keys):
            ax = axes[r, i]
            im = ax.imshow(data[f"{cond}_snap_{k}"], cmap="inferno",
                           origin="lower", interpolation="nearest")
            ax.set_title(f"({letter}{i+1}) {name} — {labels[i]}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    cbar = fig.colorbar(im, ax=axes[1:, :], shrink=0.6, location="right", pad=0.02)
    cbar.set_label(r"Ca$^{2+}$ activity $C$")

    save_fig(fig, save_stem)
    print(f"Saved {save_stem}.pdf / .png")
    plt.show()


def main(recompute=False, n_seeds=N_SEEDS_DEFAULT, T=T_DEFAULT, save_stem="Figure_3_ci"):
    if recompute or not os.path.exists(CACHE):
        data = compute(n_seeds=n_seeds, T=T)
    else:
        print(f"Loading cached results from {CACHE} (use recompute=True to rerun)")
        data = load()
    plot(data, save_stem=save_stem)


if __name__ == "__main__":
    import sys
    main(recompute="--recompute" in sys.argv)
