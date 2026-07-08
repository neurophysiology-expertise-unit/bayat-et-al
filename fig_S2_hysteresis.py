"""
Supplementary Figure S2 -- Hysteresis of the ATP-driven transition.

Tests the *nature* of the propagation->fragmentation transition. The ATP control
parameter alpha is swept quasi-statically UP (0.01 -> 1.11) and then DOWN
(1.11 -> 0.01) on a single continuous trajectory: the down-sweep starts from the
final network state of the up-sweep, so any memory of the path is retained. Two
order parameters are tracked on both legs -- the extreme-value fluctuation chi and
the Golomb-Rinzel synchrony R.

  * up and down legs overlap  ==> continuous (critical) transition, no bistability
  * a gap between the legs     ==> discontinuous / bistable (1st-order) transition

The FHN reaction-diffusion core and all model parameters are identical to
fig_3_criticality_ci.py (the up-sweep here reproduces its up-sweep); only the
alpha schedule is extended with a down-sweep and the loop records both legs.

compute -> cache (processed_data/figS2_hysteresis.npz + .csv) -> plot. Re-render
from cache in <1 s; pass --recompute to rerun the ensemble.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

from plotstyle import apply_style, clean_spines, save_fig

PROC_DIR = "processed_data"
CACHE = os.path.join(PROC_DIR, "figS2_hysteresis.npz")
CURVES_CSV = os.path.join(PROC_DIR, "figS2_hysteresis_curves.csv")

# ============================================================
# GRID / TIME / SWEEP  (matched to fig_3_criticality_ci.py)
# ============================================================
GRID_DEFAULT = 10          # main-text lattice, as in the single-seed Fig 3
dt = 0.0034
T_DEFAULT = 700.0          # per-alpha-level integration time (transient + stationary)

# up-sweep alpha grid; the full schedule is [up, down] carried continuously
alpha_up = np.linspace(0.01, 1.11, 21)

# ============================================================
# MODEL PARAMETERS  (identical to fig_3_criticality_ci.py)
# ============================================================
a = 1.0
b = 0.8
sigma = 0.4
eta = 8.0
theta_base = 0.5

N_SEEDS_DEFAULT = 20
ENSEMBLE_SEEDS = list(range(11, 11 + N_SEEDS_DEFAULT))   # 11..30


# ============================================================
# NUMBA CORE  (dynamics verbatim from fig_3_criticality_ci.py)
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
def run_one_seed_sweep(seed, disease, alpha_seq, steps, nx, ny, n):
    """
    Run one continuous quasi-static ATP schedule (any ordering of alpha) for a
    single seed, carrying (C, h) across successive levels. Returns per-level
    chi and R_sync. Dynamics/formulas mirror fig_3_criticality_ci.py exactly.
    """
    np.random.seed(seed)
    n_lvl = alpha_seq.shape[0]
    chi_out = np.zeros(n_lvl)
    Rsync_out = np.zeros(n_lvl)

    # fixed heterogeneity fields for this seed (same draws as Fig 3)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))

    t_start = int(0.3 * steps)

    for idx in range(n_lvl):
        alpha = alpha_seq[idx]

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

        sumD = 0.0
        sumD2 = 0.0
        cntD = 0
        msum = 0.0
        msum2 = 0.0
        csum = np.zeros(n)
        csum2 = np.zeros(n)

        for t in range(steps):
            noise = sigma_eff * 3.0 * np.random.standard_normal((nx, ny))
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

                Cf = C.reshape(n)
                m = 0.0
                for kk in range(n):
                    ck = Cf[kk]
                    m += ck
                    csum[kk] += ck
                    csum2[kk] += ck * ck
                m /= n
                msum += m
                msum2 += m * m

        meanD = sumD / cntD
        meanD2 = sumD2 / cntD
        chi_out[idx] = n * (meanD2 - meanD * meanD)

        var_m = msum2 / cntD - (msum / cntD) ** 2
        mean_var_i = 0.0
        for kk in range(n):
            var_kk = csum2[kk] / cntD - (csum[kk] / cntD) ** 2
            mean_var_i += var_kk
        mean_var_i /= n
        chi2 = var_m / (mean_var_i + 1e-12)
        if chi2 < 0.0:
            chi2 = 0.0
        Rsync_out[idx] = np.sqrt(chi2)

    return chi_out, Rsync_out


@njit(parallel=True, fastmath=True, cache=True)
def run_ensemble_core(seeds, disease, alpha_seq, steps, nx, ny, n):
    ns = seeds.shape[0]
    nl = alpha_seq.shape[0]
    chi = np.zeros((ns, nl))
    Rsync = np.zeros((ns, nl))
    for i in prange(ns):
        c_i, r_i = run_one_seed_sweep(seeds[i], disease, alpha_seq, steps, nx, ny, n)
        chi[i, :] = c_i
        Rsync[i, :] = r_i
    return chi, Rsync


# ============================================================
# HELPERS
# ============================================================
def ci95(stack):
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    half = 1.96 * sd / np.sqrt(n)
    return mean, half


def smooth(x, w):
    # Edge-normalised boxcar (endpoints stay honest partial averages, not pulled
    # toward zero by the convolution's implicit zero-padding).
    x = np.asarray(x, dtype=float)
    k = np.ones(w)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x), k, mode="same")
    return num / den


# ============================================================
# COMPUTE
# ============================================================
def compute(n_seeds=N_SEEDS_DEFAULT, T=T_DEFAULT, grid=GRID_DEFAULT):
    os.makedirs(PROC_DIR, exist_ok=True)
    steps = int(T / dt)
    nx = ny = grid
    n = nx * ny
    seeds = np.arange(11, 11 + n_seeds, dtype=np.int64)

    # continuous schedule: up then down (down leg starts from the up leg's end state)
    alpha_down = alpha_up[::-1]
    alpha_seq = np.concatenate([alpha_up, alpha_down])
    n_up = len(alpha_up)

    print(f"Computing S2 hysteresis: grid={grid}x{grid}, {n_seeds} seeds, "
          f"T={T}/level, {len(alpha_seq)} levels (up+down)")

    data = {"alpha": alpha_up, "n_seeds": n_seeds, "T": T, "grid": grid}
    rows = []
    for cond, disease in (("H", False), ("D", True)):
        print(f"  {cond} ensemble...")
        chi, Rsync = run_ensemble_core(seeds, disease, alpha_seq, steps, nx, ny, n)
        for obs, stack in (("chi", chi), ("Rsync", Rsync)):
            up = stack[:, :n_up]
            down = stack[:, n_up:][:, ::-1]        # realign to ascending alpha
            for leg, s in (("up", up), ("down", down)):
                mean, half = ci95(s)
                data[f"{cond}_{obs}_{leg}_mean"] = mean
                data[f"{cond}_{obs}_{leg}_ci"] = half
                for k, al in enumerate(alpha_up):
                    rows.append({"condition": cond, "observable": obs, "leg": leg,
                                 "alpha": al, "mean": mean[k],
                                 "ci_lo": mean[k] - half[k], "ci_hi": mean[k] + half[k]})

    np.savez(CACHE, **data)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    print(f"Cached -> {CACHE} and {CURVES_CSV}")
    return {k: np.asarray(v) if not np.isscalar(v) else v for k, v in data.items()}


def load():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ============================================================
# PLOT
# ============================================================
OBS_TITLE = {
    "chi": r"Extreme-value fluctuation $\chi_{ext}$",
    "Rsync": r"Synchrony order parameter $R$",
}
COND_NAME = {"H": "Healthy", "D": "Disease"}
COND_COLOR = {"H": "#0072B2", "D": "#D55E00"}


def plot(data, save_stem="Figure_S2"):
    apply_style()
    alpha = np.asarray(data["alpha"])
    observables = ["chi", "Rsync"]
    conditions = ["H", "D"]

    fig, axes = plt.subplots(len(observables), len(conditions),
                             figsize=(9, 7), squeeze=False)

    for r, obs in enumerate(observables):
        for c, cond in enumerate(conditions):
            ax = axes[r][c]
            color = COND_COLOR[cond]
            for leg, ls, lab in (("up", "-", r"up-sweep $\alpha\uparrow$"),
                                 ("down", "--", r"down-sweep $\alpha\downarrow$")):
                mean = smooth(np.asarray(data[f"{cond}_{obs}_{leg}_mean"]), 3)
                half = smooth(np.asarray(data[f"{cond}_{obs}_{leg}_ci"]), 3)
                ax.plot(alpha, mean, ls, color=color, linewidth=2, label=lab)
                ax.fill_between(alpha, mean - half, mean + half,
                                color=color, alpha=0.15, linewidth=0)
            ax.set_xlim(alpha.min(), alpha.max())
            clean_spines(ax)
            if r == 0:
                ax.set_title(f"{COND_NAME[cond]}")
            if r == len(observables) - 1:
                ax.set_xlabel(r"ATP level $\alpha$")
            if c == 0:
                ax.set_ylabel(OBS_TITLE[obs])
            if r == 0 and c == 0:
                ax.legend(loc="upper right")

    fig.suptitle("Hysteresis under up/down ATP sweeps", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, save_stem)
    print(f"Saved -> {save_stem}.pdf / .png")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    recompute = "--recompute" in sys.argv
    smoke = "--smoke" in sys.argv
    if smoke:
        data = compute(n_seeds=4, T=150.0, grid=8)
    elif recompute or not os.path.exists(CACHE):
        data = compute()
    else:
        data = load()
    plot(data)
