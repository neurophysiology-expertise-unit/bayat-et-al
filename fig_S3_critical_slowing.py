"""
Supplementary Figure S3 -- Critical slowing down across the ATP transition.

Critical slowing down is the canonical early-warning signature of a critical
transition: as a system approaches the tipping point its recovery from
perturbations slows, so the autocorrelation of its fluctuations grows. Here the
population-mean calcium signal m(t) = <C(t)>_space is recorded over the stationary
window at each ATP level of a quasi-static up-sweep, and two standard indicators
are extracted per level:

  * lag-1 autocorrelation (AR(1) coefficient) of m(t)      -- the classic EWS
  * autocorrelation time tau_ac (1/e-folding of the ACF)   -- a relaxation time

A peak/rise of these near the fluctuation (chi) maximum indicates slowing down at
the transition, complementing the spatial coherence length xi (a length) with a
time scale.

The FHN reaction-diffusion core and model parameters are identical to
fig_3_criticality_ci.py; the up-sweep here reproduces its up-sweep and additionally
records a subsampled m(t) per level. compute -> cache -> plot; --recompute reruns.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange

from plotstyle import apply_style, clean_spines, save_fig, panel_label

PROC_DIR = "processed_data"
CACHE = os.path.join(PROC_DIR, "figS3_slowing.npz")
CURVES_CSV = os.path.join(PROC_DIR, "figS3_slowing_curves.csv")

# ============================================================
# GRID / TIME / SWEEP  (matched to fig_3_criticality_ci.py)
# ============================================================
GRID_DEFAULT = 10
dt = 0.0034
T_DEFAULT = 1000.0
N_SAMP = 2048                       # subsampled m(t) points per level
AR1_LAG_TIME = 3.4                  # cadence (model time) for the lag-1 autocorrelation
#   The recorded m(t) is oversampled relative to the network's relaxation time, so a
#   lag-1 autocorrelation at the raw cadence saturates near 1 and carries no structure.
#   AR(1) is therefore evaluated at a coarser lag comparable to tau_ac, the standard
#   cadence for an early-warning indicator.

alpha_values = np.linspace(0.01, 1.11, 21)

# ============================================================
# MODEL PARAMETERS  (identical to fig_3_criticality_ci.py)
# ============================================================
a = 1.0
b = 0.8
sigma = 0.4
eta = 8.0
theta_base = 0.5

N_SEEDS_DEFAULT = 20
ENSEMBLE_SEEDS = list(range(11, 11 + N_SEEDS_DEFAULT))


# ============================================================
# NUMBA CORE  (dynamics verbatim from fig_3_criticality_ci.py)
# ============================================================
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
def run_one_seed_series(seed, disease, alpha_values, steps, nx, ny, n, n_samp):
    """
    Quasi-static up-sweep for a single seed. In addition to chi (kept for locating
    the transition) it records the subsampled population-mean signal m(t) over each
    level's stationary window. Dynamics/formulas mirror fig_3_criticality_ci.py.
    Returns chi (n_alpha,) and m_series (n_alpha, n_samp).
    """
    np.random.seed(seed)
    n_alpha = alpha_values.shape[0]
    chi_out = np.zeros(n_alpha)
    m_series = np.zeros((n_alpha, n_samp))

    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))

    t_start = int(0.3 * steps)
    stat_steps = steps - t_start
    stride = stat_steps // n_samp
    if stride < 1:
        stride = 1

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

        sumD = 0.0
        sumD2 = 0.0
        cntD = 0
        samp_i = 0

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

                rel = t - t_start
                if rel % stride == 0 and samp_i < n_samp:
                    m_series[idx, samp_i] = np.mean(C)
                    samp_i += 1

        meanD = sumD / cntD
        meanD2 = sumD2 / cntD
        chi_out[idx] = n * (meanD2 - meanD * meanD)

    return chi_out, m_series


@njit(parallel=True, fastmath=True, cache=True)
def run_ensemble_core(seeds, disease, alpha_values, steps, nx, ny, n, n_samp):
    ns = seeds.shape[0]
    na = alpha_values.shape[0]
    chi = np.zeros((ns, na))
    series = np.zeros((ns, na, n_samp))
    for i in prange(ns):
        c_i, m_i = run_one_seed_series(seeds[i], disease, alpha_values, steps, nx, ny, n, n_samp)
        chi[i, :] = c_i
        series[i, :, :] = m_i
    return chi, series


# ============================================================
# SLOWING-DOWN INDICATORS (pure python, on the subsampled signal)
# ============================================================
def ar1_coeff(x):
    """Lag-1 autocorrelation of a 1D signal."""
    x = x - x.mean()
    v = np.dot(x, x)
    if v <= 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / v)


def tau_ac_efold(x, sample_dt):
    """
    Autocorrelation time as the 1/e-folding lag of the (normalised) ACF, in model
    time units. Linearly interpolated between the two lags bracketing 1/e; censored
    at the record length if the ACF never falls to 1/e.
    """
    x = x - x.mean()
    v = np.dot(x, x)
    if v <= 0:
        return 0.0
    full = np.correlate(x, x, mode="full")[len(x) - 1:] / v   # ac[0] = 1
    thr = 1.0 / np.e
    below = np.where(full < thr)[0]
    if len(below) == 0:
        return len(full) * sample_dt
    k = below[0]
    if k == 0:
        return 0.0
    a0, a1 = full[k - 1], full[k]
    frac = (a0 - thr) / (a0 - a1) if a0 != a1 else 0.0
    return (k - 1 + frac) * sample_dt


def indicators_ensemble(series, sample_dt):
    """series: (ns, na, n_samp) -> per (seed, alpha) AR(1) and tau_ac stacks.

    tau_ac is measured on the full-resolution record; the lag-1 autocorrelation is
    measured on the record subsampled to ~AR1_LAG_TIME (a cadence comparable to the
    relaxation time), which keeps AR(1) in an informative, unsaturated range.
    """
    ns, na, _ = series.shape
    sub = max(1, int(round(AR1_LAG_TIME / sample_dt)))
    ar1 = np.zeros((ns, na))
    tau = np.zeros((ns, na))
    for s in range(ns):
        for j in range(na):
            x = series[s, j]
            ar1[s, j] = ar1_coeff(x[::sub])
            tau[s, j] = tau_ac_efold(x, sample_dt)
    return ar1, tau


def ci95(stack):
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    return mean, 1.96 * sd / np.sqrt(n)


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
def compute(n_seeds=N_SEEDS_DEFAULT, T=T_DEFAULT, grid=GRID_DEFAULT, n_samp=N_SAMP):
    os.makedirs(PROC_DIR, exist_ok=True)
    steps = int(T / dt)
    nx = ny = grid
    n = nx * ny
    seeds = np.arange(11, 11 + n_seeds, dtype=np.int64)

    t_start = int(0.3 * steps)
    stride = max(1, (steps - t_start) // n_samp)
    sample_dt = stride * dt

    print(f"Computing S3 slowing down: grid={grid}x{grid}, {n_seeds} seeds, "
          f"T={T}, n_samp={n_samp}, sample_dt={sample_dt:.4f}")

    data = {"alpha": alpha_values, "n_seeds": n_seeds, "T": T, "grid": grid,
            "sample_dt": sample_dt}
    rows = []
    for cond, disease in (("H", False), ("D", True)):
        print(f"  {cond} ensemble...")
        chi, series = run_ensemble_core(seeds, disease, alpha_values, steps, nx, ny, n, n_samp)
        ar1, tau = indicators_ensemble(series, sample_dt)
        chi_mean, _ = ci95(chi)
        data[f"{cond}_chi_mean"] = chi_mean

        # per-seed peak-over-baseline tau_ac enhancement (baseline = high-alpha tail)
        base = tau[:, -5:].mean(axis=1)
        enh = tau.max(axis=1) / base
        data[f"{cond}_tau_enh_mean"] = enh.mean()
        data[f"{cond}_tau_enh_sd"] = enh.std(ddof=1)
        data[f"{cond}_tau_enh_n"] = len(enh)
        print(f"    {cond} tau_ac peak/baseline enhancement: "
              f"{enh.mean():.2f} +/- {enh.std(ddof=1):.2f} (n={len(enh)})")
        for obs, stack in (("ar1", ar1), ("tau", tau)):
            mean, half = ci95(stack)
            data[f"{cond}_{obs}_mean"] = mean
            data[f"{cond}_{obs}_ci"] = half
            for k, al in enumerate(alpha_values):
                rows.append({"condition": cond, "observable": obs, "alpha": al,
                             "mean": mean[k], "ci_lo": mean[k] - half[k],
                             "ci_hi": mean[k] + half[k]})

    np.savez(CACHE, **data)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    print(f"Cached -> {CACHE} and {CURVES_CSV}")
    return {k: v for k, v in data.items()}


def load():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ============================================================
# PLOT
# ============================================================
OBS_TITLE = {
    "ar1": r"Lag-1 autocorrelation of $\langle C\rangle$",
    "tau": r"Autocorrelation time $\tau_{ac}$",
}
OBS_YLABEL = {"ar1": rf"AR(1) coeff. ($\Delta t\approx{AR1_LAG_TIME:g}$)",
              "tau": r"$\tau_{ac}$ (model time)"}
COND_NAME = {"H": "Healthy", "D": "Disease"}
COND_COLOR = {"H": "#0072B2", "D": "#D55E00"}


def plot(data, save_stem="Figure_S3"):
    apply_style()
    alpha = np.asarray(data["alpha"])
    observables = ["ar1", "tau"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), squeeze=False)

    # transition loci = each condition's own chi peak
    aH = alpha[int(np.argmax(np.asarray(data["H_chi_mean"])))]
    aD = alpha[int(np.argmax(np.asarray(data["D_chi_mean"])))]

    for c, obs in enumerate(observables):
        ax = axes[0][c]
        for cond in ("H", "D"):
            color = COND_COLOR[cond]
            mean = smooth(np.asarray(data[f"{cond}_{obs}_mean"]), 3)
            half = smooth(np.asarray(data[f"{cond}_{obs}_ci"]), 3)
            ax.plot(alpha, mean, "-", color=color, linewidth=2, label=COND_NAME[cond])
            ax.fill_between(alpha, mean - half, mean + half, color=color,
                            alpha=0.15, linewidth=0)
        ax.axvline(aH, color="0.6", linewidth=1.0, linestyle=":")
        ax.set_xlim(alpha.min(), alpha.max())
        ax.set_title(OBS_TITLE[obs])
        panel_label(ax, "AB"[c])
        ax.set_xlabel(r"ATP level $\alpha$")
        ax.set_ylabel(OBS_YLABEL[obs])
        clean_spines(ax)
        if c == 0:
            ax.legend(loc="upper right")
            ax.annotate(r"$\chi$ peak", xy=(aH, ax.get_ylim()[1]),
                        xytext=(4, -4), textcoords="offset points",
                        fontsize=7, color="0.4", va="top")

    # (C) baseline-normalised tau_ac with transitions aligned: isolates the SHAPE of
    # the slowing signature from the imposed recovery timescale tau_h (which disease
    # triples). Near-collapse => the slowing is comparable; the disease difference in
    # absolute tau_ac is inherited from tau_h, not an independent effect.
    axc = axes[0][2]
    tauH = np.asarray(data["H_tau_mean"]); tauD = np.asarray(data["D_tau_mean"])
    baseH = tauH[-5:].mean(); baseD = tauD[-5:].mean()
    axc.plot(alpha - aH, smooth(tauH / baseH, 3), "-", color=COND_COLOR["H"],
             linewidth=2, label=COND_NAME["H"])
    axc.plot(alpha - aD, smooth(tauD / baseD, 3), "-", color=COND_COLOR["D"],
             linewidth=2, label=COND_NAME["D"])
    axc.axvline(0.0, color="0.6", linewidth=1.0, linestyle=":")
    axc.set_xlim((alpha - aH).min(), (alpha - aH).max())
    axc.set_title(r"$\tau_{ac}/$baseline, transitions aligned")
    panel_label(axc, "C")
    axc.set_xlabel(r"$\alpha-\alpha_{\mathrm{transition}}$")
    axc.set_ylabel(r"$\tau_{ac}/\tau_{ac}^{\mathrm{base}}$")
    clean_spines(axc)

    fig.suptitle("Critical slowing down across the ATP transition",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
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
        data = compute(n_seeds=4, T=150.0, grid=8, n_samp=512)
    elif recompute or not os.path.exists(CACHE):
        data = compute()
    else:
        data = load()
    plot(data)
