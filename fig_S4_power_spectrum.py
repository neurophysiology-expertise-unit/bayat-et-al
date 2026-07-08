"""
Supplementary Figure S4 -- Power spectrum of network activity vs ATP.

Backs the Results claim that high ATP produces fast, localized calcium oscillations
(whereas low ATP produces slow, coordinated waves). The population-mean signal
m(t) = <C(t)>_space is recorded in the stationary state at a LOW and a HIGH ATP
level, and its power spectral density (Welch, ensemble-averaged over seeds) is
compared. The transition from coordinated waves to fragmented local oscillation
should move spectral power to higher frequency as ATP rises -- consistent with the
recovery variable speeding up (tau_h shrinks with alpha) and the network decoupling.

To obtain a clean stationary spectrum each level is run at FIXED alpha (long
transient discarded, then a high-resolution record), rather than sampled inside the
quasi-static sweep. The FHN reaction-diffusion core and model parameters are
identical to fig_3_criticality_ci.py. compute -> cache -> plot; --recompute reruns.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.signal import welch

from plotstyle import apply_style, clean_spines, save_fig

PROC_DIR = "processed_data"
CACHE = os.path.join(PROC_DIR, "figS4_spectrum.npz")
CURVES_CSV = os.path.join(PROC_DIR, "figS4_spectrum_curves.csv")

# ============================================================
# GRID / TIME  (matched to fig_3_criticality_ci.py)
# ============================================================
GRID_DEFAULT = 10
dt = 0.0034

ALPHA_LOW = 0.15           # near the (healthy) transition -> coordinated waves
ALPHA_HIGH = 1.00          # deep in the fragmented / high-ATP regime
REC_STRIDE = 10            # subsample m(t) every REC_STRIDE steps
N_REC = 8192               # recorded samples per run
T_TRANSIENT = 300.0        # model-time discarded before recording

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
def run_fixed_alpha(seed, disease, alpha, steps_trans, n_rec, stride, nx, ny, n):
    """
    Integrate at a FIXED alpha, discard the transient, then record the subsampled
    population-mean m(t) = <C>_space. Dynamics/formulas mirror fig_3_criticality_ci.py.
    Returns m_rec (n_rec,).
    """
    np.random.seed(seed)

    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))

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

    m_rec = np.zeros(n_rec)
    total = steps_trans + n_rec * stride
    rec_i = 0

    for t in range(total):
        noise = sigma_eff * 3.0 * np.random.standard_normal((nx, ny))
        C_active = 0.5 * (1.0 + np.tanh(eta * (C - theta)))
        diff = Deff * laplacian(C_active)

        dC = C - (C ** 3) / 3.0 - h + I0 + gamma * alpha + diff + noise
        dh = (C + a - b * h) / tau_h

        C = C + dt * dC
        h = h + dt * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)

        if t >= steps_trans:
            rel = t - steps_trans
            if rel % stride == 0 and rec_i < n_rec:
                m_rec[rec_i] = np.mean(C)
                rec_i += 1

    return m_rec


@njit(parallel=True, fastmath=True, cache=True)
def record_ensemble(seeds, disease, alpha, steps_trans, n_rec, stride, nx, ny, n):
    ns = seeds.shape[0]
    out = np.zeros((ns, n_rec))
    for i in prange(ns):
        out[i, :] = run_fixed_alpha(seeds[i], disease, alpha, steps_trans, n_rec, stride, nx, ny, n)
    return out


# ============================================================
# SPECTRUM (ensemble-averaged Welch PSD)
# ============================================================
def mean_psd(series, sample_dt):
    """series: (ns, n_rec) -> (freqs, mean PSD over seeds). fs in 1/model-time."""
    fs = 1.0 / sample_dt
    nper = min(1024, series.shape[1])
    psds = []
    freqs = None
    for x in series:
        f, p = welch(x - x.mean(), fs=fs, nperseg=nper, detrend="constant")
        freqs = f
        psds.append(p)
    return freqs, np.mean(np.vstack(psds), axis=0)


def spectral_centroid(freqs, psd):
    """
    Power-weighted mean frequency (DC bin excluded). For a 1/f-type spectrum the
    peak sits at the lowest bin regardless of ATP, so the peak frequency is
    uninformative; the centroid instead tracks how much power sits in the
    high-frequency tail and therefore captures the shift with ATP.
    """
    if len(freqs) < 2:
        return 0.0
    f, p = freqs[1:], psd[1:]
    tot = p.sum()
    if tot <= 0:
        return 0.0
    return float((f * p).sum() / tot)


# ============================================================
# COMPUTE
# ============================================================
def compute(n_seeds=N_SEEDS_DEFAULT, grid=GRID_DEFAULT, n_rec=N_REC,
            stride=REC_STRIDE, t_transient=T_TRANSIENT):
    os.makedirs(PROC_DIR, exist_ok=True)
    nx = ny = grid
    n = nx * ny
    seeds = np.arange(11, 11 + n_seeds, dtype=np.int64)
    steps_trans = int(t_transient / dt)
    sample_dt = stride * dt

    print(f"Computing S4 spectrum: grid={grid}x{grid}, {n_seeds} seeds, "
          f"n_rec={n_rec}, sample_dt={sample_dt:.4f}, transient={t_transient}")

    levels = {"low": ALPHA_LOW, "high": ALPHA_HIGH}
    data = {"n_seeds": n_seeds, "grid": grid, "sample_dt": sample_dt,
            "alpha_low": ALPHA_LOW, "alpha_high": ALPHA_HIGH}
    rows = []
    for cond, disease in (("H", False), ("D", True)):
        for lvl, al in levels.items():
            print(f"  {cond} alpha={al} ({lvl})...")
            series = record_ensemble(seeds, disease, al, steps_trans, n_rec, stride, nx, ny, n)
            f, psd = mean_psd(series, sample_dt)
            data[f"{cond}_{lvl}_freq"] = f
            data[f"{cond}_{lvl}_psd"] = psd
            data[f"{cond}_{lvl}_fcent"] = spectral_centroid(f, psd)
            for fi, pi in zip(f, psd):
                rows.append({"condition": cond, "level": lvl, "alpha": al,
                             "freq": fi, "psd": pi})

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
COND_NAME = {"H": "Healthy", "D": "Disease"}
LEVEL_STYLE = {
    "low": ("#0072B2", "-", "low ATP"),
    "high": ("#D55E00", "-", "high ATP"),
}


def plot(data, save_stem="Figure_S4"):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), squeeze=False)
    a_lo = float(np.asarray(data["alpha_low"]))
    a_hi = float(np.asarray(data["alpha_high"]))

    for c, cond in enumerate(("H", "D")):
        ax = axes[0][c]
        for lvl in ("low", "high"):
            color, ls, _ = LEVEL_STYLE[lvl]
            f = np.asarray(data[f"{cond}_{lvl}_freq"])
            psd = np.asarray(data[f"{cond}_{lvl}_psd"])
            fc = spectral_centroid(f, psd)
            al = a_lo if lvl == "low" else a_hi
            ax.loglog(f[1:], psd[1:], ls, color=color, linewidth=1.8,
                      label=rf"$\alpha={al:g}$ ($\bar f={fc:.3f}$)")
            ax.axvline(fc, color=color, linewidth=0.9, linestyle=":", alpha=0.7)
        ax.set_title(COND_NAME[cond])
        ax.set_xlabel(r"frequency (cycles / model time)")
        if c == 0:
            ax.set_ylabel(r"PSD of $\langle C\rangle$")
        ax.legend(loc="lower left")
        clean_spines(ax)

    fig.suptitle("Activity power spectrum shifts to higher frequency with ATP",
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
        data = compute(n_seeds=4, grid=8, n_rec=2048, stride=10, t_transient=60.0)
    elif recompute or not os.path.exists(CACHE):
        data = compute()
    else:
        data = load()
    plot(data)
