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
# GRID / TIME / SWEEP
# ============================================================
# Grid size is now a run-time parameter (default 20x20). The legacy single-seed
# fig_3_criticality.py runs at 10x10; the coherence length xi is capped by the
# lattice (xi <= grid/2), so a 10x10 lattice censors any xi >~ 5 lattice units and
# cannot resolve a growing coherence length near the transition. A 20x20+ lattice
# gives xi headroom to be resolvable. Override with --grid <n> on the CLI.
GRID_DEFAULT = 20
Nx = Ny = GRID_DEFAULT
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
def run_one_seed_core(seed, disease, alpha_values, steps, nx, ny, n):
    """
    Run one full quasi-static ATP sweep for a single seed on an nx-by-ny lattice.

    Returns per-alpha arrays:
      Sc     : spatial heterogeneity  (mean over full history of var/|mean|)
      chi    : extreme-value fluctuation  N*(<DeltaC^2> - <DeltaC>^2) over stationary part
      last5  : mean of last 5 stationary frames, flattened  -> used for xi outside numba
      lastf  : final frame, flattened  -> used for snapshots outside numba

    Formulas mirror fig_3_criticality.py exactly; only the loop is JIT-compiled and
    the lattice size is a parameter.
    """
    np.random.seed(seed)

    n_alpha = alpha_values.shape[0]
    Sc_out = np.zeros(n_alpha)
    chi_out = np.zeros(n_alpha)
    Rsync_out = np.zeros(n_alpha)
    last5 = np.zeros((n_alpha, n))
    lastf = np.zeros((n_alpha, n))

    # fixed heterogeneity fields for this seed (same draws as fig_3_criticality.py)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

    # state carried across ATP levels (sequential quasi-static up-sweep)
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))

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

        # synchrony (Golomb-Rinzel) accumulators over the stationary window
        msum = 0.0        # sum_t of population-mean C
        msum2 = 0.0       # sum_t of (population-mean C)^2
        csum = np.zeros(n)    # per-cell sum_t C_i
        csum2 = np.zeros(n)   # per-cell sum_t C_i^2

        ring = np.zeros((5, n))       # last-5-frame ring buffer
        ring_pos = 0
        ring_filled = 0

        for t in range(steps):
            noise = sigma_eff * 3.0 * np.random.standard_normal((nx, ny))
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

            # fluctuation + coherence + synchrony accumulate over the stationary part
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

                ring[ring_pos, :] = Cf
                ring_pos = (ring_pos + 1) % 5
                if ring_filled < 5:
                    ring_filled += 1

        Sc_out[idx] = sc_sum / steps
        meanD = sumD / cntD
        meanD2 = sumD2 / cntD
        chi_out[idx] = n * (meanD2 - meanD * meanD)

        # Golomb-Rinzel synchrony order parameter:
        #   R = sqrt( Var_t(<C>_space) / mean_i Var_t(C_i) )  in [0, 1]
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

        acc = np.zeros(n)
        for k in range(ring_filled):
            acc += ring[k, :]
        last5[idx, :] = acc / ring_filled
        lastf[idx, :] = C.reshape(n)

    return Sc_out, chi_out, Rsync_out, last5, lastf


@njit(parallel=True, fastmath=True, cache=True)
def run_ensemble_core(seeds, disease, alpha_values, steps, nx, ny, n):
    """Run all seeds in parallel (one independent quasi-static sweep each)."""
    ns = seeds.shape[0]
    na = alpha_values.shape[0]
    Sc = np.zeros((ns, na))
    chi = np.zeros((ns, na))
    Rsync = np.zeros((ns, na))
    last5 = np.zeros((ns, na, n))
    for i in prange(ns):
        s_i, c_i, r_i, l5_i, _ = run_one_seed_core(seeds[i], disease, alpha_values, steps, nx, ny, n)
        Sc[i, :] = s_i
        chi[i, :] = c_i
        Rsync[i, :] = r_i
        last5[i, :, :] = l5_i
    return Sc, chi, Rsync, last5


# ============================================================
# POST-PROCESSING HELPERS (pure python)
# ============================================================
def corr_length_from_frame(frame_flat, nx, ny):
    """
    Spatial coherence length xi via exponential fit of the radial autocorrelation.

    Computes the periodic 2D autocorrelation of the field (FFT), azimuthally
    averages it to G(r), and fits G(r) ~ exp(-r / xi) over the leading positive,
    contiguous part by log-linear least squares. This replaces the earlier
    "count of lags with autocorr > 0.2" proxy with the exponential-decay length
    scale specified in the analysis plan. xi is capped by the lattice at grid/2.
    """
    field = np.asarray(frame_flat, dtype=float).reshape(nx, ny)
    f = field - field.mean()

    F = np.fft.fft2(f)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    if peak <= 0:
        return 0.0
    ac = ac / peak                                   # zero-lag normalised to 1

    cy, cx = nx // 2, ny // 2
    yy, xx = np.indices((nx, ny))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = nx // 2
    G = np.array([ac[(r >= k - 0.5) & (r < k + 0.5)].mean() for k in range(rmax + 1)])

    rr = np.arange(1, rmax + 1)
    g = G[1:rmax + 1]
    # leading contiguous stretch where the autocorrelation stays positive
    keep = []
    for i in range(len(g)):
        if g[i] > 0.05:
            keep.append(i)
        else:
            break
    if len(keep) < 2:
        return float(rr[0]) if len(keep) == 1 else 0.5
    rr2, g2 = rr[keep], g[keep]
    slope, _ = np.polyfit(rr2, np.log(g2), 1)
    if slope >= 0:
        return float(rr2[-1])
    return float(min(-1.0 / slope, rmax))


def smooth(x, w):
    # Edge-normalised boxcar: divide by the count of real points contributing at
    # each position so the endpoints are honest partial averages rather than being
    # pulled toward zero by the implicit zero-padding of a plain convolution.
    x = np.asarray(x, dtype=float)
    k = np.ones(w)
    num = np.convolve(x, k, mode='same')
    den = np.convolve(np.ones_like(x), k, mode='same')
    return num / den


def discrimination_report(data):
    """
    For each observable, count the alpha values at which the plotted (smoothed)
    healthy and disease 95% CI bands are DISJOINT -- i.e. the ensemble cleanly
    separates the two conditions. Answers 'does xi discriminate at this lattice?'.
    """
    na = len(data["alpha"])
    grid = int(data["grid"]) if "grid" in data else Nx
    print(f"\nDiscrimination (disjoint 95% CI bands, lattice {grid}x{grid}):")
    counts = {}
    for obs in OBS:
        w = OBS_SMOOTH[obs]
        h_lo = smooth(data[f"H_{obs}_mean"] - data[f"H_{obs}_ci"], w)
        h_hi = smooth(data[f"H_{obs}_mean"] + data[f"H_{obs}_ci"], w)
        d_lo = smooth(data[f"D_{obs}_mean"] - data[f"D_{obs}_ci"], w)
        d_hi = smooth(data[f"D_{obs}_mean"] + data[f"D_{obs}_ci"], w)
        disjoint = (h_lo > d_hi) | (d_lo > h_hi)
        counts[obs] = int(disjoint.sum())
        print(f"  {OBS_LABEL[obs]:>6}: {counts[obs]:2d}/{na} alpha")
    return counts


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
def run_ensemble(disease, seeds, steps, nx, ny, n):
    na = len(alpha_values)
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    Sc, chi, Rsync, last5 = run_ensemble_core(seeds_arr, disease, alpha_values, steps, nx, ny, n)
    corr = np.zeros((len(seeds), na))
    for si in range(len(seeds)):
        for j in range(na):
            corr[si, j] = corr_length_from_frame(last5[si, j], nx, ny)
    print(f"  {'disease' if disease else 'healthy'} ensemble done ({len(seeds)} seeds)")
    return Sc, chi, Rsync, corr


def snapshots_for_seed(disease, seed, steps, nx, ny, n):
    _, _, _, _, lastf = run_one_seed_core(seed, disease, alpha_values, steps, nx, ny, n)
    snaps = {}
    for key, lvl in zip(("low", "mid", "high"), target_snaps):
        idx = int(np.argmin(np.abs(alpha_values - lvl)))
        snaps[key] = lastf[idx].reshape(nx, ny)
    return snaps


# ============================================================
# COMPUTE  ->  cache to processed_data/
# ============================================================
OBS = ("Sc", "chi", "Rsync", "corr")
OBS_SMOOTH = {"Sc": 3, "chi": 3, "corr": 5, "Rsync": 3}
OBS_LABEL = {"Sc": "S_C", "chi": "chi", "corr": "xi", "Rsync": "R_sync"}
N_BOOT = 10000
BOOT_SEED = 0


def pvalue_paired_bootstrap(H, D, n_boot=N_BOOT, rng=None):
    """
    Two-sided bootstrap p-value for the paired difference Delta = disease - healthy
    at each ATP level (healthy and disease share the same seeds -> paired).
    Resamples seeds with replacement; p = 2*min(frac<=0, frac>=0), floored at 1/n_boot.
    Returns p (n_alpha,).
    """
    rng = rng or np.random.default_rng(BOOT_SEED)
    diff = D - H                                   # (n_seeds, n_alpha), paired
    ns, na = diff.shape
    boots = np.empty((n_boot, na))
    for b in range(n_boot):
        idx = rng.integers(0, ns, ns)
        boots[b] = diff[idx].mean(axis=0)
    frac_le = (boots <= 0).mean(axis=0)
    frac_ge = (boots >= 0).mean(axis=0)
    p = 2.0 * np.minimum(frac_le, frac_ge)
    return np.clip(p, 1.0 / n_boot, 1.0)


def compute(n_seeds=N_SEEDS_DEFAULT, T=T_DEFAULT, grid=GRID_DEFAULT):
    """Run the ensemble, aggregate to mean/95%-CI, and cache to processed_data/."""
    os.makedirs(PROC_DIR, exist_ok=True)
    steps = int(T / dt)
    nx = ny = grid
    n = nx * ny
    seeds = list(range(11, 11 + n_seeds))
    print(f"Computing Fig 3 ensemble: grid={grid}x{grid}, {n_seeds} seeds, T={T}, steps={steps}")

    stacks = {}
    print("Healthy ensemble...")
    stacks["Sc", "H"], stacks["chi", "H"], stacks["Rsync", "H"], stacks["corr", "H"] = run_ensemble(False, seeds, steps, nx, ny, n)
    print("Disease ensemble...")
    stacks["Sc", "D"], stacks["chi", "D"], stacks["Rsync", "D"], stacks["corr", "D"] = run_ensemble(True, seeds, steps, nx, ny, n)

    print(f"Snapshots from seed {SNAP_SEED}...")
    snaps = {"H": snapshots_for_seed(False, SNAP_SEED, steps, nx, ny, n),
             "D": snapshots_for_seed(True, SNAP_SEED, steps, nx, ny, n)}

    data = {"alpha": alpha_values, "n_seeds": n_seeds, "T": T, "snap_seed": SNAP_SEED, "grid": grid}
    rows = []
    for obs in OBS:
        for cond in ("H", "D"):
            mean, half = ci95(stacks[obs, cond])
            data[f"{cond}_{obs}_mean"] = mean
            data[f"{cond}_{obs}_ci"] = half
            data[f"{cond}_{obs}_stack"] = stacks[obs, cond]   # (n_seeds, n_alpha) for the Delta contrast
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
    "Rsync": r"(C) Synchrony order parameter $R$",
    "corr": r"(D) Spatial coherence length $\xi$",
}
COND_STYLE = {"H": ("Healthy", "#0072B2"), "D": ("Disease", "#D55E00")}   # colorblind-safe
P_THRESH = 0.05


def plot(data, save_stem="Figure_3_ci"):
    apply_style()
    alpha = data["alpha"]
    ncol = len(OBS)                                  # 4 observables

    fig = plt.figure(figsize=(15, 11))
    # 12-col grid so 4 observable columns and 3 snapshot columns both divide evenly
    gs = fig.add_gridspec(4, 12, height_ratios=[3.0, 1.0, 2.6, 2.6],
                          hspace=0.55, wspace=0.9)

    # ---- row 0: observable curves with 95% CI bands ----
    curve_axes = []
    for col, obs in enumerate(OBS):
        ax = fig.add_subplot(gs[0, 3 * col:3 * col + 3])
        curve_axes.append(ax)
        w = OBS_SMOOTH[obs]
        for cond in ("H", "D"):
            label, color = COND_STYLE[cond]
            mean = data[f"{cond}_{obs}_mean"]
            half = data[f"{cond}_{obs}_ci"]
            ax.plot(alpha, smooth(mean, w), linewidth=2, label=label, color=color)
            ax.fill_between(alpha, smooth(mean - half, w), smooth(mean + half, w),
                            color=color, alpha=0.18, linewidth=0)
        ax.set_title(TITLES[obs])
        ax.set_xlim(alpha.min(), alpha.max())
        ax.tick_params(labelbottom=False)
        clean_spines(ax)
        if col == 0:
            ax.legend(loc="upper right")

    # ---- row 1: per-alpha significance (paired bootstrap p-value, log axis) ----
    for col, obs in enumerate(OBS):
        ax = fig.add_subplot(gs[1, 3 * col:3 * col + 3])
        H = data[f"H_{obs}_stack"]
        D = data[f"D_{obs}_stack"]
        p = pvalue_paired_bootstrap(H, D)
        sig = p < P_THRESH
        ax.set_yscale("log")
        ax.axhline(P_THRESH, color="0.5", linewidth=1.0, linestyle="--")
        ax.fill_between(alpha, p, 1.0, where=sig, step=None,
                        color="#D55E00", alpha=0.18, linewidth=0)
        ax.plot(alpha, p, color="#333333", linewidth=1.2)
        ax.plot(alpha[sig], p[sig], "o", color="#D55E00", markersize=3.5)
        ax.set_xlim(alpha.min(), alpha.max())
        ax.set_ylim(1.0 / N_BOOT * 0.5, 1.2)
        ax.invert_yaxis()                            # significant (small p) at top
        ax.set_xlabel(r"ATP level $\alpha$")
        if col == 0:
            ax.set_ylabel(r"$p$ (H vs D)", fontsize=8)
        clean_spines(ax)

    # ---- rows 2-3: representative snapshots (3 ATP levels) ----
    row_titles = {"H": ("E", "Healthy"), "D": ("F", "Disease")}
    keys = ["low", "mid", "high"]
    labels = ["Low ATP", "Mid ATP", "High ATP"]
    im = None
    snap_axes = []
    for r, cond in zip((2, 3), ("H", "D")):
        letter, name = row_titles[cond]
        for i, k in enumerate(keys):
            ax = fig.add_subplot(gs[r, 4 * i:4 * i + 4])
            snap_axes.append(ax)
            im = ax.imshow(data[f"{cond}_snap_{k}"], cmap="inferno",
                           origin="lower", interpolation="nearest")
            ax.set_title(f"({letter}{i+1}) {name} — {labels[i]}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    cbar = fig.colorbar(im, ax=snap_axes, shrink=0.7, location="right", pad=0.02)
    cbar.set_label(r"Ca$^{2+}$ activity $C$")

    save_fig(fig, save_stem)
    print(f"Saved {save_stem}.pdf / .png")
    plt.close(fig)


def main(recompute=False, n_seeds=N_SEEDS_DEFAULT, T=T_DEFAULT, grid=GRID_DEFAULT,
         save_stem="Figure_3_ci"):
    if recompute or not os.path.exists(CACHE):
        data = compute(n_seeds=n_seeds, T=T, grid=grid)
    else:
        print(f"Loading cached results from {CACHE} (use recompute=True to rerun)")
        data = load()
    discrimination_report(data)
    plot(data, save_stem=save_stem)


if __name__ == "__main__":
    import sys
    grid = GRID_DEFAULT
    if "--grid" in sys.argv:
        grid = int(sys.argv[sys.argv.index("--grid") + 1])
    main(recompute="--recompute" in sys.argv, grid=grid)
