"""
EXPLORATION (not a paper figure) — synchrony & wave-extent observables for Fig 3.

Motivation
----------
The archived literature does NOT discriminate healthy vs disease astrocyte
networks with a raw spatial coherence length xi. Instead:

  * Lallouette et al. 2019 characterise network state by the EXTENT of
    intercellular Ca2+ wave propagation -- the number / fraction of recruited
    ("activated") astrocytes -- analysed with a shell / percolation picture.
  * Nimmerjahn 2015 (in vivo) and Lapato 2018 (disease) frame the health axis
    as the DEGREE OF SYNCHRONISATION of astrocyte activity (disease = loss of
    coherence / aberrant (de)synchronisation). Peng 2026 likewise scores network
    dynamics by temporal correlation (synchrony).

So this script prototypes two better-motivated discriminating observables on the
SAME model / sweep as fig_3_criticality_ci.py, to see whether either cleanly
separates the two conditions across the ATP sweep:

  R_sync : Golomb-Rinzel-Hansel population synchrony order parameter
             R = sqrt( Var_t(<C>_space) / mean_i Var_t(C_i) ),  in [0, 1].
           R -> 1 fully synchronised; R ~ 1/sqrt(N) fully asynchronous.
           (Golomb & Hansel 2000 is already cited in Lallouette 2019.)

  P_inf  : percolation / wave-extent order parameter
             fraction of the lattice occupied by the LARGEST connected cluster of
             co-active cells (C_active > 0.5), averaged over the stationary window,
             with toroidal (periodic) wrap to match the sim's boundary conditions.

This is an exploration harness. It does NOT modify fig_3_criticality_ci.py, the
paper figures, or the manuscript. Small lattice (default 10x10) is fine here:
R_sync is a temporal-correlation measure and does not need spatial headroom the
way xi does.

Run (conda env `ece`):
    python explore_sync_observable.py --grid 10 --seeds 20
"""

import os
import sys
import numpy as np
from numba import njit, prange
from scipy import ndimage

PROC_DIR = "processed_data"
CACHE = os.path.join(PROC_DIR, "explore_sync.npz")

# --- model / sweep constants: identical to fig_3_criticality_ci.py ---
dt = 0.0034
a = 1.0
b = 0.8
sigma = 0.4
eta = 8.0
theta_base = 0.5
alpha_values = np.linspace(0.01, 1.11, 21)


@njit(fastmath=True, cache=True)
def laplacian(Z):
    """Periodic (toroidal) 5-point Laplacian (identical to the paper core)."""
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
    One quasi-static ATP up-sweep for a single seed. Model equations are copied
    verbatim from fig_3_criticality_ci.run_one_seed_core; the only additions are
    the accumulators for the synchrony order parameter and the last-5-frame mean
    used for the percolation extent.

    Returns per-alpha:
      Rsync : Golomb population synchrony order parameter in [0, 1]
      theta : the activation threshold used at this alpha (for the extent mask)
      last5 : mean of last 5 stationary frames, flattened (nx*ny)
    """
    np.random.seed(seed)
    n_alpha = alpha_values.shape[0]
    Rsync = np.zeros(n_alpha)
    theta_out = np.zeros(n_alpha)
    last5 = np.zeros((n_alpha, n))

    # fixed heterogeneity fields (same draws / order as the paper core)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

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

        # synchrony accumulators over the stationary window
        msum = 0.0       # sum_t of population-mean C
        msum2 = 0.0      # sum_t of (population-mean C)^2
        csum = np.zeros(n)    # per-cell sum_t C_i
        csum2 = np.zeros(n)   # per-cell sum_t C_i^2
        cnt = 0

        ring = np.zeros((5, n))
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

            if t >= t_start:
                Cf = C.reshape(n)
                m = 0.0
                for k in range(n):
                    ck = Cf[k]
                    m += ck
                    csum[k] += ck
                    csum2[k] += ck * ck
                m /= n
                msum += m
                msum2 += m * m
                cnt += 1

                ring[ring_pos, :] = Cf
                ring_pos = (ring_pos + 1) % 5
                if ring_filled < 5:
                    ring_filled += 1

        # Golomb synchrony:  Var_t(<C>) / mean_i Var_t(C_i)
        var_m = msum2 / cnt - (msum / cnt) ** 2
        mean_var_i = 0.0
        for k in range(n):
            var_k = csum2[k] / cnt - (csum[k] / cnt) ** 2
            mean_var_i += var_k
        mean_var_i /= n
        chi2 = var_m / (mean_var_i + 1e-12)
        if chi2 < 0.0:
            chi2 = 0.0
        Rsync[idx] = np.sqrt(chi2)
        theta_out[idx] = theta

        acc = np.zeros(n)
        for k in range(ring_filled):
            acc += ring[k, :]
        last5[idx, :] = acc / ring_filled

    return Rsync, theta_out, last5


@njit(parallel=True, fastmath=True, cache=True)
def run_ensemble_core(seeds, disease, alpha_values, steps, nx, ny, n):
    ns = seeds.shape[0]
    na = alpha_values.shape[0]
    Rsync = np.zeros((ns, na))
    theta = np.zeros((ns, na))
    last5 = np.zeros((ns, na, n))
    for i in prange(ns):
        r_i, th_i, l5_i = run_one_seed_core(seeds[i], disease, alpha_values, steps, nx, ny, n)
        Rsync[i, :] = r_i
        theta[i, :] = th_i
        last5[i, :, :] = l5_i
    return Rsync, theta, last5


def giant_cluster_fraction(frame_flat, theta, nx, ny):
    """
    Fraction of the lattice in the largest connected cluster of co-active cells,
    with toroidal (periodic) connectivity to match the sim's boundaries.

    A cell is 'active' where the mean stationary field exceeds the activation
    threshold theta (equivalently C_active = 0.5(1+tanh(eta(C-theta))) > 0.5).
    """
    field = np.asarray(frame_flat, dtype=float).reshape(nx, ny)
    active = field > theta
    n_active = int(active.sum())
    if n_active == 0:
        return 0.0, 0.0

    lbl, nlab = ndimage.label(active)          # 4-connectivity, non-periodic
    if nlab == 0:
        return 0.0, 0.0

    # union-find over labels to stitch clusters across the periodic borders
    parent = np.arange(nlab + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(nx):
        if active[i, 0] and active[i, ny - 1]:
            union(lbl[i, 0], lbl[i, ny - 1])
    for j in range(ny):
        if active[0, j] and active[nx - 1, j]:
            union(lbl[0, j], lbl[nx - 1, j])

    sizes = {}
    flat = lbl.reshape(-1)
    for v in flat:
        if v == 0:
            continue
        root = find(v)
        sizes[root] = sizes.get(root, 0) + 1
    giant = max(sizes.values())
    return giant / (nx * ny), giant / n_active   # (giant/lattice, giant/active)


def ci95(stack):
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    half = 1.96 * sd / np.sqrt(n)
    return mean, half


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


OBS = ("Rsync", "Pinf", "Pfrag")
OBS_SMOOTH = {"Rsync": 3, "Pinf": 3, "Pfrag": 3}
OBS_NAME = {"Rsync": "R_sync", "Pinf": "P_inf ", "Pfrag": "P_frag"}


def run_condition(disease, seeds, steps, nx, ny, n):
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    Rsync, theta, last5 = run_ensemble_core(seeds_arr, disease, alpha_values, steps, nx, ny, n)
    na = len(alpha_values)
    Pinf = np.zeros((len(seeds), na))
    Pfrag = np.zeros((len(seeds), na))
    for si in range(len(seeds)):
        for j in range(na):
            g_lat, g_act = giant_cluster_fraction(last5[si, j], theta[si, j], nx, ny)
            Pinf[si, j] = g_lat
            Pfrag[si, j] = g_act
    print(f"  {'disease' if disease else 'healthy'} done ({len(seeds)} seeds)")
    return {"Rsync": Rsync, "Pinf": Pinf, "Pfrag": Pfrag}


def compute(n_seeds, T, grid):
    os.makedirs(PROC_DIR, exist_ok=True)
    steps = int(T / dt)
    nx = ny = grid
    n = nx * ny
    seeds = list(range(11, 11 + n_seeds))
    print(f"Exploration: grid={grid}x{grid}, {n_seeds} seeds, T={T}, steps={steps}")

    H = run_condition(False, seeds, steps, nx, ny, n)
    D = run_condition(True, seeds, steps, nx, ny, n)

    data = {"alpha": alpha_values, "grid": grid, "n_seeds": n_seeds, "T": T}
    for obs in OBS:
        for cond, stacks in (("H", H), ("D", D)):
            mean, half = ci95(stacks[obs])
            data[f"{cond}_{obs}_mean"] = mean
            data[f"{cond}_{obs}_ci"] = half
    np.savez(CACHE, **data)
    print(f"Cached -> {CACHE}")
    return data


def discrimination_report(data):
    na = len(data["alpha"])
    grid = int(data["grid"])
    print(f"\nDiscrimination (disjoint 95% CI bands, lattice {grid}x{grid}, "
          f"{int(data['n_seeds'])} seeds):")
    for obs in OBS:
        w = OBS_SMOOTH[obs]
        h_lo = smooth(data[f"H_{obs}_mean"] - data[f"H_{obs}_ci"], w)
        h_hi = smooth(data[f"H_{obs}_mean"] + data[f"H_{obs}_ci"], w)
        d_lo = smooth(data[f"D_{obs}_mean"] - data[f"D_{obs}_ci"], w)
        d_hi = smooth(data[f"D_{obs}_mean"] + data[f"D_{obs}_ci"], w)
        disjoint = (h_lo > d_hi) | (d_lo > h_hi)
        # direction: is healthy above disease where they separate?
        h_over_d = int(((h_lo > d_hi)).sum())
        d_over_h = int(((d_lo > h_hi)).sum())
        print(f"  {OBS_NAME[obs]}: {int(disjoint.sum()):2d}/{na} alpha "
              f"(healthy>disease at {h_over_d}, disease>healthy at {d_over_h})")


def plot(data, save_stem="explore_sync"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alpha = data["alpha"]
    titles = {"Rsync": r"(A) Synchrony order parameter $R$",
              "Pinf": r"(B) Giant active cluster $P_\infty$ / lattice",
              "Pfrag": r"(C) Connected active fraction $P_\infty$ / active"}
    style = {"H": ("Healthy", "#0072B2"), "D": ("Disease", "#D55E00")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, obs in zip(axes, OBS):
        w = OBS_SMOOTH[obs]
        for cond in ("H", "D"):
            label, color = style[cond]
            mean = data[f"{cond}_{obs}_mean"]
            half = data[f"{cond}_{obs}_ci"]
            ax.plot(alpha, smooth(mean, w), lw=2, label=label, color=color)
            ax.fill_between(alpha, smooth(mean - half, w), smooth(mean + half, w),
                            color=color, alpha=0.18, lw=0)
        ax.set_title(titles[obs])
        ax.set_xlabel(r"ATP level $\alpha$")
        ax.set_xlim(alpha.min(), alpha.max())
        ax.legend(loc="best", frameon=False)
    fig.savefig(f"{save_stem}.png", dpi=150)
    print(f"Saved {save_stem}.png")


def main():
    grid = 10
    n_seeds = 20
    T = 1000.0
    if "--grid" in sys.argv:
        grid = int(sys.argv[sys.argv.index("--grid") + 1])
    if "--seeds" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    if "--T" in sys.argv:
        T = float(sys.argv[sys.argv.index("--T") + 1])
    recompute = "--recompute" in sys.argv or not os.path.exists(CACHE)
    if recompute:
        data = compute(n_seeds, T, grid)
    else:
        d = np.load(CACHE, allow_pickle=True)
        data = {k: d[k] for k in d.files}
    discrimination_report(data)
    plot(data)


if __name__ == "__main__":
    main()
