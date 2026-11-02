"""
core/model.py — the canonical ATP / astrocyte excitable-lattice model.

ONE definition of the model, extracted faithfully from the fig_3_criticality_ci.py
core (the main-result core; Fig 3 and Fig 5 use it), with two changes:

  1. the noise is scaled correctly (Euler-Maruyama, sqrt(dt)); see core/integrator.py;
  2. the base noise amplitude `sigma` is a PARAMETER, because Phase 0.2 recalibrates it.

The six ATP-dependent channels (see CHANNELS.md, Phase 0.4) are all here in one place:
  1 excitability   gamma * alpha
  2 noise          sigma_eff = sigma * (1 + 4*alpha)          [* 3.0 in the update]
  3 coupling       Deff = D0 / (1 + (kappa*alpha)^4)          (suppresses as alpha rises)
  4 threshold      theta = 0.5 + 0.7*alpha                    (suppresses coupling too)
  5 baseline       I0 = 0.05 + (1/sqrt(alpha)) * I0_base      (DECREASES as alpha rises)
  6 recovery       tau_h = 10 / ((1 + 0.8*alpha) * tau_base)

DIAGNOSTIC FLAG. `simulate_fixed_alpha` takes `legacy_dt_noise`. Default False =
the correct sqrt(dt) scheme, which is what every production caller uses. The ONLY
place that passes True is phase0_recalibrate.py, whose whole job is to compare the
old (buggy) scheme against the new one. This is not a production toggle; it is a
measurement instrument, and tests/test_noise_scaling.py guards the default.

The heterogeneity draws (gamma_base, I0_base, ...) are the SAME uniform ranges and
the SAME draw order as fig_3_criticality_ci.py, so a given seed reproduces that
script's disorder exactly.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# --- fixed model constants (identical to fig_3_criticality_ci.py) ---
A_FHN = 1.0
B_FHN = 0.8
ETA = 8.0
THETA_BASE = 0.5
DT = 0.0034
NOISE_MULT = 3.0          # the historical *3.0 on the lattice noise term

# The old nominal was sigma = 0.4 under the buggy dt-scaling. The Euler-Maruyama
# equivalent (old_sigma * sqrt(dt)) is the Phase 0.2 starting hypothesis; the
# recalibration sweep confirms or refutes it before this is trusted.
SIGMA_OLD_NOMINAL = 0.4
SIGMA_EM_PREDICTED = SIGMA_OLD_NOMINAL * np.sqrt(DT)   # ~0.0233

# --- baseline excitability, I0 at A=0 ---
# Fixed by the spontaneous-activity constraint, not chosen for network behaviour. The fine
# sweep (phase3_spontaneous_fine.py -> processed_data/i0_fine.npz, 10 seeds) resolves the
# silence-to-activity crossover to 0.01: no realization ignites at or below 0.37, ignition is
# all-or-none per realization, and the fraction of realizations that ignite rises 2/10 at 0.38
# to 10/10 at 0.42. 0.42 is therefore the lowest baseline at which EVERY seed in the figure
# ensemble satisfies the constraint, at 0.439 +/- 0.198 transients/min/cell (3.6x the somatic
# rate of Hirase et al.; the superseded 0.45 baseline produced a larger discrepancy.
I0_BASE = 0.42
I0_BASE_LEGACY = 0.45     # what Figs 1-4 were produced at before the fine sweep; kept for diffs


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
def simulate_fixed_alpha(alpha, sigma, seed, steps, nx, ny, disease, burn_frac,
                         legacy_dt_noise):
    """Run one stationary simulation at a FIXED alpha and return regime observables.

    Returns (active_fraction, chi, spike_rate):
      active_fraction : time-and-space mean of the gating variable Phi(C)  in [0, 1]
      chi             : N * Var_t(spatial-mean C) over the stationary window (susceptibility)
      spike_rate      : mean per-cell up-crossings of C=0 per 1000 steps (oscillation proxy)

    `legacy_dt_noise=True` reproduces the OLD scheme (noise * dt) for the Phase 0.2
    comparison ONLY. Production uses the default (False).
    """
    np.random.seed(seed)

    # per-seed heterogeneity, same ranges/order as fig_3_criticality_ci.py
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

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
    theta = THETA_BASE + 0.7 * alpha
    sigma_eff = sigma * (1.0 + 4.0 * alpha)
    sqrt_dt = DT ** 0.5

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))

    t_start = int(burn_frac * steps)
    n = nx * ny

    msum = 0.0
    msum2 = 0.0
    act_sum = 0.0
    crossings = 0
    cnt = 0

    for t in range(steps):
        noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        diff = Deff * laplacian(C_active)
        dC = C - (C ** 3) / 3.0 - h + I0 + gamma * alpha + diff
        dh = (C + A_FHN - B_FHN * h) / tau_h

        C_prev = C
        if legacy_dt_noise:
            C = C + DT * (dC + noise)          # OLD (buggy) scheme, diagnostic only
        else:
            C = C + DT * dC + sqrt_dt * noise  # correct Euler-Maruyama
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)

        if t >= t_start:
            m = np.mean(C)
            msum += m
            msum2 += m * m
            act_sum += np.mean(C_active)
            # count per-cell up-crossings through zero (a spike proxy)
            for i in range(nx):
                for j in range(ny):
                    if C_prev[i, j] <= 0.0 and C[i, j] > 0.0:
                        crossings += 1
            cnt += 1

    if cnt == 0:
        return 0.0, 0.0, 0.0
    mean_m = msum / cnt
    var_m = msum2 / cnt - mean_m * mean_m
    chi = n * var_m
    active_fraction = act_sum / cnt
    spike_rate = 1000.0 * crossings / (cnt * n)
    return active_fraction, chi, spike_rate


@njit(fastmath=True, cache=True)
def final_state(alpha, sigma, seed, steps, nx, ny, disease, legacy_dt_noise):
    """Return the final flattened C field after a single-alpha run.

    VALIDATION PROBE ONLY. Uses the identical RNG draw order as
    fig_3_criticality_ci.run_one_seed_core (5 heterogeneity fields incl. the
    standard_normal in gamma_base, then C, then h, then per-step noise), so in
    legacy mode it must reproduce that core's `lastf` bit-for-bit — that is the
    Gate-1 test that core/model.py is a faithful extraction of the code that made
    the published lattice figures.
    """
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

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
    theta = THETA_BASE + 0.7 * alpha
    sigma_eff = sigma * (1.0 + 4.0 * alpha)
    sqrt_dt = DT ** 0.5

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))

    for t in range(steps):
        noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        diff = Deff * laplacian(C_active)
        dC = C - (C ** 3) / 3.0 - h + I0 + gamma * alpha + diff
        dh = (C + A_FHN - B_FHN * h) / tau_h
        if legacy_dt_noise:
            C = C + DT * (dC + noise)
        else:
            C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
    return C.ravel()
