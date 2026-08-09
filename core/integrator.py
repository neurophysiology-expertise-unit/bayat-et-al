"""
core/integrator.py — the one correct stochastic integrator for this project.

WHY THIS FILE EXISTS
====================
The model is a stochastic differential equation (a noise-driven FitzHugh-Nagumo
excitable lattice). Its Methods section names the update rule "Euler-Maruyama", but
the code that was actually run did this (fig_3_criticality_ci.py:153-160):

    noise = sigma_eff * 3.0 * randn(...)
    dC    = C - C**3/3 - h + I0 + gamma*alpha + diff + noise   # noise inside the drift
    C     = C + dt * dC                                        # ... so noise *= dt

That multiplies the noise by `dt`. Euler-Maruyama for  dC = f dt + s dW  requires the
noise to be multiplied by `sqrt(dt)`, because the increment of a Wiener process over a
step of length dt has standard deviation sqrt(dt), not dt. The consequence is not
cosmetic:

  * The variance accumulated over a fixed physical time T becomes  T * s^2 * dt
    instead of  T * s^2.  The effective noise amplitude is therefore  s * sqrt(dt),
    which at the two ends of the Table-1 timestep range (dt = 0.003 .. 0.007) differs
    by more than 50%.
  * The sigma axis of the phase diagram (Fig. 5) is thus not a physical quantity, and
    the manuscript's claim of "robustness with respect to time step" cannot hold under
    the old scheme — a smaller dt is a quieter simulation.

This module provides the correct step and NOTHING ELSE. The old scheme is not kept
behind a flag; a flag is an invitation to reproduce the bug. tests/test_noise_scaling.py
is the gate that must pass before any downstream (Phase 1+) work is trusted.

References: Kloeden & Platen, *Numerical Solution of Stochastic Differential Equations*
(1992), Ch. 9-10; Higham, "An Algorithmic Introduction to Numerical Simulation of SDEs",
SIAM Review 43(3), 2001.
"""

from __future__ import annotations

import numpy as np


def em_step(state, drift, noise_coeff, dt, gaussians):
    """One Euler-Maruyama step for  dX = drift * dt + noise_coeff * dW.

    Parameters
    ----------
    state : ndarray
        Current value X[n] (any shape; the lattice field, for this model).
    drift : ndarray
        The DETERMINISTIC part f(X[n], ...) only. The noise term must NOT be folded
        into this — that is precisely the bug this function exists to prevent.
    noise_coeff : ndarray or float
        The multiplier s in front of dW (for this model, sigma_eff * 3.0).
    dt : float
        Time step.
    gaussians : ndarray
        Standard normals eta ~ N(0, 1), same shape as `state`, drawn by the caller so
        the RNG stream stays explicit and seedable (invariant #3 in new_plan.md).

    Returns
    -------
    ndarray
        X[n+1] = X[n] + drift * dt + noise_coeff * sqrt(dt) * eta.
    """
    return state + drift * dt + noise_coeff * np.sqrt(dt) * gaussians


def integrate_pure_noise(sigma, dt, T, n_real, rng):
    """Reference integrator for the pure Wiener process  dC = sigma dW,  C(0) = 0.

    With f = 0 and no coupling the SDE has the exact law  C(T) ~ N(0, T * sigma^2),
    independent of dt. This is the property tests/test_noise_scaling.py checks, and it
    is the cleanest possible probe of the noise scaling because nothing else moves.

    Memory-flat: it never materialises the full (n_real, n_steps) path, so it is safe
    to call at dt = 1e-4 with hundreds of realizations.

    Parameters
    ----------
    sigma : float
        Noise amplitude s.
    dt : float
        Time step.
    T : float
        Physical integration time. n_steps = round(T / dt).
    n_real : int
        Number of independent realizations.
    rng : np.random.Generator
        Explicit generator (e.g. np.random.default_rng(seed)).

    Returns
    -------
    ndarray, shape (n_real,)
        The terminal values C(T) for each realization.
    """
    n_steps = int(round(T / dt))
    scale = sigma * np.sqrt(dt)
    C = np.zeros(n_real)
    for _ in range(n_steps):
        C += scale * rng.standard_normal(n_real)
    return C


def _integrate_pure_noise_WRONG(sigma, dt, T, n_real, rng):
    """The OLD (incorrect) scaling, kept ONLY so the test can demonstrate the failure
    it guards against. Never import this into production code — it multiplies the noise
    by dt instead of sqrt(dt), giving a dt-dependent stationary variance."""
    n_steps = int(round(T / dt))
    scale = sigma * dt                      # <-- the bug: dt, not sqrt(dt)
    C = np.zeros(n_real)
    for _ in range(n_steps):
        C += scale * rng.standard_normal(n_real)
    return C
