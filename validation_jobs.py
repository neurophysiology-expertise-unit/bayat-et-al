"""Targeted validation jobs for the ATP manuscript review (2026-08-30).

Job A measures spontaneous nucleation across the existing decremental-release
gain sweep.  Job B repeats the deterministic focal headline result at dt/2 and
under an alternative initial-condition draw.  Each seed/condition is saved as
soon as it finishes through core.provenance.save_result; combine commands refuse
dirty, mixed-commit, or parameter-inconsistent inputs.

Usage:
  python validation_jobs.py nucleation-seed <seed>
  python validation_jobs.py nucleation-combine
  python validation_jobs.py focal-seed <seed>
  python validation_jobs.py focal-combine
"""

import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

from core.model import (A_FHN, B_FHN, DT, ETA, I0_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, THETA_BASE, laplacian)
from core.provenance import git_commit, load_result, save_result


SEEDS = tuple(range(11, 21))
GAMMAS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.20, 0.15, 0.10, 0.05)
TAU_REF = 15.0                  # matches the decremental sweep in Fig. 4
ALPHA = 0.01
L = 64
PATCH = 2
C_DOWN = -1.2
K_REF = 20.0
GAIN_SCALE = 0.15
GAIN_FLOOR = 0.02
UM_PRIMARY = 25.0
UM_ALTERNATIVE = 50.0
OUT = Path("processed_data")


def gamma_tag(gamma):
    return f"{gamma:.2f}".replace(".", "p")


def require_clean_commit():
    commit = git_commit()
    if commit == "unknown" or commit.endswith("-dirty"):
        raise RuntimeError(f"validation requires a clean committed tree, got {commit}")
    return commit


@njit(fastmath=True, cache=False)
def decremental_nucleation_count(seed, gamma_regen, steps, stride):
    """Unprovoked noisy run; count the same sampled-frame events as nucleation_events."""
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (L, L))
                  * (1.0 + 2.0 * np.random.standard_normal((L, L))))
    i0_slope = np.random.uniform(0.1, 0.5, (L, L))
    tau_base = np.random.uniform(0.5, 1.1, (L, L))
    d0_base = np.random.uniform(0.05, 0.5, (L, L))
    kappa_base = np.random.uniform(1.0, 4.0, (L, L))

    i0 = I0_BASE + ALPHA * i0_slope
    tau_h = 10.0 / ((1.0 + 0.8 * ALPHA) * tau_base)
    deff = d0_base / (1.0 + (kappa_base * ALPHA) ** 4)
    theta = THETA_BASE + 0.7 * ALPHA
    sigma_eff = SIGMA_EM_PREDICTED * (1.0 + 4.0 * ALPHA)
    gdrive = gamma_base * ALPHA

    # Same unprovoked initial-condition distribution as run_ref(..., do_focal=False).
    C = np.random.uniform(-0.1, 0.3, (L, L))
    h = np.random.uniform(0.4, 1.2, (L, L))
    gain = np.zeros((L, L))
    reft = np.zeros((L, L))
    prev_step = np.zeros((L, L), dtype=np.bool_)
    prev_sample = np.zeros((L, L), dtype=np.bool_)
    have_sample = False
    events = 0
    sqrt_dt = np.sqrt(DT)

    for t in range(steps):
        c_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        active = c_active > 0.5
        inref = reft > 0.0
        broadcast = c_active * gain * (1.0 - inref)
        coupling_in = deff * laplacian(broadcast)
        rose = active & ~prev_step
        fell = prev_step & ~active
        gain_new = np.minimum(1.0, GAIN_FLOOR + gamma_regen * coupling_in / GAIN_SCALE)
        gain = np.where(rose, np.maximum(gain_new, 0.0), gain)
        reft = np.where(fell, TAU_REF, reft)
        clamp = K_REF * inref * (C - C_DOWN)
        noise = sigma_eff * NOISE_MULT * np.random.standard_normal((L, L))
        dC = C - C ** 3 / 3.0 - h + i0 + gdrive + coupling_in - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - DT, 0.0)
        prev_step = active

        if t % stride == 0:
            sampled = 0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5
            if have_sample:
                for i in range(L):
                    for j in range(L):
                        if sampled[i, j] and not prev_sample[i, j]:
                            neighbours = (prev_sample[(i - 1) % L, j] or
                                          prev_sample[(i + 1) % L, j] or
                                          prev_sample[i, (j - 1) % L] or
                                          prev_sample[i, (j + 1) % L])
                            if not neighbours:
                                events += 1
            prev_sample = sampled
            have_sample = True
    return events


@njit(fastmath=True, cache=False)
def focal_activation_times(seed, dt, stride, random_initial, init_seed, T):
    """Deterministic focal protocol with dt passed explicitly (never a reassigned global)."""
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (L, L))
                  * (1.0 + 2.0 * np.random.standard_normal((L, L))))
    i0_slope = np.random.uniform(0.1, 0.5, (L, L))
    tau_base = np.random.uniform(0.5, 1.1, (L, L))
    d0_base = np.random.uniform(0.05, 0.5, (L, L))
    kappa_base = np.random.uniform(1.0, 4.0, (L, L))

    i0 = I0_BASE + ALPHA * i0_slope
    tau_h = 10.0 / ((1.0 + 0.8 * ALPHA) * tau_base)
    deff = d0_base / (1.0 + (kappa_base * ALPHA) ** 4)
    theta = THETA_BASE + 0.7 * ALPHA
    gdrive = gamma_base * ALPHA

    if random_initial:
        # Separate RNG stream: perturb initial state without changing the disorder draw.
        np.random.seed(init_seed)
        C = np.random.uniform(-0.1, 0.3, (L, L))
        h = np.random.uniform(0.4, 1.2, (L, L))
    else:
        C = np.full((L, L), C_DOWN)
        h = (C + A_FHN) / B_FHN

    c = L // 2
    for i in range(c - PATCH, c + PATCH + 1):
        for j in range(c - PATCH, c + PATCH + 1):
            C[i, j] = 1.5

    tact = np.full((L, L), -1, dtype=np.int64)
    frame = 0
    steps = int(T / dt)
    for t in range(steps):
        c_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        diff = deff * laplacian(c_active)
        dC = C - C ** 3 / 3.0 - h + i0 + gdrive + diff
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + dt * dC
        h = h + dt * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        if t % stride == 0:
            sampled = 0.5 * (1.0 + np.tanh(ETA * (C - theta))) > 0.5
            tact = np.where((tact < 0) & sampled, frame, tact)
            frame += 1
    return tact


def focal_metrics(tact, dt_frame):
    c = L // 2
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c))
    dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    radius = np.sqrt(dx ** 2 + dy ** 2)
    reached = (tact >= 0) & (radius > PATCH)
    extent = float(radius[reached].max()) if reached.any() else 0.0
    fraction = float((tact >= 0).mean())
    rr = radius[reached]
    tt = tact[reached].astype(float) * dt_frame
    band = (rr >= 2) & (rr <= L // 2 - 2)
    speed50 = np.nan
    r2 = np.nan
    if band.sum() > 20 and np.ptp(tt[band]) > 1e-9:
        slope, intercept = np.polyfit(rr[band], tt[band], 1)
        fitted = slope * rr[band] + intercept
        r2 = 1.0 - np.sum((tt[band] - fitted) ** 2) / np.sum((tt[band] - tt[band].mean()) ** 2)
        if slope > 1e-9:
            speed50 = UM_ALTERNATIVE / slope
    return extent, fraction, speed50, r2


def nucleation_seed(seed):
    commit = require_clean_commit()
    T = 300.0
    stride = 5
    steps = int(T / DT)
    for gamma in GAMMAS:
        events = int(decremental_nucleation_count(seed, gamma, steps, stride))
        rate = events / (L * L) * 1000.0 / T
        path = OUT / f"validation_nucleation_seed{seed}_g{gamma_tag(gamma)}.npz"
        save_result(path,
                    {"job": "gamma_regen_vs_nucleation", "seed": seed,
                     "gamma_regen": gamma, "tau_ref": TAU_REF, "alpha": ALPHA,
                     "baseline": I0_BASE, "L": L, "T": T, "dt": DT,
                     "steps": steps, "stride": stride, "dt_frame": stride * DT,
                     "sigma": SIGMA_EM_PREDICTED, "noise_on": True,
                     "protocol": "unprovoked; phase3_nucleation sampled-frame definition",
                     "expected_commit": commit},
                    event_count=np.array(events), nucleation_rate=np.array(rate))
        print(f"seed={seed} gamma={gamma:.2f} events={events} rate={rate:.6f} wrote={path}",
              flush=True)


def focal_seed(seed):
    commit = require_clean_commit()
    T = 100.0
    conditions = (
        ("reference", DT, 5, False, -1),
        ("dt_half", DT / 2.0, 10, False, -1),
        ("random_initial", DT, 5, True, seed + 100000),
    )
    for name, dt, stride, random_initial, init_seed in conditions:
        tact = focal_activation_times(seed, dt, stride, random_initial, init_seed, T)
        extent, fraction, speed50, r2 = focal_metrics(tact, stride * dt)
        path = OUT / f"validation_focal_seed{seed}_{name}.npz"
        save_result(path,
                    {"job": "focal_sensitivity", "condition": name, "seed": seed,
                     "init_seed": init_seed if random_initial else None,
                     "initial_condition": ("C~U(-0.1,0.3), h~U(0.4,1.2)"
                                           if random_initial else "uniform down state"),
                     "alpha": ALPHA, "baseline": I0_BASE, "L": L, "T": T,
                     "dt": dt, "steps": int(T / dt), "stride": stride,
                     "dt_frame": stride * dt, "noise_on": False, "patch": PATCH,
                     "um_per_cell_primary": UM_PRIMARY,
                     "um_per_cell_alternative": UM_ALTERNATIVE,
                     "expected_commit": commit},
                    extent=np.array(extent), activated_fraction=np.array(fraction),
                    speed25=np.array(speed50 / 2.0), speed50=np.array(speed50),
                    r2=np.array(r2))
        print(f"seed={seed} condition={name} extent={extent:.6f} "
              f"speed25={speed50/2:.6f} fraction={fraction:.6f} r2={r2:.6f} wrote={path}",
              flush=True)


def load_checked(path, expected_commit, expected_job):
    arrays, params, commit = load_result(path)
    if commit != expected_commit or commit.endswith("-dirty"):
        raise RuntimeError(f"{path}: commit {commit}, expected clean {expected_commit}")
    if params.get("job") != expected_job:
        raise RuntimeError(f"{path}: wrong job {params.get('job')}")
    if params.get("expected_commit") != expected_commit:
        raise RuntimeError(f"{path}: parameter-stamped commit mismatch")
    return arrays, params


def nucleation_combine():
    commit = require_clean_commit()
    all_rates = np.empty((len(SEEDS), len(GAMMAS)))
    all_counts = np.empty_like(all_rates, dtype=int)
    for si, seed in enumerate(SEEDS):
        for gi, gamma in enumerate(GAMMAS):
            path = OUT / f"validation_nucleation_seed{seed}_g{gamma_tag(gamma)}.npz"
            arrays, params = load_checked(path, commit, "gamma_regen_vs_nucleation")
            if params["seed"] != seed or not np.isclose(params["gamma_regen"], gamma):
                raise RuntimeError(f"{path}: seed/gamma mismatch")
            all_rates[si, gi] = float(arrays["nucleation_rate"])
            all_counts[si, gi] = int(arrays["event_count"])
    mean = all_rates.mean(0)
    sd = all_rates.std(0)
    path = save_result(OUT / "validation_gamma_regen_nucleation_ens.npz",
                       {"job": "gamma_regen_vs_nucleation_ensemble", "seeds": list(SEEDS),
                        "gammas": list(GAMMAS), "tau_ref": TAU_REF, "alpha": ALPHA,
                        "baseline": I0_BASE, "L": L, "T": 300.0, "dt": DT,
                        "stride": 5, "sigma": SIGMA_EM_PREDICTED,
                        "aggregation": "mean and population SD over seeds",
                        "source_commit": commit},
                       seeds=np.array(SEEDS), gammas=np.array(GAMMAS),
                       nucleation_rate_all=all_rates, event_count_all=all_counts,
                       nucleation_rate_mean=mean, nucleation_rate_sd=sd)
    for gamma, m, s in zip(GAMMAS, mean, sd):
        print(f"gamma={gamma:.2f} nucleation={m:.6f}+/-{s:.6f}")
    print(f"wrote={path}")


def focal_combine():
    commit = require_clean_commit()
    names = ("reference", "dt_half", "random_initial")
    keys = ("extent", "activated_fraction", "speed25", "speed50", "r2")
    stacked = {key: np.empty((len(names), len(SEEDS))) for key in keys}
    for ci, name in enumerate(names):
        for si, seed in enumerate(SEEDS):
            path = OUT / f"validation_focal_seed{seed}_{name}.npz"
            arrays, params = load_checked(path, commit, "focal_sensitivity")
            if params["seed"] != seed or params["condition"] != name:
                raise RuntimeError(f"{path}: seed/condition mismatch")
            for key in keys:
                stacked[key][ci, si] = float(arrays[key])
    output = {}
    for key, values in stacked.items():
        output[f"{key}_all"] = values
        output[f"{key}_mean"] = values.mean(1)
        output[f"{key}_sd"] = values.std(1)
    path = save_result(OUT / "validation_focal_sensitivity_ens.npz",
                       {"job": "focal_sensitivity_ensemble", "seeds": list(SEEDS),
                        "conditions": list(names), "alpha": ALPHA, "baseline": I0_BASE,
                        "L": L, "T": 100.0, "dt_reference": DT,
                        "dt_half": DT / 2.0, "dt_frame_all": 5 * DT,
                        "alternative_initial_condition": "C~U(-0.1,0.3), h~U(0.4,1.2)",
                        "aggregation": "mean and population SD over seeds",
                        "source_commit": commit},
                       seeds=np.array(SEEDS), conditions=np.array(names), **output)
    for ci, name in enumerate(names):
        print(f"{name}: extent={output['extent_mean'][ci]:.6f}+/-{output['extent_sd'][ci]:.6f} "
              f"speed25={output['speed25_mean'][ci]:.6f}+/-{output['speed25_sd'][ci]:.6f} "
              f"fraction={output['activated_fraction_mean'][ci]:.6f}"
              f"+/-{output['activated_fraction_sd'][ci]:.6f} "
              f"r2={output['r2_mean'][ci]:.6f}+/-{output['r2_sd'][ci]:.6f}")
    print(f"wrote={path}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    command = sys.argv[1]
    if command == "nucleation-seed" and len(sys.argv) == 3:
        nucleation_seed(int(sys.argv[2]))
    elif command == "nucleation-combine" and len(sys.argv) == 2:
        nucleation_combine()
    elif command == "focal-seed" and len(sys.argv) == 3:
        focal_seed(int(sys.argv[2]))
    elif command == "focal-combine" and len(sys.argv) == 2:
        focal_combine()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
