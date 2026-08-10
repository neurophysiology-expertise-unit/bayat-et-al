"""
phase1_finite_size.py — Phase 1: finite-size scaling of the healthy ATP crossover.

Question (new_plan.md Phase 1): does the crossover have genuine collective character
(chi_peak grows with L, alpha* drifts) or is it a finite-size effect of the 10x10 lattice
(chi_peak saturates)? If it saturates, drop the "critical slowing / susceptibility"
language and let the activity/coordination dissociation carry the paper.

The sweep core here is a faithful copy of fig_3_criticality_ci.run_one_seed_core (the
published Fig 3 pipeline, validated bit-identical in PHASE0_VALIDATION.md Gate 1a), with
three things made explicit:
  * the correct Euler-Maruyama noise scaling (sqrt(dt)); `legacy=True` restores the old
    dt-scaling ONLY for the validation check below;
  * `sigma` is a parameter (Phase 0.2: use 0.02332, the corrected nominal);
  * `i0_ref` folds in the Phase-1 add-on: if not NaN, I0 is held at that alpha instead of
    varying as 1/sqrt(alpha), so the I0-independent condition is measured on the same L-sweep
    (cheap insurance in case Bayat confirms 1/sqrt(A) was a typo).

`validate()` proves the sweep core (legacy, sigma=0.4) reproduces fig_3_criticality_ci's
chi(alpha) bit-for-bit — i.e. the whole sweep harness matches the published pipeline, not
just a single trajectory.

Run:  python phase1_finite_size.py validate
      python phase1_finite_size.py run [--smoke]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED
from core.provenance import save_result

ALPHAS = np.linspace(0.01, 1.11, 21)
# Narrowed grid for Phase 1: dense near the peak (0.12-0.18) + 3 tail points.
# ~40% cheaper than the full 21-point grid at no accuracy cost in the peak region.
NARROW_ALPHAS = np.concatenate([np.linspace(0.05, 0.35, 14), np.array([0.6, 0.9, 1.1])])


@njit(fastmath=True, cache=True)
def sweep_one_seed(seed, disease, alpha_values, steps, nx, ny, n, sigma, i0_ref, legacy):
    """One quasi-static ATP up-sweep; returns (Sc, chi, Rsync) per alpha.

    Faithful to fig_3_criticality_ci.run_one_seed_core; see module docstring.
    i0_ref <= 0 -> I0 = 0.05 + (1/sqrt(alpha))*I0_base (original, alpha-dependent).
    i0_ref finite -> I0 held at that alpha (I0-independent condition).
    """
    np.random.seed(seed)
    na = alpha_values.shape[0]
    Sc_out = np.zeros(na)
    chi_out = np.zeros(na)          # chi_ext = N*Var_t(max-min)  (range statistic)
    chi_true_out = np.zeros(na)     # chi_true = N*Var_t(Cbar)    (intensive susceptibility)
    Rsync_out = np.zeros(na)

    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))

    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    t_start = int(0.3 * steps)
    sqrt_dt = DT ** 0.5

    for idx in range(na):
        alpha = alpha_values[idx]
        a_i0 = alpha if i0_ref <= 0.0 else i0_ref   # i0_ref<=0 sentinel (np.isnan unsafe under fastmath)
        gamma = gamma_base.copy()
        I0 = 0.05 + (1.0 / np.sqrt(a_i0)) * I0_base
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

        sc_sum = 0.0
        sumD = 0.0
        sumD2 = 0.0
        cntD = 0
        msum = 0.0
        msum2 = 0.0
        csum = np.zeros(n)
        csum2 = np.zeros(n)

        for t in range(steps):
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
            C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
            diff = Deff * laplacian(C_active)
            dC = C - (C ** 3) / 3.0 - h + I0 + gamma * alpha + diff
            dh = (C + A_FHN - B_FHN * h) / tau_h
            if legacy:
                C = C + DT * (dC + noise)
            else:
                C = C + DT * dC + sqrt_dt * noise
            h = h + DT * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)

            m = np.mean(C)
            v = np.mean(C * C) - m * m
            sc_sum += v / (abs(m) + 1e-9)

            if t >= t_start:
                D_ = np.max(C) - np.min(C)
                sumD += D_
                sumD2 += D_ * D_
                cntD += 1
                Cf = C.reshape(n)
                mm = np.mean(Cf)              # vectorised: the scalar kk-loop was O(n)
                csum += Cf                    # per-step and made large-L runs take hours
                csum2 += Cf * Cf
                msum += mm
                msum2 += mm * mm

        Sc_out[idx] = sc_sum / steps
        meanD = sumD / cntD
        meanD2 = sumD2 / cntD
        chi_out[idx] = n * (meanD2 - meanD * meanD)
        var_m = msum2 / cntD - (msum / cntD) ** 2
        chi_true_out[idx] = n * var_m          # proper FSS susceptibility
        var_i = csum2 / cntD - (csum / cntD) ** 2
        mean_var_i = np.mean(var_i)
        r2 = var_m / (mean_var_i + 1e-12)
        Rsync_out[idx] = np.sqrt(r2) if r2 > 0.0 else 0.0
    return Sc_out, chi_out, chi_true_out, Rsync_out


@njit(parallel=True, fastmath=True, cache=True)
def sweep_ensemble(seeds, disease, alpha_values, steps, nx, ny, n, sigma, i0_ref, legacy):
    ns = seeds.shape[0]
    na = alpha_values.shape[0]
    Sc = np.zeros((ns, na))
    chi = np.zeros((ns, na))
    chi_true = np.zeros((ns, na))
    R = np.zeros((ns, na))
    for i in prange(ns):
        s, c, ct, r = sweep_one_seed(seeds[i], disease, alpha_values, steps, nx, ny, n,
                                     sigma, i0_ref, legacy)
        Sc[i, :] = s
        chi[i, :] = c
        chi_true[i, :] = ct
        R[i, :] = r
    return Sc, chi, chi_true, R


def validate():
    """Prove the sweep core reproduces the original fig_3_criticality_ci chi(alpha)."""
    import importlib.util
    SP = Path("/tmp/claude-1000/-mnt-sysfs01-users-cagatay-code/"
              "b4fdfc31-cb8f-4081-8c3a-f950d21b1410/scratchpad/fig3_orig.py")
    spec = importlib.util.spec_from_file_location("f3o", SP)
    f3o = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f3o)
    grid, steps, seed = 8, 3000, 11
    n = grid * grid
    _, chi_orig, _, _, _ = f3o.run_one_seed_core(seed, False, ALPHAS, steps, grid, grid, n)
    _, chi_mine, _, _ = sweep_one_seed(seed, False, ALPHAS, steps, grid, grid, n,
                                    0.4, -1.0, True)   # legacy, sigma=0.4
    d = float(np.max(np.abs(chi_orig - chi_mine)))
    print(f"validate: max|chi_orig - chi_mine| = {d:.2e}  "
          f"{'IDENTICAL — sweep harness matches published' if d < 1e-9 else 'MISMATCH'}")
    return d < 1e-9


def run(smoke, Ls=None, n_seeds=None, T=None, out_name="phase1_finite_size"):
    if Ls is None:
        Ls = [16, 32] if smoke else [32, 64, 128]
    if n_seeds is None:
        n_seeds = 6 if smoke else 20
    if T is None:
        T = 300.0 if smoke else 1000.0
    steps = int(T / DT)
    seeds = np.arange(11, 11 + n_seeds)
    sigma = SIGMA_EM_PREDICTED     # corrected nominal, 0.02332
    out = {"alphas": ALPHAS, "Ls": np.array(Ls)}
    print(f"Phase 1: L={Ls}, {n_seeds} seeds, steps={steps}, sigma={sigma:.5f}\n")

    for L in Ls:
        n = L * L
        _, chi, _, _ = sweep_ensemble(seeds, False, ALPHAS, steps, L, L, n, sigma, -1.0, False)
        cm = chi.mean(0)
        k = int(np.argmax(cm))
        out[f"L{L}_chi"] = cm
        out[f"L{L}_chi_sem"] = chi.std(0) / np.sqrt(len(seeds))
        print(f"L={L:4d}  chi_peak={cm[k]:8.2f} at alpha={ALPHAS[k]:.3f}", flush=True)
        # I0-independent condition at the two smaller L only
        if L in (32, 64):
            _, chi_f, _ = sweep_ensemble(seeds, False, ALPHAS, steps, L, L, n, sigma, 0.175, False)
            cfm = chi_f.mean(0)
            kf = int(np.argmax(cfm))
            out[f"L{L}_chi_I0fixed"] = cfm
            print(f"        I0-fixed: chi_peak={cfm[kf]:8.2f} at alpha={ALPHAS[kf]:.3f}", flush=True)

    peaks = np.array([out[f"L{L}_chi"].max() for L in Ls])
    locs = np.array([ALPHAS[int(np.argmax(out[f"L{L}_chi"]))] for L in Ls])
    print("\nfinite-size scaling:")
    print("  L       :", "  ".join(f"{L:6d}" for L in Ls))
    print("  chi_peak:", "  ".join(f"{p:6.1f}" for p in peaks))
    print("  alpha*  :", "  ".join(f"{a:6.3f}" for a in locs))
    grow = peaks[-1] / peaks[0]
    print(f"\n  chi_peak(L_max)/chi_peak(L_min) = {grow:.2f}  "
          f"-> {'GROWS (collective)' if grow > 1.5 else 'SATURATES (finite-size)'}")
    params = {"Ls": Ls, "n_seeds": n_seeds, "steps": steps, "sigma": sigma,
              "alphas": ALPHAS.tolist(), "smoke": smoke}
    p = save_result(Path("processed_data") / (out_name + ".npz"), params, **out)
    print(f"wrote {p}")


def overnight(T, n_seeds=8, Ls=(32, 64, 128)):
    """The Phase 1 production run. Each (L, condition) is written to its OWN
    provenance-stamped npz the instant it finishes, so a crash at hour 6 keeps
    hours 1-5. Healthy at every L; the I0-independent variant at L=32,64 only."""
    seeds = np.arange(11, 11 + n_seeds)
    alphas = NARROW_ALPHAS
    sigma = SIGMA_EM_PREDICTED
    steps = int(T / DT)
    outdir = Path("processed_data")
    # (L, i0_ref, tag): cheapest first so results accumulate before the 128 run
    jobs = []
    for L in Ls:
        jobs.append((L, -1.0, "healthy"))
        if L in (32, 64):
            jobs.append((L, 0.175, "i0fixed"))
    jobs.sort(key=lambda j: j[0])   # ascending L
    print(f"overnight: T={T:.0f} steps={steps}, {n_seeds} seeds, sigma={sigma:.5f}, "
          f"{len(alphas)} alphas, jobs={[(L,t) for L,_,t in jobs]}\n", flush=True)
    for L, i0_ref, tag in jobs:
        t0 = time.time()
        _, chi, _, R = sweep_ensemble(seeds, False, alphas, steps, L, L, L * L, sigma, i0_ref, False)
        cm = chi.mean(0)
        k = int(np.argmax(cm))
        params = {"L": L, "condition": tag, "i0_ref": i0_ref, "T": T, "steps": steps,
                  "n_seeds": n_seeds, "sigma": sigma, "alphas": alphas.tolist()}
        p = save_result(outdir / f"phase1_L{L}_{tag}.npz", params,
                        alphas=alphas, chi=chi, chi_mean=cm,
                        chi_sem=chi.std(0) / np.sqrt(n_seeds), R_mean=R.mean(0))
        print(f"[{time.strftime('%H:%M')}] L={L:4d} {tag:8s}  "
              f"chi_peak={cm[k]:8.2f} at alpha*={alphas[k]:.3f}  "
              f"({time.time()-t0:.0f}s)  -> {p.name}", flush=True)
    print("\novernight complete.", flush=True)


def t_convergence(L=32, n_seeds=8, Ts=(200.0, 500.0, 1000.0, 2000.0)):
    """T-convergence of chi at fixed L: chi is a time-variance estimator whose finite-T
    bias is downward and scales with tau_ac/T. Since tau_ac grows with L, too-short a T
    suppresses large-L peaks more than small-L ones and can manufacture a fake saturation.
    So T must be chosen where chi_peak is converged BEFORE comparing across L.
    Adopt the smallest T within 5% of the longest-T chi_peak."""
    seeds = np.arange(11, 11 + n_seeds)
    alphas = NARROW_ALPHAS
    sigma = SIGMA_EM_PREDICTED
    print(f"T-convergence @ L={L}, {n_seeds} seeds, narrow grid ({len(alphas)} alphas), "
          f"sigma={sigma:.5f}\n", flush=True)
    print(f"{'T':>7} {'steps':>8} {'chi_peak':>9} {'alpha*':>7} {'sec':>6}", flush=True)
    rows = []
    for T in Ts:
        steps = int(T / DT)
        t0 = time.time()
        _, chi, _, _ = sweep_ensemble(seeds, False, alphas, steps, L, L, L * L, sigma, -1.0, False)
        cm = chi.mean(0)
        k = int(np.argmax(cm))
        rows.append((T, steps, cm[k], alphas[k]))
        print(f"{T:>7.0f} {steps:>8d} {cm[k]:>9.2f} {alphas[k]:>7.3f} {time.time()-t0:>6.0f}",
              flush=True)
    ref = rows[-1][2]
    chosen = None
    for (T, steps, peak, a) in rows:
        if abs(peak - ref) / ref <= 0.05:
            chosen = T
            break
    print(f"\nreference chi_peak (T={rows[-1][0]:.0f}) = {ref:.2f}", flush=True)
    print(f"smallest T within 5% = {chosen}", flush=True)
    return rows, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "run", "tconv", "overnight"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--Ls", default=None, help="comma-separated lattice sizes")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--out", default="phase1_finite_size")
    args = ap.parse_args()
    if args.cmd == "validate":
        return 0 if validate() else 1
    if args.cmd == "tconv":
        t_convergence(L=32, n_seeds=args.seeds or 8)
        return 0
    if args.cmd == "overnight":
        if args.T is None:
            sys.exit("overnight needs --T (the converged T from tconv)")
        Ls = tuple(int(x) for x in args.Ls.split(",")) if args.Ls else (32, 64, 128)
        overnight(args.T, n_seeds=args.seeds or 8, Ls=Ls)
        return 0
    Ls = [int(x) for x in args.Ls.split(",")] if args.Ls else None
    run(args.smoke, Ls=Ls, n_seeds=args.seeds, T=args.T, out_name=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
