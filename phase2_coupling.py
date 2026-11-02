"""
phase2_coupling.py — Phase 2: is the ATP crossover just weaker coupling?

Pre-registered in new_plan.md (2026-08-10). Three sweeps, all sharing the SAME alpha
sequence so they share the D_eff axis automatically (D_eff always follows alpha):

  mode 0  A  (full ATP)        : every channel follows alpha
  mode 1  B  (coupling only)   : channels 1,2,4,5,6 held at alpha_ref; only D_eff follows alpha
  mode 2  B' (both suppressors): as B, but theta (channel 4) ALSO follows alpha

Per-alpha observables: active fraction (population activity), R (Golomb-Rinzel synchrony
= coordination), mean D_eff (the x-axis). Primary test is on active fraction: coupling is
diffusive and generates no spikes, so if activity rises while D_eff falls, it cannot be a
coupling effect. Submitted model (I0 = 1/sqrt(alpha)).

Run: python phase2_coupling.py <L> <alpha_ref> <n_seeds>  (writes processed_data/phase2_*.npz)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from numba import njit, prange
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)
from core.provenance import save_result


@njit(fastmath=True, cache=True)
def sweep_p2(seed, mode, alpha_values, alpha_ref, steps, nx, ny, n, sigma, i0_form=0, theta_ovr=-1.0,
             kappa_scale=1.0, i0_baseline=-1.0):
    np.random.seed(seed)
    na = alpha_values.shape[0]
    act = np.zeros(na)
    Rsync = np.zeros(na)
    dmean = np.zeros(na)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    # I0_base range is variant-dependent; uniform(lo,hi) consumes one draw/cell either way,
    # so the RNG stream position (and every subsequent base array) is identical across variants.
    if i0_form == 1:
        I0_base = np.random.uniform(0.1, 0.5, (nx, ny))    # Bayat's proposed correction
    else:
        I0_base = np.random.uniform(0.01, 0.15, (nx, ny))  # submitted / bounded / const
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny)) * kappa_scale  # scale AFTER draw: RNG stream preserved
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    t_start = int(0.3 * steps)
    sqrt_dt = DT ** 0.5

    for idx in range(na):
        alpha = alpha_values[idx]
        # effective alphas per channel group:
        aD = alpha                                     # D_eff always follows alpha
        aT = alpha if (mode == 0 or mode == 2) else alpha_ref   # theta: A and B' follow alpha
        if theta_ovr >= 0.0:                                     # Phase 3.1 readout test: freeze theta
            aT = theta_ovr
        aO = alpha if mode == 0 else alpha_ref         # other channels: only A follows alpha

        gamma = gamma_base
        # I0 variants (new_plan.md 2026-08-11 robustness table); aO = alpha (A) or alpha_ref (B/B')
        if i0_form == 0:
            I0 = 0.05 + (1.0 / np.sqrt(aO)) * I0_base    # 0 submitted (1/sqrt sign error)
        elif i0_form == 1:
            base = i0_baseline if i0_baseline >= 0.0 else 0.2   # swept baseline excitability (test 2026-08-12)
            I0 = base + aO * I0_base                       # 1 Bayat's proposed (I0_base=U(0.1,0.5))
        elif i0_form == 2:
            I0 = 0.05 + I0_base * (1.0 + np.sqrt(aO))     # 2 bounded, preserves a=0 heterogeneity
        else:
            I0 = 0.05 + I0_base                           # 3 constant, ATP-independent control
        tau_h = 10.0 / ((1.0 + 0.8 * aO) * tau_base)
        Deff = D0_base / (1.0 + (kappa_base * aD) ** 4)
        theta = THETA_BASE + 0.7 * aT
        sigma_eff = sigma * (1.0 + 4.0 * aO)
        gdrive = gamma * aO                             # gamma*alpha excitability term

        asum = 0.0
        msum = 0.0
        msum2 = 0.0
        csum = np.zeros(n)
        csum2 = np.zeros(n)
        cnt = 0
        for t in range(steps):
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
            C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
            diff = Deff * laplacian(C_active)
            dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff
            dh = (C + A_FHN - B_FHN * h) / tau_h
            C = C + DT * dC + sqrt_dt * noise
            h = h + DT * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)
            if t >= t_start:
                asum += np.mean(C_active)
                Cf = C.reshape(n)
                mm = np.mean(Cf)
                csum += Cf
                csum2 += Cf * Cf
                msum += mm
                msum2 += mm * mm
                cnt += 1
        act[idx] = asum / cnt
        var_m = msum2 / cnt - (msum / cnt) ** 2
        var_i = csum2 / cnt - (csum / cnt) ** 2
        mean_var_i = np.mean(var_i)
        r2 = var_m / (mean_var_i + 1e-12)
        Rsync[idx] = np.sqrt(r2) if r2 > 0.0 else 0.0
        dmean[idx] = np.mean(Deff)
    return act, Rsync, dmean


@njit(parallel=True, fastmath=True, cache=True)
def ensemble_p2(seeds, mode, alpha_values, alpha_ref, steps, nx, ny, n, sigma, i0_form=0, theta_ovr=-1.0,
                kappa_scale=1.0, i0_baseline=-1.0):
    ns = seeds.shape[0]
    na = alpha_values.shape[0]
    A = np.zeros((ns, na))
    R = np.zeros((ns, na))
    D = np.zeros((ns, na))
    for i in prange(ns):
        a, r, d = sweep_p2(seeds[i], mode, alpha_values, alpha_ref, steps, nx, ny, n, sigma, i0_form,
                           theta_ovr, kappa_scale, i0_baseline)
        A[i, :] = a
        R[i, :] = r
        D[i, :] = d
    return A, R, D


ALPHAS = np.linspace(0.01, 1.11, 21)


def main():
    L = int(sys.argv[1]); aref = float(sys.argv[2]); ns = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    i0_form = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # I0 variant 0..3 (new_plan robustness table)
    # i0_form 1 carries a baseline excitability. It used to fall through to the sweep_p2 default of
    # 0.2, which was recorded nowhere, so the file could not disclose what it ran at -- and
    # figdata_fig2.py then paired that sweep against a recruited fraction computed at I0_BASE.
    # The baseline is now explicit, defaulted to the canonical constant, and stamped into params.
    i0_baseline = float(sys.argv[5]) if len(sys.argv) > 5 else (I0_BASE if i0_form == 1 else -1.0)
    vtag = {0: "submitted", 1: "bayat", 2: "bounded", 3: "const"}[i0_form]
    pfx = f"phase2v{i0_form}_{vtag}"                          # distinct per variant; never overwrites
    if i0_baseline >= 0.0:
        pfx += f"_b{i0_baseline:.2f}"                         # never clobbers the earlier 0.2 sweeps
    T = 200.0; steps = int(T / DT); seeds = np.arange(11, 11 + ns); sig = SIGMA_EM_PREDICTED
    names = {0: "A_full", 1: "B_coupling", 2: "Bprime_coupθ"}
    for mode in (0, 1, 2):
        if mode == 0 and abs(aref - 0.10) > 1e-9:
            continue    # Sweep A is alpha_ref-independent; run it once (with the 0.10 batch)
        t0 = time.time()
        A, R, D = ensemble_p2(seeds, mode, ALPHAS, aref, steps, L, L, L * L, sig, i0_form,
                              i0_baseline=i0_baseline)
        tag = names[mode] if mode == 0 else f"{names[mode]}_aref{aref:.2f}"
        p = save_result(Path("processed_data") / f"{pfx}_L{L}_{tag}.npz",
                        {"L": L, "mode": mode, "alpha_ref": aref, "n_seeds": ns, "T": T, "i0_form": i0_form,
                         "i0_baseline": i0_baseline, "seeds": seeds.tolist(),
                         "steps": steps, "sigma": sig, "alphas": ALPHAS.tolist()},
                        alphas=ALPHAS, active=A, R=R, Deff=D)
        print(f"L={L} {tag} (i0_baseline={i0_baseline}): active/R/Deff {A.shape} "
              f"({time.time()-t0:.0f}s) -> {p.name}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
