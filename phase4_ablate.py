"""
phase4_ablate.py — which excitability-side variable decorrelates the lattice under drive?

phase4_constrain.py showed that drive decorrelates the lattice even with coupling held fixed at its strongest
(rho-bar 0.046 -> 0.004 over A = 0.01..1.11). Three channels of the drive could do it; each run below holds
coupling fixed (as in `excit`) and switches ONE of them off, plus one run with all three off:
  fix_noise : noise amplitude held at its A = 0.01 value (sigma*(1 + 4*0.01)) instead of sigma*(1 + 4A)
  uni_gamma : drive gamma_i*A replaced by mean(gamma)*A (same mean drive, no spread in intrinsic drive)
  fix_theta : threshold held at THETA_BASE + 0.7*0.01 instead of THETA_BASE + 0.7A
Reading: the switch whose removal stops (or reverses) the decorrelation is the variable responsible.

--- original header of phase4_constrain.py follows ---
phase4_constrain.py — which physical variable stops the model from synchronizing under drive?

Data constraint (Cahill 2024 reanalysis, data_cahill2024_dose.py): strong mGluR drive switches some
slices into strongly synchronized astrocyte activity (rho-bar 0 -> 0.5-0.65), all-or-none across slices.
The ATP model instead decorrelates with drive, and the coupling-only control (phase3_rhobar mode 1) showed
that the ATP-driven fall in D_eff alone produces that decorrelation.

Experiment. Same lattice and channels as phase2_coupling (i0_form 1, I0_BASE), with two switches:
  fix_D  : coupling held at its value for A = A_D (default 0.01, strongest coupling) while drive A moves
           every other channel (gamma*A drive, I0(A), tau_h(A), theta(A), noise(A)): excitability only.
  d_mult : multiplies D_eff, to map rho-bar over (drive, coupling strength).
Prediction to discriminate:
  rho-bar rises with drive at fixed coupling, abruptly above some drive and coupling  -> the model can make
     collective synchrony and the ATP-uncoupling channel is what suppresses it;
  rho-bar stays ~0 at every coupling                                                   -> the missing
     ingredient is not coupling strength (regenerative release, refractoriness, noise structure).


"""
from __future__ import annotations
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
from numba import njit, prange
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result
from phase3_rhobar import rho_exact

ALPHAS = np.linspace(0.01, 1.11, 21)
STRIDE = 10


@njit(fastmath=True, cache=True)
def sweep_c(seed, alpha_values, fix_D, a_D, d_mult, steps, nx, ny, n, sigma, i0_baseline, stride,
            fix_noise, uni_gamma, fix_theta):
    np.random.seed(seed)
    na = alpha_values.shape[0]
    act = np.zeros(na); rho = np.zeros(na); dmean = np.zeros(na)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    gamma_eff = gamma_base.copy()
    if uni_gamma:
        gamma_eff[:] = np.mean(gamma_base)
    t_start = int(0.3 * steps)
    sqrt_dt = DT ** 0.5
    nsamp = (steps - t_start + stride - 1) // stride
    buf = np.zeros((nsamp, n))
    for idx in range(na):
        alpha = alpha_values[idx]
        aD = a_D if fix_D else alpha
        I0 = i0_baseline + alpha * I0_base
        tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
        Deff = d_mult * D0_base / (1.0 + (kappa_base * aD) ** 4)
        theta = THETA_BASE + 0.7 * (a_D if fix_theta else alpha)
        sigma_eff = sigma * (1.0 + 4.0 * (a_D if fix_noise else alpha))
        gdrive = gamma_eff * alpha
        asum = 0.0; cnt = 0; k = 0
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
                asum += np.mean(C_active); cnt += 1
                if (t - t_start) % stride == 0:
                    buf[k, :] = C.reshape(n); k += 1
        act[idx] = asum / cnt
        rho[idx] = rho_exact(buf[:k])
        dmean[idx] = np.mean(Deff)
    return act, rho, dmean


@njit(parallel=True, cache=True)
def ensemble_c(seeds, alpha_values, fix_D, a_D, d_mult, steps, nx, ny, n, sigma, i0_baseline, stride,
               fix_noise, uni_gamma, fix_theta):
    ns = seeds.shape[0]; na = alpha_values.shape[0]
    A = np.zeros((ns, na)); P = np.zeros((ns, na)); D = np.zeros((ns, na))
    for i in prange(ns):
        a, p, d = sweep_c(seeds[i], alpha_values, fix_D, a_D, d_mult, steps, nx, ny, n, sigma,
                          i0_baseline, stride, fix_noise, uni_gamma, fix_theta)
        A[i, :] = a; P[i, :] = p; D[i, :] = d
    return A, P, D


def run(tag, fix_D, a_D, mults, ns, fix_noise=False, uni_gamma=False, fix_theta=False, L=32, T=200.0):
    steps = int(T / DT); seeds = np.arange(11, 11 + ns)
    for m in mults:
        t0 = time.time()
        A, P, D = ensemble_c(seeds, ALPHAS, fix_D, a_D, m, steps, L, L, L * L, SIGMA_EM_PREDICTED,
                             I0_BASE, STRIDE, fix_noise, uni_gamma, fix_theta)
        out = f"ablate_{tag}_L{L}.npz"
        save_result(Path("processed_data") / out,
                    {"L": L, "T": T, "seeds": seeds.tolist(), "fix_D": fix_D, "a_D": a_D, "d_mult": m,
                     "i0_baseline": I0_BASE, "stride": STRIDE, "alphas": ALPHAS.tolist(),
                     "fix_noise": fix_noise, "uni_gamma": uni_gamma, "fix_theta": fix_theta},
                    alphas=ALPHAS, active=A, rhobar=P, Deff=D)
        print(f"{tag} d_mult={m:g} ({time.time()-t0:.0f}s) -> {out}", flush=True)
        for a, r, s, q in zip(ALPHAS[::4], P.mean(0)[::4], P.std(0)[::4], A.mean(0)[::4]):
            print(f"   A={a:.3f}  rho={r:.4f}+-{s:.4f}  active={q:.3f}", flush=True)


if __name__ == "__main__":
    for tag, kw in (("noise", dict(fix_noise=True)), ("gamma", dict(uni_gamma=True)),
                    ("theta", dict(fix_theta=True)),
                    ("all3", dict(fix_noise=True, uni_gamma=True, fix_theta=True))):
        run(tag, True, 0.01, [1.0], 20, **kw)
