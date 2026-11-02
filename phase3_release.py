"""Released-ATP demonstration (new_plan 2026-08-12, framing bound LOCKED before numbers).
On top of the corrected const-I0 model, add an activity-released, diffusing ATP field and test
whether spatial coordination recovers. DRIVE-ONLY: alpha_eff feeds only the excitability drive
gamma*(alpha_base + A_local); all other channels stay at alpha_base (so recovered coordination
cannot come from locally-modulated D_eff/theta).

  A_local field (mode 0, DIFFUSING): dA/dt = D_atp*lap(A) - A/tau_atp + s*C_active ; drive uses A(x,y)
  A_scalar     (mode 1, UNIFORM NULL): dA/dt = -A/tau_atp + s*<C_active>          ; drive uses <A>
The uniform null injects the SAME KIND and rate of released ATP (source s*<C_active>, same decay)
but with NO spatial gradients. Attribution to spatial structure requires: diffusing recovers rho-bar,
uniform does NOT. s=0 must reduce exactly to the corrected model.

Quasi-static s-ramp (mirrors the paper's alpha up-sweep); alpha_base fixed. Draw order matches
phase2_coupling.sweep_p2 const path so s=0 is bit-identical to it at alpha=alpha_base.
Run: python phase3_release.py            (grid + null + s=0 check)
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit, prange
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path


@njit(fastmath=True, cache=False)
def sweep_release(seed, s_values, alpha_base, D_atp, tau_atp, mode, steps, nx, ny, n, sigma):
    np.random.seed(seed)
    ns = s_values.shape[0]
    act = np.zeros(ns); Rs = np.zeros(ns); Aused = np.zeros(ns)
    # draw order identical to sweep_p2 const path
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.01, 0.15, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    # alpha_base is FIXED, so every non-drive channel is constant across the whole run:
    I0 = 0.05 + I0_base                               # const-I0
    tau_h = 10.0 / ((1.0 + 0.8 * alpha_base) * tau_base)
    Deff = D0_base / (1.0 + (kappa_base * alpha_base) ** 4)
    theta = THETA_BASE + 0.7 * alpha_base
    sigma_eff = sigma * (1.0 + 4.0 * alpha_base)
    A_local = np.zeros((nx, ny)); A_scalar = 0.0      # released-ATP state (carries across s-ramp)
    t_start = int(0.3 * steps); sqrt_dt = DT ** 0.5
    for idx in range(ns):
        s = s_values[idx]
        asum = 0.0; msum = 0.0; msum2 = 0.0; csum = np.zeros(n); csum2 = np.zeros(n)
        ause = 0.0; cnt = 0
        for t in range(steps):
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
            C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
            # released-ATP update
            if mode == 0:
                A_local = A_local + DT * (D_atp * laplacian(A_local) - A_local / tau_atp + s * C_active)
                A_local = np.maximum(A_local, 0.0)
                gdrive = gamma_base * (alpha_base + A_local)
                a_mean = np.mean(A_local)
            else:
                A_scalar = A_scalar + DT * (-A_scalar / tau_atp + s * np.mean(C_active))
                if A_scalar < 0.0:
                    A_scalar = 0.0
                gdrive = gamma_base * (alpha_base + A_scalar)
                a_mean = A_scalar
            diff = Deff * laplacian(C_active)
            dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff
            dh = (C + A_FHN - B_FHN * h) / tau_h
            C = C + DT * dC + sqrt_dt * noise
            h = h + DT * dh
            C = np.minimum(np.maximum(C, -4.0), 4.0)
            if t >= t_start:
                asum += np.mean(C_active)
                Cf = C.reshape(n); mm = np.mean(Cf)
                csum += Cf; csum2 += Cf * Cf; msum += mm; msum2 += mm * mm
                ause += a_mean; cnt += 1
        act[idx] = asum / cnt
        var_m = msum2 / cnt - (msum / cnt) ** 2
        mean_var_i = np.mean(csum2 / cnt - (csum / cnt) ** 2)
        r2 = var_m / (mean_var_i + 1e-12)
        Rs[idx] = np.sqrt(r2) if r2 > 0.0 else 0.0
        Aused[idx] = ause / cnt
    return act, Rs, Aused


@njit(parallel=True, fastmath=True, cache=False)
def ensemble_release(seeds, s_values, alpha_base, D_atp, tau_atp, mode, steps, nx, ny, n, sigma):
    nseed = seeds.shape[0]; ns = s_values.shape[0]
    A = np.zeros((nseed, ns)); R = np.zeros((nseed, ns)); AU = np.zeros((nseed, ns))
    for i in prange(nseed):
        a, r, au = sweep_release(seeds[i], s_values, alpha_base, D_atp, tau_atp, mode,
                                 steps, nx, ny, n, sigma)
        A[i] = a; R[i] = r; AU[i] = au
    return A, R, AU


def main():
    L = 32; ns_seed = 40; T = 200.0; steps = int(T / DT); seeds = np.arange(11, 11 + ns_seed)
    sig = SIGMA_EM_PREDICTED; N = L * L
    tau_atp = 5.0
    S = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6])
    Datps = [0.1, 0.5, 2.0]
    outdir = Path("processed_data")
    for alpha_base in (0.30, 0.50):
        ab = f"a{alpha_base:.2f}"
        for D_atp in Datps:                            # diffusing grid
            t0 = time.time()
            A, R, AU = ensemble_release(seeds, S, alpha_base, D_atp, tau_atp, 0, steps, L, L, N, sig)
            save_result(outdir / f"phase3_release_{ab}_diff_D{D_atp}.npz",
                        {"L": L, "alpha_base": alpha_base, "tau_atp": tau_atp, "D_atp": D_atp, "mode": 0,
                         "n_seeds": ns_seed, "T": T, "s_values": S.tolist()}, s_values=S, active=A, R=R, Aused=AU)
            print(f"{ab} diffusing D_atp={D_atp}: active {A.mean(0).round(3)}  ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()                               # uniform null (D_atp irrelevant)
        A, R, AU = ensemble_release(seeds, S, alpha_base, 0.0, tau_atp, 1, steps, L, L, N, sig)
        save_result(outdir / f"phase3_release_{ab}_uniform.npz",
                    {"L": L, "alpha_base": alpha_base, "tau_atp": tau_atp, "mode": 1,
                     "n_seeds": ns_seed, "T": T, "s_values": S.tolist()}, s_values=S, active=A, R=R, Aused=AU)
        print(f"{ab} uniform null: active {A.mean(0).round(3)}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
