"""
phase3_rhobar.py — exact mean pairwise correlation rho-bar versus ATP.

The Phase-2 sweep (phase2_coupling.py, mode 0, i0_form 1, I0_BASE) stores only the
Golomb-Rinzel R on raw C, from which (N R^2 - 1)/(N - 1) is a covariance-normalized synchrony,
not the mean correlation coefficient (commit e388503). This re-runs the same sweep with the
same seeds and, per ATP level, also keeps every STRIDE-th post-burn-in field so the z-scored
two-pass rho-bar of the manuscript Methods can be computed exactly on that sample:

    rho_bar = [ (1/T) sum_t (sum_i z_i(t))^2 - N ] / (N (N - 1))

sweep_rb is sweep_p2 restricted to mode 0 / i0_form 1, with the same RNG consumption and
arithmetic; the run reproduces the stored `active` and `R` and fails loudly if it does not.

Run:  python phase3_rhobar.py            (full: 40 seeds x 21 ATP levels, L=32, ~15-20 min)
      python phase3_rhobar.py check      (self-check: formula vs brute force, stride 1 vs 10)
"""
from __future__ import annotations
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
from numba import njit, prange
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result

SRC = "processed_data/phase2v1_bayat_b0.42_L32_A_full.npz"
STRIDE = 10   # 10 * DT = 0.034 model s between samples; single-cell dynamics evolve over ~10 s


@njit(cache=True)
def rho_exact(Z):
    """Mean pairwise correlation of the columns of Z (T x N), cells with zero variance dropped."""
    T, n = Z.shape
    mu = np.zeros(n); sd = np.zeros(n)
    for i in range(n):
        mu[i] = Z[:, i].mean()
        sd[i] = np.sqrt(np.mean((Z[:, i] - mu[i]) ** 2))
    keep = sd > 1e-12
    m = keep.sum()
    acc = 0.0
    for t in range(T):
        s = 0.0
        for i in range(n):
            if keep[i]:
                s += (Z[t, i] - mu[i]) / sd[i]
        acc += s * s
    return (acc / T - m) / (m * (m - 1.0))


@njit(fastmath=True, cache=True)
def sweep_rb(seed, alpha_values, steps, nx, ny, n, sigma, i0_baseline, stride):
    np.random.seed(seed)
    na = alpha_values.shape[0]
    act = np.zeros(na); Rsync = np.zeros(na); dmean = np.zeros(na); rho = np.zeros(na)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny)) * 1.0
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    t_start = int(0.3 * steps)
    sqrt_dt = DT ** 0.5
    nsamp = (steps - t_start + stride - 1) // stride
    buf = np.zeros((nsamp, n))

    for idx in range(na):
        alpha = alpha_values[idx]
        aD = alpha; aT = alpha; aO = alpha
        gamma = gamma_base
        I0 = i0_baseline + aO * I0_base
        tau_h = 10.0 / ((1.0 + 0.8 * aO) * tau_base)
        Deff = D0_base / (1.0 + (kappa_base * aD) ** 4)
        theta = THETA_BASE + 0.7 * aT
        sigma_eff = sigma * (1.0 + 4.0 * aO)
        gdrive = gamma * aO

        asum = 0.0; msum = 0.0; msum2 = 0.0
        csum = np.zeros(n); csum2 = np.zeros(n)
        cnt = 0; k = 0
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
                if (t - t_start) % stride == 0:
                    buf[k, :] = Cf
                    k += 1
        act[idx] = asum / cnt
        var_m = msum2 / cnt - (msum / cnt) ** 2
        var_i = csum2 / cnt - (csum / cnt) ** 2
        mean_var_i = np.mean(var_i)
        r2 = var_m / (mean_var_i + 1e-12)
        Rsync[idx] = np.sqrt(r2) if r2 > 0.0 else 0.0
        dmean[idx] = np.mean(Deff)
        rho[idx] = rho_exact(buf[:k])
    return act, Rsync, dmean, rho


@njit(parallel=True, cache=True)
def ensemble_rb(seeds, alpha_values, steps, nx, ny, n, sigma, i0_baseline, stride):
    ns = seeds.shape[0]; na = alpha_values.shape[0]
    A = np.zeros((ns, na)); R = np.zeros((ns, na)); D = np.zeros((ns, na)); P = np.zeros((ns, na))
    for i in prange(ns):
        a, r, d, p = sweep_rb(seeds[i], alpha_values, steps, nx, ny, n, sigma, i0_baseline, stride)
        A[i, :] = a; R[i, :] = r; D[i, :] = d; P[i, :] = p
    return A, R, D, P


def check():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 40)) + 0.7 * rng.standard_normal((500, 1))   # shared component
    cc = np.corrcoef(Z.T); brute = cc[np.triu_indices(40, 1)].mean()
    assert abs(rho_exact(Z) - brute) < 1e-12, (rho_exact(Z), brute)
    Z[:, 3] = 1.0                                                              # flat cell is dropped
    keep = np.r_[0:3, 4:40]; cc = np.corrcoef(Z[:, keep].T)
    assert abs(rho_exact(Z) - cc[np.triu_indices(39, 1)].mean()) < 1e-12
    # sampling: stride 10 vs stride 1 on one seed, three ATP levels, short run
    al = np.array([0.065, 0.56, 1.11]); steps = int(60.0 / DT)
    r1 = sweep_rb(11, al, steps, 32, 32, 1024, SIGMA_EM_PREDICTED, I0_BASE, 1)[3]
    r10 = sweep_rb(11, al, steps, 32, 32, 1024, SIGMA_EM_PREDICTED, I0_BASE, STRIDE)[3]
    print("stride 1 :", np.round(r1, 5)); print("stride 10:", np.round(r10, 5))
    assert np.max(np.abs(r1 - r10)) < 2e-3, np.abs(r1 - r10)
    print("check ok")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        return check()
    src = np.load(SRC, allow_pickle=True)
    import json
    pr = json.loads(str(src["__params__"]))
    L = pr["L"]; seeds = np.array(pr["seeds"]); alphas = np.asarray(src["alphas"])
    steps = pr["steps"]; sig = pr["sigma"]; i0b = pr["i0_baseline"]
    assert pr["mode"] == 0 and pr["i0_form"] == 1
    t0 = time.time()
    A, R, D, P = ensemble_rb(seeds, alphas, steps, L, L, L * L, sig, i0b, STRIDE)
    dA = np.max(np.abs(A - src["active"])); dR = np.max(np.abs(R - src["R"]))
    print(f"reproduction vs {SRC}: max|d active| = {dA:.2e}  max|d R| = {dR:.2e}  ({time.time()-t0:.0f}s)")
    assert dA < 1e-9 and dR < 1e-9, "re-run does not reproduce the stored sweep"
    N = L * L; proxy = (N * R ** 2 - 1.0) / (N - 1.0)
    print(f"max |rho_exact - covariance proxy| = {np.max(np.abs(P - proxy)):.4f}")
    p = save_result(Path("processed_data") / "rhobar_vs_atp_L32.npz",
                    {"source": SRC, "L": L, "seeds": seeds.tolist(), "steps": steps, "sigma": sig,
                     "i0_baseline": i0b, "stride": STRIDE, "alphas": alphas.tolist(),
                     "definition": "mean pairwise Pearson correlation of z-scored C over the post-burn-in window, "
                                   "every STRIDE-th step"},
                    alphas=alphas, active=A, R=R, Deff=D, rhobar=P, rho_proxy=proxy)
    for a, m, s, q in zip(alphas, P.mean(0), P.std(0), A.mean(0)):
        print(f"  A={a:.3f}  rho_bar={m:.4f} +- {s:.4f}   active={q:.3f}")
    print(f"wrote {p.name}")


if __name__ == "__main__":
    raise SystemExit(main())
