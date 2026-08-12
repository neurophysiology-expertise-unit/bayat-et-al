"""Save the full spatiotemporal C_active field to test for a PROPAGATING wavefront (not just
correlation). bayat-I0 + baseline excitability, Sweep-A dynamics (all channels follow alpha),
L=64 (room for a front to travel), baseline=0.45, alpha in {0.01, 0.12}. One representative seed
per alpha (ensemble averaging would destroy a wave; Fig 2D is a single field).
Field saved subsampled in time (stride) as float32. Run: python phase3_wavefield_run.py
"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from numba import njit
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED
from pathlib import Path


@njit(fastmath=True, cache=False)
def run_field(seed, alpha, baseline, steps, nx, ny, sigma, stride):
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))          # bayat range
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    # bayat-I0 with swept baseline; all channels at this fixed alpha
    I0 = baseline + alpha * I0_base
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
    Deff = D0_base / (1.0 + (kappa_base * alpha) ** 4)
    theta = THETA_BASE + 0.7 * alpha
    sigma_eff = sigma * (1.0 + 4.0 * alpha)
    gdrive = gamma_base * alpha
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    sqrt_dt = DT ** 0.5; fi = 0
    for t in range(steps):
        noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        diff = Deff * laplacian(C_active)
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld


def main():
    L = 64; T = 200.0; steps = int(T / DT); sig = SIGMA_EM_PREDICTED
    baseline = 0.45; stride = 10; seed = 11
    dt_frame = stride * DT                                   # model-time between saved frames
    out = Path("processed_data")
    for alpha in (0.01, 0.12):
        t0 = time.time()
        fld = run_field(seed, alpha, baseline, steps, L, L, sig, stride)
        p = out / f"phase3_wavefield_b0.45_a{alpha:.2f}_L64.npz"
        np.savez_compressed(p, field=fld, alpha=alpha, baseline=baseline, L=L, seed=seed,
                            stride=stride, dt_frame=dt_frame, cell_um=50.0)
        print(f"alpha={alpha:.2f}: field {fld.shape} mean_act={fld.mean():.4f} "
              f"({time.time()-t0:.0f}s) -> {p.name}", flush=True)


if __name__ == "__main__":
    main()
