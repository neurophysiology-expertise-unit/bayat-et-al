"""Spontaneous event rate at alpha=0 per I0 baseline, to CONSTRAIN the baseline against the
measured 0.1-0.65 events/min/cell (NOT to tune toward rho-bar=0.08). Uses the SAME event-detection
thresholds and time mapping as fig_1_single_cell.py: SEC_PER_AU=1.0 (1 model unit ~ 1 s),
PK_HEIGHT=0.5, PK_PROM=1.0, PK_DIST_AU=2.0. Records per-cell C(t), detects peaks per cell,
reports events/min/cell. bayat-I0 at alpha=0 -> I0=baseline, no drive; L=32, a few seeds.
Run: python phase3_spontaneous_run.py"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from numba import njit
from scipy.signal import find_peaks
from core.model import laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT, SIGMA_EM_PREDICTED

SEC_PER_AU = 1.0; PK_HEIGHT = 0.5; PK_PROM = 1.0; PK_DIST_AU = 2.0


@njit(fastmath=True, cache=False)
def run_C(seed, baseline, steps, nx, ny, sigma, stride):
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    C = np.random.uniform(-0.1, 0.3, (nx, ny))
    h = np.random.uniform(0.4, 1.2, (nx, ny))
    alpha = 0.0                                              # no applied ATP
    I0 = baseline + alpha * I0_base                          # = baseline
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
    Deff = D0_base / (1.0 + (kappa_base * alpha) ** 4)       # = D0_base
    theta = THETA_BASE + 0.7 * alpha                         # = 0.5
    sigma_eff = sigma * (1.0 + 4.0 * alpha)                  # = sigma
    n = nx * ny; nframes = steps // stride
    rec = np.zeros((nframes, n), dtype=np.float32); sqrt_dt = DT ** 0.5; fi = 0
    for t in range(steps):
        noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        diff = Deff * laplacian(C_active)
        dC = C - (C ** 3) / 3.0 - h + I0 + gamma_base * alpha + diff
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        if t % stride == 0 and fi < nframes:
            rec[fi] = C.reshape(n).astype(np.float32); fi += 1
    return rec


def main():
    L = 32; T = 600.0; steps = int(T / DT); sig = SIGMA_EM_PREDICTED   # 600 s = 10 min record
    stride = 10; dt_frame = stride * DT; T_min = (steps * DT * SEC_PER_AU) / 60.0
    dist_frames = max(1, int(PK_DIST_AU / dt_frame))
    seeds = [11, 12, 13]
    print(f"Spontaneous rate at alpha=0, L={L}, T={T:.0f}s={T_min:.1f}min, {len(seeds)} seeds, "
          f"thresholds H={PK_HEIGHT} P={PK_PROM} dist={PK_DIST_AU}AU. Measured band 0.1-0.65 /min/cell.\n")
    print(f"  {'baseline':>8} {'rate/min/cell':>14} {'active_frac(a=0)':>16} {'%cells active':>13}")
    for base in (0.05, 0.15, 0.25, 0.35, 0.45):
        rates = []; afracs = []; pcactive = []
        for sd in seeds:
            rec = run_C(sd, base, steps, L, L, sig, stride)             # (nframes, n)
            npk = 0; n_active_cells = 0
            for c in range(rec.shape[1]):
                pk, _ = find_peaks(rec[:, c], height=PK_HEIGHT, prominence=PK_PROM, distance=dist_frames)
                npk += len(pk); n_active_cells += (len(pk) > 0)
            rates.append(npk / rec.shape[1] / T_min)
            afracs.append(float((0.5 * (1 + np.tanh(ETA * (rec - 0.5)))).mean()))
            pcactive.append(100.0 * n_active_cells / rec.shape[1])
        print(f"  {base:8.2f} {np.mean(rates):8.3f}+/-{np.std(rates):.3f} "
              f"{np.mean(afracs):16.4f} {np.mean(pcactive):12.1f}%", flush=True)


if __name__ == "__main__":
    main()
