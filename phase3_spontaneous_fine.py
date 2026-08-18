"""Fine I0^base sweep of the spontaneous transient rate at alpha=0, one seed per process.

The coarse sweep in phase3_spontaneous_run.py stepped I0^base by 0.10 and therefore reported
an apparent jump from exactly 0.000 to 0.536 events/min/cell between 0.35 and 0.45. That step
is too large to locate the crossover: this script resolves it to 0.01-0.02 so the baseline can
be constrained against the measured somatic rate rather than merely bracketed by it.

Two quantities are recorded per baseline and both matter:
  rate         -- detected transients/min/cell, SAME thresholds as fig_1_single_cell.py
                  (PK_HEIGHT=0.5, PK_PROM=1.0, PK_DIST_AU=2.0, SEC_PER_AU=1.0)
  active_frac  -- the time-and-cell mean of the transmission variable Phi(C), which is a
                  detector-free measure of subthreshold occupancy. The zero rates at low
                  baselines are a property of the peak detector, not of the dynamics:
                  active_frac is smooth and monotone across the whole grid.

Run: python phase3_spontaneous_fine.py <seed>      (then figdata_i0fine_combine.py)
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from core.model import DT, SIGMA_EM_PREDICTED, ETA
from core.provenance import save_result
from phase3_spontaneous_run import run_C, PK_HEIGHT, PK_PROM, PK_DIST_AU, SEC_PER_AU

BASES = (0.05, 0.15, 0.25, 0.30, 0.35, 0.37, 0.38, 0.39, 0.40,
         0.41, 0.42, 0.43, 0.45, 0.47, 0.50)


def main(seed):
    L = 32; T = 600.0; steps = int(T / DT); sig = SIGMA_EM_PREDICTED
    stride = 10; dt_frame = stride * DT
    T_min = (steps * DT * SEC_PER_AU) / 60.0
    dist_frames = max(1, int(PK_DIST_AU / dt_frame))
    t0 = time.time()
    rate = []; afrac = []; pcact = []
    for base in BASES:
        rec = run_C(seed, base, steps, L, L, sig, stride)
        npk = 0; nac = 0
        for c in range(rec.shape[1]):
            pk, _ = find_peaks(rec[:, c], height=PK_HEIGHT, prominence=PK_PROM,
                               distance=dist_frames)
            npk += len(pk); nac += (len(pk) > 0)
        rate.append(npk / rec.shape[1] / T_min)
        afrac.append(float((0.5 * (1 + np.tanh(ETA * (rec - 0.5)))).mean()))
        pcact.append(100.0 * nac / rec.shape[1])
        del rec
        print(f"  seed {seed} I0={base:4.2f}: rate {rate[-1]:6.3f}  "
              f"active_frac {afrac[-1]:.3e}  cells {pcact[-1]:5.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
    p = save_result(Path("processed_data") / f"i0fine_seed{seed}.npz",
                    {"L": L, "T": T, "steps": steps, "sigma": sig, "seed": seed,
                     "alpha": 0.0, "i0_form": 1, "baselines": list(BASES),
                     "PK_HEIGHT": PK_HEIGHT, "PK_PROM": PK_PROM,
                     "PK_DIST_AU": PK_DIST_AU, "SEC_PER_AU": SEC_PER_AU},
                    baselines=np.array(BASES), rate=np.array(rate),
                    active_frac=np.array(afrac), pct_cells_active=np.array(pcact))
    print(f"seed {seed}: wrote {p.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
