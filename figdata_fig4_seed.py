"""Fig 4 panel data, ONE seed per process (the ensemble version of figdata_fig4.py).

The locked figure convention requires an error bar on every displayed number, so the
refractory and decremental sweeps are repeated over the same 10 seeds used for Fig 3
(heterogeneous parameters are seed-drawn, so every quantity varies across seeds).
Each process writes processed_data/fig4_seed<SEED>.npz; figdata_fig4_combine.py
aggregates them into one provenance-stamped file.

Run: python figdata_fig4_seed.py <seed>
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result
from phase3_refractory import run_ref, focal_stats
from phase3_nucleation import nucleation_events
from phase3_decremental import run_dr, focal_lin
from figdata_fig4 import r2_within, TAUS, GAMMAS, UM_50, UM_25

RMAXES = (5, 6, 8, 12, 35)


def main(seed):
    L = 64; sig = SIGMA_EM_PREDICTED; baseline = I0_BASE; stride = 5; dtf = stride * DT
    patch = 2; alpha = 0.01
    t0 = time.time()

    nucl = []; ext_t = []; frac_n = []; spd_t = []
    for tau in TAUS:
        As = run_ref(seed, alpha, baseline, tau, True, False, int(300.0 / DT), L, L, sig, stride, patch)
        rate = len(nucleation_events(As.astype(np.float64))) / (L * L) * 1000.0 / 300.0
        del As
        fd = run_ref(seed, alpha, baseline, tau, False, True, int(100.0 / DT), L, L, sig, stride, patch)
        fn = run_ref(seed, alpha, baseline, tau, True, True, int(100.0 / DT), L, L, sig, stride, patch)
        _, rd, sd = focal_stats(fd, dtf, UM_50, L)
        fr_n, _, _ = focal_stats(fn, dtf, UM_50, L)
        del fd, fn
        nucl.append(rate); ext_t.append(rd); frac_n.append(fr_n); spd_t.append(sd)
        print(f"  seed {seed} tau={tau:4.0f}: nucl {rate:5.2f}  ext {rd:5.1f}  "
              f"frac_noisy {fr_n*100:4.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    ext_g = []; spd_g = []; r2_g = []; ctrl = []
    for gr in GAMMAS:
        fd = run_dr(seed, alpha, baseline, gr, 15.0, False, int(120.0 / DT), L, L, sig, stride, patch)
        e, s, r2, _ = focal_lin(fd, dtf, L, patch)
        ext_g.append(e); spd_g.append(s); r2_g.append(r2)
        if gr == 1.0:                       # r<=R linearity control on the known-good front
            ctrl = [r2_within(fd, L, dtf, patch, rm)[0] for rm in RMAXES]
        del fd
        print(f"  seed {seed} gamma={gr:5.2f}: ext {e:5.1f}  spd50 {s:6.1f}  R2 {r2:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    p = save_result(Path("processed_data") / f"fig4_seed{seed}.npz",
                    {"L": L, "alpha": alpha, "baseline": baseline, "seed": seed, "stride": stride,
                     "dt_frame": dtf, "patch": patch, "sigma": sig, "taus": TAUS, "gammas": GAMMAS,
                     "tau_ref_for_decremental": 15.0, "control_rmaxes": list(RMAXES),
                     "um_per_cell_primary": UM_25, "um_per_cell_alt": UM_50},
                    taus=np.array(TAUS), nucleation_rate=np.array(nucl),
                    extent_vs_tau=np.array(ext_t), frac_noisy_vs_tau=np.array(frac_n),
                    speed50_vs_tau=np.array(spd_t),
                    gammas=np.array(GAMMAS), extent_vs_gamma=np.array(ext_g),
                    speed50_vs_gamma=np.array(spd_g), r2_vs_gamma=np.array(r2_g),
                    control_rmax=np.array(RMAXES), control_r2=np.array(ctrl))
    print(f"seed {seed}: wrote {p.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]))
