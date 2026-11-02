"""Termination TIME of the bounded front (the extent result's missing companion).

The 120 s figure quoted alongside the extent is the recording window, not the time at which
the front stops. Bowser et al. report that the biological wave ceases after ~15 s as well as
within ~100-250 um, so the termination time is a second, independent comparison the extent
result does not by itself make. Ten seeds (11-20), persisted, per the project convention.
Run: python figdata_termination_time.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result
from phase3_decremental import run_dr

L, STRIDE, PATCH = 64, 5, 2
UM, ALPHA, TAU_REF, T_WIN = 25.0, 0.01, 15.0, 120.0
GAMMAS = [0.10, 0.15, 0.25]
SEEDS = list(range(11, 21))
dtf = STRIDE * DT


def stop_time(gr, seed, frac=0.99):
    f = run_dr(seed, ALPHA, I0_BASE, gr, TAU_REF, False, int(T_WIN / DT), L, L,
               SIGMA_EM_PREDICTED, STRIDE, PATCH)
    c = L // 2
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(abs(xx - c), L - abs(xx - c)); dy = np.minimum(abs(yy - c), L - abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    cum = np.zeros((L, L), bool); ext = []
    for t in range(f.shape[0]):
        cum |= f[t] > 0.5
        m = cum & (r > PATCH)
        ext.append(r[m].max() if m.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    t99 = float(np.argmax(ext >= frac * final) * dtf)
    return t99, float(final)


ts = np.zeros((len(GAMMAS), len(SEEDS))); ex = np.zeros_like(ts)
for i, gr in enumerate(GAMMAS):
    for j, s in enumerate(SEEDS):
        ts[i, j], ex[i, j] = stop_time(gr, s)
    print(f"  gamma_regen={gr:.2f}: stops at {ts[i].mean():5.1f} +/- {ts[i].std():4.1f} s   "
          f"extent {ex[i].mean():.2f} +/- {ex[i].std():.2f} cells "
          f"({ex[i].mean()*UM:.0f} um)", flush=True)

p = save_result(Path("processed_data") / "termination_time.npz",
                {"L": L, "alpha": ALPHA, "baseline": I0_BASE, "tau_ref": TAU_REF,
                 "gammas": GAMMAS, "seeds": SEEDS, "T_window": T_WIN, "dt_frame": dtf,
                 "um_per_cell_primary": UM, "criterion": "first frame at >=99% of final extent"},
                gammas=np.array(GAMMAS), seeds=np.array(SEEDS),
                stop_time=ts, extent=ex,
                stop_time_mean=ts.mean(1), stop_time_sd=ts.std(1),
                extent_mean=ex.mean(1), extent_sd=ex.std(1))
print(f"wrote {p.name}")
