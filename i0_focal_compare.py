"""Focal-protocol behaviour at the candidate baselines, before committing to one.

The fine spontaneous sweep (phase3_spontaneous_fine.py) constrains I0^base from BELOW: 0.38 is
the first baseline with a detectable transient and 0.39-0.41 brackets the measured somatic rate.
It says nothing about whether the focal protocol still works there. A medium in which only half
the cells can fire may transmit a front differently from one in which nearly all can, so the
choice between 0.40 and 0.41 is settled here, not by the rate alone.

Reports, per baseline, exactly the quantities Figs 3-4 display (same estimators as
figdata_fig3.one_run): front extent in cells, activated fraction, and the linear-fit speed with
its R^2, deterministic and noisy, over the same 10 seeds. 0.45 is included as the reference point
the current manuscript numbers were produced at.

Run: python i0_focal_compare.py [baseline ...]      (default 0.40 0.41 0.45)
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from phase3_focal_run import focal
from figdata_fig3 import tact_and_radius, UM_50, UM_25


def one_run(seed, alpha, baseline, steps, L, sig, noise_on, patch, stride, dtf):
    """As figdata_fig3.one_run, but also returns the fit R^2 — the pre-set R^2>0.9 criterion is
    part of what makes a speed reportable, so it must travel with the speed."""
    fld = focal(seed, alpha, baseline, steps, L, L, sig, noise_on, patch, stride)
    tact, r = tact_and_radius(fld, L, patch)
    reached = (tact >= 0) & (r > patch)
    extent = float(r[reached].max()) if reached.any() else 0.0
    frac = float((tact >= 0).mean())
    rr = r[reached]; tt = tact[reached].astype(float) * dtf
    band = (rr >= 2) & (rr <= L // 2 - 2)
    speed50 = np.nan; r2 = np.nan
    if band.sum() > 20 and np.ptp(tt[band]) > 1e-9:
        slope, icpt = np.polyfit(rr[band], tt[band], 1)
        pred = slope * rr[band] + icpt
        ss_res = float(((tt[band] - pred) ** 2).sum())
        ss_tot = float(((tt[band] - tt[band].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if slope > 1e-9:
            speed50 = (1.0 / slope) * UM_50
    return extent, frac, speed50, r2


def main(bases):
    L = 64; T = 100.0; steps = int(T / DT); sig = SIGMA_EM_PREDICTED
    stride = 5; dtf = stride * DT; patch = 2; alpha = 0.01
    seeds = list(range(11, 21))
    print(f"Focal comparison: alpha={alpha}, L={L}, T={T:.0f}, {len(seeds)} seeds, "
          f"baselines {list(bases)}")
    print(f"  {'I0base':>7} {'cond':>14} {'extent(cells)':>16} {'frac':>13} "
          f"{'speed_25um':>15} {'R2':>13}")
    out = {}
    for base in bases:
        for noise_on, tag in ((False, "deterministic"), (True, "noisy")):
            t0 = time.time(); E = []; F = []; S = []; Q = []
            for sd in seeds:
                e, f, s, q = one_run(sd, alpha, base, steps, L, sig, noise_on, patch, stride, dtf)
                E.append(e); F.append(f); S.append(s); Q.append(q)
            E = np.array(E); F = np.array(F); S = np.array(S); Q = np.array(Q)
            key = f"b{base:.2f}_{tag}"
            out[f"extent_{key}"] = E; out[f"frac_{key}"] = F
            out[f"speed50_{key}"] = S; out[f"r2_{key}"] = Q
            print(f"  {base:7.2f} {tag:>14} {np.nanmean(E):8.1f}+/-{np.nanstd(E):<6.1f} "
                  f"{np.nanmean(F)*100:7.1f}+/-{np.nanstd(F)*100:<4.1f}% "
                  f"{np.nanmean(S)/2:8.1f}+/-{np.nanstd(S)/2:<5.1f} "
                  f"{np.nanmean(Q):7.3f}+/-{np.nanstd(Q):<5.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
    tagb = "_".join(f"{b:.2f}" for b in bases)
    p = save_result(Path("processed_data") / f"i0_focal_compare_{tagb}.npz",
                    {"L": L, "alpha": alpha, "baselines": list(bases), "seeds": seeds, "T": T,
                     "steps": steps, "stride": stride, "dt_frame": dtf, "patch": patch,
                     "sigma": sig, "um_per_cell_primary": UM_25, "um_per_cell_alt": UM_50,
                     "note": "baseline selection control; speeds stored at 50 um/cell, "
                             "printed at the 25 um/cell primary convention"},
                    **out)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    argv = [float(a) for a in sys.argv[1:]]
    main(tuple(argv) if argv else (0.40, 0.41, 0.45))
