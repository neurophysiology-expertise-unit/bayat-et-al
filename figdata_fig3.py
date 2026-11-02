"""Fig 3 panel data: focal-initiation front (A: t_act vs radius; B: activation map).
Re-runs phase3_focal_run.focal and persists a provenance-stamped npz, so no figure is built from a
log. Prints the quantities previously reported so they can be checked against the logged values:
  alpha=0.01 deterministic: speed 13.3 um/s, extent 31.0 cells, activated frac 60%
  alpha=0.01 noisy        : speed 134.7 um/s, extent 44.6 cells, activated frac 98%
Run: python figdata_fig3.py
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result
from phase3_focal_run import focal

UM_50, UM_25 = 50.0, 25.0


def tact_and_radius(fld, L, patch):
    act = fld > 0.5
    tact = np.where(act.any(0), act.argmax(0), -1)
    c = L // 2
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c))
    dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    return tact, r


def one_run(seed, alpha, baseline, steps, L, sig, noise_on, patch, stride, dtf):
    fld = focal(seed, alpha, baseline, steps, L, L, sig, noise_on, patch, stride)
    tact, r = tact_and_radius(fld, L, patch)
    reached = (tact >= 0) & (r > patch)
    extent = float(r[reached].max()) if reached.any() else 0.0
    frac = float((tact >= 0).mean())
    rr = r[reached]; tt = tact[reached].astype(float) * dtf
    band = (rr >= 2) & (rr <= L // 2 - 2)
    speed50 = np.nan
    if band.sum() > 20 and np.ptp(tt[band]) > 1e-9:
        slope = np.polyfit(rr[band], tt[band], 1)[0]
        if slope > 1e-9:
            speed50 = (1.0 / slope) * UM_50
    return fld, tact, r, extent, frac, speed50


def main():
    L = 64; T = 100.0; steps = int(T / DT); sig = SIGMA_EM_PREDICTED
    baseline = I0_BASE; stride = 5; dtf = stride * DT; patch = 2; alpha = 0.01
    seeds = list(range(11, 21))            # 10 seeds; heterogeneous params are seed-drawn, so BOTH
    rep_seed = 11                          # conditions need an ensemble, not only the noisy one
    out = {}
    print(f"Fig-3 focal data: alpha={alpha}, baseline={baseline}, L={L}, {len(seeds)} seeds, "
          f"dt_frame={dtf:.4f}")
    print(f"  {'cond':>14} {'extent(cells)':>16} {'speed_25um':>16} {'speed_50um':>16} {'frac':>14}")
    for noise_on, tag in ((False, "deterministic"), (True, "noisy")):
        t0 = time.time(); E = []; F = []; S = []
        for sd in seeds:
            fld, tact, r, extent, frac, speed50 = one_run(sd, alpha, baseline, steps, L, sig,
                                                          noise_on, patch, stride, dtf)
            E.append(extent); F.append(frac); S.append(speed50)
            if sd == rep_seed:             # representative realization for the map/scatter panels
                out[f"tact_{tag}"] = tact.astype(np.int32)
                out[f"field_{tag}"] = fld[::4].astype(np.float32)
                out["radius"] = r
        E = np.array(E); F = np.array(F); S = np.array(S)
        out[f"extent_{tag}_all"] = E; out[f"frac_{tag}_all"] = F; out[f"speed50_{tag}_all"] = S
        m = lambda v: (np.nanmean(v), np.nanstd(v))
        (em, es), (fm, fs), (sm, ss) = m(E), m(F), m(S)
        print(f"  {tag:>14} {em:8.1f}+/-{es:<6.1f} {sm/2:8.1f}+/-{ss/2:<6.1f} "
              f"{sm:8.1f}+/-{ss:<6.1f} {fm*100:7.0f}+/-{fs*100:<4.0f}% ({time.time()-t0:.0f}s)",
              flush=True)
    p = save_result(Path("processed_data") / "fig3_focal.npz",
                    {"L": L, "alpha": alpha, "baseline": baseline, "seeds": seeds,
                     "representative_seed": rep_seed, "T": T,
                     "steps": steps, "stride": stride, "dt_frame": dtf, "patch": patch,
                     "sigma": sig, "um_per_cell_primary": UM_25, "um_per_cell_alt": UM_50},
                    **out)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
