"""Fig 4 panel data: refractory does part of the job, decremental release does the rest.
Panels: (A) nucleation rate vs tau_ref; (B) focal extent vs tau_ref (unchanged); (C) extent vs
gamma_regen; (D) speed vs gamma_regen; (E) r<=5 linearity control on the known-good front.
Persists one provenance-stamped npz. Prints each quantity previously reported from logs for
checking:
  A: nucleation 1.43 (tau=0) -> 0.26 / 0.27 / 0.25 per 1e3 cell/s
  B: focal-det extent 31.0 cells at every tau_ref; noisy activated frac 98 -> 40 -> 30 -> 30 %
  C: extent 35.0, 34.9, 34.9, 34.5, 31.6, 10.4, 9.2, 6.1, 5.7, 5.0, 4.2 cells
  D: speed 13.2 ... 14.1 um/s (50um convention), no systematic trend
  E: gamma=1.0 front R^2 0.964 full-range, 0.690 restricted to r<=5
Run: python figdata_fig4.py
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result
from phase3_refractory import run_ref, focal_stats
from phase3_nucleation import nucleation_events
from phase3_decremental import run_dr, focal_lin

UM_50, UM_25 = 50.0, 25.0
TAUS = [0.0, 5.0, 15.0, 30.0]
GAMMAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.20, 0.15, 0.10, 0.05]


def r2_within(fld, L, dtf, patch, rmax):
    act = fld > 0.5
    tact = np.where(act.any(0), act.argmax(0), -1)
    c = L // 2
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c))
    dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    m = (tact >= 0) & (r > patch) & (r <= rmax)
    if m.sum() < 8:
        return np.nan, int(m.sum())
    rr = r[m]; tt = tact[m].astype(float) * dtf
    sl, ic = np.polyfit(rr, tt, 1)
    return float(1 - np.sum((tt - (sl * rr + ic)) ** 2) / (np.sum((tt - tt.mean()) ** 2) + 1e-12)), int(m.sum())


def main():
    L = 64; sig = SIGMA_EM_PREDICTED; baseline = I0_BASE; stride = 5; dtf = stride * DT
    patch = 2; alpha = 0.01; seed = 11
    out = {}

    print("(A,B) refractory sweep")
    print(f"  {'tau_ref':>7} {'nucl/1e3cell/s':>14} {'ext_det':>8} {'frac_noisy':>10}")
    nucl = []; ext_t = []; frac_n = []
    for tau in TAUS:
        t0 = time.time()
        As = run_ref(seed, alpha, baseline, tau, True, False, int(300.0 / DT), L, L, sig, stride, patch)
        ev = nucleation_events(As.astype(np.float64))
        rate = len(ev) / (L * L) * 1000.0 / 300.0
        fd = run_ref(seed, alpha, baseline, tau, False, True, int(100.0 / DT), L, L, sig, stride, patch)
        fn = run_ref(seed, alpha, baseline, tau, True, True, int(100.0 / DT), L, L, sig, stride, patch)
        _, rd, _ = focal_stats(fd, dtf, UM_50, L)      # (frac, extent, speed)
        fr_n, _, _ = focal_stats(fn, dtf, UM_50, L)
        nucl.append(rate); ext_t.append(rd); frac_n.append(fr_n)
        print(f"  {tau:7.0f} {rate:14.2f} {rd:8.1f} {fr_n*100:9.0f}% ({time.time()-t0:.0f}s)", flush=True)

    print("\n(C,D) decremental sweep + (E) linearity control")
    print(f"  {'gamma':>6} {'extent':>7} {'speed50':>8} {'speed25':>8} {'R2':>6}")
    ext_g = []; spd_g = []; r2_g = []
    for gr in GAMMAS:
        fd = run_dr(seed, alpha, baseline, gr, 15.0, False, int(120.0 / DT), L, L, sig, stride, patch)
        e, s, r2, fr = focal_lin(fd, dtf, L, patch)
        ext_g.append(e); spd_g.append(s); r2_g.append(r2)
        print(f"  {gr:6.2f} {e:7.1f} {s:8.1f} {s/2:8.1f} {r2:6.3f}", flush=True)
        if gr == 1.0:
            out["control_field_gamma1"] = fd[::4].astype(np.float32)
            print("     r<=R control on this known-good front:")
            for rmax in (5, 6, 8, 12, 35):
                rr2, n = r2_within(fd, L, dtf, patch, rmax)
                out[f"control_r2_rmax{rmax}"] = rr2
                print(f"       r<={rmax:2d}: R2={rr2:.3f} (n={n})", flush=True)

    p = save_result(Path("processed_data") / "fig4_mechanisms.npz",
                    {"L": L, "alpha": alpha, "baseline": baseline, "seed": seed, "stride": stride,
                     "dt_frame": dtf, "patch": patch, "sigma": sig, "taus": TAUS, "gammas": GAMMAS,
                     "tau_ref_for_decremental": 15.0,
                     "um_per_cell_primary": UM_25, "um_per_cell_alt": UM_50},
                    taus=np.array(TAUS), nucleation_rate=np.array(nucl),
                    extent_vs_tau=np.array(ext_t), frac_noisy_vs_tau=np.array(frac_n),
                    gammas=np.array(GAMMAS), extent_vs_gamma=np.array(ext_g),
                    speed50_vs_gamma=np.array(spd_g), r2_vs_gamma=np.array(r2_g), **out)
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()
