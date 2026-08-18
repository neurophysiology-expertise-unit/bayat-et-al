"""Combine the per-seed spontaneous-field lag checks into one provenance-stamped ensemble npz.

The claim this supports is a negative control with TWO signatures, and the combine reports both:
  tau_ref = 0   lags are large and mutually inconsistent across separations -> no lag-vs-distance
                relation, so no propagation geometry can be fitted;
  tau_ref > 0   lags are exactly zero wherever the correlation is resolvable, and the peak falls
                to the noise floor beyond two to three cell spacings.
Ten seeds (11-20), mean +/- SD, matching every other ensemble in the figures.
Run: python figdata_lagcheck_combine.py
"""
import sys; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from pathlib import Path
from core.model import I0_BASE
from core.provenance import save_result

SEEDS = list(range(11, 21))


def main():
    files = [Path("processed_data") / f"lagcheck_seed{s}.npz" for s in SEEDS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing per-seed results, refusing to combine: {missing}")

    d0 = np.load(files[0], allow_pickle=True)
    taus = d0["taus"]; ds = d0["separations"]; dtf = float(d0["dt_frame"][0])
    lag = np.stack([np.load(f, allow_pickle=True)["lag_frames"] for f in files])   # (seed, tau, d)
    corr = np.stack([np.load(f, allow_pickle=True)["peakcorr"] for f in files])
    res = np.stack([np.load(f, allow_pickle=True)["resolved"] for f in files])

    print(f"lag check, {len(SEEDS)} seeds, baseline I0={I0_BASE}, dt_frame={dtf:.4f} s\n")
    for i, tau in enumerate(taus):
        print(f"  [tau_ref={tau:.0f}s]")
        print(f"    {'d':>3} {'lag(fr) mean+/-SD':>22} {'peakcorr':>18} {'seeds resolved':>15}")
        for j, dsep in enumerate(ds):
            print(f"    {dsep:>3.0f} {lag[:,i,j].mean():>11.1f} +/- {lag[:,i,j].std():<7.1f}"
                  f" {corr[:,i,j].mean():>10.4f} +/- {corr[:,i,j].std():<5.4f}"
                  f" {int(res[:,i,j].sum()):>10d}/{len(SEEDS)}")
        # the two signatures, quantified
        rmask = res[:, i, :] > 0.5
        zero_frac = (lag[:, i, :][rmask] == 0).mean() if rmask.any() else np.nan
        maxres = np.array([ds[r].max() if r.any() else 0.0 for r in rmask])
        print(f"    -> lag is exactly zero in {100*zero_frac:.0f}% of resolvable (seed, d) pairs; "
              f"resolvable out to d = {maxres.mean():.1f} +/- {maxres.std():.1f} cells\n")

    p = save_result(Path("processed_data") / "lagcheck.npz",
                    {"seeds": SEEDS, "baseline": I0_BASE, "alpha": 0.01, "L": 64, "T": 300.0,
                     "taus": taus.tolist(), "separations": ds.tolist(), "dt_frame": dtf,
                     "resolved_rule": str(np.load(files[0], allow_pickle=True)["__params__"])[:0] or
                                      "peakcorr>0.08 and >0.25*corr(d=1) and lag<0.95*max_lag"},
                    taus=taus, separations=ds, lag_frames=lag, peakcorr=corr, resolved=res,
                    lag_mean=lag.mean(0), lag_sd=lag.std(0),
                    peakcorr_mean=corr.mean(0), peakcorr_sd=corr.std(0),
                    n_resolved=res.sum(0), dt_frame=np.array([dtf]))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
