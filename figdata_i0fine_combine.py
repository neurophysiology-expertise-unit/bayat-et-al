"""Combine the per-seed fine I0^base sweeps into one provenance-stamped ensemble file.

phase3_spontaneous_fine.py writes one npz per seed (one process per seed). This collects
them into processed_data/i0_fine.npz with mean/SD across seeds for each baseline, which is
what a figure panel or appendix table reads. Every displayed number therefore carries an
error bar over the same seed set as Figs 3 and 4 (11-20), per the locked conventions.

Run: python figdata_i0fine_combine.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from core.provenance import save_result, load_result

MEASURED_LO, MEASURED_HI = 0.1, 0.65        # somatic band, Stobart et al. (see fig_3 panel E)


def main():
    files = sorted(Path("processed_data").glob("i0fine_seed*.npz"))
    if not files:
        raise SystemExit("no i0fine_seed*.npz found — run phase3_spontaneous_fine.py <seed> first")
    bases = None; R = []; A = []; P = []; seeds = []; commits = set()
    for f in files:
        arr, par, commit = load_result(f)
        b = arr["baselines"]
        if bases is None:
            bases = b
        elif not np.array_equal(bases, b):
            raise SystemExit(f"{f.name} has a different baseline grid — refusing to average")
        R.append(arr["rate"]); A.append(arr["active_frac"]); P.append(arr["pct_cells_active"])
        seeds.append(int(par["seed"])); commits.add(commit)
    R = np.array(R); A = np.array(A); P = np.array(P)
    print(f"{len(seeds)} seeds {min(seeds)}-{max(seeds)}, {len(bases)} baselines, "
          f"commit(s) {sorted(commits)}")
    print(f"  {'I0base':>7} {'rate/min/cell':>18} {'%cells active':>18} {'active_frac':>14}  in-band")
    for i, b in enumerate(bases):
        inband = MEASURED_LO <= R[:, i].mean() <= MEASURED_HI
        print(f"  {b:7.2f} {R[:, i].mean():8.3f}+/-{R[:, i].std():<8.3f} "
              f"{P[:, i].mean():8.1f}+/-{P[:, i].std():<7.1f} {A[:, i].mean():14.3e}  "
              f"{'YES' if inband else '-'}")
    p = save_result(Path("processed_data") / "i0_fine.npz",
                    {"seeds": seeds, "n_seeds": len(seeds), "source_commits": sorted(commits),
                     "measured_band": [MEASURED_LO, MEASURED_HI],
                     "note": "alpha=0, L=32, T=600 s; per-seed files from "
                             "phase3_spontaneous_fine.py; rate uses the fig_1 peak thresholds"},
                    baselines=bases,
                    rate_mean=R.mean(0), rate_sd=R.std(0), rate_all=R,
                    pct_active_mean=P.mean(0), pct_active_sd=P.std(0), pct_active_all=P,
                    active_frac_mean=A.mean(0), active_frac_sd=A.std(0), active_frac_all=A)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
