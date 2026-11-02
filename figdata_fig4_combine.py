"""Aggregate the per-seed Fig 4 runs into one provenance-stamped npz (mean and SD over seeds).

Reads processed_data/fig4_seed<S>.npz for each seed written by figdata_fig4_seed.py and writes
processed_data/fig4_mechanisms_ens.npz. Prints the ensemble against the single-seed values already
reported (seed 11), so any divergence is visible before a figure is built.
Run: python figdata_fig4_combine.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json
import numpy as np
from pathlib import Path
from core.provenance import save_result

SEEDS = list(range(11, 21))
KEYS = ("nucleation_rate", "extent_vs_tau", "frac_noisy_vs_tau", "speed50_vs_tau",
        "extent_vs_gamma", "speed50_vs_gamma", "r2_vs_gamma", "control_r2")


def main():
    files = {s: Path("processed_data") / f"fig4_seed{s}.npz" for s in SEEDS}
    have = [s for s in SEEDS if files[s].exists()]
    missing = [s for s in SEEDS if s not in have]
    if missing:
        print(f"missing seeds: {missing}")
        sys.exit(1)
    D = {s: np.load(files[s], allow_pickle=True) for s in have}
    P = json.loads(str(D[have[0]]["__params__"]))
    stacks = {k: np.stack([D[s][k] for s in have]) for k in KEYS}
    ref = D[11]

    out = {}
    for k, v in stacks.items():
        out[f"{k}_mean"] = np.nanmean(v, 0)
        out[f"{k}_sd"] = np.nanstd(v, 0)
        out[f"{k}_all"] = v

    print(f"{len(have)} seeds: {have}\n")
    print("(A,B) refractory sweep — ensemble vs seed 11")
    print(f"  {'tau':>5} {'nucl/1e3cell/s':>20} {'extent(cells)':>18} {'frac_noisy':>16}")
    for i, tau in enumerate(ref["taus"]):
        print(f"  {tau:5.0f} "
              f"{out['nucleation_rate_mean'][i]:8.2f}+/-{out['nucleation_rate_sd'][i]:<5.2f}"
              f"[{ref['nucleation_rate'][i]:5.2f}] "
              f"{out['extent_vs_tau_mean'][i]:7.1f}+/-{out['extent_vs_tau_sd'][i]:<4.1f}"
              f"[{ref['extent_vs_tau'][i]:5.1f}] "
              f"{100*out['frac_noisy_vs_tau_mean'][i]:6.0f}+/-{100*out['frac_noisy_vs_tau_sd'][i]:<3.0f}%"
              f"[{100*ref['frac_noisy_vs_tau'][i]:3.0f}%]")

    print("\n(C,D) decremental sweep — ensemble vs seed 11")
    print(f"  {'gamma':>6} {'extent(cells)':>18} {'speed25(um/s)':>18} {'R2':>16}")
    for i, g in enumerate(ref["gammas"]):
        print(f"  {g:6.2f} "
              f"{out['extent_vs_gamma_mean'][i]:7.1f}+/-{out['extent_vs_gamma_sd'][i]:<4.1f}"
              f"[{ref['extent_vs_gamma'][i]:5.1f}] "
              f"{out['speed50_vs_gamma_mean'][i]/2:7.1f}+/-{out['speed50_vs_gamma_sd'][i]/2:<4.1f}"
              f"[{ref['speed50_vs_gamma'][i]/2:5.1f}] "
              f"{out['r2_vs_gamma_mean'][i]:6.3f}+/-{out['r2_vs_gamma_sd'][i]:<5.3f}"
              f"[{ref['r2_vs_gamma'][i]:5.3f}]")

    print("\n(E) r<=R control on the known-good gamma=1.0 front — ensemble vs seed 11")
    for i, rm in enumerate(ref["control_rmax"]):
        print(f"  r<={rm:2d}: R2 = {out['control_r2_mean'][i]:.3f}+/-{out['control_r2_sd'][i]:.3f}"
              f"  [{ref['control_r2'][i]:.3f}]")

    P.update({"seeds": have, "aggregated": "mean and SD over seeds"})
    P.pop("seed", None)
    p = save_result(Path("processed_data") / "fig4_mechanisms_ens.npz", P,
                    taus=ref["taus"], gammas=ref["gammas"], control_rmax=ref["control_rmax"],
                    seeds=np.array(have), **out)
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()
