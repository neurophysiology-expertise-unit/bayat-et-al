"""Fig 2E data: leave-one-ATP-channel-out tests under affine ATP-dependent current.
Deterministic single-unit calculation over the sampled parameter population; no simulation.
Each freeze holds one ATP-dependent channel at its A=0 value; this is distinct
from holding population heterogeneity at its mean.
Run: python figdata_fig2d.py
"""
import sys; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from core.model import I0_BASE
from core.provenance import save_result
from phase3_onset_duty import onset, Cstar

AGRID = np.linspace(0.0, 3.0, 601)
M = 40000
SEED = 11
WINDOW = 1.11


def main():
    rng = np.random.default_rng(SEED)
    g = rng.uniform(0.05, 0.34, M) * (1.0 + 2.0 * rng.standard_normal(M))
    i0_slope = rng.uniform(0.10, 0.50, M)
    tb = rng.uniform(0.5, 1.1, M)
    i0_intercept = np.full(M, I0_BASE)
    ons = onset(g + i0_slope, i0_intercept, tb, AGRID)
    fin = np.isfinite(ons)
    base_recruit = float(np.mean(ons <= WINDOW))
    print(f"baseline recruited inside [0,{WINDOW}]: {100*base_recruit:.2f}%  "
          f"(never bifurcate on [0,3]: {100*np.mean(~fin):.1f}%)")

    print("\nSpearman(onset, parameter) over finite-onset units:")
    rho = {}
    for name, arr in (("gamma", g), ("tau_base", tb), ("I0_slope", i0_slope)):
        r, _ = spearmanr(arr[fin], ons[fin]); rho[name] = float(r)
        print(f"  {name:9s} {r:+.3f}")

    print("\nleave-one-ATP-channel-out test (channel held at its A=0 value):")
    frozen = {}
    for name in ("gamma", "tau_base", "I0_slope"):
        # onset() represents drive as intercept + effective_slope*A.
        ge, tt = g + i0_slope, tb.copy()
        if name == "gamma":       ge = i0_slope
        elif name == "I0_slope": ge = g
        if name == "tau_base":
            # tau_h is held at each unit's A=0 value, 10/tau_base.
            traces = np.stack([(1.0 - Cstar(a, ge, i0_intercept) ** 2) - 0.8 * tb / 10.0
                               for a in AGRID])
            past = traces >= 0.0
            idx = np.where(past.any(0), past.argmax(0), len(AGRID))
            o2 = np.where(idx < len(AGRID), AGRID[np.clip(idx, 0, len(AGRID) - 1)], np.inf)
        else:
            o2 = onset(ge, i0_intercept, tt, AGRID)
        frozen[name] = float(np.mean(o2 <= WINDOW))
        print(f"  freeze {name:9s} -> recruited {100*frozen[name]:5.2f}%  (baseline {100*base_recruit:.2f}%)")

    p = save_result(Path("processed_data") / "fig2d_gamma_freeze.npz",
                    {"model": "ATP-dependent affine current", "freeze_definition": "ATP channel held at A=0",
                     "i0_intercept": I0_BASE, "i0_slope_range": [0.10, 0.50],
                     "M": M, "seed": SEED, "window": WINDOW, "agrid_n": len(AGRID),
                     "spearman": rho, "frozen_recruited": frozen,
                     "baseline_recruited": base_recruit},
                    onset=ons, gamma=g, i0_slope=i0_slope, tau_base=tb,
                    freeze_names=np.array(list(frozen.keys())),
                    freeze_recruited=np.array([frozen[k] for k in frozen]),
                    baseline_recruited=np.array([base_recruit]),
                    spearman_names=np.array(list(rho.keys())),
                    spearman_vals=np.array([rho[k] for k in rho]))
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()
