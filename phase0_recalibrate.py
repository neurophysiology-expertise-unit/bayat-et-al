"""
phase0_recalibrate.py — Phase 0.2: find the sigma that reproduces the published regimes
under the corrected (sqrt-dt) noise scheme.

The old figures were produced by the buggy scheme at sigma = 0.4. That scheme is
equivalent to the correct scheme at sigma * sqrt(dt) = 0.4 * sqrt(0.0034) ~ 0.0233
(core/model.SIGMA_EM_PREDICTED). This script does NOT assume that: it runs the LEGACY
scheme at sigma=0.4 as the reference, sweeps the CORRECT scheme over a set of candidate
sigmas across the ATP (alpha) axis, and reports which candidate best reproduces the
reference regime curves. If the answer lands near 0.0233, the paper's phenomenology
survives with a corrected number; if nothing matches, the old figures were an artifact.

Observables per (sigma, alpha), averaged over seeds:
  active_fraction  — mean gating Phi(C); non-monotonic in alpha in the published Fig 2H
  chi              — susceptibility N*Var_t(mean C); peaks at the transition
  spike_rate       — per-cell zero up-crossings / 1000 steps; oscillation proxy

Run:  /opt/conda/envs/ece/bin/python phase0_recalibrate.py [--smoke]
Writes processed_data/phase0_recalibration.npz (provenance-stamped) and NUMERICS_NOTE.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import simulate_fixed_alpha, DT, SIGMA_EM_PREDICTED, SIGMA_OLD_NOMINAL
from core.provenance import save_result

CANDIDATE_SIGMAS = [0.02, 0.0233, 0.034, 0.05, 0.1, 0.2, 0.4]


def sweep(sigma, alphas, seeds, steps, nx, ny, legacy):
    """Return (active, chi, spike) arrays of shape (n_alpha,), seed-averaged."""
    na = len(alphas)
    act = np.zeros(na)
    chi = np.zeros(na)
    spk = np.zeros(na)
    for ai, a in enumerate(alphas):
        for s in seeds:
            af, ch, sr = simulate_fixed_alpha(a, sigma, s, steps, nx, ny, False, 0.3, legacy)
            act[ai] += af
            chi[ai] += ch
            spk[ai] += sr
    return act / len(seeds), chi / len(seeds), spk / len(seeds)


def rms_norm(a, b):
    """Range-normalised RMS distance between two curves (0 = identical shape+level)."""
    scale = max(np.ptp(b), 1e-9)
    return float(np.sqrt(np.mean(((a - b) / scale) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny/fast sanity run")
    args = ap.parse_args()

    if args.smoke:
        alphas = np.linspace(0.05, 1.1, 6)
        seeds = list(range(11, 14))
        steps = 4000
    else:
        alphas = np.linspace(0.05, 1.1, 12)
        seeds = list(range(11, 19))       # 8 seeds
        steps = 15000                     # T ~ 51, burn 30%

    grids = {"single_unit": (1, 1), "lattice_10x10": (10, 10)}
    out = {}
    print(f"dt={DT}  predicted EM-equivalent sigma = {SIGMA_EM_PREDICTED:.4f} "
          f"(= {SIGMA_OLD_NOMINAL} * sqrt(dt))\n")

    for gname, (nx, ny) in grids.items():
        print(f"=== {gname} ({nx}x{ny}) ===", flush=True)
        # reference: legacy scheme at the old nominal sigma
        ref = sweep(SIGMA_OLD_NOMINAL, alphas, seeds, steps, nx, ny, legacy=True)
        out[f"{gname}__ref_legacy_active"] = ref[0]
        out[f"{gname}__ref_legacy_chi"] = ref[1]
        out[f"{gname}__ref_legacy_spike"] = ref[2]

        print(f"{'sigma':>8} | {'d(active)':>9} {'d(chi)':>8} {'d(spike)':>9} | {'total':>7}")
        print("-" * 52)
        best_sigma, best_d = None, np.inf
        for sig in CANDIDATE_SIGMAS:
            cur = sweep(sig, alphas, seeds, steps, nx, ny, legacy=False)
            out[f"{gname}__sig{sig}_active"] = cur[0]
            out[f"{gname}__sig{sig}_chi"] = cur[1]
            out[f"{gname}__sig{sig}_spike"] = cur[2]
            da = rms_norm(cur[0], ref[0])
            dc = rms_norm(cur[1], ref[1])
            ds = rms_norm(cur[2], ref[2])
            tot = da + dc + ds
            mark = "  <-- best so far" if tot < best_d else ""
            if tot < best_d:
                best_d, best_sigma = tot, sig
            print(f"{sig:>8.4f} | {da:>9.3f} {dc:>8.3f} {ds:>9.3f} | {tot:>7.3f}{mark}",
                  flush=True)
        out[f"{gname}__best_sigma"] = np.array([best_sigma])
        print(f"  best match to legacy(0.4): sigma = {best_sigma}  "
              f"(prediction {SIGMA_EM_PREDICTED:.4f})\n")

    out["alphas"] = alphas
    params = {"dt": DT, "candidate_sigmas": CANDIDATE_SIGMAS, "seeds": seeds,
              "steps": steps, "alphas": alphas.tolist(),
              "sigma_em_predicted": SIGMA_EM_PREDICTED, "smoke": args.smoke}
    p = save_result(Path("processed_data") / "phase0_recalibration.npz", params, **out)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
