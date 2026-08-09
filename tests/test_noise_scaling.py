"""
tests/test_noise_scaling.py — the Phase-0 gate.

new_plan.md, Phase 0.1: integrate the pure noise process (f = 0, D = 0) for a fixed
physical time T at several dt, and assert that Var[C(T)] is (a) constant across dt to
within 5% and (b) equal to T * sigma^2. This must pass before any Phase 1 work begins.

No pytest dependency: run it directly

    /opt/conda/envs/ece/bin/python tests/test_noise_scaling.py

Exit status 0 = pass, 1 = fail. It also prints, for contrast, what the OLD (dt-scaled)
scheme produces, so the failure the fix prevents is visible rather than asserted.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.integrator import integrate_pure_noise, _integrate_pure_noise_WRONG  # noqa: E402

# --- test configuration -------------------------------------------------------
SIGMA = 0.4            # nominal noise amplitude (Table 1)
T = 1.0               # fixed physical time; expected Var[C(T)] = T * SIGMA**2 = 0.16
DTS = [1e-4, 3e-4, 1e-3, 3e-3]
# The blueprint specified 500 realizations, but a sample variance from N draws has a
# relative standard error of sqrt(2/(N-1)) — 6.3% at N=500, which cannot resolve a 5%
# across-dt tolerance (an initial run failed at 7.6% spread on pure sampling noise).
# N=4000 brings that standard error to ~2.2%, so the 5% physics tolerance is testable.
N_REAL = 4000
SEED = 20260809
REL_TOL_ACROSS_DT = 0.05     # variance constant across dt to within 5% (the physics)
REL_TOL_ABSOLUTE = 0.06      # and within 6% of T*sigma^2 (resolvable at N=4000)

EXPECTED = T * SIGMA ** 2


def _variances(integrator):
    rng = np.random.default_rng(SEED)
    return {dt: float(np.var(integrator(SIGMA, dt, T, N_REAL, rng), ddof=1)) for dt in DTS}


def main():
    print(f"pure Wiener process  dC = sigma dW,  sigma={SIGMA}, T={T}, "
          f"{N_REAL} realizations")
    print(f"exact stationary law: Var[C(T)] = T*sigma^2 = {EXPECTED:.5f}\n")

    correct = _variances(integrate_pure_noise)
    wrong = _variances(_integrate_pure_noise_WRONG)

    print(f"{'dt':>10} | {'CORRECT (sqrt dt)':>18} | {'OLD (dt) — buggy':>18}")
    print("-" * 54)
    for dt in DTS:
        print(f"{dt:>10.1e} | {correct[dt]:>18.5f} | {wrong[dt]:>18.3e}")
    print()

    vals = np.array([correct[dt] for dt in DTS])
    spread = (vals.max() - vals.min()) / vals.mean()
    abs_err = abs(vals.mean() - EXPECTED) / EXPECTED

    ok_across = spread <= REL_TOL_ACROSS_DT
    ok_absolute = abs_err <= REL_TOL_ABSOLUTE

    print(f"[across-dt]  spread {spread*100:5.2f}%  (<= {REL_TOL_ACROSS_DT*100:.0f}%)   "
          f"-> {'PASS' if ok_across else 'FAIL'}")
    print(f"[absolute ]  mean {vals.mean():.5f} vs {EXPECTED:.5f}  "
          f"({abs_err*100:5.2f}% off, <= {REL_TOL_ABSOLUTE*100:.0f}%)   "
          f"-> {'PASS' if ok_absolute else 'FAIL'}")

    # Sanity: the OLD scheme must actually be dt-dependent, or the test proves nothing.
    wvals = np.array([wrong[dt] for dt in DTS])
    wspread = (wvals.max() - wvals.min()) / wvals.mean()
    print(f"[contrast ]  OLD scheme spread across dt: {wspread*100:5.1f}%  "
          f"(should be large — this is the bug)")

    passed = ok_across and ok_absolute
    print("\n" + ("PASS — noise scaling is correct; Phase 1 may proceed."
                  if passed else
                  "FAIL — integrator does not reproduce Var[C(T)] = T*sigma^2."))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
