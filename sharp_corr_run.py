"""Phase-1 sharpness under a chosen I0 variant (new_plan.md 2026-08-11 robustness table).
HELD: the Phase-1 sharpness rerun waits until the I0 form question is settled by the Phase 2.1
four-variant collapse table. Kept variant-parameterized so it is correct whenever it is run.
Same extended grid as sharp_ext_run.py so the chi_true interior check is directly comparable.
Usage: python sharp_corr_run.py <L> <i0_form 0..3> [n_seeds]"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from phase1_finite_size import sweep_ensemble, DT, SIGMA_EM_PREDICTED
from core.provenance import save_result
from pathlib import Path
GRID = np.concatenate([np.geomspace(0.005, 0.05, 12), np.linspace(0.062, 0.35, 9)])
L = int(sys.argv[1]); IF = int(sys.argv[2]); NS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
VTAG = {0: "submitted", 1: "bayat", 2: "bounded", 3: "const"}[IF]
T = 200.0; steps = int(T/DT); seeds = np.arange(11, 11+NS); sig = SIGMA_EM_PREDICTED
t0 = time.time()
Sc, chi_ext, chi_true, R = sweep_ensemble(seeds, False, GRID, steps, L, L, L*L, sig, -1.0, False, IF)
p = save_result(Path("processed_data")/f"sharp_v{IF}_{VTAG}_L{L}.npz",
                {"L": L, "n_seeds": NS, "T": T, "steps": steps, "sigma": sig, "i0_form": IF,
                 "grid": "geom[0.005,0.05]+lin[0.062,0.35]", "alphas": GRID.tolist()},
                alphas=GRID, chi=chi_ext, chi_true=chi_true, R=R, Sc=Sc)
k = int(chi_true.mean(0).argmax())
print(f"L={L} v{IF}({VTAG}): alpha0(chi_true)={GRID[k]:.4f} {'BOUNDARY' if k == 0 else 'interior'} "
      f"({time.time()-t0:.0f}s) -> {p.name}", flush=True)
