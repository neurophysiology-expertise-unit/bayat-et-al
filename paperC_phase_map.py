"""
paperC_phase_map.py — FEASIBILITY for a short paper C (pure excitable-media dynamics; not part of paper A).
Coupling x regeneration-factor phase map of the focal front in the paper-A lattice.

Model: EXACTLY phase4_front_coupling.run_drm (imported, not copied: same compiled code), focal patch at
alpha = 0.01, tau_ref = 15 s, deterministic (noise off), L = 64, 120 s window, 25 um/cell. Seeds 11-20 are
the quenched cell-to-cell heterogeneity (D0 ~ U(0.05,0.5), etc.), the only randomness in these runs.

Grid: d_mult in D_MULTS (10) x gamma_regen in GAMMAS (19) x seeds 11-20 = 1900 runs.
The grid contains every (d_mult, gamma) point of processed_data/front_vs_coupling.npz (d_mult 1, 2, 4 and its
12 gammas); those 360 runs MUST reproduce the stored extent and stop_time exactly (asserted before saving).

Measures per run (as phase4_front_coupling.one): final extent (cells, radius beyond the seed patch counted from
the patch centre), t99 (time to 99% of final extent), speed, R^2.
"Propagates" (fixed in advance): t99 >= 0.9 * T_WIN, i.e. the front was still advancing at the end of the
window. "Stops": otherwise.

PRE-REGISTERED PREDICTIONS (written 2026-10-11 before any run of this grid):
 P1. The non-monotonic extent-vs-gamma curve at x4 (front_vs_coupling.npz) is a ridge, not a single-point
     accident: for every d_mult >= 4 the extent at an intermediate gamma exceeds the extent at gamma = 1 by more
     than 2 seed-SD of the gamma=1 extent, and the arg-max gamma of mean extent moves DOWN as d_mult grows
     (the decremental gain gamma*D/SCALE is what is held fixed, so gamma_peak ~ 1/d_mult, within a factor 2).
 P2. For d_mult <= 2 the extent is monotone non-decreasing in gamma within seed noise (no ridge): the
     non-monotonicity is a strong-coupling effect.
 P3. At gamma = 1 (full regeneration) there is a coupling d* between 3 and 5 above which no seed propagates
     (fraction propagating drops from >= 0.5 to 0 across one or two grid steps).
If a prediction fails it is reported as failed; the criteria are not revised after the data land.

Checkpoints go OUTSIDE the repo (/tmp/paperC_ckpt) so no untracked file appears in the tree while the final
provenance-stamped save runs. Final result: processed_data/paperC_phase_map.npz.
Run: python -s paperC_phase_map.py        (about 1 h on 11 cores)
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from core.provenance import save_result
from phase4_front_coupling import one, L, ALPHA, TAU_REF, T_WIN, UM, PATCH, SEEDS

D_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
GAMMAS = [1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1, 0.09, 0.075, 0.06, 0.05, 0.04, 0.03,
          0.025, 0.015]
CKPT = Path("/tmp/paperC_ckpt")


def main():
    CKPT.mkdir(exist_ok=True)
    E = np.full((len(D_MULTS), len(GAMMAS), len(SEEDS)), np.nan); T = E.copy(); S = E.copy(); R = E.copy()
    with Pool(11) as p:
        for i, m in enumerate(D_MULTS):
            ck = CKPT / f"row_{m:g}.npy"
            if ck.exists():
                rows = np.load(ck)
            else:
                t0 = time.time()
                res = p.map(one, [(m, gr, s) for gr in GAMMAS for s in SEEDS], chunksize=1)
                rows = np.array([r[3:] for r in res]).reshape(len(GAMMAS), len(SEEDS), 4)
                np.save(ck, rows); print(f"d_mult {m:g} done in {time.time()-t0:.0f}s", flush=True)
            E[i], T[i], S[i], R[i] = rows[..., 0], rows[..., 1], rows[..., 2], rows[..., 3]
    # reproduction of the stored paper-A runs (exact)
    ref = np.load("processed_data/front_vs_coupling.npz")
    for a, m in enumerate(ref["mults"]):
        for b, gr in enumerate(ref["gammas"]):
            i, j = D_MULTS.index(float(m)), GAMMAS.index(float(gr))
            assert np.array_equal(E[i, j], ref["extent"][a, b]), (m, gr)
            assert np.array_equal(T[i, j], ref["stop_time"][a, b]), (m, gr)
    print("stored front_vs_coupling.npz reproduced exactly (360 runs)")
    prop = T >= 0.9 * T_WIN
    for i, m in enumerate(D_MULTS):
        print(f"d_mult {m:g}: " + " ".join(f"{E[i,j].mean():5.1f}{'*' if prop[i,j].mean()>=0.5 else ' '}"
                                            for j in range(len(GAMMAS))))
    out = save_result(Path("processed_data") / "paperC_phase_map.npz",
                      {"model": "phase4_front_coupling.run_drm (imported)", "L": L, "alpha": ALPHA,
                       "tau_ref": TAU_REF, "T_window": T_WIN, "um_per_cell": UM, "patch": PATCH,
                       "d_mults": D_MULTS, "gammas": GAMMAS, "seeds": SEEDS, "noise": False,
                       "propagates": "t99 >= 0.9*T_window", "extent": "cells from patch centre, 99% criterion"},
                      d_mults=np.array(D_MULTS), gammas=np.array(GAMMAS), seeds=np.array(SEEDS),
                      extent=E, stop_time=T, speed=S, r2=R, propagates=prop)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
