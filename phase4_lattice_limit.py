"""
phase4_lattice_limit.py -- review S3: is the gamma_regen = 1 reference lattice-limited on the 64 x 64 torus?
PRE-REGISTERED 2026-10-15 (this docstring, predictions included, is committed before the first run).

Why. At 120 s the full-regeneration front reaches 34.1 cells at x1 coupling and 43.3 at x2
(front_vs_coupling.npz), more than L/2 = 32 on the 64 x 64 torus (diagonal maximum 45.25), so these arms
have met their periodic images. The abstract's contraction (34.1 -> 3.9 cells) and the matched
gamma_regen interval use this reference.

Method (fixed). phase4_front_coupling.run_drm IMPORTED (no copy), deterministic, alpha 0.01, I0_BASE,
tau_ref 15, focal 5 x 5 patch, 120 s, seeds 11-20. Arms: d_mult 1 x gamma_regen in
{1, 0.3, 0.25, 0.15, 0.10, 0.075, 0.05} (the matched set plus both neighbours that bound it at L = 64) and
d_mult 2 x gamma_regen 1. Measures as phase4_front_coupling.one (final extent = max torus radius reached
outside the patch; stop time = first time the running extent reaches 99 % of its final value; speed and
R^2 from phase3_decremental.focal_lin), with L as a parameter, plus: still advancing = running extent grew
during the last 5 s of the window (the S4 reading of "not stopped").
Step 1, reproduction gate: the same arms at L = 64 must reproduce front_vs_coupling.npz extent, stop_time,
speed and r2 per seed exactly (speed with NaN = NaN); otherwise raise before anything is saved.
Step 2: the same arms at L = 128 (L/2 = 64, diagonal maximum 90.5). The heterogeneity fields are drawn at
the new size, so L = 128 is a new ensemble with the same seeds, not the same cells.
Reported: the L = 128 gamma = 1 reference extent (x1, x2); contraction factor ext(gamma=1)/ext(gamma) at both
sizes; the matched interval (mean extent 4-10 cells = 100-250 um at 25 um/cell) at both sizes; stop times.

Predictions (written before running):
  Q1  Bounded fronts (x1, gamma <= 0.25): mean extent within 1.0 cell and mean stop time within 25 % of
      the L = 64 values.
  Q2  x1, gamma = 1: the L = 64 reference is window-limited, not lattice-limited: L = 128 extent within 3
      cells of 34.1, still advancing in >= 8/10 seeds. x2, gamma = 1: lattice-limited at L = 64: L = 128
      extent > 45.25 cells, still advancing in >= 8/10 seeds.
  Q3  Contraction factor at x1 within 10 % of the L = 64 value for every gamma; matched interval unchanged,
      {0.075, 0.10, 0.15, 0.25} (0.05 and 0.3 outside).
Whatever the outcome, every arm is saved; a failed prediction is reported as failed.

Run: python -s phase4_lattice_limit.py [workers]   (clean tree; writes processed_data/lattice_limit_L128.npz
and commits it at once, so the stamp is clean)
"""
import sys, time, subprocess, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result, git_commit
from phase4_front_coupling import run_drm, STRIDE, PATCH, UM, ALPHA, TAU_REF, T_WIN
from phase3_decremental import focal_lin

dtf = STRIDE * DT
SEEDS = list(range(11, 21))
ARMS = [(1.0, g) for g in (1.0, 0.3, 0.25, 0.15, 0.10, 0.075, 0.05)] + [(2.0, 1.0)]
LAST = int(5.0 / dtf)


def one(job):
    L, m, gr, seed = job
    f = run_drm(seed, ALPHA, I0_BASE, gr, TAU_REF, False, int(T_WIN / DT), L, L, SIGMA_EM_PREDICTED,
                STRIDE, PATCH, m)
    # phase4_front_coupling.one with L as a parameter
    c = L // 2; yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(abs(xx - c), L - abs(xx - c)); dy = np.minimum(abs(yy - c), L - abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    cum = np.zeros((L, L), bool); ext = []
    for t in range(f.shape[0]):
        cum |= f[t] > 0.5; mm = cum & (r > PATCH); ext.append(r[mm].max() if mm.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    t99 = float(np.argmax(ext >= 0.99 * final) * dtf)
    e2, sp, r2, frac = focal_lin(f, dtf, L, PATCH)
    return (L, m, gr, seed, final, t99, sp * UM / 50.0 if np.isfinite(sp) else np.nan, r2,
            bool(ext[-1] > ext[-1 - LAST]), float(ext[int(60.0 / dtf)]))


def run(L, workers):
    t0 = time.time()
    with Pool(workers) as p:
        res = p.map(one, [(L, m, g, s) for m, g in ARMS for s in SEEDS], chunksize=1)
    shp = (len(ARMS), len(SEEDS))
    out = {k: np.zeros(shp) for k in ("extent", "stop_time", "speed", "r2", "advancing", "extent60")}
    for L_, m, g, s, e, t, sp, r2, adv, e60 in res:
        i, k = ARMS.index((m, g)), SEEDS.index(s)
        for key, v in zip(out, (e, t, sp, r2, adv, e60)):
            out[key][i, k] = v
    print(f"L={L} done ({time.time()-t0:.0f}s)", flush=True)
    return out


def report(L, o):
    print(f"\nL = {L}  (L/2 = {L/2:g}, torus max {L/np.sqrt(2):.2f})")
    for i, (m, g) in enumerate(ARMS):
        e = o["extent"][i]; t = o["stop_time"][i]
        print(f"  x{m:g} gamma {g:5.3f}: extent {e.mean():6.2f}+-{e.std(ddof=1):5.2f} cells ({e.mean()*UM:5.0f} um)"
              f"  stop {t.mean():6.1f}+-{t.std(ddof=1):5.1f} s  advancing {int(o['advancing'][i].sum())}/10"
              f"  contraction {o['extent'][0].mean()/e.mean():5.2f}", flush=True)


def main():
    assert not git_commit().endswith("-dirty"), "tree is dirty; commit first"
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    o64 = run(64, w)
    fc = np.load("processed_data/front_vs_coupling.npz")
    fm, fg = list(fc["mults"]), list(fc["gammas"])
    for i, (m, g) in enumerate(ARMS):
        a, b = fm.index(m), fg.index(g)
        for key in ("extent", "stop_time", "r2"):
            assert np.array_equal(o64[key][i], fc[key][a, b]), (m, g, key)
        assert np.array_equal(o64["speed"][i], fc["speed"][a, b], equal_nan=True), (m, g, "speed")
    print("gate passed: L=64 reproduces front_vs_coupling exactly for all 8 arms", flush=True)
    report(64, o64)
    o128 = run(128, w)
    report(128, o128)
    assert not git_commit().endswith("-dirty"), "tree dirty at save time"
    p = save_result(Path("processed_data") / "lattice_limit_L128.npz",
                    {"Ls": [64, 128], "alpha": ALPHA, "tau_ref": TAU_REF, "T_window": T_WIN, "um_per_cell": UM,
                     "arms_dmult_gamma": ARMS, "seeds": SEEDS, "criterion": "99% of final extent",
                     "advancing": "running extent grew in the last 5 s", "deterministic": True,
                     "axis": "[arm, seed]", "preregistered": "phase4_lattice_limit.py docstring"},
                    arms=np.array(ARMS), seeds=np.array(SEEDS),
                    **{f"{k}_L64": v for k, v in o64.items()}, **{f"{k}_L128": v for k, v in o128.items()})
    assert not str(np.load(p)["__git_commit__"]).endswith("-dirty")
    subprocess.run(["git", "add", str(p)], check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"S3: {p.name} (pre-registered lattice-limit test)\n\n"
                    "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"], check=True)
    print(f"wrote and committed {p}", flush=True)


if __name__ == "__main__":
    main()
