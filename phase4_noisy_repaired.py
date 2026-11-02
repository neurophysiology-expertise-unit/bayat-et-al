"""
phase4_noisy_repaired.py -- review S8: the repaired model (refractory + decremental release) UNDER NOISE.
PRE-REGISTERED 2026-10-15 (this docstring, predictions included, is committed before the first run).

Question. Every bounded-front number in paper A is deterministic. Under noise the unrepaired model ignites
everywhere and no front is identifiable (Fig. 3C). Does refractory state + decremental release give an
identifiable, bounded evoked front when stochastic forcing is on?

Model and protocol (fixed). phase4_front_coupling.run_drm IMPORTED (no copy; = phase3_decremental.run_dr
with d_mult), alpha = 0.01, baseline I0_BASE, d_mult = 1, L = 64, 120 s, focal 5x5 patch, seeds 11-20,
sigma = SIGMA_EM_PREDICTED, frames every 5 steps. Each arm is run deterministic AND noisy (same seed, so the
same heterogeneity draw):
  unrepaired        tau_ref 0,  gamma_regen 1     (the comparison the review asks for)
  refractory_only   tau_ref 15, gamma_regen 1
  repaired_g        tau_ref 15, gamma_regen in {0.25, 0.15, 0.10, 0.075}
  base_fig3         phase3_focal_run.focal IMPORTED (the model of Fig. 3, no refractory, no gain): reference.
Reproduction gates (raise before saving): the deterministic tau_ref = 15 arms must reproduce
front_vs_coupling.npz (d_mult 1) extent and stop_time per seed exactly; the first 100 s of base_fig3 must
reproduce fig3_focal.npz extent and activated fraction per seed (det and noisy) exactly.

Measures (per run; active = Phi > 1/2; onset = rising edge of active, frame-0 active cells = the patch):
  activated fraction   cells with any onset / L^2 (as Fig. 3D).
  naive extent, R^2    phase3_decremental.focal_lin on the whole field (as in the paper; NOT separating).
  nucleation count     phase3_nucleation.nucleation_events over the window (rising edge with no active
                       von Neumann neighbour in the previous frame), also as a rate per 1e3 cells per s.
  EVOKED FRONT (primary, the separating definition): a causal connected cluster. Nodes are onset events
  (cell, frame). An edge joins event (i, t_i) to (j, t_j) when j is a von Neumann neighbour of i and
  0 <= t_j - t_i <= Delta. The evoked set is every event reachable from the patch's frame-0 onsets.
    evoked extent      max radius (torus) over evoked cells outside the patch (patch radius 2 if none).
    evoked R^2         linear fit of each evoked cell's first evoked onset time vs radius (cells outside the
                       patch; needs > 6 cells), the same fit as focal_lin.
    evoked stop time   time at which the running evoked extent first reaches 99 % of its final value.
    evoked cells       number of distinct evoked cells; non-evoked onsets = spontaneous in the window.
  Delta (fixed by rule, not tuned): the largest causal lag in the DETERMINISTIC repaired runs (4 gammas x 10
  seeds), where the causal lag of an onset is its time minus the latest neighbour onset at or before it.
  This makes the deterministic cluster equal to the whole deterministic activation by construction.
  Sensitivity: the evoked extent is also reported at Delta/2 and 2*Delta.
  Secondary separating measure: excess-activation radius. P(k) = activated fraction in annulus floor(r)=k;
  background P_bg = activated fraction over r >= 20; R_excess = k*+1 for the last annulus k* of the
  contiguous run k = 3, 4, ... with P(k) - P_bg >= 0.25 (2 = patch radius if annulus 3 already fails).

Predictions (written before running):
  P1  Repaired arms: noisy evoked extent (Delta) within 25 % of the deterministic extent at the same
      gamma_regen (means over seeds), and inside 4-10 cells (100-250 um) at every gamma in the set.
  P2  Activated fraction: repaired noisy <= 10 %; refractory_only noisy >= 60 %; unrepaired noisy >= 80 %;
      base_fig3 noisy >= 90 %.
  P3  Radial timing: repaired noisy evoked R^2 within 0.10 of the deterministic R^2 at the same gamma;
      naive whole-field R^2 of the repaired noisy runs < 0.3 (the naive metric does not separate).
  P4  Nucleation in the window: repaired arms and refractory_only within 10 % of each other; unrepaired
      >= 5x the repaired arms.
  P5  Unrepaired and refractory_only noisy: evoked extent >= 30 cells (no bounded front).
Whatever the outcome, every arm is saved; a failed prediction is reported as failed.

Run: python -s phase4_noisy_repaired.py   (clean tree; writes processed_data/noisy_repaired_L64.npz and
commits it at once, so the stamp is clean)
"""
import sys, json, time, subprocess, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result, git_commit
from phase4_front_coupling import run_drm
from phase3_decremental import focal_lin
from phase3_nucleation import nucleation_events
from phase3_focal_run import focal

L, STRIDE, PATCH, ALPHA, T_WIN = 64, 5, 2, 0.01, 120.0
dtf = STRIDE * DT
SEEDS = list(range(11, 21))
ARMS = [("unrepaired", 0.0, 1.0), ("refractory_only", 15.0, 1.0),
        ("repaired_g0.25", 15.0, 0.25), ("repaired_g0.15", 15.0, 0.15),
        ("repaired_g0.10", 15.0, 0.10), ("repaired_g0.075", 15.0, 0.075), ("base_fig3", None, None)]
REPAIRED = [a for a in ARMS if a[0].startswith("repaired")]
c = L // 2; _yy, _xx = np.mgrid[0:L, 0:L]
_dx = np.minimum(abs(_xx - c), L - abs(_xx - c)); _dy = np.minimum(abs(_yy - c), L - abs(_yy - c))
R = np.sqrt(_dx ** 2 + _dy ** 2)


def ext_t99(act):
    """phase4_front_coupling.one: running extent outside the patch and the 99 % stopping time."""
    cum = np.zeros((L, L), bool); ext = []
    for t in range(act.shape[0]):
        cum |= act[t]; mm = cum & (R > PATCH); ext.append(R[mm].max() if mm.any() else 0.0)
    ext = np.array(ext); final = ext[-1]
    return final, float(np.argmax(ext >= 0.99 * final) * dtf)


def onsets(act):
    rise = np.zeros_like(act); rise[0] = act[0]; rise[1:] = act[1:] & ~act[:-1]
    t, x, y = np.nonzero(rise)
    return np.stack([t, x, y], 1).astype(np.int32)


def one(job):
    arm, tau, gr, seed, noisy = job
    if arm == "base_fig3":
        f = focal(seed, ALPHA, I0_BASE, int(T_WIN / DT), L, L, SIGMA_EM_PREDICTED, noisy, PATCH, STRIDE)
    else:
        f = run_drm(seed, ALPHA, I0_BASE, gr, tau, noisy, int(T_WIN / DT), L, L, SIGMA_EM_PREDICTED,
                    STRIDE, PATCH, 1.0)
    act = f > 0.5
    out = dict(arm=arm, seed=seed, noisy=noisy, onsets=onsets(act))
    out["extent"], out["t99"] = ext_t99(act)
    _, _, out["r2_naive"], _ = focal_lin(f, dtf, L, PATCH)
    out["frac"] = float(act.any(0).mean())
    out["n_nucl"] = len(nucleation_events(f.astype(np.float64)))
    if arm == "base_fig3":                       # gate: first 100 s = fig3_focal one_run
        a100 = act[: int(100.0 / DT) // STRIDE]
        tact = np.where(a100.any(0), a100.argmax(0), -1); reached = (tact >= 0) & (R > PATCH)
        out["g_extent100"] = float(R[reached].max()) if reached.any() else 0.0
        out["g_frac100"] = float((tact >= 0).mean())
    return out


def causal_lags(ev):
    """lag of every non-frame-0 onset to the latest neighbour onset at or before it (inf if none)."""
    by = {}
    for t, x, y in ev:
        by.setdefault((x, y), []).append(t)
    by = {k: np.sort(v) for k, v in by.items()}
    lags = []
    for t, x, y in ev:
        if t == 0:
            continue
        best = np.inf
        for nx, ny in (((x + 1) % L, y), ((x - 1) % L, y), (x, (y + 1) % L), (x, (y - 1) % L)):
            v = by.get((nx, ny))
            if v is None:
                continue
            k = np.searchsorted(v, t, side="right") - 1
            if k >= 0:
                best = min(best, t - v[k])
        lags.append(best)
    return np.array(lags)


def evoked(ev, delta):
    """causal cluster from the frame-0 onsets: (extent, R^2, stop time, n cells, n non-evoked onsets)."""
    by = {}
    for i, (t, x, y) in enumerate(ev):
        by.setdefault((x, y), []).append(i)
    by = {k: np.array(sorted(v, key=lambda i: ev[i, 0])) for k, v in by.items()}
    tim = {k: ev[v, 0] for k, v in by.items()}
    seen = np.zeros(len(ev), bool); stack = list(np.nonzero(ev[:, 0] == 0)[0]); seen[stack] = True
    while stack:
        i = stack.pop(); t, x, y = ev[i]
        for nb in (((x + 1) % L, y), ((x - 1) % L, y), (x, (y + 1) % L), (x, (y - 1) % L)):
            v = tim.get(nb)
            if v is None:
                continue
            lo, hi = np.searchsorted(v, t, "left"), np.searchsorted(v, t + delta, "right")
            for j in by[nb][lo:hi]:
                if not seen[j]:
                    seen[j] = True; stack.append(j)
    e = ev[seen]
    first = {}
    for t, x, y in e:
        if (x, y) not in first or t < first[(x, y)]:
            first[(x, y)] = t
    xy = np.array(list(first.keys())); tt = np.array(list(first.values()), float)
    rr = R[xy[:, 0], xy[:, 1]]; out = rr > PATCH
    ext = float(rr[out].max()) if out.any() else float(PATCH)
    r2 = np.nan
    if out.sum() > 6 and np.ptp(tt[out]) > 1e-6 and np.ptp(rr[out]) > 0.5:
        m, b0 = np.polyfit(rr[out], tt[out], 1); yh = m * rr[out] + b0
        r2 = 1.0 - np.sum((tt[out] - yh) ** 2) / (np.sum((tt[out] - tt[out].mean()) ** 2) + 1e-12)
    # running evoked extent -> 99 % stop time
    o = np.argsort(tt); run = np.maximum.accumulate(np.where(rr[o] > PATCH, rr[o], 0.0))
    stop = float(tt[o][np.argmax(run >= 0.99 * run[-1])] * dtf) if run[-1] > 0 else 0.0
    return ext, r2, stop, len(first), int((~seen).sum())


def r_excess(ev):
    a = np.zeros((L, L), bool); a[ev[:, 1], ev[:, 2]] = True
    pbg = a[R >= 20].mean(); k = PATCH + 1; last = PATCH
    while k < 20:
        ring = np.floor(R) == k
        if a[ring].mean() - pbg < 0.25:
            break
        last = k + 1; k += 1
    return float(last), float(pbg)


def save_and_commit(path, params, **arrays):
    assert not git_commit().endswith("-dirty"), "tree dirty at save time"
    p = save_result(path, params, **arrays)
    assert not str(np.load(p)["__git_commit__"]).endswith("-dirty")
    subprocess.run(["git", "add", str(p)], check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"S8: {p.name} (pre-registered noisy repaired-model test)\n\n"
                    "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"], check=True)
    print(f"wrote and committed {p}", flush=True)


def main():
    assert not git_commit().endswith("-dirty"), "tree is dirty; commit first"
    t0 = time.time()
    jobs = [(a, tau, gr, s, nz) for nz in (False, True) for a, tau, gr in ARMS for s in SEEDS]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        res = p.map(one, jobs, chunksize=1)
    print(f"simulations done ({time.time()-t0:.0f}s)", flush=True)
    get = {(r["arm"], r["seed"], r["noisy"]): r for r in res}
    names = [a[0] for a in ARMS]

    # --- reproduction gates ---
    fc = np.load("processed_data/front_vs_coupling.npz"); fg = list(fc["gammas"])
    for a, tau, gr in ARMS:
        if tau == 15.0:
            j = fg.index(gr)
            assert np.array_equal([get[(a, s, False)]["extent"] for s in SEEDS], fc["extent"][0, j]), a
            assert np.array_equal([get[(a, s, False)]["t99"] for s in SEEDS], fc["stop_time"][0, j]), a
    f3 = np.load("processed_data/fig3_focal.npz")
    for nz, tag in ((False, "deterministic"), (True, "noisy")):
        assert np.array_equal([get[("base_fig3", s, nz)]["g_extent100"] for s in SEEDS], f3[f"extent_{tag}_all"])
        assert np.array_equal([get[("base_fig3", s, nz)]["g_frac100"] for s in SEEDS], f3[f"frac_{tag}_all"])
    print("gates passed: front_vs_coupling (d_mult 1) and fig3_focal reproduced exactly", flush=True)

    # --- Delta from the deterministic repaired runs ---
    lags = np.concatenate([causal_lags(get[(a, s, False)]["onsets"]) for a, _, _ in REPAIRED for s in SEEDS])
    assert np.isfinite(lags).all(), "a deterministic onset with no neighbour onset before it"
    delta = int(lags.max())
    print(f"Delta = {delta} frames = {delta*dtf:.2f} s", flush=True)

    shp = (len(ARMS), 2, len(SEEDS))
    K = ["extent", "t99", "r2_naive", "frac", "n_nucl", "ev_extent", "ev_r2", "ev_stop", "ev_cells",
         "n_nonevoked", "ev_extent_half", "ev_extent_double", "r_excess", "p_bg"]
    A = {k: np.full(shp, np.nan) for k in K}
    for i, a in enumerate(names):
        for nz in (0, 1):
            for k_, s in enumerate(SEEDS):
                r = get[(a, s, bool(nz))]; ev = r["onsets"]
                for k in ("extent", "t99", "r2_naive", "frac", "n_nucl"):
                    A[k][i, nz, k_] = r[k]
                (A["ev_extent"][i, nz, k_], A["ev_r2"][i, nz, k_], A["ev_stop"][i, nz, k_],
                 A["ev_cells"][i, nz, k_], A["n_nonevoked"][i, nz, k_]) = evoked(ev, delta)
                A["ev_extent_half"][i, nz, k_] = evoked(ev, max(delta // 2, 1))[0]
                A["ev_extent_double"][i, nz, k_] = evoked(ev, 2 * delta)[0]
                A["r_excess"][i, nz, k_], A["p_bg"][i, nz, k_] = r_excess(ev)
    for nz, tag in ((0, "deterministic"), (1, "noisy")):
        print(f"\n{tag}")
        for i, a in enumerate(names):
            m = lambda k: f"{np.nanmean(A[k][i, nz]):.3g}+-{np.nanstd(A[k][i, nz], ddof=1):.2g}"
            print(f"  {a:16s} frac {m('frac')} naive ext {m('extent')} R2n {m('r2_naive')} nucl {m('n_nucl')} | "
                  f"evoked ext {m('ev_extent')} (D/2 {m('ev_extent_half')}, 2D {m('ev_extent_double')}) "
                  f"R2 {m('ev_r2')} stop {m('ev_stop')} cells {m('ev_cells')} | R_exc {m('r_excess')}", flush=True)
    save_and_commit(Path("processed_data") / "noisy_repaired_L64.npz",
                    {"L": L, "alpha": ALPHA, "baseline": I0_BASE, "T_window": T_WIN, "stride": STRIDE,
                     "dt_frame": dtf, "patch": PATCH, "sigma": SIGMA_EM_PREDICTED, "d_mult": 1.0,
                     "seeds": SEEDS, "arms": [list(map(str, a)) for a in ARMS], "delta_frames": delta,
                     "delta_s": delta * dtf, "axis": "[arm, (deterministic, noisy), seed]",
                     "um_per_cell": 25.0, "preregistered": "phase4_noisy_repaired.py docstring"},
                    arms=np.array(names), seeds=np.array(SEEDS), delta_frames=np.array(delta), **A)


if __name__ == "__main__":
    main()
