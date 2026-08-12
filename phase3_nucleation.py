"""Nucleation-source analysis (2026-08-12) — reconcile the pooled lag-vs-distance (possibly b~2)
with the focal-initiation front (linear, 13.3 um/s). Hypothesis: the spontaneous field has many
simultaneous nucleation sites, so fronts collide/annihilate and a pooled pairwise lag statistic
mixes different fronts -> apparent super-linearity is a MULTI-SOURCE artifact, not diffusion.

Reports per tau_ref (alpha=0.01, L=64, noise ON):
 (1) nucleation rate: rising-edge cells with NO active neighbour in the previous frame (spontaneous
     ignitions, not front-driven), per 1000 cells per model-second.
 (2) pooled power-law exponent b +/- CI (same as lagcheck) -> does b move toward 1 as tau_ref rises
     (fewer sources)? That trend is itself the answer.
 (3) LOCAL single-front speed: near each ignition, activation-time vs radius in a small window before
     neighbouring fronts arrive -> local speed (um/s) and linearity R^2. If local fronts are linear
     at ~13 um/s even when pooled b~2, the pooled statistic is the multi-source artifact.
Run: python phase3_nucleation.py
"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from core.model import DT, SIGMA_EM_PREDICTED
from phase3_refractory import run_ref
from phase3_lagcheck import lag_quality

UM = 50.0

def neighbours_active(prevact):
    """4-neighbour dilation (periodic): True where any von-Neumann neighbour was active."""
    return (np.roll(prevact, 1, 0) | np.roll(prevact, -1, 0) |
            np.roll(prevact, 1, 1) | np.roll(prevact, -1, 1))

def nucleation_events(A):
    """list of (t, x, y) spontaneous ignitions + total rate. active=A>0.5."""
    act = A > 0.5
    evs = []
    for t in range(1, act.shape[0]):
        rising = act[t] & ~act[t - 1]
        nucl = rising & ~neighbours_active(act[t - 1])
        xs, ys = np.where(nucl)
        for x, y in zip(xs, ys):
            evs.append((t, int(x), int(y)))
    return evs

def local_front_speed(A, evs, dtf, L, W=120, Rmax=7, sample=200):
    """for a sample of ignitions, activation-time vs radius in a local window -> slope & R^2."""
    act = A > 0.5; T = A.shape[0]
    speeds = []; r2s = []
    yy, xx = np.mgrid[0:L, 0:L]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(evs), size=min(sample, len(evs)), replace=False) if evs else []
    for k in idx:
        t0, x0, y0 = evs[k]
        if t0 + W >= T:
            continue
        dx = np.minimum(np.abs(xx - x0), L - np.abs(xx - x0))
        dy = np.minimum(np.abs(yy - y0), L - np.abs(yy - y0))
        r = np.sqrt(dx ** 2 + dy ** 2)
        win = act[t0:t0 + W]                         # (W, L, L)
        # first activation frame within window, for cells within Rmax
        ever = win.any(0); first = np.where(ever, win.argmax(0), -1)
        m = (r <= Rmax) & (first >= 0)
        if m.sum() < 12:
            continue
        rr = r[m]; tt = first[m].astype(float) * dtf
        if np.ptp(tt) < 1e-6 or np.ptp(rr) < 1.0:
            continue
        b1, b0 = np.polyfit(tt, rr, 1)               # radius vs time -> slope = speed (cells/s)
        yhat = b1 * tt + b0
        ss = 1.0 - np.sum((rr - yhat) ** 2) / (np.sum((rr - rr.mean()) ** 2) + 1e-12)
        if b1 > 0:
            speeds.append(b1 * UM); r2s.append(ss)
    return (np.median(speeds) if speeds else np.nan,
            np.percentile(speeds, [25, 75]) if speeds else [np.nan, np.nan],
            np.median(r2s) if r2s else np.nan, len(speeds))

def pooled_b(A, dtf, L):
    T = A.shape[0]; max_lag = min(T // 2 - 1, 3000)
    q = lag_quality(A, dtf, L, max_lag)
    ref = q[0][2]
    dd = [d for d, lag, pc in q if pc > 0.08 and pc > 0.25 * ref and 0 < lag < max_lag * 0.95]
    ll = [lag for d, lag, pc in q if pc > 0.08 and pc > 0.25 * ref and 0 < lag < max_lag * 0.95]
    qual = [(d, lag, pc) for d, lag, pc in q]
    if len(dd) >= 3:
        x = np.log(dd); y = np.log(ll); n = len(x)
        b, _ = np.polyfit(x, y, 1)
        resid = y - (b * x + np.polyfit(x, y, 1)[1])
        se = np.sqrt(np.sum(resid ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2))
        return b, 1.96 * se, n, qual
    return np.nan, np.nan, len(dd), qual

def main():
    L = 64; sig = SIGMA_EM_PREDICTED; baseline = 0.45; stride = 5; dtf = stride * DT; patch = 2
    T = 300.0; steps = int(T / DT); Tsec = T
    print(f"Nucleation-source analysis, alpha=0.01, L={L}, T={T:.0f}s. Focal ref: 13.3 um/s linear.\n")
    print(f"  {'tau_ref':>7} | {'nucl/1e3cell/s':>14} | {'pooled b (CI)':>16} | "
          f"{'local speed um/s (IQR)':>24} | {'local R^2':>9} | {'nfit':>5}")
    for tau in (0.0, 5.0, 15.0, 30.0):
        t0 = time.time()
        A = run_ref(11, 0.01, baseline, tau, True, False, steps, L, L, sig, stride, patch).astype(np.float64)
        evs = nucleation_events(A)
        rate = len(evs) / (L * L) * 1000.0 / Tsec
        b, ci, n, _ = pooled_b(A, dtf, L)
        v, iqr, r2, nf = local_front_speed(A, evs, dtf, L)
        print(f"  {tau:7.0f} | {rate:14.2f} | {b:6.2f} +/-{ci:5.2f} ({n}) | "
              f"{v:8.1f} [{iqr[0]:.0f}-{iqr[1]:.0f}] | {r2:9.2f} | {nf:5d} ({time.time()-t0:.0f}s)", flush=True)
    print("\n  b->1 as tau rises (fewer sources) => pooled super-linearity is a MULTI-SOURCE artifact,")
    print("  and local fronts ~13 um/s linear => genuine propagation. b stays ~2 => diffusive.")

if __name__ == "__main__":
    main()
