"""
data_cahill2024_invivo_switch.py — test 7: does in vivo astrocyte coordination track FLUCTUATION of the shared input
(how often behavioural state switches), beyond its LEVEL (fraction of time running)?

Model prediction (phase4_shared.py, test 6): coordination is created by fluctuation of a shared input (rho-bar rises with
its amplitude a_c), not by its static level. In vivo the shared input is locomotion/arousal (Slezak et al. 2019,
Paukert et al. 2014). Cahill et al. 2024 in vivo events carry only a running/stationary flag (no speed), so the
fluctuation proxy is the number of state switches.

Written BEFORE the analysis. Data and tiling as data_cahill2024_invivo.py (40 um tiles, 2 s bins, tiles >= 20 events;
all 30 sessions; the CNO manipulation is ignored). Bin state = majority flag of its events, gaps filled from the nearest
labelled bin, then a 5-bin (10 s) running median so label flicker is not counted as switching. Each session is cut into
non-overlapping 3 min windows (90 bins); per window: rho-bar between tiles (tiles with >= 5 events in the window),
running fraction f, and switch count s (state changes). Windows with fewer than 3 usable tiles are dropped.
Primary statistic: within-session partial correlation of rho-bar with s controlling for f (all three demeaned within
session, then rho-bar and s residualised on f). Prediction: positive. CI by bootstrap over sessions (2000); also
the same with f alone (level) for comparison.

Run: python data_cahill2024_invivo_switch.py
"""
import sys, glob, re, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, h5py
from scipy.ndimage import median_filter
from core.provenance import save_result
from data_cahill2024_invivo import D, FS, TILE, BIN, MINEV

WINB = 90                  # bins per window (3 min)
rng = np.random.default_rng(20261007)


def rb(X):
    X = X[X.std(1) > 0]
    if X.shape[0] < 3:
        return np.nan
    C = np.corrcoef(X); n = C.shape[0]
    return (C.sum() - n) / (n * (n - 1))


def windows(f):
    h = h5py.File(f, "r")
    fr = h["eventFrames"][()].ravel(); x = h["mark_centroidXUM"][()].ravel(); y = h["mark_centroidYUM"][()].ravel()
    run = h["mark_isRunning"][()].ravel()
    n = int(fr.max()) + 1; nb = int(np.ceil(n / FS / BIN))
    b = np.minimum((fr / FS // BIN).astype(int), nb - 1)
    st = np.full(nb, np.nan)
    for k in np.unique(b):
        st[k] = np.nanmean(run[b == k])
    lab = np.where(~np.isnan(st))[0]
    st = median_filter((np.interp(np.arange(nb), lab, st[lab]) > 0.5).astype(float), size=5, mode="nearest") > 0.5
    tile = (np.minimum(x // TILE, 9) * 10 + np.minimum(y // TILE, 9)).astype(int)
    R = np.zeros((100, nb)); np.add.at(R, (tile, b), 1)
    R = R[R.sum(1) >= MINEV]
    out = []
    for w0 in range(0, nb - WINB + 1, WINB):
        sl = slice(w0, w0 + WINB); Rw = R[:, sl]; Rw = Rw[Rw.sum(1) >= 5]
        r = rb(Rw)
        if np.isfinite(r):
            out.append((r, st[sl].mean(), np.sum(st[sl][1:] != st[sl][:-1])))
    return np.array(out)


def partial(Ws, use_f=True):
    """Within-session partial correlation of rho with s (controlling f) or with f alone."""
    r_all, s_all, f_all = [], [], []
    for W in Ws:
        if W.shape[0] < 3:
            continue
        r, f, s = (W[:, 0] - W[:, 0].mean(), W[:, 1] - W[:, 1].mean(), W[:, 2] - W[:, 2].mean())
        r_all.append(r); s_all.append(s); f_all.append(f)
    r, s, f = map(np.concatenate, (r_all, s_all, f_all))
    if not use_f:
        return np.corrcoef(r, f)[0, 1]
    A = np.vstack([f, np.ones_like(f)]).T
    rr = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]; ss = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]
    return np.corrcoef(rr, ss)[0, 1]


def main():
    Ws, names = [], []
    for f in sorted(glob.glob(D + "/*.mat")):
        Ws.append(windows(f)); names.append(Path(f).name)
    Ws_ok = [W for W in Ws if W.shape[0] >= 3]
    nwin = sum(W.shape[0] for W in Ws_ok)
    pc = partial(Ws_ok); pf = partial(Ws_ok, use_f=False)
    bs = [partial([Ws_ok[i] for i in rng.integers(0, len(Ws_ok), len(Ws_ok))]) for _ in range(2000)]
    bf = [partial([Ws_ok[i] for i in rng.integers(0, len(Ws_ok), len(Ws_ok))], use_f=False) for _ in range(2000)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); lf, hf = np.nanpercentile(bf, [2.5, 97.5])
    print(f"{len(Ws_ok)} sessions, {nwin} windows of 3 min")
    print(f"PRIMARY  partial corr(rho, switches | running fraction), within session: {pc:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"level    corr(rho, running fraction), within session:                   {pf:+.3f}  [{lf:+.3f}, {hf:+.3f}]")
    allW = np.vstack(Ws_ok)
    print(f"windows: rho median {np.median(allW[:,0]):.3f}, switches median {np.median(allW[:,2]):.0f} (max {allW[:,2].max():.0f}), "
          f"running fraction median {np.median(allW[:,1]):.2f}")
    p = save_result(Path("processed_data") / "cahill2024_invivo_switch.npz",
                    {"source": "Cahill et al. 2024 events.zip/invivo", "window_bins": WINB, "bin_s": BIN,
                     "label_median_bins": 5, "boot_seed": 20261007},
                    windows=np.array(Ws, dtype=object), names=np.array(names),
                    partial=np.array([pc, lo, hi]), level=np.array([pf, lf, hf]))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
