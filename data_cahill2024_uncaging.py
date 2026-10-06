"""
data_cahill2024_uncaging.py — test the model's focal-front predictions on Cahill et al. 2024.

Data: Cahill et al., Nature 629:146 (2024), Dryad 10.5061/dryad.83bk3jb0j (CC0), `events.zip` +
`ramping.zip`: AQuA event tables from acute V1 slices (TTX), cyto-GCaMP6f astrocytes, 1.42 Hz,
427 frames, 2P uncaging of RuBi-GABA or RuBi-glutamate onto one astrocyte at frame 215.
Conditions: wt, cbx (carbenoxolone 50 uM), cx43fl (Cx43 knockdown, uncaged cell RFP-cre+),
laser_control (uncaging laser, no caged compound).

Written BEFORE looking at results. Predictions from the model (manuscript_v2):
  T1 front    : after a point stimulus, the excess response in a band arrives later the farther the
                band is from the stimulus (latency increases with distance). Scattered ignition would
                give no distance dependence.
  T2 bounded  : the excess event rate falls with distance and approaches the laser-control level
                within ~100-250 um.
  T3 coupling : CBX and Cx43 knockdown reduce the excess response at distance (relative to wt).

Method (event-level, as Cahill's Sholl analysis): events of the uncaged cell and of 'ramping'
cells (p <= 0.1 and n_events > 5, Cahill's rule) are dropped; each remaining event is placed in a
distance band by its own minimum distance to the uncaging site (`mark_uncageDistMin`). Bands
25-75, 75-125, 125-175 um (Cahill's start and end). Per recording and band: event rate before
(frames < 215) and after (frames > 217) uncaging; excess = post - pre, events/min. Latency per
band = time after uncaging at which the cumulative excess count, pooled over recordings, reaches
half of its value at +120 s. CIs: bootstrap over recordings (n = 2000).

POST-HOC CORRECTION (after the first run): the event rate rises during the pre-uncaging period in every
condition, from a depressed first minute of each recording, so `exc` against the whole pre-period
overstates the response. The primary comparison is therefore MATCHED windows: the 60 s before uncaging
versus the 60 s after (`m60_*`), and the cumulative excess over time is computed against the same 60 s
baseline rate (`cum_*`). The pre-registered full-baseline numbers are kept (`*_exc`) and labelled as such.

Run: python data_cahill2024_uncaging.py
"""
import sys, glob, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, h5py, pandas as pd
from core.provenance import save_result

ROOT = Path("/mnt/sysfs01/users/cagatay/external/public/cahill2024/dryad")
CONDS = ["wt", "cbx", "cx43fl", "laser_control"]
RAMP = {("wt", "GABA"): "GABA", ("wt", "Glu"): "Glu", ("cbx", "GABA"): "CBX-GABA",
        ("cbx", "Glu"): "CBX-Glu", ("cx43fl", "GABA"): "Cx43-GABA", ("cx43fl", "Glu"): "Cx43-Glu",
        ("laser_control", "None"): "Control-Laser"}
BANDS = [(25, 75), (75, 125), (125, 175)]
U0, U1 = 215, 217                      # uncaging frames (all recordings)
TBIN = 10.0                            # s, PSTH bin
LAT_WIN = 120.0                        # s, window for the half-rise latency
rng = np.random.default_rng(20261006)


def nt_of(fname):
    return "GABA" if "RuBiGABA" in fname else ("Glu" if "RuBiGlu" in fname else "None")


def ramping_set(cond, nt):
    f = ROOT / "ramping" / f"ramping_{RAMP[(cond, nt)]}.csv"
    t = pd.read_csv(f)
    bad = t[(t.p <= 0.1) & (t.n_events > 5)]
    return set(zip(bad.filename, bad.cell_recording))


def load(cond):
    recs = []
    for f in sorted(glob.glob(str(ROOT / "events" / cond / "*.mat"))):
        name = Path(f).name; nt = nt_of(name)
        h = h5py.File(f, "r")
        fr = h["eventFrames"][()].ravel(); c = h["eventCells"][()].ravel()
        d = h["mark_uncageDistMin"][()].ravel(); fs = float(h["fs"][()].item())
        ucell = float(h["uncageCell"][()].item()); nfr = 427
        bad = ramping_set(cond, nt)
        keep = np.array([(cc != ucell) and ((name, int(cc)) not in bad) for cc in c], bool) if c.size else np.zeros(0, bool)
        recs.append(dict(name=name, nt=nt, fs=fs, t=(fr[keep] - U0) / fs, d=d[keep],
                         pre=U0 / fs, post=(nfr - U1) / fs))
    return recs


def band_rates(recs):
    """(n_rec, n_band) pre and post rates in events/min."""
    pre = np.zeros((len(recs), len(BANDS))); post = np.zeros_like(pre)
    for i, r in enumerate(recs):
        for j, (a, b) in enumerate(BANDS):
            m = (r["d"] >= a) & (r["d"] < b)
            pre[i, j] = np.sum(m & (r["t"] < 0)) / r["pre"] * 60
            post[i, j] = np.sum(m & (r["t"] > (U1 - U0) / r["fs"])) / r["post"] * 60
    return pre, post


def latency(recs, idx):
    """Half-rise time of pooled cumulative excess events per band, for recordings idx."""
    out = np.full(len(BANDS), np.nan)
    tt = np.arange(0, LAT_WIN + 1e-9, 1.0)
    for j, (a, b) in enumerate(BANDS):
        cum = np.zeros_like(tt); base = 0.0
        for i in idx:
            r = recs[i]; m = (r["d"] >= a) & (r["d"] < b)
            ev = r["t"][m & (r["t"] > 0)]
            cum += np.searchsorted(np.sort(ev), tt, side="right")
            base += np.sum(m & (r["t"] < 0)) / r["pre"]          # events/s expected
        exc = cum - base * tt
        if exc[-1] <= 0:
            continue
        out[j] = tt[np.argmax(exc >= 0.5 * exc[-1])]
    return out


def psth(recs):
    edges = np.arange(-150, 150 + TBIN, TBIN)
    H = np.zeros((len(BANDS), len(edges) - 1))
    for j, (a, b) in enumerate(BANDS):
        for r in recs:
            m = (r["d"] >= a) & (r["d"] < b)
            H[j] += np.histogram(r["t"][m], edges)[0]
        H[j] = H[j] / len(recs) / TBIN * 60                        # events/min per recording
    return edges, H


def early(recs, step=2.0, tmax=30.0):
    """Excess events per recording per bin in the first tmax s, baseline rate subtracted (added after the
    pre-registered run, to resolve a fast front that 10 s bins would hide)."""
    edges = np.arange(0, tmax + step, step)
    E = np.zeros((len(BANDS), len(edges) - 1))
    for j, (a, b) in enumerate(BANDS):
        for r in recs:
            m = (r["d"] >= a) & (r["d"] < b)
            E[j] += np.histogram(r["t"][m], edges)[0] - np.sum(m & (r["t"] < 0)) / r["pre"] * step
        E[j] /= len(recs)
    return edges, E


W = 60.0                                   # s, matched pre/post window
TCUM = np.arange(0, 91, 2.0)               # s, cumulative-excess time grid


def matched(recs):
    """(n_rec, n_band) excess events/min, 60 s after (from end of uncaging) minus 60 s before."""
    out = np.zeros((len(recs), len(BANDS)))
    for i, r in enumerate(recs):
        t_end = (U1 - U0) / r["fs"]
        for j, (a, b) in enumerate(BANDS):
            m = (r["d"] >= a) & (r["d"] < b)
            out[i, j] = (np.sum(m & (r["t"] > t_end) & (r["t"] <= t_end + W))
                         - np.sum(m & (r["t"] >= -W) & (r["t"] < 0))) * 60 / W
    return out


def cumulative(recs):
    """(n_rec, n_band, len(TCUM)) cumulative excess events after uncaging vs the 60 s baseline rate."""
    X = np.zeros((len(recs), len(BANDS), len(TCUM)))
    for i, r in enumerate(recs):
        for j, (a, b) in enumerate(BANDS):
            m = (r["d"] >= a) & (r["d"] < b)
            rate = np.sum(m & (r["t"] >= -W) & (r["t"] < 0)) / W
            ev = np.sort(r["t"][m & (r["t"] > 0)])
            X[i, j] = np.searchsorted(ev, TCUM, side="right") - rate * TCUM
    return X


def boot(f, n, B=2000):
    vals = np.array([f(rng.integers(0, n, n)) for _ in range(B)])
    return np.nanpercentile(vals, [2.5, 97.5], axis=0)


def main():
    res = {}; arrays = {}
    for cond in CONDS:
        for nt in (["None"] if cond == "laser_control" else ["GABA", "Glu", "both"]):
            recs = [r for r in load(cond) if nt in ("both", r["nt"])]
            if not recs:
                continue
            pre, post = band_rates(recs); exc = post - pre
            lat = latency(recs, np.arange(len(recs)))
            lat_ci = boot(lambda ix: latency(recs, ix), len(recs), 500)
            exc_ci = boot(lambda ix: exc[ix].mean(0), len(recs))
            edges, H = psth(recs)
            eedges, E = early(recs)
            m60 = matched(recs); m60_ci = boot(lambda ix: m60[ix].mean(0), len(recs))
            cum = cumulative(recs); cum_ci = boot(lambda ix: cum[ix].mean(0), len(recs))
            key = f"{cond}_{nt}"
            res[key] = dict(n=len(recs), exc=exc.mean(0), exc_ci=exc_ci, lat=lat, lat_ci=lat_ci,
                            pre=pre.mean(0))
            arrays.update({f"{key}_exc": exc, f"{key}_pre": pre, f"{key}_post": post,
                           f"{key}_lat": lat, f"{key}_latci": lat_ci, f"{key}_psth": H, f"{key}_early": E,
                           f"{key}_excci": exc_ci, f"{key}_m60": m60, f"{key}_m60ci": m60_ci,
                           f"{key}_cum": cum.mean(0), f"{key}_cumci": cum_ci})
            print(f"{key:20s} n={len(recs):3d}   matched 60 s excess/min: " + "  ".join(
                f"{m60[:, j].mean():+.2f}[{m60_ci[0, j]:+.2f},{m60_ci[1, j]:+.2f}]" for j in range(len(BANDS))))
            k30 = np.searchsorted(TCUM, 30); k60 = np.searchsorted(TCUM, 60)
            print("      cumulative excess at 30 s / 60 s: " + "  ".join(
                f"{cum[:, j, k30].mean():+.2f}/{cum[:, j, k60].mean():+.2f}" for j in range(len(BANDS))))
            for j, (a, b) in enumerate(BANDS):
                print(f"   {a:3d}-{b:3d} um  pre {pre[:, j].mean():5.2f}/min  excess {exc[:, j].mean():+6.2f} "
                      f"[{exc_ci[0, j]:+.2f},{exc_ci[1, j]:+.2f}]  half-rise {lat[j]:5.0f} s "
                      f"[{lat_ci[0, j]:.0f},{lat_ci[1, j]:.0f}]")
    arrays["tcum"] = TCUM; arrays["early_edges"] = eedges; arrays["psth_edges"] = edges; arrays["bands"] = np.array(BANDS)
    p = save_result(Path("processed_data") / "cahill2024_uncaging_tests.npz",
                    {"source": "Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j events.zip+ramping.zip",
                     "bands_um": BANDS, "uncage_frames": [U0, U1], "lat_window_s": LAT_WIN,
                     "psth_bin_s": TBIN, "matched_window_s": W, "ramp_rule": "p<=0.1 and n_events>5", "boot_seed": 20261006},
                    **arrays)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
