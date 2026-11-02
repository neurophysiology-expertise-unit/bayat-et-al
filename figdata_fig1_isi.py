"""Fig 1 ISI coefficient of variation per ATP regime (review A5) -- PRE-REGISTERED 2026-10-15.

Why: the text quotes ISI CVs 0.87 / 0.44 / 0.06 (low / intermediate / high ATP) with no committed
source. fig1_single_cell.npz stores only rate/peak/baseline and one displayed trace per regime, and
the low-ATP displayed trace has 2 events (CV undefined).

Method (fixed before running):
  * Model: the ORIGINAL `simulate` and `event_stats` imported from figdata_fig1 (no copies), same
    sigma, dt, ATP levels (0.19 / 0.27 / 0.90) and the same 10 seeds.
  * Reproduction gate: the 600 s records must reproduce the stored rate, peak and baseline arrays
    of fig1_single_cell.npz exactly (np.array_equal) and the stored displayed traces; otherwise
    raise before anything is saved.
  * Peaks: the text's criteria (height 0.5, prominence 1.0, min separation 2 model units), i.e.
    the same find_peaks call as figdata_fig1.event_stats.
  * ISI CV per seed = SD(ISI, ddof=0) / mean(ISI); defined when the seed has >= 2 intervals.
  * Two record lengths: (a) the 600 s records of Fig 1C-E; (b) 6000 s records (same seeds; the
    first 600 s are the same trajectory), because at low ATP a 600 s record holds only ~5 events.
  * PRIMARY statistic (the one to quote): per-seed CV over the 6000 s records, mean +- SD (ddof=1)
    over the 10 seeds. Secondary: (a) per seed over 600 s, and the pooled-interval CV of each.

Predictions (written before running):
  P1  CV falls monotonically low > intermediate > high (both lengths).
  P2  high ATP (limit cycle) CV < 0.1.
  P3  low ATP (noise-triggered) CV in [0.6, 1.1] (near-Poisson excursions with a refractory gap).
  P4  intermediate CV lies between, in [0.25, 0.7].
  The text values 0.87 / 0.44 / 0.06 are NOT a target; whatever is stored replaces them.

Run: python -s figdata_fig1_isi.py  (clean tree; writes processed_data/fig1_isi_cv.npz)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from core.provenance import save_result, git_commit
import figdata_fig1 as f1

T_LONG = 6000.0


def isis(C):
    pk, _ = find_peaks(C, height=f1.PK_HEIGHT, prominence=f1.PK_PROM,
                       distance=int(f1.PK_DIST_AU / f1.DT))
    return np.diff(pk) * f1.DT * f1.SEC_PER_AU


def cv(x):
    return float(np.std(x) / np.mean(x)) if len(x) >= 2 else np.nan


def main():
    assert not git_commit().endswith("-dirty"), "tree is dirty; commit first"
    ref = np.load("processed_data/fig1_single_cell.npz")
    n_long = int((T_LONG / f1.SEC_PER_AU) / f1.DT)
    n_disp = ref["traces"].shape[1]
    nc, ns = len(f1.ATP_LEVELS), len(f1.SEEDS)
    cv600 = np.full((nc, ns), np.nan); cv6000 = np.full((nc, ns), np.nan)
    nint600 = np.zeros((nc, ns), int); nint6000 = np.zeros((nc, ns), int)
    pooled600, pooled6000 = np.zeros(nc), np.zeros(nc)
    for ci, A0 in enumerate(f1.ATP_LEVELS):
        all6, all60 = [], []
        for si, seed in enumerate(f1.SEEDS):
            C600 = f1.simulate(A0, seed, f1.N, f1.DT, f1.SIGMA)
            r, p, b = f1.event_stats(C600)
            assert r == ref["rate"][ci, si] and b == ref["baseline"][ci, si], (A0, seed)
            assert (np.isnan(p) and np.isnan(ref["peak"][ci, si])) or p == ref["peak"][ci, si]
            if seed == ref["rep_seed"][ci]:
                assert np.array_equal(C600[:n_disp].astype(np.float32), ref["traces"][ci])
            CL = f1.simulate(A0, seed, n_long, f1.DT, f1.SIGMA)
            assert np.array_equal(CL[:f1.N], C600), "long record does not extend the 600 s one"
            i6, i60 = isis(C600), isis(CL)
            cv600[ci, si], cv6000[ci, si] = cv(i6), cv(i60)
            nint600[ci, si], nint6000[ci, si] = len(i6), len(i60)
            all6.append(i6); all60.append(i60)
        pooled600[ci] = cv(np.concatenate(all6)); pooled6000[ci] = cv(np.concatenate(all60))
        print(f"A={A0:.2f}: CV600 {np.nanmean(cv600[ci]):.3f}+/-{np.nanstd(cv600[ci], ddof=1):.3f}"
              f" (n_int {nint600[ci].min()}-{nint600[ci].max()}, pooled {pooled600[ci]:.3f})"
              f" | CV6000 {np.nanmean(cv6000[ci]):.3f}+/-{np.nanstd(cv6000[ci], ddof=1):.3f}"
              f" (n_int {nint6000[ci].min()}-{nint6000[ci].max()}, pooled {pooled6000[ci]:.3f})",
              flush=True)
    p = save_result(Path("processed_data") / "fig1_isi_cv.npz",
                    {"source": "figdata_fig1.simulate/event_stats (imported)", "dt": f1.DT,
                     "sigma": f1.SIGMA, "atp_levels": f1.ATP_LEVELS, "seeds": f1.SEEDS,
                     "pk_height": f1.PK_HEIGHT, "pk_prom": f1.PK_PROM, "pk_dist_au": f1.PK_DIST_AU,
                     "T_short": f1.T_SEC, "T_long": T_LONG, "cv": "std(ddof=0)/mean per seed",
                     "primary": "cv6000 mean+-SD(ddof=1) over seeds"},
                    atp=np.array(f1.ATP_LEVELS), seeds=np.array(f1.SEEDS),
                    cv600=cv600, cv6000=cv6000, nint600=nint600, nint6000=nint6000,
                    pooled600=pooled600, pooled6000=pooled6000)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
