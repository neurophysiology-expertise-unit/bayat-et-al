"""
data_cahill2024_invivo.py — astrocyte coordination in awake mice: behavioural state vs chemogenetic drive.

EXPLORATORY: the analysis was chosen after a first look at these data (2026-10-06), not pre-registered.

Data: Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j, events.zip/invivo: AQuA events from awake head-fixed
mice (cortex, 400 x 400 um, 1.8 Hz, 30 min sessions); astrocytic Gi-DREADD, CNO saline / 1 / 5 mg/kg, the
same five mice on different days, a Baseline and a Post session each. Events carry position and a
running / stationary flag; there are no cell identities.

Method. Field tiled into 10 x 10 tiles of 40 um; per tile, event-onset counts in 2 s bins; tiles with >= 20
events kept. Coordination = mean pairwise Pearson correlation rho-bar between tiles. Each bin is labelled
running if the majority of its events are flagged running; bins without events take the label of the
nearest labelled bin. rho-bar is computed separately over running and stationary bins (>= 2 min of each).
Reported: (1) paired running vs stationary rho-bar over all sessions with both; (2) Post - Baseline change
in event rate and rho-bar per dose (within mouse). Bootstrap 95% CIs over sessions or mice.

Run: python data_cahill2024_invivo.py
"""
import sys, glob, re, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, h5py
from core.provenance import save_result

D = "/mnt/sysfs01/users/cagatay/external/public/cahill2024/dryad/events/invivo"
FS = 1.8; TILE = 40.0; BIN = 2.0; MINEV = 20; MINBINS = 60
rng = np.random.default_rng(20261006)


def rhobar(X):
    X = X[X.std(1) > 0]
    if X.shape[0] < 3:
        return np.nan
    C = np.corrcoef(X); n = C.shape[0]
    return (C.sum() - n) / (n * (n - 1))


def session(f):
    h = h5py.File(f, "r")
    fr = h["eventFrames"][()].ravel(); x = h["mark_centroidXUM"][()].ravel(); y = h["mark_centroidYUM"][()].ravel()
    run = h["mark_isRunning"][()].ravel()
    n = int(fr.max()) + 1; nb = int(np.ceil(n / FS / BIN))
    b = np.minimum((fr / FS // BIN).astype(int), nb - 1)
    st = np.full(nb, np.nan)
    for k in np.unique(b):
        st[k] = np.nanmean(run[b == k])
    lab = np.where(~np.isnan(st))[0]
    st = np.interp(np.arange(nb), lab, st[lab]) > 0.5
    tile = (np.minimum(x // TILE, 9) * 10 + np.minimum(y // TILE, 9)).astype(int)
    R = np.zeros((100, nb)); np.add.at(R, (tile, b), 1)
    R = R[R.sum(1) >= MINEV]
    out = dict(rate=fr.size / (n / FS / 60), run_frac=st.mean(), ntile=R.shape[0], rho_all=rhobar(R))
    for name, m in (("run", st), ("stat", ~st)):
        out[f"rho_{name}"] = rhobar(R[:, m]) if m.sum() >= MINBINS else np.nan
    return out


def ci(x, B=2000):
    x = np.asarray(x); x = x[~np.isnan(x)]
    return np.percentile([x[rng.integers(0, x.size, x.size)].mean() for _ in range(B)], [2.5, 97.5]), x.size


def main():
    rows = []
    for f in sorted(glob.glob(D + "/*.mat")):
        m = re.match(r".*/\d+_MRAcytoNGi\((\d)\)_(saline|Saline|1mgKg|5mgKg)_.*_(Baseline|Post)_", f)
        o = session(f); o.update(mouse=int(m.group(1)), dose=m.group(2).lower(), phase=m.group(3)); rows.append(o)
    rr = np.array([r["rho_run"] for r in rows]); rs = np.array([r["rho_stat"] for r in rows])
    both = ~np.isnan(rr) & ~np.isnan(rs); diff = rr[both] - rs[both]
    (lo, hi), k = ci(diff)
    print(f"running vs stationary, {k} sessions: rho_run {np.nanmean(rr[both]):.3f}  rho_stat {np.nanmean(rs[both]):.3f}  "
          f"diff {diff.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  run>stat in {np.sum(diff > 0)}/{k}")
    arrays = dict(rho_run=rr, rho_stat=rs, rho_all=np.array([r["rho_all"] for r in rows]),
                  rate=np.array([r["rate"] for r in rows]), run_frac=np.array([r["run_frac"] for r in rows]),
                  mouse=np.array([r["mouse"] for r in rows]), ntile=np.array([r["ntile"] for r in rows]),
                  dose=np.array([r["dose"] for r in rows]), phase=np.array([r["phase"] for r in rows]))
    for dose in ("saline", "1mgkg", "5mgkg"):
        dr, dq, dl = [], [], []
        for mo in range(1, 6):
            B = [r for r in rows if r["mouse"] == mo and r["dose"] == dose and r["phase"] == "Baseline"]
            P = [r for r in rows if r["mouse"] == mo and r["dose"] == dose and r["phase"] == "Post"]
            if B and P:
                dr.append(np.log(P[0]["rate"] / B[0]["rate"])); dq.append(P[0]["rho_all"] - B[0]["rho_all"])
                dl.append(P[0]["run_frac"] - B[0]["run_frac"])
        (l1, h1), _ = ci(dr); (l2, h2), n2 = ci(dq)
        print(f"{dose:7s} n={len(dr)}  log rate ratio {np.mean(dr):+.2f} [{l1:+.2f},{h1:+.2f}]  "
              f"d rho_all {np.nanmean(dq):+.3f} [{l2:+.3f},{h2:+.3f}] (n={n2})  d running fraction {np.mean(dl):+.2f}")
        arrays.update({f"{dose}_dlograte": np.array(dr), f"{dose}_drho": np.array(dq), f"{dose}_drun": np.array(dl)})
    p = save_result(Path("processed_data") / "cahill2024_invivo_tests.npz",
                    {"source": "Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j events.zip/invivo",
                     "status": "exploratory", "tile_um": TILE, "bin_s": BIN, "min_events_tile": MINEV,
                     "min_bins_state": MINBINS, "boot_seed": 20261006}, **arrays)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
