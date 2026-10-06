"""
data_cahill2024_dose.py — test 4: does astrocyte network coordination change abruptly with agonist dose?

Data: Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j, AQuA_CytoGCaMP_ReceptorAgonistBathApp_Fig1.mat:
acute V1 slices (TTX), cyto-GCaMP6f astrocytes, 1.42 Hz, 853 frames (10 min); bath baclofen (GABA_B) and
t-ACPD (mGluR) applied in turn to each slice at one of 5/25/50/100 uM. Per-cell FIJI traces
(`ImageJByRegion.RawTrace`, cells x frames). Mice 20210624 and 20210626 excluded (authors' DatesToExclude):
16 slices, 4 per dose.

Written BEFORE the analysis. Prediction from the model (manuscript_v2): driving the population harder
raises activity without a collective onset, so mean pairwise correlation does not jump at any dose.
With 4 slices per dose an abrupt step cannot be resolved finely; the test is (i) whether coordination
rises or falls with activation and (ii) whether its change is graded across the four doses.

Method. Agonist entry frame = first frame of the Alexa-594 trace >= mean + 3 SD of frames 1-300 (authors'
rule in Ext. Data Fig. 1k). Windows: 120 s before entry (pre) and 120 s from entry (post), the authors'
comparison window. dF/F per cell with F0 = median of the pre window. Activity = mean dF/F over cells
and window. Coordination = mean pairwise Pearson correlation rho-bar of
  (a) PRIMARY: fluctuations, each trace minus its 60 s running median (so a shared slow rise in
      calcium does not count as coordination);
  (b) linearly detrended traces within each window.
Reported per dose: post - pre change in activity and in rho-bar (mean, bootstrap 95% CI over slices),
and the Spearman correlation of the change with dose across slices.

Run: python data_cahill2024_dose.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, h5py
from scipy.ndimage import median_filter
from scipy.stats import spearmanr
from core.provenance import save_result

F = Path("/mnt/sysfs01/users/cagatay/external/public/cahill2024/dryad/"
         "AQuA_CytoGCaMP_ReceptorAgonistBathApp_Fig1.mat")
FS = 1.42; WIN = int(round(120 * FS)); HP = int(round(60 * FS)) | 1
EXCLUDE = ("20210624", "20210626")
rng = np.random.default_rng(20261006)


def rhobar(X):
    """Mean off-diagonal Pearson correlation of rows of X (cells x time); flat rows dropped."""
    X = X[np.std(X, 1) > 1e-12]
    C = np.corrcoef(X); n = C.shape[0]
    return (C.sum() - n) / (n * (n - 1))


def detrend(X):
    t = np.arange(X.shape[1]); A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    return X - (A @ coef).T


def entry_frame(alexa):
    y = alexa[:, 1] if alexa.ndim == 2 and alexa.shape[1] == 2 else alexa.ravel()
    b = y[:300]; idx = np.nonzero(y[300:] >= b.mean() + 3 * b.std())[0]
    return 300 + int(idx[0]) if idx.size else None


def main():
    f = h5py.File(F, "r"); md = f["mydata"]; rows = []
    for ag in ("Baclofen", "tACPD"):
        for i in range(md[ag].shape[0]):
            g = f[md[ag][i, 0]]
            name = "".join(map(chr, g["file"][()].ravel())).split("\\")[-1]
            if name.startswith(EXCLUDE):
                continue
            raw = np.asarray(g["ImageJByRegion"]["RawTrace"][()], float)    # (cells, frames)
            al = np.asarray(g["Alexa594"]["RawTraceXY"][()], float)
            al = al.T if al.shape[0] == 2 else al
            e = entry_frame(al)
            if e is None or e - WIN < 0 or e + WIN > raw.shape[1]:
                print("skip", name, e); continue
            pre = slice(e - WIN, e); post = slice(e, e + WIN)
            F0 = np.median(raw[:, pre], 1, keepdims=True)
            dff = (raw - F0) / F0
            hp = dff - median_filter(dff, size=(1, HP), mode="nearest")
            rows.append(dict(agonist=ag, conc=float(g["concentration"][()].ravel()[0]), name=name,
                             n_cells=raw.shape[0], entry=e,
                             act_pre=dff[:, pre].mean(), act_post=dff[:, post].mean(),
                             rho_hp_pre=rhobar(hp[:, pre]), rho_hp_post=rhobar(hp[:, post]),
                             rho_dt_pre=rhobar(detrend(dff[:, pre])), rho_dt_post=rhobar(detrend(dff[:, post]))))
    arrays = {}
    for ag in ("Baclofen", "tACPD"):
        R = [r for r in rows if r["agonist"] == ag]
        conc = np.array([r["conc"] for r in R])
        d = {k: np.array([r[f"{k}_post"] - r[f"{k}_pre"] for r in R]) for k in ("act", "rho_hp", "rho_dt")}
        pre = {k: np.array([r[f"{k}_pre"] for r in R]) for k in ("act", "rho_hp", "rho_dt")}
        print(f"\n{ag}: {len(R)} slices, cells/slice median {np.median([r['n_cells'] for r in R]):.0f}")
        for c in sorted(set(conc)):
            m = conc == c
            out = []
            for k in ("act", "rho_hp", "rho_dt"):
                x = d[k][m]; ci = np.percentile([x[rng.integers(0, m.sum(), m.sum())].mean() for _ in range(2000)], [2.5, 97.5])
                out.append(f"d{k} {x.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] (pre {pre[k][m].mean():.4f})")
            print(f"  {c:5.0f} uM n={m.sum()}  " + "  ".join(out))
        for k in ("act", "rho_hp", "rho_dt"):
            sr = spearmanr(conc, d[k]); print(f"  Spearman(dose, d{k}) = {sr.correlation:+.2f}  p={sr.pvalue:.3f}")
        sr = spearmanr(d["act"], d["rho_hp"]); print(f"  Spearman(dact, drho_hp) across slices = {sr.correlation:+.2f}  p={sr.pvalue:.3f}")
        arrays.update({f"{ag}_conc": conc, **{f"{ag}_d{k}": v for k, v in d.items()},
                       **{f"{ag}_pre_{k}": v for k, v in pre.items()},
                       f"{ag}_ncells": np.array([r["n_cells"] for r in R]), f"{ag}_entry": np.array([r["entry"] for r in R])})
    p = save_result(Path("processed_data") / "cahill2024_dose_tests.npz",
                    {"source": "Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j, ReceptorAgonistBathApp_Fig1",
                     "window_s": 120, "highpass_running_median_s": 60, "exclude": list(EXCLUDE),
                     "entry_rule": "Alexa594 >= mean+3SD of frames 1-300", "boot_seed": 20261006}, **arrays)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
