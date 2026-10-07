"""
data_cahill2024_cbx.py — test 5: does drive-induced slice synchrony survive gap-junction block?

Data: Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j, FIJIAQuA_CytoGCaMP_BathAppBacLY_ExtFig1k.mat:
9 acute V1 slices (3 mice), TTX + carbenoxolone (CBX, 100 uM), bath baclofen and LY379268 (selective mGluR2/3
agonist) at 100 uM in turn; per-cell FIJI traces; agonist-entry frame given by the authors
(Alexa594 'FirstFrameAboveBL', same rule as test 4).

Written BEFORE the analysis. Comparator: test 4 (data_cahill2024_dose.py), no CBX, t-ACPD 50-100 uM: 3 of 8 slices
switched to strong synchrony (change in fluctuation rho-bar +0.53 to +0.65), the rest < +0.08.
Prediction: if the synchrony is carried by gap-junction coupling, LY + CBX gives no slice with a change above
+0.2; if it is carried by a non-junctional route (released messenger, common drive), some slices still
switch. Caveats fixed in advance: LY379268 is not t-ACPD (no LY-without-CBX arm exists), so a NEGATIVE result
cannot separate the agonist from the block; only a positive result is decisive.

Method: identical to data_cahill2024_dose.py (120 s windows around entry, dF/F with F0 = median of pre window,
rho-bar of fluctuations after a 60 s running-median high-pass = primary, linearly detrended = secondary).
Run: python data_cahill2024_cbx.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, h5py
from scipy.ndimage import median_filter
from core.provenance import save_result
from data_cahill2024_dose import rhobar, detrend, WIN, HP

F = Path("/mnt/sysfs01/users/cagatay/external/public/cahill2024/dryad/"
         "FIJIAQuA_CytoGCaMP_BathAppBacLY_ExtFig1k.mat")


def main():
    f = h5py.File(F, "r"); md = f["mydata"]; arrays = {}
    for ag in ("Baclofen", "LY379268"):
        rows = []
        for i in range(md[ag].shape[0]):
            g = f[md[ag][i, 0]]
            name = "".join(map(chr, g["file"][()].ravel())).split("\\")[-1]
            raw = np.asarray(g["ImageJByRegion"]["RawTrace"][()], float)
            e = int(g["Alexa594"]["FirstFrameAboveBL"][()].ravel()[0])
            pre = slice(e - WIN, e); post = slice(e, e + WIN)
            F0 = np.median(raw[:, pre], 1, keepdims=True); dff = (raw - F0) / F0
            hp = dff - median_filter(dff, size=(1, HP), mode="nearest")
            rows.append((name, raw.shape[0], dff[:, post].mean() - dff[:, pre].mean(),
                         rhobar(hp[:, post]) - rhobar(hp[:, pre]), rhobar(hp[:, pre]),
                         rhobar(detrend(dff[:, post])) - rhobar(detrend(dff[:, pre]))))
        print(f"\n{ag} + CBX, {len(rows)} slices")
        for r in rows:
            print(f"  {r[0][:44]:44s} cells {r[1]:3d}  dact {r[2]:+.3f}  drho_hp {r[3]:+.3f} (pre {r[4]:+.4f})  drho_dt {r[5]:+.3f}")
        d = np.array([r[3] for r in rows])
        print(f"  slices with drho_hp > +0.2: {np.sum(d > 0.2)}/{len(d)}   mean {d.mean():+.3f}")
        arrays.update({f"{ag}_dact": np.array([r[2] for r in rows]), f"{ag}_drho_hp": d,
                       f"{ag}_pre_rho_hp": np.array([r[4] for r in rows]),
                       f"{ag}_drho_dt": np.array([r[5] for r in rows]), f"{ag}_ncells": np.array([r[1] for r in rows])})
    p = save_result(Path("processed_data") / "cahill2024_cbx_tests.npz",
                    {"source": "Cahill et al. 2024, Dryad 10.5061/dryad.83bk3jb0j, BathAppBacLY_ExtFig1k (CBX)",
                     "window_s": 120, "highpass_running_median_s": 60, "threshold_switch": 0.2}, **arrays)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
