"""
fig_slice_data.py — published slice data against the model's focal-front prediction.

Reads processed_data/cahill2024_uncaging_tests.npz (data_cahill2024_uncaging.py; Cahill et al. 2024,
Dryad 10.5061/dryad.83bk3jb0j). Panels:
  A  excess event rate, 60 s after vs 60 s before uncaging, versus distance (mean, 95% bootstrap CI)
  B  event-rate time course, wild type by distance band and laser-only control (10 s bins), each
     relative to its own last 60 s before uncaging
  C  cumulative excess events after uncaging, wild type by band (95% CI), with the time the model's
     deterministic front would take to reach each band from the nearest (6.3 um/s, Fig. 3A)
Run: python fig_slice_data.py   ->  Figure_slice.pdf / .png
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import apply_style, clean_spines, panel_label, save_fig

V_MODEL = 6.3                                   # um/s, deterministic front, Fig. 3A (25 um spacing)
apply_style()
d = np.load("processed_data/cahill2024_uncaging_tests.npz", allow_pickle=True)
bands = d["bands"]; mid = bands.mean(1)
COND = [("wt_both", "wild type", "#1F4E79", "-o"),
        ("cbx_both", "carbenoxolone", "#C0392B", "-s"),
        ("cx43fl_both", "Cx43 knockdown", "#E08E0B", "-^"),
        ("laser_control_None", "laser only", "0.55", "--D")]
BCOL = ["#1F4E79", "#4A90C2", "#9CC3E6"]

fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5), gridspec_kw=dict(wspace=0.38))

a = ax[0]
for k, lab, col, fmt in COND:
    e = d[f"{k}_m60"].mean(0); ci = d[f"{k}_m60ci"]
    a.errorbar(mid, e, yerr=[e - ci[0], ci[1] - e], fmt=fmt, color=col, ms=4, lw=1.4,
               capsize=2, elinewidth=0.9, label=f"{lab} (n={d[f'{k}_m60'].shape[0]})")
a.axhline(0, color="0.6", lw=0.7)
a.set_xticks(mid); a.set_xticklabels([f"{b0}–{b1}" for b0, b1 in bands])
a.set_xlabel(r"distance from uncaging site ($\mu$m)"); a.set_ylabel("excess events min$^{-1}$")
a.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
clean_spines(a); panel_label(a, "A")

b = ax[1]
edges = d["psth_edges"]; c = (edges[:-1] + edges[1:]) / 2
for j, col in enumerate(BCOL):
    H = d["wt_both_psth"][j]; b.plot(c, H - H[(c < 0) & (c > -60)].mean(), "-", color=col, lw=1.5,
                                     label=f"{bands[j][0]}–{bands[j][1]} $\\mu$m")
H = d["laser_control_None_psth"].mean(0)
b.plot(c, H - H[(c < 0) & (c > -60)].mean(), "-", color="0.6", lw=1.2, label="laser only (all)")
b.axvline(0, color="k", ls=":", lw=0.9); b.axhline(0, color="0.6", lw=0.7)
b.set_xlabel("time from uncaging (s)"); b.set_ylabel("excess events min$^{-1}$")
b.legend(fontsize=6.6, loc="upper left", frameon=False)
clean_spines(b); panel_label(b, "B")

g = ax[2]
tc = d["tcum"]
for j, col in enumerate(BCOL):
    m = d["wt_both_cum"][j]; ci = d["wt_both_cumci"][:, j]
    g.fill_between(tc, ci[0], ci[1], color=col, alpha=0.15, lw=0)
    g.plot(tc, m, "-", color=col, lw=1.5)
    if j:
        g.axvline((mid[j] - mid[0]) / V_MODEL, color=col, ls="--", lw=1.0)
g.axhline(0, color="0.6", lw=0.7)
g.text(0.03, 0.97, "dashed: model front would\nreach this band from the nearest", transform=g.transAxes,
       ha="left", va="top", fontsize=6.4, color="0.3")
g.set_xlabel("time from uncaging (s)"); g.set_ylabel("cumulative excess events per recording")
g.set_xlim(0, 90)
clean_spines(g); panel_label(g, "C")

save_fig(fig, "Figure_slice")
print("wrote Figure_slice.pdf / Figure_slice.png")
