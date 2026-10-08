"""Graphical abstract: three properties of a calcium wave, three mechanisms.

Built entirely from the provenance-stamped npz files behind Figs 3 and 4 --
the same seeds, the same observables, the same 25 um per cell convention that
appears in the paper. Nothing here is redrawn by hand.

  (1,2) focal activation maps, deterministic vs noisy: native coupling carries a
        front; under noise the field ignites throughout and there is no front.
  (3)   spontaneous nucleation rate vs refractory duration.
  (4)   front extent vs decremental gain, against the reported wave extent.

Elsevier spec: minimum 1328 x 531 px (w x h), legible at 13 x 5 cm.
Run: python graphical_abstract.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotstyle import apply_style

UM = 25.0
COL_DET = "#1F4E79"
COL_NOISY = "#C0392B"
COL_HL = "#2E7D5B"
EXT_LO, EXT_HI = 100.0, 250.0        # reported astrocyte wave extent (um)


def main():
    apply_style()
    d3 = np.load("processed_data/fig3_focal.npz", allow_pickle=True)
    d4 = np.load("processed_data/fig4_mechanisms_ens.npz", allow_pickle=True)
    P3 = json.loads(str(d3["__params__"]))
    dtf, L = P3["dt_frame"], P3["L"]
    tact_d, tact_n = d3["tact_deterministic"], d3["tact_noisy"]
    taus, gam = d4["taus"], d4["gammas"]

    fig = plt.figure(figsize=(18 / 2.54, 6.4 / 2.54))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.15, 1.25], wspace=0.55,
                          left=0.045, right=0.985, top=0.80, bottom=0.20)

    # ---- (1,2) activation maps: a front, and no front -----------------------
    tmax = max(tact_d.max(), tact_n.max()) * dtf
    ext = [-L / 2 * UM, L / 2 * UM, -L / 2 * UM, L / 2 * UM]
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("0.88")
    for i, (tac, title, col) in enumerate(((tact_d, "a front", COL_DET),
                                           (tact_n, "no front", COL_NOISY))):
        ax = fig.add_subplot(gs[0, i])
        img = np.where(tac >= 0, tac.astype(float) * dtf, np.nan)
        ax.imshow(img, cmap=cm, vmin=0, vmax=tmax, extent=ext, origin="lower",
                  interpolation="nearest")
        ax.set_title(("deterministic: " if i == 0 else "with noise: ") + title,
                     fontsize=7.2, color=col, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("0.5")
    fig.text(0.205, 0.045, "focal initiation, native gap-junction coupling",
             ha="center", fontsize=6.8, color="0.3")

    # ---- (3) refractory state removes spurious nucleation -------------------
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(taus))
    ax.bar(x, d4["nucleation_rate_mean"], yerr=d4["nucleation_rate_sd"], width=0.6,
           color=[COL_NOISY] + [COL_DET] * (len(taus) - 1), capsize=2,
           error_kw=dict(elinewidth=0.8))
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.0f}" for t in taus], fontsize=6.5)
    ax.set_xlabel(r"refractory $\tau_{\rm ref}$ (s)", fontsize=7.2, labelpad=1)
    ax.set_ylabel("spurious ignitions\n" + r"(10$^{3}$ cells)$^{-1}$ s$^{-1}$",
                  fontsize=7.0, labelpad=2)
    ax.set_title("refractory state:\nsuppressed nucleation", fontsize=7.2, pad=3)
    ax.tick_params(labelsize=6.5, length=2.5, width=0.8, pad=1.5)
    ax.set_ylim(0, max(d4["nucleation_rate_mean"]) * 1.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (4) decremental release bounds the extent --------------------------
    ax = fig.add_subplot(gs[0, 3])
    ax.axhspan(EXT_LO / UM, EXT_HI / UM, color=COL_HL, alpha=0.18, lw=0)
    ax.axhline(EXT_LO / UM, color=COL_HL, lw=0.7); ax.axhline(EXT_HI / UM, color=COL_HL, lw=0.7)
    ax.errorbar(gam, d4["extent_vs_gamma_mean"], yerr=d4["extent_vs_gamma_sd"], fmt="-o",
                color=COL_DET, ecolor=COL_DET, ms=2.6, lw=1.3, elinewidth=0.8, capsize=2)
    ax.text(0.36, 0.09, "reported extent", transform=ax.transAxes, fontsize=6.2, color=COL_HL)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(r"regeneration factor $\gamma_{\rm regen}$", fontsize=7.2, labelpad=1)
    ax.set_ylabel("front extent (cells)", fontsize=7.2, labelpad=2)
    ax.set_title("decremental release:\ncontracted front extent", fontsize=7.2, pad=3)
    ax.tick_params(labelsize=6.5, length=2.5, width=0.8, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.5, 0.955, "Distinct mechanisms suppress ignition and bound extent",
             ha="center", fontsize=8.4, fontweight="bold")

    fig.savefig("graphical_abstract.pdf", dpi=600)
    fig.savefig("graphical_abstract.png", dpi=600)
    w, h = fig.get_size_inches() * 600
    print(f"saved graphical_abstract.pdf/.png  ->  {w:.0f} x {h:.0f} px "
          f"(Elsevier minimum 1328 x 531)")
    plt.close(fig)


if __name__ == "__main__":
    main()
