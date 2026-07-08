"""
Supplementary Figure S1: disease - healthy contrast with bootstrap CIs.

For each observable (spatial heterogeneity S_C, susceptibility chi, coherence
length xi) we form the paired difference Delta = disease - healthy at every ATP
level and estimate a 95% confidence interval by resampling seeds (a paired
hierarchical bootstrap; healthy and disease share the same 20 seeds). ATP levels
where the 95% CI excludes zero are marked: this makes explicit *where* each
observable separates the two conditions, and confirms that xi is the least
discriminating of the three.

Reads the cached per-seed stacks written by fig_3_criticality_ci.py
(processed_data/fig3_ci.npz), so it never re-simulates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotstyle import apply_style, clean_spines, save_fig

PROC_DIR = "processed_data"
FIG3_CACHE = os.path.join(PROC_DIR, "fig3_ci.npz")
OUT_CSV = os.path.join(PROC_DIR, "figS1_delta.csv")

OBS = ("Sc", "chi", "corr")
TITLES = {
    "Sc": r"(A) $\Delta S_C$",
    "chi": r"(B) $\Delta \chi_{ext}$",
    "corr": r"(C) $\Delta \xi$",
}
N_BOOT = 5000
SEED = 0


def paired_bootstrap(H, D, n_boot=N_BOOT, rng=None):
    """
    H, D : (n_seeds, n_alpha) paired seed stacks (same seeds in both).
    Returns mean Delta, lo95, hi95 across ATP from resampling seeds.
    """
    rng = rng or np.random.default_rng(SEED)
    diff = D - H                                  # (n_seeds, n_alpha), paired
    ns = diff.shape[0]
    mean = diff.mean(axis=0)
    boots = np.empty((n_boot, diff.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, ns, ns)
        boots[b] = diff[idx].mean(axis=0)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return mean, lo, hi


def main(save_stem="Figure_S1"):
    if not os.path.exists(FIG3_CACHE):
        raise FileNotFoundError(
            f"{FIG3_CACHE} not found - run `python fig_3_criticality_ci.py --recompute` first.")
    d = np.load(FIG3_CACHE, allow_pickle=True)
    alpha = d["alpha"]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    rows = []
    for ax, obs in zip(axes, OBS):
        H = d[f"H_{obs}_stack"]
        D = d[f"D_{obs}_stack"]
        mean, lo, hi = paired_bootstrap(H, D)
        sig = (lo > 0) | (hi < 0)                 # 95% CI excludes zero

        ax.axhline(0, color="0.5", linewidth=1.0, linestyle="--")
        ax.plot(alpha, mean, color="#333333", linewidth=2)
        ax.fill_between(alpha, lo, hi, color="#999999", alpha=0.35, linewidth=0)
        # mark significant ATP levels
        if sig.any():
            y = ax.get_ylim()[1]
            ax.plot(alpha[sig], np.full(sig.sum(), y), marker="v", linestyle="none",
                    color="#D55E00", markersize=5, clip_on=False)
        ax.set_title(TITLES[obs])
        ax.set_xlabel(r"ATP level $\alpha$")
        ax.set_xlim(alpha.min(), alpha.max())
        clean_spines(ax)

        for k, al in enumerate(alpha):
            rows.append({"observable": obs, "alpha": al, "delta_mean": mean[k],
                         "ci_lo": lo[k], "ci_hi": hi[k], "significant": bool(sig[k])})

    axes[0].set_ylabel("disease - healthy")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    save_fig(fig, save_stem)
    print(f"Saved {save_stem}.pdf / .png ; contrast table -> {OUT_CSV}")
    plt.show()


if __name__ == "__main__":
    main()
