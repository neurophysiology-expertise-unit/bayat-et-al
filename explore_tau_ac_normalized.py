"""
EXPLORATION (not a paper figure) -- is the disease vs healthy difference in the
critical-slowing indicator tau_ac an INDEPENDENT effect, or is it inherited from
the imposed recovery timescale tau_h (which disease triples by hand)?

The disease parametrization sets tau_h -> 3*tau_h. Since tau_ac (the autocorrelation
/ relaxation time of the population signal) is set to leading order by the slow
recovery variable's timescale, a longer disease tau_ac is largely BUILT IN. To
separate inherited from genuine we rescale onto a common dynamic range and compare
SHAPE, not absolute value:

  (A) raw tau_ac(alpha)                      -- reproduces Fig S3B (confounded)
  (B) tau_ac(alpha) / tau_h(alpha)           -- divide out the imposed recovery time
                                                (tau_h analytic, mean tau_base=0.8;
                                                 disease x3). Collapse => pure confound.
  (C) tau_ac / baseline, transitions aligned -- each condition normalised by its own
                                                far-from-transition baseline and shifted
                                                so its chi peak sits at 0. Collapse of
                                                the near-transition PEAK => the slowing
                                                signature has the same shape in both.

Reads the cached S3 means (processed_data/figS3_slowing.npz); no re-simulation.
Run:  python explore_tau_ac_normalized.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = os.path.join("processed_data", "figS3_slowing.npz")

# tau_h(alpha) = 10 / ((1 + 0.8*alpha) * tau_base); mean tau_base = (0.5+1.1)/2 = 0.8
TAU_BASE_MEAN = 0.8
DISEASE_TAUH_FACTOR = 3.0


def tau_h_mean(alpha, disease):
    th = 10.0 / ((1.0 + 0.8 * alpha) * TAU_BASE_MEAN)
    return th * DISEASE_TAUH_FACTOR if disease else th


def smooth(x, w=3):
    x = np.asarray(x, float)
    k = np.ones(w)
    return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")


def main():
    d = np.load(CACHE, allow_pickle=True)
    alpha = np.asarray(d["alpha"])
    tauH = np.asarray(d["H_tau_mean"]); tauH_ci = np.asarray(d["H_tau_ci"])
    tauD = np.asarray(d["D_tau_mean"]); tauD_ci = np.asarray(d["D_tau_ci"])
    chiH = np.asarray(d["H_chi_mean"]); chiD = np.asarray(d["D_chi_mean"])

    # transition loci = each condition's own chi peak
    aH = alpha[int(np.argmax(chiH))]
    aD = alpha[int(np.argmax(chiD))]

    # (B) divide out imposed tau_h
    thH = tau_h_mean(alpha, False)
    thD = tau_h_mean(alpha, True)
    rH = tauH / thH
    rD = tauD / thD

    # (C) baseline-normalise (baseline = mean of far-from-transition tail, high alpha)
    baseH = tauH[-5:].mean()
    baseD = tauD[-5:].mean()
    nH = tauH / baseH
    nD = tauD / baseD

    # ---- quantitative summary ----
    print("=== tau_ac confound test (from cached S3 means) ===")
    print(f"transition (chi peak):  healthy alpha={aH:.3f}   disease alpha={aD:.3f}")
    print(f"baseline tau_ac (high-alpha tail):  H={baseH:.3f}  D={baseD:.3f}  "
          f"ratio D/H={baseD/baseH:.2f}   (imposed tau_h ratio = {DISEASE_TAUH_FACTOR:.0f})")
    pkH = tauH.max(); pkD = tauD.max()
    print(f"peak tau_ac:  H={pkH:.3f}  D={pkD:.3f}  ratio D/H={pkD/pkH:.2f}")
    print(f"peak-over-baseline enhancement:  H={pkH/baseH:.2f}x  D={pkD/baseD:.2f}x")
    print(f"tau_ac/tau_h  mean over sweep:  H={rH.mean():.3f}  D={rD.mean():.3f}  "
          f"ratio D/H={rD.mean()/rH.mean():.2f}")
    print(f"tau_ac/tau_h  max abs H-D difference across alpha: {np.max(np.abs(rH-rD)):.3f} "
          f"(relative to H mean {rH.mean():.3f})")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    cH, cD = "#0072B2", "#D55E00"

    ax[0].plot(alpha, smooth(tauH), color=cH, lw=2, label="Healthy")
    ax[0].fill_between(alpha, smooth(tauH - tauH_ci), smooth(tauH + tauH_ci), color=cH, alpha=.15, lw=0)
    ax[0].plot(alpha, smooth(tauD), color=cD, lw=2, label="Disease")
    ax[0].fill_between(alpha, smooth(tauD - tauD_ci), smooth(tauD + tauD_ci), color=cD, alpha=.15, lw=0)
    ax[0].axvline(aH, color="0.6", ls=":", lw=1)
    ax[0].set_title(r"(A) Raw $\tau_{ac}$ (confounded)")
    ax[0].set_ylabel(r"$\tau_{ac}$ (model time)"); ax[0].legend(frameon=False)

    ax[1].plot(alpha, smooth(rH), color=cH, lw=2, label="Healthy")
    ax[1].plot(alpha, smooth(rD), color=cD, lw=2, label="Disease")
    ax[1].axvline(aH, color="0.6", ls=":", lw=1)
    ax[1].set_title(r"(B) $\tau_{ac}/\tau_h$ (imposed timescale removed)")
    ax[1].set_ylabel(r"$\tau_{ac}/\tau_h$")

    ax[2].plot(alpha - aH, smooth(nH), color=cH, lw=2, label="Healthy")
    ax[2].plot(alpha - aD, smooth(nD), color=cD, lw=2, label="Disease")
    ax[2].axvline(0.0, color="0.6", ls=":", lw=1)
    ax[2].set_title(r"(C) $\tau_{ac}/$baseline, transitions aligned")
    ax[2].set_ylabel(r"$\tau_{ac}/\tau_{ac}^{\rm base}$")
    ax[2].set_xlabel(r"$\alpha - \alpha_{\rm transition}$")

    for a in (ax[0], ax[1]):
        a.set_xlabel(r"ATP level $\alpha$")

    fig.savefig("explore_tau_ac_normalized.png", dpi=150)
    print("Saved explore_tau_ac_normalized.png")


if __name__ == "__main__":
    main()
