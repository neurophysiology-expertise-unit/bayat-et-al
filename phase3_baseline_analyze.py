"""Baseline-excitability readout: active fraction and rho-bar vs alpha, per I0 baseline.
Hypothesis check: does a mid baseline give LOW-alpha coordination (waves: strong D_eff + near-
threshold cells) that collapses at high alpha (fragmentation)? Fig 2 benchmark rho-bar ~0.08."""
import numpy as np, glob, os
N = 32 * 32
def rho(R): return (N * R ** 2 - 1) / (N - 1)
files = sorted(glob.glob("processed_data/phase3_baseline_b*.npz"))
data = {}
al = None
for f in files:
    z = np.load(f); b = float(f.split("_b")[1].replace(".npz", ""))
    data[b] = (z["active"].mean(0), rho(z["R"]).mean(0)); al = z["alphas"]
bs = sorted(data)
print("active fraction vs alpha, per baseline:")
print(f"{'alpha':>6} " + " ".join(f"b={b:.2f}" for b in bs))
for i in range(len(al)):
    print(f"{al[i]:6.3f} " + " ".join(f"{data[b][0][i]:6.3f}" for b in bs))
print("\nrho-bar (coordination) vs alpha, per baseline  (Fig2 ~0.08):")
print(f"{'alpha':>6} " + " ".join(f"b={b:.2f}" for b in bs))
for i in range(len(al)):
    print(f"{al[i]:6.3f} " + " ".join(f"{data[b][1][i]:6.4f}" for b in bs))
print("\nper-baseline: peak rho-bar and where; low-alpha (a<=0.23) mean rho-bar:")
lowmask = al <= 0.23
for b in bs:
    ac, rb = data[b]
    k = int(rb.argmax())
    print(f"  b={b:.2f}: peak rho-bar {rb.max():.4f} @a={al[k]:.3f} (act {ac[k]:.3f}) | "
          f"low-a mean rho {rb[lowmask].mean():.4f} act {ac[lowmask].mean():.3f} | "
          f"high-a(>=0.9) rho {rb[al>=0.9].mean():.4f}")
