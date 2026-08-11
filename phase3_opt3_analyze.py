"""Option 3 readout: rho-bar(alpha) + active(alpha) for bayat-I0 Sweep A under
  baseline (theta,kappa both alpha-dep) | theta=0.5 fixed | theta=0.5 + kappa_scale=0.25.
Published Fig 2 mean pairwise correlation ~0.08 at low ATP."""
import numpy as np
N = 32 * 32
def rho(R): return (N * R ** 2 - 1) / (N - 1)
jobs = [("baseline (theta&kappa alpha-dep)", "processed_data/phase2v1_bayat_L32_A_full.npz"),
        ("theta=0.5 fixed (kappa alpha-dep)", "processed_data/phase3_opt3_bayat_theta05.npz"),
        ("theta=0.5 + kappa_scale=0.25",      "processed_data/phase3_opt3_bayat_theta05_kap025.npz")]
al = np.load(jobs[0][1])["alphas"]
D = {lab: (rho(np.load(f)["R"]).mean(0), np.load(f)["active"].mean(0), np.load(f)["Deff"].mean(0))
     for lab, f in jobs}
print("bayat-I0 Sweep A, L=32, 40 seeds.  rho-bar / active / D_eff vs alpha. Fig2 ~0.08 low ATP.\n")
for lab, _ in jobs:
    r, a, d = D[lab]; k = int(r.argmax())
    print(f"  {lab}")
    print(f"    rho-bar: peak {r.max():.4f} @a={al[k]:.3f}   |  active peak {a.max():.4f}   "
          f"|  D_eff {d.max():.3f}->{d.min():.3f}")
print(f"\n{'alpha':>6} " + " ".join(f"{lab.split('(')[0].strip()[:12]:>13}" for lab,_ in jobs))
print("        " + " ".join(f"{'rho / act':>13}" for _ in jobs))
for i in range(len(al)):
    cells = " ".join(f"{D[lab][0][i]:6.4f}/{D[lab][1][i]:5.3f}" for lab, _ in jobs)
    print(f"{al[i]:6.3f} {cells}")
