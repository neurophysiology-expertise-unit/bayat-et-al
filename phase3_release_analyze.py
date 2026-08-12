"""Released-ATP grid readout: rho-bar and active fraction vs s, for diffusing (each D_atp) and the
uniform null, per alpha_base. Numbers before interpretation. Published Fig 2 ~0.08 at low ATP.
s=0 row = the corrected model (must match across all conditions). Attribution to spatial structure
requires diffusing rho-bar to rise while the uniform null does NOT."""
import numpy as np, glob, os
N = 32 * 32
def rho(R): return (N * R ** 2 - 1) / (N - 1)
for ab in ("a0.30", "a0.50"):
    print(f"\n===== alpha_base = {ab[1:]} =====")
    files = sorted(glob.glob(f"processed_data/phase3_release_{ab}_diff_D*.npz")) + \
            [f"processed_data/phase3_release_{ab}_uniform.npz"]
    conds = []
    for f in files:
        if not os.path.exists(f): continue
        z = np.load(f); S = z["s_values"]
        rb = rho(z["R"]).mean(0); ac = z["active"].mean(0); au = z["Aused"].mean(0)
        lab = "uniform-null" if "uniform" in f else "diffuse D=" + f.split("_D")[1].replace(".npz", "")
        conds.append((lab, S, rb, ac, au))
    print(f"  {'s':>5} | " + " | ".join(f"{c[0]:>22}" for c in conds))
    print(f"        | " + " | ".join(f"{'rho / act / Aused':>22}" for _ in conds))
    S0 = conds[0][1]
    for j in range(len(S0)):
        cells = " | ".join(f"{c[2][j]:7.4f} /{c[3][j]:6.3f} /{c[4][j]:6.3f}" for c in conds)
        print(f"  {S0[j]:5.2f} | {cells}")
    print(f"  (Fig2 benchmark rho-bar ~0.08; s=0 row must equal the corrected model & match across cols)")
    # peak diffusing vs uniform
    for lab, S, rb, ac, au in conds:
        print(f"    {lab:>14}: peak rho-bar {rb.max():.4f} @s={S[int(rb.argmax())]:.2f}  (s=0 rho={rb[0]:.4f})")
