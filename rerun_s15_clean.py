"""
rerun_s15_clean.py — re-stamp the nine `-dirty` results on a clean tree (pre-submission review S15).

The scripts behind these files are unchanged since the commits in their stamps (git diff empty); the
`-dirty` came from untracked siblings at save time. This re-runs each one by importing the ORIGINAL
njit functions from their own modules (no copies), with the parameters read back from the stored
__params__, asserts that every stored array is reproduced bit-for-bit (np.array_equal), and only then
overwrites the file through save_result. A mismatch raises before anything is written.

Files: constrain_map_dmult{0.25,0.5,2,4}_L32 (phase4_constrain.ensemble_c),
       ablate_{gamma,theta,all3}_L32 (phase4_ablate.ensemble_c),
       rhobar_vs_atp_L32_B_coupling_aref{0.10,0.90} (phase3_rhobar.ensemble_rb).

Run: python -s rerun_s15_clean.py   (refuses to start on a dirty tree)
"""
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from core.provenance import save_result, git_commit
import phase4_constrain as pc, phase4_ablate as pa, phase3_rhobar as pr

PD = Path("processed_data")


def check_and_save(name, params, arrays):
    z = np.load(PD / name)
    for k, v in arrays.items():
        assert np.array_equal(z[k], v), f"{name}: {k} not reproduced, max|d|={np.max(np.abs(z[k]-v)):.3e}"
    stored = json.loads(str(z["__params__"]))
    assert stored == json.loads(json.dumps(params, sort_keys=True, default=str)), f"{name}: params differ"
    save_result(PD / name, params, **arrays)
    print(f"{name}: identical, re-stamped {git_commit()}", flush=True)


def main():
    assert not git_commit().endswith("-dirty"), "tree is dirty; commit first"
    for m in (0.25, 0.5, 2.0, 4.0):
        name = f"constrain_map_dmult{m:g}_L32.npz"
        p = json.loads(str(np.load(PD / name)["__params__"]))
        t0 = time.time()
        A, P, D = pc.ensemble_c(np.array(p["seeds"]), pc.ALPHAS, p["fix_D"], p["a_D"], p["d_mult"],
                                int(p["T"] / DT), p["L"], p["L"], p["L"] ** 2, SIGMA_EM_PREDICTED,
                                I0_BASE, pc.STRIDE)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        check_and_save(name, p, dict(alphas=pc.ALPHAS, active=A, rhobar=P, Deff=D))
    for tag in ("gamma", "theta", "all3"):
        name = f"ablate_{tag}_L32.npz"
        p = json.loads(str(np.load(PD / name)["__params__"]))
        t0 = time.time()
        A, P, D = pa.ensemble_c(np.array(p["seeds"]), pa.ALPHAS, p["fix_D"], p["a_D"], p["d_mult"],
                                int(p["T"] / DT), p["L"], p["L"], p["L"] ** 2, SIGMA_EM_PREDICTED,
                                I0_BASE, pa.STRIDE, p["fix_noise"], p["uni_gamma"], p["fix_theta"])
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        check_and_save(name, p, dict(alphas=pa.ALPHAS, active=A, rhobar=P, Deff=D))
    for aref in (0.10, 0.90):
        name = f"rhobar_vs_atp_L32_B_coupling_aref{aref:.2f}.npz"
        z = np.load(PD / name); p = json.loads(str(z["__params__"]))
        alphas = np.asarray(z["alphas"]); L = p["L"]
        t0 = time.time()
        A, R, D, P = pr.ensemble_rb(np.array(p["seeds"]), p["mode"], alphas, p["alpha_ref"], p["steps"],
                                    L, L, L * L, p["sigma"], p["i0_baseline"], p["stride"])
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        N = L * L
        check_and_save(name, p, dict(alphas=alphas, active=A, R=R, Deff=D, rhobar=P,
                                     rho_proxy=(N * R ** 2 - 1.0) / (N - 1.0)))


if __name__ == "__main__":
    main()
