"""
paperC_clip.py — paper C, NEXT ACTION 2: the Q4 clipping intervention re-run at the EMPIRICAL sink threshold.
(Not part of paper A.)

Feasibility Q4 (paperC_mechanism M4) clipped every site whose D_i exceeded the quasi-static M_sus = 1.29 and
restored propagation in 0/10 seeds at d = 4, 5, 6 (gamma_regen = 1). The per-cell 'full first episode' fraction
(M3) crossed 50 % at M_emp = 0.678 of x = D_i*g_i (g = 1 at gamma = 1), not at 1.29. Here the same intervention
uses M_emp, read from processed_data/paperC_mechanism.npz (not re-estimated).

MODEL: paperC_model.draw + run_gen (verbatim copies; run_gen(draw) == run_drm asserted in paperC_model.check()),
heterogeneous lattice, deterministic, alpha 0.01, tau_ref 15 s, gamma_regen = 1, L = 64, 120 s, 5x5 seed,
seeds 11-20, d in {2.5, 3, 4, 5, 6}.
 CLIP:   D_i -> 0.95*M_emp wherever D_i > M_emp (all other draws unchanged), exactly as M4 with M_sus -> M_emp.
 MATCH:  control for the drop in mean coupling that clipping causes: every D_i scaled by
         mean(D_clipped)/mean(D) (disorder kept, high sites kept, same mean as CLIP).
 Propagates := t99 >= 0.9*T_window (the Q4 criterion); 'advancing' (t99 >= 0.9 T or extent >= 0.95 r_max) also
 reported.

PRE-REGISTERED PREDICTIONS (written 2026-10-12 before any run of this script):
 Q4'. (the original Q4 criterion, threshold changed only) CLIP restores propagation: >= 5/10 seeds propagate at
      each of d = 4, 5, 6.
 S.   Specificity. The statement 'the high-coupling sites cause the gamma = 1 stop' is supported at a given d
      only if CLIP propagates in >= 5/10 seeds AND MATCH propagates in <= 2/10. If MATCH also propagates, the
      clip result is explained by the lower mean coupling and is not evidence for the high-D-site mechanism.
      (Expectation, stated so it can be checked: at d >= 4 most sites are clipped (D_i > 0.678 for ~70 % of
      sites at d = 4), so CLIP is close to a homogeneous lattice at D ~ 0.64, which propagates; we expect Q4'
      to pass and S to be decided by MATCH.)
Failed predictions are reported as failed; criteria are not revised after the data land.

Output: processed_data/paperC_clip.npz.   Run: python -s paperC_clip.py   (about 10 min on 11 cores)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from core.provenance import save_result
from paperC_model import (draw, run_gen, extent_t99, check, ALPHA, I0_BASE, THETA, TAU_REF, STEPS, STRIDE, L,
                          PATCH, SEEDS, T_WIN)

D_CLIP = [2.5, 3.0, 4.0, 5.0, 6.0]
M_EMP = float(np.load(pathlib.Path(__file__).resolve().parent / "processed_data" / "paperC_mechanism.npz")["M_emp"])


def seedmask():
    m = np.zeros((L, L), np.bool_); c = L // 2; m[c - PATCH:c + PATCH + 1, c - PATCH:c + PATCH + 1] = True
    return m


def job(args):
    d, seed, mode = args
    I0, th, De, gd, D0 = draw(seed, ALPHA, I0_BASE, L, L, d)
    Dc = np.where(De > M_EMP, 0.95 * M_EMP, De)
    if mode == "clip":
        Du = Dc
    elif mode == "match":
        Du = De * (Dc.mean() / De.mean())
    else:
        Du = De
    fld = run_gen(I0, th, Du, gd, THETA, seedmask(), 1.0, TAU_REF, STEPS, STRIDE)[0]
    e, t = extent_t99(fld, L)
    return e, t, float((De > M_EMP).mean()), float(Du.mean()), float(De.mean())


def main():
    check()
    print(f"M_emp = {M_EMP:.4f}", flush=True)
    modes = ["clip", "match", "none"]
    jobs = [(d, s, m) for m in modes for d in D_CLIP for s in SEEDS]
    with Pool(11) as p:
        R = np.array(p.map(job, jobs, chunksize=1)).reshape(len(modes), len(D_CLIP), len(SEEDS), 5)
    pm = np.load(pathlib.Path(__file__).resolve().parent / "processed_data" / "paperC_phase_map.npz")
    for i, d in enumerate(D_CLIP):      # unmodified runs must equal the stored phase map at gamma = 1, exactly
        k = list(pm["d_mults"]).index(d)
        assert np.array_equal(R[2, i, :, 0], pm["extent"][k, 0]) and np.array_equal(R[2, i, :, 1], pm["stop_time"][k, 0]), d
    print("unmodified runs reproduce paperC_phase_map.npz (gamma = 1) exactly", flush=True)
    rmax = (L // 2) * np.sqrt(2)
    prop = R[..., 1] >= 0.9 * T_WIN
    adv = prop | (R[..., 0] >= 0.95 * rmax)
    q4 = {d: int(prop[0, i].sum()) for i, d in enumerate(D_CLIP)}
    for i, d in enumerate(D_CLIP):
        print(f"d {d:3.1f}: clipped frac {R[0,i,:,2].mean():.2f}  mean D {R[0,i,0,4]:.3f} -> {R[0,i,:,3].mean():.3f} | "
              + " | ".join(f"{m}: ext {R[k,i,:,0].mean():5.1f}+-{R[k,i,:,0].std():4.1f}  prop {prop[k,i].sum()}/10"
                           f"  adv {adv[k,i].sum()}/10" for k, m in enumerate(modes)), flush=True)
    q4p = all(q4[d] >= 5 for d in (4.0, 5.0, 6.0))
    spec = {d: bool(prop[0, i].sum() >= 5 and prop[1, i].sum() <= 2) for i, d in enumerate(D_CLIP)}
    print(f"Q4' (clip >= 5/10 at d 4,5,6): {'PASS' if q4p else 'FAIL'}  {q4}")
    print(f"S (clip >= 5/10 and match <= 2/10): {spec}")
    out = save_result(Path("processed_data") / "paperC_clip.npz",
                      {"model": "paperC_model.draw + run_gen (verbatim), heterogeneous, noise off, gamma_regen 1",
                       "M_emp": M_EMP, "clip_to": 0.95 * M_EMP, "d": D_CLIP, "seeds": SEEDS, "modes": modes,
                       "L": L, "T_window": T_WIN, "propagates": "t99 >= 0.9*T_window",
                       "Q4prime_pass": q4p, "S": spec},
                      d=np.array(D_CLIP), extent=R[..., 0], t99=R[..., 1], frac_clipped=R[0, :, :, 2],
                      mean_D_used=R[..., 3], mean_D_orig=R[..., 4], propagates=prop, advancing=adv)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
