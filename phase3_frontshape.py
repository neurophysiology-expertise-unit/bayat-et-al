"""Front-shape analysis (2026-08-12): is the bounded (decremental) front COHERENT-but-decelerating,
or incoherent? The linear t_act-vs-radius test assumes CONSTANT speed; an attenuating front is
expected to decelerate, so a low linear R^2 may be the right physics failing the wrong test.

Per gamma_regen (focal, alpha=0.01, baseline=I0_BASE, tau_ref=15s, L=64):
  (a) LINEAR fit R^2  -- reported as-is, pre-registered threshold NOT adjusted
  (b) DECELERATING fit t_act = a*(1 - exp(-r/lambda)) -- R^2 and decay length lambda; does lambda
      track the extent?
  (c) LOCAL SPEED vs radius (binned) -- smooth decrease = coherent attenuating front;
      scatter = incoherent
  (d) INITIAL speed from the first two radial bins -- what an experimental measurement captures
Run: python phase3_frontshape.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import curve_fit
from core.model import DT, SIGMA_EM_PREDICTED, I0_BASE
from phase3_decremental import run_dr

UM = 50.0


def decel(r, a, lam):
    """t_act for a front whose local speed decays with radius; t->a as r->inf (front stalls)."""
    return a * (1.0 - np.exp(-r / lam))


def shape(fld, dtf, L, patch):
    T = fld.shape[0]; c = L // 2
    act = fld > 0.5
    tact = np.where(act.any(0), act.argmax(0), -1)
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c))
    dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2)
    m = (tact >= 0) & (r > patch)
    if m.sum() < 8:
        return None
    rr = r[m]; tt = tact[m].astype(float) * dtf
    extent = rr.max()
    # (a) linear fit
    sl, ic = np.polyfit(rr, tt, 1)
    r2_lin = 1 - np.sum((tt - (sl * rr + ic)) ** 2) / (np.sum((tt - tt.mean()) ** 2) + 1e-12)
    v_lin = (1.0 / sl) * UM if sl > 1e-9 else np.nan
    # (b) decelerating fit
    r2_dec = np.nan; lam = np.nan
    try:
        # bound lambda to the physical range (unbounded -> degenerates to linear-through-origin)
        p, _ = curve_fit(decel, rr, tt, p0=[tt.max() * 1.5, max(extent / 2, 1.0)],
                         bounds=([0.0, 0.3], [tt.max() * 100, max(extent * 3, 5.0)]), maxfev=40000)
        a_, lam = p
        r2_dec = 1 - np.sum((tt - decel(rr, a_, lam)) ** 2) / (np.sum((tt - tt.mean()) ** 2) + 1e-12)
    except Exception:
        pass
    # (c) local speed vs radius (binned means of t_act)
    edges = np.arange(patch, np.ceil(extent) + 1.0, 1.0)
    cent = []; mt = []
    for i in range(len(edges) - 1):
        b = (rr >= edges[i]) & (rr < edges[i + 1])
        if b.sum() >= 3:
            cent.append(0.5 * (edges[i] + edges[i + 1])); mt.append(tt[b].mean())
    cent = np.array(cent); mt = np.array(mt)
    loc = []
    for i in range(len(cent) - 1):
        dtv = mt[i + 1] - mt[i]
        loc.append(((cent[i + 1] - cent[i]) / dtv * UM) if dtv > 1e-9 else np.nan)
    v_init = loc[0] if loc else np.nan
    return dict(extent=extent, r2_lin=r2_lin, v_lin=v_lin, r2_dec=r2_dec, lam=lam,
                cent=cent, loc=np.array(loc), v_init=v_init)


def main():
    L = 64; sig = SIGMA_EM_PREDICTED; stride = 5; dtf = stride * DT; patch = 2
    steps = int(120.0 / DT)
    print("Front-shape analysis: linear (as-is) vs decelerating fit, focal alpha=0.01, tau_ref=15s.")
    print("Linear R^2 threshold NOT adjusted; reported alongside the decelerating fit.\n")
    print(f"  {'gamma':>6} {'extent':>7} {'v_init':>8} {'v_lin':>7} {'R2_lin':>7} {'R2_decel':>9} "
          f"{'lambda':>7} {'lam/ext':>8}")
    rows = {}
    for gr in (0.30, 0.20, 0.10):
        f = run_dr(11, 0.01, I0_BASE, gr, 15.0, False, steps, L, L, sig, stride, patch)
        s = shape(f, dtf, L, patch)
        if s is None:
            print(f"  {gr:6.2f}  (too few activated cells)"); continue
        rows[gr] = s
        print(f"  {gr:6.2f} {s['extent']:7.1f} {s['v_init']:8.1f} {s['v_lin']:7.1f} "
              f"{s['r2_lin']:7.3f} {s['r2_dec']:9.3f} {s['lam']:7.2f} {s['lam']/s['extent']:8.2f}",
              flush=True)
    # local speed vs radius detail at gamma=0.10
    for gr in (0.10, 0.20):
        if gr in rows:
            s = rows[gr]
            print(f"\n  local speed vs radius, gamma={gr:.2f} (smooth decrease = coherent attenuating "
                  f"front; scatter = incoherent):")
            print("    r(cells): " + " ".join(f"{x:6.1f}" for x in s['cent'][:-1]))
            print("    v(um/s) : " + " ".join(f"{x:6.1f}" for x in s['loc']))


if __name__ == "__main__":
    main()
