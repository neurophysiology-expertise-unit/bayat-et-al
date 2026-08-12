"""Lag-quality re-check (2026-08-12): is the refractory spontaneous field a TRAVELLING WAVE
(lag ~ distance, exponent b~1) or DIFFUSIVE spread (lag ~ distance^2, b~2)? The earlier grid used
max_lag=400 frames and the lags (33,89,241,396) saturated near that cap, so the apparent
superlinearity may be truncation. Redo with a large lag window and report, per separation d:
  peak lag, peak CORRELATION value (resolution quality), and local speed.
Fit log(lag)=log a + b log d -> exponent b with 95% CI over WELL-RESOLVED d only. Compare the
implied speed to the focal-initiation 13.3 um/s (which was linear in radius = a clean front).
alpha=0.01, tau_ref in {0,5,15}, L=64, T=300, noise ON. Run: python phase3_lagcheck.py
"""
import sys, time; sys.path.insert(0, '/mnt/sysfs01/users/cagatay/code/bayat-et-al')
import numpy as np
from core.model import DT
from phase3_refractory import run_ref

UM = 50.0

def lag_quality(A, dtf, L, max_lag):
    """per d: (peak_lag_frames, peakcorr). Large symmetric window; peak over positive lags."""
    T = A.shape[0]
    dA = (A - A.mean(0, keepdims=True)).reshape(T, L * L)
    var = np.mean(dA * dA)
    nfft = 1
    while nfft < 2 * T:
        nfft *= 2
    F = np.fft.rfft(dA, n=nfft, axis=0)
    ds = [1, 2, 3, 4, 6, 8, 10, 12]
    out = []
    for d in ds:
        s = np.roll(A.reshape(T, L, L), -d, axis=1).reshape(T, L * L)
        s = s - s.mean(0, keepdims=True)
        cc = np.fft.irfft(F * np.conj(np.fft.rfft(s, n=nfft, axis=0)), n=nfft, axis=0).sum(1)
        cc = cc / (var * T * L * L + 1e-12)
        pos = cc[:max_lag + 1]                       # positive lags only (propagation +x direction)
        k = int(np.argmax(pos))
        out.append((d, k, float(pos[k])))
    return out

def analyze(A, dtf, L, tag):
    T = A.shape[0]; max_lag = min(T // 2 - 1, 3000)
    q = lag_quality(A, dtf, L, max_lag)
    print(f"\n  [{tag}]  max_lag={max_lag} frames ({max_lag*dtf:.1f} model-t)")
    print(f"    {'d':>3} {'lag(fr)':>8} {'lag(s)':>8} {'peakcorr':>9} {'localspeed(um/s)':>16} {'resolved?':>10}")
    prev = None
    ref_corr = q[0][2]
    dd = []; ll = []
    for i, (d, lag, pc) in enumerate(q):
        lag_s = lag * dtf
        loc = ""
        if prev is not None:
            dl = (lag - prev[1]) * dtf
            loc = f"{(d-prev[0])/dl*UM:16.1f}" if dl > 1e-9 else f"{'inf/flat':>16}"
        else:
            loc = f"{'-':>16}"
        resolved = "yes" if (pc > 0.08 and pc > 0.25 * ref_corr and lag < max_lag * 0.95) else "NO"
        print(f"    {d:>3} {lag:>8} {lag_s:>8.2f} {pc:>9.4f} {loc} {resolved:>10}")
        if resolved == "yes" and lag > 0:
            dd.append(d); ll.append(lag)
        prev = (d, lag, pc)
    # power-law fit log(lag)=log a + b log d over resolved points
    if len(dd) >= 3:
        x = np.log(np.array(dd)); y = np.log(np.array(ll))
        n = len(x); b, loga = np.polyfit(x, y, 1)
        yhat = b * x + loga; resid = y - yhat
        se = np.sqrt(np.sum(resid**2) / (n - 2) / np.sum((x - x.mean())**2))
        ci = 1.96 * se
        interp = "travelling wave (b~1)" if abs(b - 1) < abs(b - 2) else "DIFFUSIVE (b~2)"
        print(f"    power-law exponent b = {b:.2f} +/- {ci:.2f} (95% CI, n={n} resolved pts) -> {interp}")
    else:
        print(f"    too few resolved points ({len(dd)}) for a power-law fit -> lag ill-defined = NOT a clean wave")

def main():
    L = 64; sig = __import__('core.model', fromlist=['SIGMA_EM_PREDICTED']).SIGMA_EM_PREDICTED
    baseline = 0.45; stride = 5; dtf = stride * DT; patch = 2
    print(f"Spontaneous-field lag quality, alpha=0.01, L={L}, T=300 (dt_frame={dtf:.4f} model-t=s).")
    print("Focal-initiation reference: 13.3 um/s, linear in radius.")
    for tau in (0.0, 5.0, 15.0):
        t0 = time.time()
        A = run_ref(11, 0.01, baseline, tau, True, False, int(300.0 / DT), L, L, sig, stride, patch).astype(np.float64)
        analyze(A, dtf, L, f"tau_ref={tau:.0f}s")
        print(f"    ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
