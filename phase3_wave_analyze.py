"""Is the low-alpha coordination a PROPAGATING WAVEFRONT or cluster synchrony? Discriminators on the
saved C_active field (b=0.45, L=64):
 1. Space-time correlation lag vs distance: for spatial separation d, the temporal lag of peak
    cross-correlation. Wave -> lag rises linearly with d (speed=d/lag); synchrony -> lag~0 for all d.
 2. Propagation speed: from the lag-distance slope, in cells/model-time -> um/s (1 cell=50um,
    1 model-unit=1s). Compare to measured 15-28 um/s (=0.3-0.56 cells/s). Falsifiable, not a fit.
 3. Extent / wraparound: spatial correlation length; does correlation persist at d=L/2 (periodic
    wrap = unphysical over-propagation, the Bennett/Gibson-Farnell failure mode)?
 4. Fig-2D analogue: activity heatmap of a 1D transect (row) vs time.
Run: python phase3_wave_analyze.py"""
import numpy as np
from pathlib import Path

def analyze(path):
    z = np.load(path); A = z["field"].astype(np.float64)     # (T, L, L)
    a = float(z["alpha"]); dtf = float(z["dt_frame"]); um = float(z["cell_um"]); L = int(z["L"])
    T = A.shape[0]
    print(f"\n===== alpha={a:.2f}  (field {A.shape}, dt_frame={dtf:.4f} model-units, 1 cell={um:.0f}um) =====")
    print(f"  mean activity {A.mean():.4f}")
    dA = (A - A.mean(axis=0, keepdims=True)).reshape(T, L * L)  # de-mean each cell in time
    var = np.mean(dA * dA)
    max_lag = min(400, T // 3)                                # frames of interest
    # FFT cross-correlation: all lags at once, summed over cells. Zero-pad to avoid circular wrap.
    nfft = 1
    while nfft < 2 * T:
        nfft *= 2
    F = np.fft.rfft(dA, n=nfft, axis=0)                        # (freq, N)
    print(f"  {'d(cells)':>8} {'peak_lag(frames)':>16} {'lag(model-t)':>13} {'peakcorr':>9}")
    ds = [1, 2, 3, 4, 6, 8, L // 2]
    lag_d = []
    Aroll = A.reshape(T, L, L)
    for d in ds:
        sroll = np.roll(Aroll, -d, axis=1).reshape(T, L * L)  # neighbor at +d in x (periodic)
        sroll = sroll - sroll.mean(0, keepdims=True)
        Fs = np.fft.rfft(sroll, n=nfft, axis=0)
        cc_full = np.fft.irfft(F * np.conj(Fs), n=nfft, axis=0).sum(axis=1)   # summed over cells
        # cc_full[k] = sum_t dA[t]*sroll[t+k] for k=0..; negative lags at tail
        pos = cc_full[:max_lag + 1]
        neg = cc_full[-max_lag:][::-1]                         # lag = -1..-max_lag
        cc = np.concatenate([neg[::-1], pos])                 # lag from -max_lag..+max_lag
        lags = np.arange(-max_lag, max_lag + 1)
        cc = cc / (var * T * L * L + 1e-12)
        kbest = int(np.argmax(cc)); pl = int(lags[kbest])
        lag_d.append((d, pl, cc[kbest]))
        tag = " <- L/2 (wraparound check)" if d == L // 2 else ""
        print(f"  {d:8d} {pl:16d} {pl*dtf:13.3f} {cc[kbest]:9.4f}{tag}")
    # speed from small-d slope (use d=1..6 where a front would be coherent)
    core = [(d, pl) for d, pl, _ in lag_d if d <= 6 and abs(pl) > 0]
    if len(core) >= 2:
        dd = np.array([c[0] for c in core]); ll = np.array([c[1] for c in core]) * dtf  # model-time
        if np.ptp(ll) > 1e-9:
            slope = np.polyfit(ll, dd, 1)[0]                  # cells per model-time
            speed_um = slope * um                             # um/s (1 model-t=1s)
            print(f"  --> front speed ~ {slope:.3f} cells/model-t = {speed_um:.1f} um/s "
                  f"(measured 15-28 um/s){'  IN RANGE' if 15 <= speed_um <= 28 else '  OUT OF RANGE'}")
        else:
            print("  --> lags ~ constant across distance => CLUSTER SYNCHRONY, no propagating front")
    else:
        print("  --> peak lag ~0 at all distances => CLUSTER SYNCHRONY / zero-lag, NOT a travelling wave")
    # wraparound: correlation at L/2 relative to d=1
    c1 = [c for d, _, c in lag_d if d == 1][0]; chalf = [c for d, _, c in lag_d if d == L // 2][0]
    print(f"  spatial corr: peakcorr(d=1)={c1:.4f}, peakcorr(d=L/2)={chalf:.4f}  "
          f"ratio={chalf/(c1+1e-12):.3f} {'(persists at L/2 = WRAPS, over-propagation)' if chalf/(c1+1e-12) > 0.5 else '(decays = bounded extent)'}")
    # extent: distance where peakcorr falls to half of d=1
    return A, a, dtf, um

def heatmap(A, a, path):
    """Fig-2D analogue: activity of the middle row (64 cells) vs time, ASCII, downsampled."""
    L = A.shape[1]; row = A[:, L // 2, :]                     # (T, L)
    tds = max(1, row.shape[0] // 60)                          # ~60 time columns
    small = row[::tds].T                                      # (L cells, ~60 t)
    chars = " .:-=+*#%@"
    hi = small.max() + 1e-9
    print(f"\n  Fig-2D analogue (alpha={a:.2f}): middle-row activity, cell(row) x time(col). "
          f"diagonal stripes = wavefront; vertical = synchrony; speckle = independent")
    for i in range(0, L, 2):                                  # every other cell to fit
        line = "".join(chars[min(len(chars) - 1, int(v / hi * (len(chars) - 1)))] for v in small[i])
        print(f"  {i:2d}|{line}")

def main():
    for a in ("0.01", "0.12"):
        p = Path(f"processed_data/phase3_wavefield_b0.45_a{a}_L64.npz")
        if not p.exists():
            print(f"missing {p}"); continue
        A, av, dtf, um = analyze(p)
        heatmap(A, av, p)

if __name__ == "__main__":
    main()
