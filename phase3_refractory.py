"""Refractory-state test (2026-08-12) — does adding a refractory state convert zero-lag cluster
synchrony into a spatially bounded travelling front at physiological speed under noise?

Standard excitable-medium refractory (cf. Lallouette-De Pitta-Berry UAR): when a cell completes a
spike (falling edge of C_active through 0.5), start a timer tau_ref during which (a) its coupling
BROADCAST is gated off (refractory cells don't drive neighbours) and (b) a hyperpolarizing clamp
holds it near rest so it cannot re-fire. Then restore. Everything else = baseline=I0_BASE corrected
model. Sweep tau_ref, alpha in {0.01,0.12}, L=64.

Reports per (alpha, tau_ref): spontaneous lag-vs-distance (discriminator), focal front extent &
speed (deterministic + noisy). 1 cell=50um, 1 model-t=1s; wave 15-28 um/s, extent 2-5 cells.
Run: python phase3_refractory.py
"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from numba import njit
from core.model import (laplacian, DT, ETA, A_FHN, B_FHN, THETA_BASE, NOISE_MULT,
                        SIGMA_EM_PREDICTED, I0_BASE)

C_DOWN = -1.2; K_REF = 20.0


@njit(fastmath=True, cache=False)
def run_ref(seed, alpha, baseline, tau_ref, noise_on, do_focal, steps, nx, ny, sigma, stride, patch):
    np.random.seed(seed)
    gamma_base = (np.random.uniform(0.05, 0.34, (nx, ny))
                  * (1.0 + 2.0 * np.random.standard_normal((nx, ny))))
    I0_base = np.random.uniform(0.1, 0.5, (nx, ny))
    tau_base = np.random.uniform(0.5, 1.1, (nx, ny))
    D0_base = np.random.uniform(0.05, 0.5, (nx, ny))
    kappa_base = np.random.uniform(1.0, 4.0, (nx, ny))
    I0 = baseline + alpha * I0_base
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)
    Deff = D0_base / (1.0 + (kappa_base * alpha) ** 4)
    theta = THETA_BASE + 0.7 * alpha
    sigma_eff = sigma * (1.0 + 4.0 * alpha)
    gdrive = gamma_base * alpha
    if do_focal:
        C = np.full((nx, ny), C_DOWN); h = (C + A_FHN) / B_FHN
        c0 = nx // 2; c1 = ny // 2
        for i in range(c0 - patch, c0 + patch + 1):
            for j in range(c1 - patch, c1 + patch + 1):
                C[i, j] = 1.5
    else:
        C = np.random.uniform(-0.1, 0.3, (nx, ny)); h = np.random.uniform(0.4, 1.2, (nx, ny))
    reft = np.zeros((nx, ny)); prev = np.zeros((nx, ny))
    nframes = steps // stride
    fld = np.zeros((nframes, nx, ny), dtype=np.float32)
    sqrt_dt = DT ** 0.5; fi = 0
    for t in range(steps):
        C_active = 0.5 * (1.0 + np.tanh(ETA * (C - theta)))
        active = (C_active > 0.5).astype(np.float64)
        # falling edge (was active, now not) -> enter refractory
        fell = prev * (1.0 - active)
        reft = np.where(fell > 0.5, tau_ref, reft)
        inref = (reft > 0.0).astype(np.float64)
        broadcast = C_active * (1.0 - inref)                 # refractory cells don't broadcast
        diff = Deff * laplacian(broadcast)
        clamp = K_REF * inref * (C - C_DOWN)                 # hold refractory cells near rest
        if noise_on:
            noise = sigma_eff * NOISE_MULT * np.random.standard_normal((nx, ny))
        else:
            noise = np.zeros((nx, ny))
        dC = C - (C ** 3) / 3.0 - h + I0 + gdrive + diff - clamp
        dh = (C + A_FHN - B_FHN * h) / tau_h
        C = C + DT * dC + sqrt_dt * noise
        h = h + DT * dh
        C = np.minimum(np.maximum(C, -4.0), 4.0)
        reft = np.maximum(reft - DT, 0.0)
        prev = active
        if t % stride == 0 and fi < nframes:
            fld[fi] = (0.5 * (1.0 + np.tanh(ETA * (C - theta)))).astype(np.float32)
            fi += 1
    return fld


def lag_vs_distance(A, dtf, L):
    """FFT space-time cross-corr; return list of (d, peak_lag_frames, peakcorr)."""
    T = A.shape[0]
    dA = (A - A.mean(0, keepdims=True)).reshape(T, L * L)
    var = np.mean(dA * dA); max_lag = min(400, T // 3)
    nfft = 1
    while nfft < 2 * T:
        nfft *= 2
    F = np.fft.rfft(dA, n=nfft, axis=0)
    out = []
    for d in (1, 2, 3, 4, 6, 8):
        s = np.roll(A.reshape(T, L, L), -d, axis=1).reshape(T, L * L)
        s = s - s.mean(0, keepdims=True)
        cc = np.fft.irfft(F * np.conj(np.fft.rfft(s, n=nfft, axis=0)), n=nfft, axis=0).sum(1)
        pos = cc[:max_lag + 1]; neg = cc[-max_lag:]
        full = np.concatenate([neg, pos]); lags = np.arange(-max_lag, max_lag + 1)
        full = full / (var * T * L * L + 1e-12)
        k = int(np.argmax(full))
        out.append((d, int(lags[k]), float(full[k])))
    return out


def focal_stats(A, dtf, um, L):
    T = A.shape[0]; c = L // 2
    act = A > 0.5
    tact = np.where(act.any(0), act.argmax(0), -1)
    yy, xx = np.mgrid[0:L, 0:L]
    dx = np.minimum(np.abs(xx - c), L - np.abs(xx - c)); dy = np.minimum(np.abs(yy - c), L - np.abs(yy - c))
    r = np.sqrt(dx ** 2 + dy ** 2); reached = tact >= 0
    frac = reached.mean()
    max_r = r[reached].max() if reached.any() else 0.0
    rr = r[reached]; tt = tact[reached].astype(float) * dtf
    band = (rr >= 2) & (rr <= L // 2 - 2)
    speed_um = np.nan
    if band.sum() > 20 and np.ptp(tt[band]) > 1e-6:
        slope = np.polyfit(rr[band], tt[band], 1)[0]
        if slope > 1e-6:
            speed_um = (1.0 / slope) * um
    return frac, max_r, speed_um


def main():
    L = 64; sig = SIGMA_EM_PREDICTED; baseline = I0_BASE; stride = 5; patch = 2; um = 50.0
    dtf = stride * DT
    TAUS = [0.0, 5.0, 15.0, 30.0]          # tau_ref in model-time (s); 0 = control (no refractory)
    for alpha in (0.01, 0.12):
        print(f"\n===== alpha={alpha:.2f} =====")
        for tau in TAUS:
            # spontaneous, noise ON, for lag discriminator
            t0 = time.time()
            fs = run_ref(11, alpha, baseline, tau, True, False, int(200.0 / DT), L, L, sig, stride, patch)
            lag = lag_vs_distance(fs, dtf, L)
            lagstr = " ".join(f"d{d}:{pl}" for d, pl, _ in lag)
            mono = all(lag[i][1] <= lag[i + 1][1] for i in range(len(lag) - 1)) and lag[-1][1] > 0
            # focal deterministic + noisy for extent/speed
            fd = run_ref(11, alpha, baseline, tau, False, True, int(100.0 / DT), L, L, sig, stride, patch)
            fn = run_ref(11, alpha, baseline, tau, True, True, int(100.0 / DT), L, L, sig, stride, patch)
            frac_d, r_d, v_d = focal_stats(fd, dtf, um, L)
            frac_n, r_n, v_n = focal_stats(fn, dtf, um, L)
            print(f" tau_ref={tau:4.0f}s | spont lag[{lagstr}] {'PROPAGATING' if mono else 'zero-lag/synch'} "
                  f"| focal-det extent {r_d:.1f}cells({r_d*um:.0f}um) speed {v_d:.1f}um/s frac {frac_d*100:.0f}%"
                  f" | focal-noisy extent {r_n:.1f}cells speed {v_n:.1f}um/s frac {frac_n*100:.0f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
