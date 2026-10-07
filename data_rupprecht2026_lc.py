"""
data_rupprecht2026_lc.py — test 8: does a causal shared input (optogenetic locus coeruleus stimulation) raise astrocyte
network coordination, does natural arousal FLUCTUATION track coordination beyond arousal LEVEL, and is coordination
higher awake than under anaesthesia?

Data: Rupprecht, Duss, Maria, Helmchen, Bohacek 2026, Zenodo 10.5281/zenodo.18672975 (CC-BY 4.0), preprint
doi 10.64898/2026.01.16.699885; file astrocytes.mat (= 2024-Nov-26_astrocytes_..._activity_bouts.mat), variable
astromice_sian_batch2: 22 sessions of hippocampal astrocyte somata, 2-photon, 3 planes, ~10 Hz, from 4 mice
(4364, 4381, 4389, 4413). sessionID ...A1 = anaesthesia (4, one per mouse), B1-B3 = awake (12, three per mouse),
P1-P3 = pharmacology under anaesthesia (6, mice 4381/4413). Each session is a concatenation of ~143 s files
(nb_frames_all); opto_timepoints are FRAME indices (the Zenodo text says seconds, but they run to ~T frames, and the
population response starts 4-12 frames after them), opto_type_all is the LC stimulation frequency (1-40 Hz),
typically ~60 s into a file. Awake sessions also carry reward and airpuff times (frames).

Written BEFORE the analysis (after inspecting only structure, NaN pattern and the population response to 3 stimuli
in 3 sessions to confirm the frame units). P sessions are excluded from every test: the drug and its timing are not in
the dataset documentation.

Common method. Accepted ROIs only (noise_level_accept; none has NaNs). somata_dFF as given. Fluctuations = each trace
minus its 60 s running median, computed within each file (not across file joins) -- as data_cahill2024_dose.py.
Coordination rho-bar = mean off-diagonal Pearson correlation between ROIs in a window. Bootstrap 95% CI over sessions
(2000 resamples, seed 20261007). n = sessions; the 4 mice are reported, not resampled (4 is too few).

(a) CAUSAL. Per stimulation trial: pre window [onset-40 s, onset), post window [onset, onset+40 s), both inside the
    onset's file; trials with an airpuff in [onset-40 s, onset+40 s) dropped. delta = rho-bar(post) - rho-bar(pre).
    PREDICTION: session-mean delta > 0 (CI above 0), in awake (B) and in anaesthesia (A) separately; and delta rises
    with frequency: mean over sessions of the within-session Spearman(frequency, delta) > 0 (sessions with >= 2
    frequencies). Per-frequency mean delta with CIs is reported.
    Secondary (no directional prediction; says how much of (a) is the stimulus-locked mean response): the same after
    subtracting from each ROI its mean fluctuation trace over all trials of that frequency in that session (pre and post
    alike); frequencies with a single trial in a session are dropped. Sensitivity: (a) without trials having a reward in
    [onset-40 s, onset+40 s).
(b) NATURAL AROUSAL (B sessions). Non-overlapping 30 s windows inside files, excluding any frame within
    [onset, onset+90 s) of a stimulation or [airpuff, airpuff+60 s) (gaps between files ignored, i.e. conservative).
    Per window: rho-bar, LEVEL = mean and FLUCTUATION = SD of pupil, and the same for paw_movement. Windows with
    < 80 % finite behaviour samples are dropped for that signal. Within each session all variables are z-scored; pooled.
    PRIMARY (two, both reported): partial correlation of rho-bar with fluctuation controlling for level, for pupil and
    for paw. PREDICTION: positive (CI above 0) for both. Level-only correlation reported for comparison. Sensitivity:
    windows containing a reward dropped.
(c) STATE. Per session mean rho-bar over the (b) windows (same exclusions; for A sessions this is the pre-stimulation
    part of each file). PREDICTION: awake > anaesthesia; per mouse, mean(B) - A > 0 in all 4 mice, and the session-level
    group difference has a CI above 0. Mean dF/F is reported alongside (coordination depends on activity/SNR).

Sanity check only: the authors' Pair_wise_correlations (+ _only_LC, _wo_LC) averaged over accepted pairs, compared with
our unfiltered whole-session rho-bar (Pearson across sessions) and with the sign of our (a) delta.

Run: /opt/conda/envs/ece/bin/python -s data_rupprecht2026_lc.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, h5py
from scipy.ndimage import median_filter
from scipy.stats import spearmanr
from core.provenance import save_result

F = Path("/mnt/sysfs01/users/cagatay/external/public/rupprecht2026/astrocytes.mat")
HP_S, WS_S, WN_S, EXCL_STIM_S, EXCL_PUFF_S, MINFIN = 60, 40, 30, 90, 60, 0.8
SEED = 20261007
rng = np.random.default_rng(SEED)


def rhobar(X):
    X = X[np.std(X, 1) > 1e-12]
    C = np.corrcoef(X); n = C.shape[0]
    return (C.sum() - n) / (n * (n - 1))


def chars(d):
    return "".join(map(chr, d[()].ravel()))


def times(g, k):
    v = np.asarray(g[k][()], float).ravel()
    return v[v > 0]                          # [0, 0] marks "none"


def load(f, ref):
    g = f[ref]
    sid = chars(g["sessionID"]); fs = float(g["frame_rate"][()].item())
    acc = g["noise_level_accept"][()].ravel().astype(bool)
    X = np.asarray(g["somata_dFF"][()], float)[acc]          # (ROIs, T)
    assert not np.isnan(X).any(), sid
    edges = np.concatenate([[0], np.cumsum(g["nb_frames_all"][()].ravel().astype(int))])
    assert edges[-1] == X.shape[1], sid
    hp = np.empty_like(X); k = int(round(HP_S * fs)) | 1
    for a, b in zip(edges[:-1], edges[1:]):
        hp[:, a:b] = X[:, a:b] - median_filter(X[:, a:b], size=(1, k), mode="nearest")
    T = X.shape[1]
    beh = {}
    for name in ("pupil", "paw_movement"):
        v = np.asarray(g[name][()], float).ravel()
        beh[name] = v if v.size == T else np.full(T, np.nan)
    N = acc.size; m = np.outer(acc, acc) & ~np.eye(N, dtype=bool)
    theirs = {k: np.nanmean(np.asarray(g[k][()], float).reshape(N, N)[m])
              for k in ("Pair_wise_correlations", "Pair_wise_correlations_only_LC", "Pair_wise_correlations_wo_LC")}
    return dict(sid=sid, kind=sid.split("_")[-1][0], mouse=sid.split("_")[1], fs=fs, X=X, hp=hp, edges=edges,
                onsets=np.asarray(g["opto_timepoints"][()], float).ravel().astype(int),
                freqs=np.asarray(g["opto_type_all"][()], float).ravel(),
                puffs=times(g, "airpuff_timepoints"), rewards=times(g, "reward_timepoints"),
                beh=beh, theirs=theirs, rho_raw=rhobar(X))


def file_of(S, t):
    i = np.searchsorted(S["edges"], t, side="right") - 1
    return S["edges"][i], S["edges"][i + 1]


def trials(S):
    """Per-trial pre/post rho-bar (test a), plus evoked-subtracted secondary."""
    W = int(round(WS_S * S["fs"])); rows, segs = [], []
    for t, fq in zip(S["onsets"], S["freqs"]):
        a, b = file_of(S, t)
        if t - W < a or t + W > b or np.any((S["puffs"] >= t - W) & (S["puffs"] < t + W)):
            continue
        pre, post = S["hp"][:, t - W:t], S["hp"][:, t:t + W]
        rew = bool(np.any((S["rewards"] >= t - W) & (S["rewards"] < t + W)))
        rows.append([fq, rhobar(pre), rhobar(post), float(rew),
                     S["X"][:, t - W:t].mean(), S["X"][:, t:t + W].mean(), np.nan, np.nan])
        segs.append((pre, post))
    rows = np.array(rows).reshape(-1, 8)
    for fq in np.unique(rows[:, 0]) if len(rows) else []:
        idx = np.where(rows[:, 0] == fq)[0]
        if idx.size < 2:
            continue
        mpre = np.mean([segs[i][0] for i in idx], 0); mpost = np.mean([segs[i][1] for i in idx], 0)
        for i in idx:
            rows[i, 6] = rhobar(segs[i][0] - mpre); rows[i, 7] = rhobar(segs[i][1] - mpost)
    return rows   # cols: freq, rho_pre, rho_post, reward, act_pre, act_post, rho_pre_evsub, rho_post_evsub


def nat_windows(S):
    """Spontaneous 30 s windows (tests b, c). cols: rho, act, pupil mean, pupil sd, paw mean, paw sd, reward."""
    fs = S["fs"]; T = S["X"].shape[1]; W = int(round(WN_S * fs)); bad = np.zeros(T, bool)
    for t in S["onsets"]:
        bad[t:t + int(round(EXCL_STIM_S * fs))] = True
    for t in S["puffs"].astype(int):
        bad[t:t + int(round(EXCL_PUFF_S * fs))] = True
    rew = np.zeros(T, bool); rew[np.clip(S["rewards"].astype(int), 0, T - 1)] = True
    out = []
    for a, b in zip(S["edges"][:-1], S["edges"][1:]):
        for w0 in range(a, b - W + 1, W):
            sl = slice(w0, w0 + W)
            if bad[sl].any():
                continue
            r = [rhobar(S["hp"][:, sl]), S["X"][:, sl].mean()]
            for name in ("pupil", "paw_movement"):
                v = S["beh"][name][sl]
                r += [np.nanmean(v), np.nanstd(v)] if np.isfinite(v).mean() >= MINFIN else [np.nan, np.nan]
            out.append(r + [float(rew[sl].any())])
    return np.array(out).reshape(-1, 7)


def boot(vals, stat=np.mean):
    vals = list(vals)
    bs = [stat([vals[i] for i in rng.integers(0, len(vals), len(vals))]) for _ in range(2000)]
    return np.array([stat(vals), *np.nanpercentile(bs, [2.5, 97.5])])


def zs(v):
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 0 else v * np.nan


def partial(Ws, xcol, ccol, use_x=True):
    """Pooled within-session (z-scored) partial corr of rho (col 0) with col xcol controlling ccol;
    use_x=False: plain corr of rho with ccol (level only)."""
    r_, x_, c_ = [], [], []
    for W in Ws:
        W = W[np.isfinite(W[:, xcol]) & np.isfinite(W[:, ccol])]
        if W.shape[0] < 4:
            continue
        r_.append(zs(W[:, 0])); x_.append(zs(W[:, xcol])); c_.append(zs(W[:, ccol]))
    r, x, c = map(np.concatenate, (r_, x_, c_))
    ok = np.isfinite(r) & np.isfinite(x) & np.isfinite(c); r, x, c = r[ok], x[ok], c[ok]
    if not use_x:
        return np.corrcoef(r, c)[0, 1]
    A = np.vstack([c, np.ones_like(c)]).T
    rr = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]; xx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return np.corrcoef(rr, xx)[0, 1]


def main():
    f = h5py.File(F, "r"); refs = f["astromice_sian_batch2"][()].ravel()
    S_all = []
    for ref in refs:
        S = load(f, ref)
        if S["kind"] == "P":
            print("excluded (pharmacology):", S["sid"]); continue
        S["trials"] = trials(S); S["win"] = nat_windows(S)
        del S["X"], S["hp"]
        S_all.append(S)
        print(f"{S['sid']}: {len(S['trials'])} trials, {len(S['win'])} spontaneous windows")
    out = {}

    # ---- (a) causal LC stimulation
    print("\n(a) LC stimulation: delta rho-bar = post - pre (40 s windows)")
    for label, kinds in (("awake B", "B"), ("anaesthesia A", "A"), ("pooled A+B", "AB")):
        Ss = [S for S in S_all if S["kind"] in kinds and len(S["trials"])]
        for tag, sel, c0, c1 in (("primary", lambda R: R, 1, 2), ("no-reward", lambda R: R[R[:, 3] == 0], 1, 2),
                                 ("evoked-subtracted", lambda R: R[np.isfinite(R[:, 6])], 6, 7)):
            d = [np.mean(sel(S["trials"])[:, c1] - sel(S["trials"])[:, c0]) for S in Ss if len(sel(S["trials"]))]
            res = boot(d); npos = int(np.sum(np.array(d) > 0))
            print(f"  {label:14s} {tag:18s} n={len(d):2d} sessions  delta {res[0]:+.4f} [{res[1]:+.4f},{res[2]:+.4f}]  "
                  f"{npos}/{len(d)} positive")
            out[f"a_{kinds}_{tag}"] = res; out[f"a_{kinds}_{tag}_per_session"] = np.array(d)
        pre = boot([S["trials"][:, 1].mean() for S in Ss]); post = boot([S["trials"][:, 2].mean() for S in Ss])
        print(f"  {label:14s} rho-bar pre {pre[0]:.4f} [{pre[1]:.4f},{pre[2]:.4f}]  post {post[0]:.4f} [{post[1]:.4f},{post[2]:.4f}]")
        out[f"a_{kinds}_pre"], out[f"a_{kinds}_post"] = pre, post
        sp = [spearmanr(S["trials"][:, 0], S["trials"][:, 2] - S["trials"][:, 1]).correlation
              for S in Ss if len(np.unique(S["trials"][:, 0])) >= 2 and len(S["trials"]) >= 3]
        res = boot(sp)
        print(f"  {label:14s} within-session Spearman(freq, delta): {res[0]:+.3f} [{res[1]:+.3f},{res[2]:+.3f}] "
              f"over {len(sp)} sessions, {int(np.sum(np.array(sp) > 0))} positive")
        out[f"a_{kinds}_freq_spearman"] = res; out[f"a_{kinds}_freq_spearman_per_session"] = np.array(sp)
        for fq in sorted(set(np.concatenate([S["trials"][:, 0] for S in Ss]))):
            d = [np.mean((S["trials"][:, 2] - S["trials"][:, 1])[S["trials"][:, 0] == fq])
                 for S in Ss if np.any(S["trials"][:, 0] == fq)]
            res = boot(d)
            print(f"      {fq:4.0f} Hz  n={len(d):2d} sessions  delta {res[0]:+.4f} [{res[1]:+.4f},{res[2]:+.4f}]")
            out[f"a_{kinds}_freq{int(fq)}"] = np.append(res, len(d))

    # ---- (b) natural arousal, awake
    print("\n(b) awake spontaneous windows: rho-bar vs fluctuation (SD) beyond level (mean), within session")
    B = [S for S in S_all if S["kind"] == "B"]
    for tag, Ws in (("primary", [S["win"] for S in B]), ("no-reward", [S["win"][S["win"][:, 6] == 0] for S in B])):
        for name, xc, cc in (("pupil", 3, 2), ("paw", 5, 4)):
            Wk = [W for W in Ws if np.isfinite(W[:, xc]).sum() >= 4]
            pc = partial(Wk, xc, cc); pl = partial(Wk, xc, cc, use_x=False)
            bs = [partial([Wk[i] for i in rng.integers(0, len(Wk), len(Wk))], xc, cc) for _ in range(2000)]
            bl = [partial([Wk[i] for i in rng.integers(0, len(Wk), len(Wk))], xc, cc, False) for _ in range(2000)]
            ps = np.array([partial([W], xc, cc) for W in Wk])
            nw = sum(int(np.sum(np.isfinite(W[:, xc]))) for W in Wk)
            print(f"  {tag:9s} {name:5s} {len(Wk)} sessions, {nw} windows: partial r(rho, SD | mean) {pc:+.3f} "
                  f"[{np.nanpercentile(bs, 2.5):+.3f},{np.nanpercentile(bs, 97.5):+.3f}]  ({int(np.sum(ps > 0))}/{len(ps)} "
                  f"sessions positive);  level r(rho, mean) {pl:+.3f} [{np.nanpercentile(bl, 2.5):+.3f},"
                  f"{np.nanpercentile(bl, 97.5):+.3f}]")
            out[f"b_{tag}_{name}_partial"] = np.array([pc, *np.nanpercentile(bs, [2.5, 97.5])])
            out[f"b_{tag}_{name}_level"] = np.array([pl, *np.nanpercentile(bl, [2.5, 97.5])])
            out[f"b_{tag}_{name}_partial_per_session"] = ps

    # ---- (c) state
    print("\n(c) state: session-mean rho-bar of spontaneous windows")
    sm = {S["sid"]: (S["kind"], S["mouse"], np.mean(S["win"][:, 0]), np.mean(S["win"][:, 1]), len(S["win"])) for S in S_all}
    for k in ("A", "B"):
        v = [x[2] for x in sm.values() if x[0] == k]; a = [x[3] for x in sm.values() if x[0] == k]
        r = boot(v); ra = boot(a)
        print(f"  {k}: n={len(v)} sessions  rho-bar {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}]  mean dF/F {ra[0]:.3f} [{ra[1]:.3f},{ra[2]:.3f}]")
        out[f"c_{k}_rho"], out[f"c_{k}_act"] = r, ra
    vB = [x[2] for x in sm.values() if x[0] == "B"]; vA = [x[2] for x in sm.values() if x[0] == "A"]
    bs = [np.mean(rng.choice(vB, len(vB))) - np.mean(rng.choice(vA, len(vA))) for _ in range(2000)]
    diff = np.array([np.mean(vB) - np.mean(vA), *np.percentile(bs, [2.5, 97.5])])
    mice = sorted({x[1] for x in sm.values()})
    pm = np.array([np.mean([x[2] for x in sm.values() if x[1] == m and x[0] == "B"]) -
                   np.mean([x[2] for x in sm.values() if x[1] == m and x[0] == "A"]) for m in mice])
    print(f"  B - A {diff[0]:+.4f} [{diff[1]:+.4f},{diff[2]:+.4f}]; per mouse {dict(zip(mice, np.round(pm, 4)))}")
    out["c_diff"], out["c_per_mouse"] = diff, pm

    # ---- sanity vs authors' correlations
    th = np.array([[S["theirs"][k] for k in ("Pair_wise_correlations", "Pair_wise_correlations_only_LC",
                                               "Pair_wise_correlations_wo_LC")] for S in S_all])
    ours = np.array([S["rho_raw"] for S in S_all])
    dA = np.array([np.mean(S["trials"][:, 2] - S["trials"][:, 1]) if len(S["trials"]) else np.nan for S in S_all])
    print(f"\nsanity: Pearson(our raw whole-session rho-bar, their Pair_wise_correlations) = "
          f"{np.corrcoef(ours, th[:, 0])[0, 1]:+.3f}; mean ours {ours.mean():.4f} vs theirs {th[:, 0].mean():.4f}")
    print(f"sanity: their only_LC > wo_LC in {int(np.sum(th[:, 1] > th[:, 2]))}/{len(th)} sessions; "
          f"sign agreement with our (a) delta {int(np.sum(np.sign(th[:, 1] - th[:, 2]) == np.sign(dA)))}/{len(th)}")
    out["sanity_theirs"], out["sanity_ours_raw"] = th, ours

    p = save_result(Path("processed_data") / "rupprecht2026_lc_tests.npz",
                    {"source": "Rupprecht et al. 2026, Zenodo 10.5281/zenodo.18672975, astrocytes.mat",
                     "highpass_running_median_s": HP_S, "stim_window_s": WS_S, "nat_window_s": WN_S,
                     "excl_after_stim_s": EXCL_STIM_S, "excl_after_puff_s": EXCL_PUFF_S, "min_finite_beh": MINFIN,
                     "excluded": "P sessions", "boot_seed": SEED},
                    sessions=np.array([S["sid"] for S in S_all]),
                    # first column = index into `sessions`
                    trials=np.vstack([np.c_[np.full(len(S["trials"]), i), S["trials"]] for i, S in enumerate(S_all)]),
                    windows=np.vstack([np.c_[np.full(len(S["win"]), i), S["win"]] for i, S in enumerate(S_all)]), **out)
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
