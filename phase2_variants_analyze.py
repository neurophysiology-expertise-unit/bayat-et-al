"""Phase 2.1 collapse table across the four I0 variants (new_plan.md 2026-08-11).
Same pre-registered statistic as phase2_analyze.py: %% of D_eff-matched alpha points where
Sweep A's mean lies inside the control's 95%% bootstrap band (resample seeds). Pre-reg
thresholds, UNADJUSTED: >=90%% COLLAPSE, <=50%% SEPARATION, between = AMBIGUOUS.
Coordination = (N*R^2-1)/(N-1) (validated proxy, max dev 0.0025 at L=32).
Reading rule: separation in EVERY variant = robust to parameterization. Separation in SOME =
report which and narrow. Variant 3 (const) isolates whether separation needs any I0-ATP coupling.
Usage: python phase2_variants_analyze.py"""
import numpy as np, os
rng = np.random.default_rng(20260811); B = 20000; L = 32; N = L * L
VTAG = {0: "submitted", 1: "bayat", 2: "bounded", 3: "const"}
def rho(Rarr): return (N * Rarr ** 2 - 1.0) / (N - 1.0)
def load(f):
    z = np.load(f, allow_pickle=False); return z['active'], rho(z['R']), z['alphas']
def band(X):
    ns, na = X.shape; lo = np.zeros(na); hi = np.zeros(na)
    for j in range(na):
        bs = np.array([X[rng.integers(0, ns, ns), j].mean() for _ in range(B)])
        lo[j], hi[j] = np.percentile(bs, [2.5, 97.5])
    return lo, hi
def pct_inside(mA, X):
    lo, hi = band(X); return 100.0 * np.mean((mA >= lo) & (mA <= hi))
def verdict(p): return "SEP" if p <= 50 else ("COLL" if p >= 90 else "AMB")

print(f"Phase 2.1 collapse across four I0 variants (L={L}, 40 seeds). "
      f"%% D_eff-matched points where Sweep A's mean is inside control's 95%% band.")
print(f"Pre-reg (unadjusted): >=90%% COLLAPSE, <=50%% SEPARATION.\n")
hdr = f"{'variant':>10} {'aref':>5} {'control':>20} | {'active %in':>12} | {'coord %in':>12}"
print(hdr); print('-' * len(hdr))
summary = {}
for V in range(4):
    vt = VTAG[V]; pfx = f"processed_data/phase2v{V}_{vt}_L{L}"
    fA = f"{pfx}_A_full.npz"
    if not os.path.exists(fA):
        print(f"{vt:>10} {'--':>5} {'(Sweep A pending)':>20} |"); continue
    actA, rhoA, al = load(fA); mActA = actA.mean(0); mRhoA = rhoA.mean(0)
    for aref in (0.10, 0.90):
        for tag, lab in [(f'B_coupling_aref{aref:.2f}', 'B(coupling)'),
                         (f'Bprime_coupθ_aref{aref:.2f}', "B'(coup+theta)")]:
            f = f"{pfx}_{tag}.npz"
            if not os.path.exists(f):
                print(f"{vt:>10} {aref:>5.2f} {lab:>20} | pending"); continue
            actB, rhoB, _ = load(f)
            pa = pct_inside(mActA, actB); pc = pct_inside(mRhoA, rhoB)
            summary[(V, aref, lab)] = (pa, pc)
            print(f"{vt:>10} {aref:>5.2f} {lab:>20} | {pa:6.1f}% [{verdict(pa):4}] | "
                  f"{pc:6.1f}% [{verdict(pc):4}]")
    print()

# robustness read: does activity SEPARATE in every variant / every control?
print("=" * len(hdr))
if summary:
    all_sep = all(pa <= 50 for (pa, pc) in summary.values())
    any_sep = any(pa <= 50 for (pa, pc) in summary.values())
    persep = {}
    for (V, aref, lab), (pa, pc) in summary.items():
        persep.setdefault(V, []).append(pa <= 50)
    sep_variants = [VTAG[V] for V, flags in persep.items() if all(flags)]
    print(f"active SEPARATES in EVERY (variant x control x aref): {all_sep}")
    print(f"variants where activity separates from BOTH controls at BOTH aref: {sep_variants}")
    print(f"  -> constant-I0 (variant 3) present in that list = separation needs no I0-ATP coupling")
