# CHANNELS — every ATP-dependent term in the model

**Phase 0.3–0.4** (`neubrain/projects/astro_atp/new_plan.md`). Audited 2026-08-09 by
reading the code (`core/model.py`, faithfully matching `fig_3_criticality_ci.py`) against
the manuscript's parameter table and Methods.

`A` is the ATP control parameter (called `alpha` in the code). `U(a,b)` is a per-cell
uniform draw (static heterogeneity, fixed per seed).

## The six channels

| # | Quantity | Form (code = manuscript Table 1) | Direction as A↑ | Effect | In the manuscript's "three"? |
|---|----------|----------------------------------|-----------------|--------|------------------------------|
| 1 | `γ·A` excitability drive | `γ = U(0.05,0.34)`, enters as `γ·A` | **↑** | more excitation | **yes** (γA) |
| 2 | `σ_eff` noise | `σ·(1 + 4A)`, σ=0.4 | **↑** | more noise | **yes** (σ_eff) |
| 3 | `D_eff` coupling | `D0/(1+(κA)^4)`, `D0=U(0.05,0.5)`, `κ=U(1,4)` | **↓↓** (quartic) | **suppresses coupling** | **yes** (D_eff) |
| 4 | `θ` activation threshold | `0.5 + 0.7A` (inside `Φ=½[1+tanh(η(C−θ))]`) | **↑** | **suppresses coupling** (raises the threshold a cell must cross to transmit) | **no** |
| 5 | `I0` baseline excitability | `0.05 + (1/√A)·U(0.01,0.15)` | **↓** (diverges as A→0) | baseline drive **highest at low ATP** | **no** |
| 6 | `τ_h` recovery timescale | `10/((1+0.8A)·U(0.5,1.1))` | **↓** | faster recovery / faster oscillation at high ATP | **no** |

The manuscript (Results, "By varying A we simultaneously modify…") names **three** channels:
excitability `γA`, noise `σ_eff`, coupling `D_eff`. The parameter table nonetheless makes
**θ, I0 and τ_h** functions of A as well. So three ATP channels are undisclosed in the text
while being present in both the table and the code. Code and table agree on every form; the
gap is between the *table* and the *prose*, not between code and table.

## Why the editor's objection has force (Phase 2 target)

**Two of the six channels suppress effective coupling as ATP rises** — `D_eff` (channel 3,
a quartic cutoff) and `θ` (channel 4, raising the transmission threshold in the gating
function `Φ` that the diffusive term acts through). So "increase ATP" drives "reduce
coupling" through *two independent routes*. That is exactly why the ATP crossover can look
"largely an expected consequence of progressively weaker coupling": the sweep moves two
coupling knobs at once, plus the noise and excitability knobs.

The **disease perturbation compounds this**: it multiplies `D0 × 0.5` and `κ × 1.5`, both of
which further reduce `D_eff`. So the disease "exacerbation" is, mechanically, mostly *more
coupling suppression*. Phase 2 must show whether anything survives after the coupling
trajectory is matched.

## Phase 0.3 finding — the `I0 = 1/√A` term

**Not a code/table transcription error** — the code (`I0 = 0.05 + (1.0/np.sqrt(alpha))*I0_base`)
matches the manuscript table (`0.05 + (1/√A)·U(0.01,0.15)`) exactly. But the *sign* is the one
the plan feared, and it matters:

- Baseline excitability is **largest at low ATP** and falls as ATP rises. With `I0_base` mean
  ≈ 0.08 and max 0.15: `I0` ranges from ≈ 0.13 (A=1.1) up to ≈ 0.85 mean / **1.55 max at
  A=0.01**. This is a large constant drive at *low* ATP.
- It acts **opposite to channel 1** (`γA`, which vanishes as A→0). So the two "excitability"
  channels pull in opposite directions across the ATP axis.
- **Most likely a deliberate device, not a typo:** because `γA → 0` at low ATP, *something*
  has to make the low-ATP regime active (the paper's "low ATP → wave nucleation/propagation"
  story). The `1/√A` baseline supplies that drive. But this is nowhere stated as its purpose,
  and it means the low-ATP "active, wave-propagating" regime may be **imposed by the baseline
  drive rather than emergent**.
- **This is a Phase 2 confound, and Phase 0.2 already saw its footprint:** the 10×10 `χ`
  outlier at α=0.05 (~38 vs ~1 elsewhere) is consistent with this low-ATP `I0` blow-up, not a
  transition. Phase 2's leave-one-out (§2.2) should hold `I0` fixed and check whether the
  low-ATP activity persists without it.

**Recommendation:** keep the code/table as they agree, but (a) state the sign of the `I0`
channel explicitly in Methods, (b) disclose θ, I0, τ_h as ATP channels (the count is six, not
three), and (c) in Phase 2 treat `I0` as one of the leave-one-out channels.

## Separate code-vs-manuscript discrepancy (noise magnitude)

Every integration core multiplies the noise by an extra **`3.0`** (`noise = σ_eff * 3.0 *
η`), which is absent from the manuscript, where `ξ = σ_eff·η`. So the code's noise amplitude
is 3× the stated `σ_eff`. Combined with the Phase 0.1/0.2 result (effective σ was
mis-scaled by √dt), the noise term is the least faithfully documented part of the model.
`core/model.py` keeps the `3.0` (as `NOISE_MULT`) to stay faithful to what was actually run,
but Methods should either include the factor or fold it into σ. Recorded here so Phase 4's
Methods rewrite fixes it.
