# PHASE 0 VALIDATION — two gates before Phase 1

Both gates raised because Phase 0.1–0.4 had never checked `core/model.py` or the model
against *published* output — only against itself. Run 2026-08-09, env `ece`.

## Gate 1 — is `core/model.py` the code that made the figures?

### 1a. Trajectory identity vs the original lattice core — **PASS**

`core/model.py::final_state` uses the identical RNG draw order as the original
`fig_3_criticality_ci.py::run_one_seed_core` (recovered from git at `adbed58~1`, i.e.
before any of this session's edits). Run in **legacy mode** (noise × dt) at matched
seed/alpha/grid, the final C field is **bit-identical**:

| alpha | seed | grid | max &#124;core − original&#124; |
|-------|------|------|----------------------------|
| 0.40  | 11   | 10×10 | 0.0e+00 |
| 0.175 | 17   | 10×10 | 0.0e+00 |
| 0.90  | 23   | 10×10 | 7.7e-15 |

So `core/model.py` **is** a faithful extraction of the lattice code that produced Fig 3
and Fig 5. Anything that code produced, this file reproduces (that is what makes the
Gate-2 sweep below, run on the original, transfer to it).

### 1b. Fig 1 rates — a SEPARATE model, and it partly fails to reproduce

The factor-of-two the review flagged is real and has a clean cause: **Fig 1 is not the
lattice model.** `fig_1_single_cell.py` is a hand-tuned single cell with fixed `I0` per
regime (0.38/0.5/0.62), `gamma=0.72`, `sigma_eff=sigma(1+A)` — none of which is the
lattice parameterization. So `core/model.py` at 1×1 is NOT expected to match Fig 1, and
the Phase 0.2 "single-unit" onset at alpha≈0.6–0.9 was the *lattice* unit, not Fig 1's.

Running Fig 1's OWN original model (old scheme, its 10 seeds) at its published ATP levels:

| alpha | reproduced | published |
|-------|-----------|-----------|
| 0.19  | 0.13 ± 0.05 | 0.13 ± 0.05  ✓ |
| **0.40**  | **1.96 ± 0.10** | **1.33 ± 0.18**  ✗ |
| 0.90  | 2.86 ± 0.05 | 2.86 ± 0.05  ✓ |

The endpoints reproduce to two decimals (std included), so the machinery and seeds are
right — but the **intermediate, oscillation-onset point does not** (1.96 vs 1.33). Onset
is the most bifurcation-sensitive point, so a small parameter drift between the git
`fig_1` and the version that made the figure is amplified there. **This is a Fig-1
reproducibility gap, not a `core/model.py` problem** — but it should be run down before
Fig 1 is reused. Likely a one-line parameter difference; a question for Bayat (who wrote
the code per CRediT).

**RESOLVED 2026-08-14 — no email needed.** The drift is the ATP level itself, not a model
parameter. The repo script carried `ATP_LEVELS = [0.19, 0.40, 0.9]`; the published figure
used the intermediate point at **0.27**. Sweeping the intermediate level under the
corrected-EM scheme at `SIGMA_EM_PREDICTED = 0.02332` gives `A=0.27 -> 1.33 ± 0.18`, an
exact match to the published value including its SD, with the endpoints unchanged
(0.13 ± 0.05, 2.86 ± 0.05). All three published rates now reproduce. `figdata_fig1.py`
carries the corrected level and the sigma import; the ATP axis of Figure 1 is therefore
0.19 / 0.27 / 0.90, and the manuscript caption states those values.

## Gate 2 — is the chi peak an artifact of the `I0 = 1/sqrt(A)` drive?

Healthy chi sweep (fig_3_criticality_ci protocol, reduced budget: grid 10, T=200, 10
seeds, 21 alphas), original `I0` vs `I0` held alpha-independent at the peak's alpha (0.175):

| I0 form | chi peak height | chi peak alpha |
|---------|-----------------|----------------|
| `0.05 + (1/sqrt(A))·I0_base` (original) | 62.1 | 0.120 |
| held alpha-independent | 55.2 | 0.120 |

**The peak survives.** Location unchanged; height falls ~11%. The `1/sqrt(A)` term does
**not** create the susceptibility peak — so Fig 3B and the Fig 5 ridge are not an I0
artifact. This is the "incidental, proceed" branch.

What the I0 term *does* do: inflate the low-alpha tail (chi at alpha=0.065 drops from
45 → 10 when I0 is held fixed). That is the Phase 0.2 alpha=0.05 outlier, now explained —
a low-ATP baseline-drive effect, not a transition — but it sits away from the peak and
does not carry the headline.

(Budget note: this reduced run peaks at alpha=0.120, one grid point below the published
0.175; a finite-budget shift. The confound conclusion is invariant to it.)

## Decisions and open items

- **`core/model.py` is validated as canonical for the lattice results** (Gate 1a, bit-exact).
- **The `I0 = 1/sqrt(A)` form is NOT changed.** Gate 2 shows it does not affect the
  headline, and code == manuscript table, so there is nothing to reconcile mechanically.
  Whether the `1/sqrt(A)` sign was *intended* is a question for Bayat, not a simulation —
  **left for the authors to confirm before the Phase 4 Methods rewrite.** Do not silently
  pick a different form.
- ~~**Fig 1 intermediate rate (1.96 vs 1.33) is unresolved**~~ — RESOLVED 2026-08-14: the
  published intermediate level is A=0.27, not the 0.40 the script carried; at 0.27 the rate
  reproduces exactly (1.33 ± 0.18). See section 1b. Not a question for Bayat.
- Phase 1 (finite-size scaling) is clear to start on `core/model.py` at sigma=0.02332.
