# NUMERICS_NOTE — noise scheme and the recalibrated sigma

**Phase 0.1–0.2 of the revision plan** (`neubrain/projects/astro_atp/new_plan.md`).
This file records the one number every downstream run must use, and why.

## The fix (Phase 0.1)

The integration cores multiplied the stochastic term by `dt` instead of `sqrt(dt)`
(noise folded into the drift `dC`, then `C += dt*dC`). Euler–Maruyama for
`dC = f dt + s dW` requires `sqrt(dt)`. See `core/integrator.py`; the gate
`tests/test_noise_scaling.py` proves `Var[C(T)] = T*s^2` under the corrected scheme.
All ten cores are fixed in place.

## The recalibrated sigma (Phase 0.2)

**Use `sigma = 0.02332` (= 0.4 · sqrt(0.0034)) everywhere downstream, in place of the
old nominal `sigma = 0.4`.**

The old figures were produced by the buggy scheme at `sigma = 0.4`. That scheme is not
merely *similar* to the corrected scheme at a smaller sigma — it is **algebraically
identical** to the corrected scheme at `sigma * sqrt(dt)`:

```
legacy :  C += dt * (dC + noise)             , noise = sigma_eff * 3 * eta
correct:  C += dt * dC + sqrt(dt) * noise'    , noise'= sigma_eff' * 3 * eta

  dt * sigma_eff            (legacy, sigma = 0.4)
  = sqrt(dt) * sigma_eff'   (correct)   iff   sigma_eff' = sigma_eff * sqrt(dt)
  i.e.  sigma' = 0.4 * sqrt(0.0034) = 0.02332
```

On the same seed the two produce the **same trajectory**. So the noise fix does not
change the model's behaviour at all — it only **relabels** the noise axis. The published
phenomenology is preserved exactly; the number on the sigma axis was wrong by a factor
of `sqrt(dt) ~ 0.058`.

### Empirical confirmation

`phase0_recalibrate.py` runs the LEGACY scheme at `sigma=0.4` as a reference and sweeps
the CORRECTED scheme over `sigma in {0.02, 0.0233, 0.034, 0.05, 0.1, 0.2, 0.4}` across the
ATP (alpha) axis, comparing three regime observables (active fraction, susceptibility chi,
spike rate). Best match to the reference, seed-averaged, 8 seeds, 15000 steps:

| lattice      | best sigma | distance | 2nd best |
|--------------|-----------|----------|----------|
| single unit  | **0.0233**| 0.038    | 0.020 (0.46) |
| 10×10        | **0.0233**| 0.011    | 0.020 (0.28) |

The distance rises monotonically as sigma moves away from 0.0233 in either direction. The
tiny residual at 0.0233 is only the rounding of 0.02332. Result: `processed_data/
phase0_recalibration.npz` (provenance-stamped).

### The regimes are recovered

At `sigma = 0.0233` the single unit traverses the published sequence along ATP:

- `alpha <~ 0.5` — quiescent / noise-driven excitable (activity ~0.02, sparse spikes);
- `alpha ~ 0.6–0.9` — oscillation onset (spike rate climbs);
- `alpha >~ 1.0` — high-frequency / irregular (spike-count proxy peaks near alpha=1.0 then
  falls as the dynamics become fast and irregular).

So Phase 0.2's question — "do the known regimes reappear at some sigma in 0.022–0.034?" —
is answered **yes, exactly at 0.02332**, by construction rather than by luck.

## What this does and does NOT settle

- **Does:** confirm the model is faithfully re-implemented in `core/model.py` (the identity
  holds numerically → no transcription error), and that the sigma to use downstream is
  0.02332, not 0.4.
- **Does NOT:** change any scientific conclusion, and in particular does **not** address the
  editor's objection (that the crossover may be just weaker coupling). Because the fix is a
  pure relabeling, the phenomenology — and its vulnerability — is exactly as before. That
  question is Phase 2's job.

## A flag for Phase 0.3

In the 10×10 sweep, `chi` at the lowest ATP (`alpha = 0.05`) is a large outlier
(~38 vs ~1 elsewhere). This is consistent with the `I0 = 0.05 + (1/sqrt(alpha)) * I0_base`
term blowing up as `alpha -> 0` (up to `I0 ~ 0.72` at alpha=0.05), driving strong activity
at *low* ATP — i.e. a baseline drive that acts opposite to the stated ATP effect. This is
exactly the term Phase 0.3 must audit; the low-alpha chi feature is likely that artifact,
not a transition. (Note also: this sweep uses independent fixed-alpha runs, not the
quasi-static up-sweep of `fig_3_criticality_ci.py`, so its chi(alpha) shape is not directly
comparable to the paper's Fig 3.)

## Action items created by this note

1. Downstream scripts still hardcode `sigma = 0.4`. When figures are regenerated under the
   corrected scheme, set `sigma = 0.02332` (or import it from `core/model`), or they will be
   ~17× too noisy. **The repo's current figures/caches are still the old-scheme outputs.**
2. Phase 0.3: audit the `1/sqrt(alpha)` I0 term (sign + the low-alpha blow-up above).

## Phase 1 — T-convergence of chi (choosing the sweep length)

chi is a time-variance estimator, so its finite-T bias is downward and scales with
tau_ac/T. tau_ac peaks near the transition (Fig S3) and grows with L, so a too-short
T suppresses large-L peaks MORE than small-L ones and can manufacture an artificial
saturation — which under the Phase-1 decision rule would wrongly strip the criticality
claim. So T is calibrated at L=32 BEFORE the L-comparison, and the smallest T within 5%
of the T=2000 chi_peak is adopted. This belongs in Methods: "T was chosen such that chi
estimates were converged to within 5%."

L=32, 8 seeds, narrow grid (17 alphas), sigma=0.02332:

| T    | steps  | chi_peak | alpha* |
|------|--------|----------|--------|
| 200  | 58823  | 32.07    | 0.165  |
| 500  | 147058 | 33.48    | 0.188  |
| 1000 | 294117 | 35.37    | 0.188  |
| 2000 | 588235 | *(running)* | |

**chi_peak is still climbing at T=1000 (500->1000 is +5.6%, not decelerating)** — a
direct demonstration of the downward finite-T bias. T=2000 is genuinely required to
apply the 5% rule; the converged T is read by the automated overnight launcher. This is
also why launching the overnight run at T=500 or T=1000 would have been an error.

**alpha\* has converged to 0.188 at L=32** (0.165 at T=200 was under-resolved; 0.188 at
both T=500 and T=1000). Combined with the L=10 legacy check (alpha*=0.120, firm), the
peak LOCATION is drifting UP with L: 0.120 (L=10) -> 0.188 (L=32). alpha*-drift with L is
one of the two collective-transition signatures; whether chi_peak also grows with L is
the Phase 1 question, decided from the L=32/64/128 run (a judgement call, not automated).

## Phase 1 run configuration

- Model: `core/model.py` (bit-identical to the published fig_3_criticality_ci core;
  see PHASE0_VALIDATION.md), sigma=0.02332, correct sqrt(dt) scheme.
- L in {32, 64, 128}, healthy; L in {32, 64} additionally with I0 held alpha-independent
  (the cheap insurance if 1/sqrt(A) turns out to be a typo).
- 8 seeds, narrow grid, converged T (from the table above).
- Each (L, condition) written to its own provenance-stamped
  `processed_data/phase1_L<L>_<cond>.npz` as it completes (crash-resilient).

## Phase 1 — observable and estimator constraints (Methods-ready draft)

**The Fig-3B statistic loses discrimination at physical lattice sizes.** The susceptibility
plotted in the submitted Fig. 3B is χ_ext = N·Var_t(max_i C_i − min_i C_i), a *range*
statistic. Its finite-size behaviour is governed by extreme-value statistics rather than
by a thermodynamic susceptibility: as the lattice grows, some cell is almost always near the
maximum and some near rest, so the instantaneous range ΔC(t) self-averages and its
α-dependence washes out. Empirically the peak-to-tail ratio of χ_ext falls from 3.85 (10×10)
to 1.46 (64×64) with disjoint bootstrap CIs, while the peak *height* does not fall — the
contrast collapses because the tail rises. Any susceptibility claim must therefore be made
on an intensive quantity, χ_true = N·Var_t(C̄), not on χ_ext.

**Sweep length T was set by a convergence rule but χ is a finite-T underestimate.** T=200 was
adopted as the smallest T within 5% of the T=2000 χ_peak. However χ_peak was still rising at
+5.6% per doubling at T=1000, and the T=2000 reference was itself not converged. We therefore
treat all χ values as finite-T underestimates and make finite-size comparisons only at fixed
T. The T-scan of χ_peak was non-monotonic (32.1, 33.5, 35.4, 33.1 at T=200/500/1000/2000);
this is expected because χ_peak = max_α χ(α) carries an upward argmax bias that grows with
estimator noise and works against the downward finite-T bias, so χ_peak must be reported with
a confidence interval, not as a point estimate.

**Peak-fitting was rejected in favour of argmax + bootstrap.** Parabola-in-log-χ and Lorentzian
fits to the peak region produced CIs 10–15× narrower than the argmax CIs on identical data,
with fitted amplitude systematically ≈25% below and fitted α₀ ≈+0.04 above the argmax at both
L — the signature of a symmetric functional form pulled by an asymmetric right tail rather than
of genuine variance reduction. A form-sensitivity check confirmed this: at L=32 the fitted α₀
was 0.186 (parabola) / 0.191 (Lorentzian) / 0.131 (log-normal), a spread of 0.060 against the
parabola's own bootstrap CI of 0.022. All finite-size statistics are therefore reported as
argmax-based with inside-resample bootstrap CIs.

## sigma bookkeeping — the ONE authoritative place (2026-08-11)

Three separate factors sit between the manuscript's "sigma" and the physical noise amplitude.
Recording them here so the number that goes into Methods is unambiguous and the Fig-5 axis is
labelled correctly. This is the same error class as the original dt-vs-sqrt(dt) bug: right
internally, wrong externally, would repeat it.

  sigma_nominal      = 0.4            (Table 1 "baseline noise amplitude", old-scheme convention)
  NOISE_MULT         = 3.0            (undocumented x3 in every core; absent from Methods/Table 1)
  dt                 = 0.0034         (confirmed from figfig1/figfig2; NOT Table 1's 0.003-0.007)
  SIGMA_EM_PREDICTED = 0.4*sqrt(dt)   = 0.02332   INTERNAL CODE CONSTANT (pre-x3), not physical
  sigma_true         = 3*0.4*sqrt(dt) = 0.06997 ~ 0.0700   PHYSICAL Euler-Maruyama amplitude

**Report sigma_true ~ 0.0700 in Methods** (the coefficient of dW at the base). SIGMA_EM_PREDICTED
(0.02332) and NOISE_MULT (3.0) are internal code constants; sigma_true = NOISE_MULT *
SIGMA_EM_PREDICTED. Do NOT print 0.02332 or 0.4 as "the noise amplitude" externally.

**Conversion for the old figures:** any nominal sigma maps to physical via
sigma_true = sigma_nominal * 3*sqrt(dt) = sigma_nominal * 0.17493.
  Fig 5 plotted axis [0.05, 0.8] (nominal)  ->  sigma_true [0.0087, 0.140].

**Convention audit — every place a sigma is printed/plotted/written:**
- Fig 5 phase-diagram sigma axis: currently NOMINAL; must be relabelled to sigma_true (x0.1749).
- T-convergence table + all sweep stdout ("sigma=0.02332"): that is SIGMA_EM_PREDICTED, the
  internal code nominal, NOT sigma_true. Fine internally; never quote it as the amplitude.
- phase-diagram config / any sigma arg to the sweeps: SIGMA_EM_PREDICTED convention.

## Model facts confirmed from Bayat's originals (figfig1/figfig2, 2026-08-11)

- **Two different I0 schemes, undocumented.** Single-cell (figfig1) uses a three-tier CONSTANT
  I0 (0.38 / 0.5 / 0.62 by regime), no alpha dependence. Network (figfig2) uses
  I0 = 0.05 + (1/sqrt(A))*U(0.01,0.15). Methods states neither split.
- **dt = 0.0034** in both files (confirms the back-solved 0.00339; contradicts Table 1's
  "0.003-0.007" range, which should be corrected to the single value used).
- **np.clip(C, -4, 4)** in both — a stabilization step absent from Methods. core/model.py has it
  (as np.minimum(np.maximum(...))). Document it in Methods either way.
