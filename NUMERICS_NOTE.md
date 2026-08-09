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
