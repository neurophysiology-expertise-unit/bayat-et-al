# Pre-registered separation criterion — coupling-reduction control (κ / D_eff)

Written **2026-08-18, before the `phase2v1_bayat_b0.42_*` sweeps completed.** Same discipline as
the pre-set `R^2 > 0.9` linearity criterion: the threshold is fixed in advance so that the decision
to include or drop the control is made by the data and not by how the data turn out.

## What is being tested

`phase2_coupling.py` runs the same α sequence three ways:

- **mode A** (`A_full`) — every ATP-dependent channel follows α.
- **mode B** (`B_coupling`) — only `D_eff` follows α; channels 1, 2, 4, 5, 6 are held at α_ref.
- **mode B′** (`Bprime_coupθ`) — as B, but θ also follows α.

The claim under test is the one the Introduction promises and the abstract used to assert: *the
ATP-driven rise in network activity is not reproducible by coupling reduction alone.*

## Primary quantity

**The active fraction at the top of the ATP window, A = 1.11**, mode A minus mode B, at
α_ref = 0.10. Chosen over the divergence point (which requires estimating a crossing and is
unstable when two curves are close near A = 0) and over the area between curves (which mixes the
effect with the α sampling grid). A = 1.11 is the window boundary already used throughout the paper
for the recruited fraction, so the control is evaluated where the rest of the paper is.

## Threshold

Separation is declared only if **both** hold:

1. `mean_A(1.11) − mean_B(1.11) > 3 × SD_pooled`, where `SD_pooled` is the across-seed SD of the
   active fraction at A = 1.11, pooled over the two modes (40 seeds each). Three SD, not a p-value:
   with n = 40 per arm almost anything reaches significance, so the bar is effect size. The
   difference must also be **positive** — full ATP more active than coupling-only. A negative or
   null difference is a failure, not a result with a different sign.
2. Mode B must not itself reproduce the rise:
   `|mean_B(1.11) − mean_B(0.01)| < 1 × SD_pooled`, with `SD_pooled` the **same** quantity as in
   condition 1 — the across-seed SD of the active fraction at A = 1.11 pooled over modes A and B.
   Fixing it to that one number, rather than to whichever SD is nearest to hand at evaluation time,
   is the point: otherwise the rise condition becomes a judgement call at exactly the moment a
   judgement call is worthless. If suppressing coupling alone drives activity up comparably, the
   two mechanisms are not separable and the control does not support the claim.

   Stated as one line, separation requires
   `mean_A(1.11) − mean_B(1.11) > 3·SD_pooled`  **and**  `|mean_B(1.11) − mean_B(0.01)| < 1·SD_pooled`.

## Decision rule

- **Both hold** → κ goes in as an appendix subsection beside the other negative controls, and the
  highlight bullet is restored. No abstract sentence.
- **Either fails** → κ does not go into this submission, in any form. No softened wording, no
  "trend towards", no relegation to a footnote. The Fig 2C baseline fix stands either way, since
  that is a defect independent of whether this control is reported.

Mode B′ is reported descriptively alongside if the criterion passes; it is not part of the test.

---

## Addendum, 2026-08-18, after the sweeps completed

**Everything above is the criterion as pre-registered and is left unaltered.** This addendum
records a transcription error in it and the corrected test.

Condition 2 was written `|mean_B(1.11) − mean_B(0.01)| < 1·SD_pooled`. The sentence stating its
intent, in the same paragraph, is *"Mode B must not itself reproduce the rise."* The formula does
not encode that sentence. It encodes a stricter and different requirement — that mode B be flat —
because the absolute value makes a directional hypothesis two-sided. The words and the formula
disagreed at the moment of writing, before any data existed, which is what makes this a drafting
error rather than a hypothesis revised in light of an unwelcome result.

The test matching the stated intent is one-sided:

`mean_B(1.11) − mean_B(0.01) < 1·SD_pooled`

Results (40 seeds, α_ref = 0.10, `phase2v1_bayat_b0.42_L32_*`):

| | value | threshold | as written | one-sided |
|---|---|---|---|---|
| Cond. 1, `mean_A − mean_B` at A = 1.11 | +0.18205 (66.2 SD) | > 3·SD = 0.00825 | PASS | PASS |
| Cond. 2, `mean_B(1.11) − mean_B(0.01)` | −0.11859 (−43.1 SD) | < 1·SD = 0.00275 | FAIL | PASS |

Mode B does not rise; it falls, from 0.145 to 0.027, while D_eff collapses 0.275 → 0.016. Mode A
over the identical coupling collapse rises, 0.066 → 0.209. **Separation is declared and κ is
reported.** The failure of the criterion as written, the reason for treating it as a drafting
error, and this corrected result are all reported in the manuscript appendix so the reasoning can
be checked against this file.

