# bayat-et-al — durable rules

Simulation code for the astro_atp paper. The manuscript lives in
`neubrain/projects/astro_atp/analysis/manuscript_v2/`. Read `../AGENTS.md` first — the scope
rule there applies here.

## Provenance is the invariant

Every result that reaches a figure or the manuscript must be written by
`core.provenance.save_result`, which stamps the full parameter dict and the git commit.

- **No panel and no manuscript claim may be built from a logfile.** If a script prints a
  number the text relies on, make it save an npz instead.
- **An unstamped or mis-stamped result is deleted, not reused.**
- **Stamp the parameters that identify the run, especially the baseline.** A file that does
  not record what it ran at cannot be checked. In August 2026 `phase2_coupling` fell through
  to an unrecorded default of `i0_baseline=0.2` and a figure paired it against a 0.42
  computation for weeks without anything failing.
- Superseded results go to `processed_data/legacy_b0.45/` (gitignored), never overwritten in
  place.

## Numbers

- `I0_BASE` in `core/model.py` is canonical. Never hardcode a baseline excitability anywhere
  else — import it. `I0_BASE_LEGACY` exists only for deliberate old-vs-new diffs.
- Every displayed number carries an error bar over the same **10 seeds (11–20)**.
- Distances: **25 µm/cell** primary; the 50 µm convention appears in captions only.
- 1 model time unit ≈ 1 s.
- **Never fabricate a result the data do not show.** If a sweep does not support a claim,
  report that it does not.

## Criteria are fixed in advance

Pre-set thresholds (`R^2 > 0.9` for a propagating front; `KAPPA_CRITERION.md` for the
coupling control) are written down before the data land and are **not** revised afterwards.
If a criterion turns out to be mis-drafted, say so explicitly, show both readings, and let
the human decide — do not silently substitute the corrected form.

## Traps found the hard way

- **numba freezes module globals at compile time.** Reassigning a global between two arms of
  a comparison does nothing after the first `@njit` call — both arms run identical code with
  no error. Pass values as arguments instead.
- Prefer `pathlib.Path(__file__).resolve().parent` over an absolute `sys.path.insert`; note
  that not every script imports pathlib, so import it inline in the same line.
- Long runs: detached with `nohup`, own logfile, one result saved per step as it finishes.
  A crash at hour four must not cost hours one through three.

## Before you claim it works

`python -m py_compile` every file you touched, and re-run at least one consumer of any data
format you changed to confirm the output is byte-identical where it should be.
