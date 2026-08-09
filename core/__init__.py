"""
core/ — the single source of truth for the ATP / astrocyte excitable-lattice model.

Created 2026-08-09 for the revision blueprint (new_plan.md, Phase 0). Before this,
every fig_*.py script carried its own verbatim copy of the FHN integration core,
which is how a single-character numerical error (dt vs sqrt(dt) noise scaling) came
to sit in nine places at once. This package exists so there is one integrator, one
model definition, and one place to fix.

Migration status: the integrator and provenance helpers live here now. The fig_*.py
scripts have NOT yet been repointed at them — that is the next Phase-0 task. Until
they are, treat the figures in the repo as produced by the OLD (incorrect) scheme.
"""
