#!/usr/bin/env bash
# Phase 2.1 collapse under FOUR I0 variants (new_plan.md 2026-08-11 robustness table).
# L=32, 40 seeds, T=200, both controls (B, B'), both alpha_ref (0.10, 0.90).
# Sequential, one numba-parallel ensemble at a time (not oversubscribed).
set -e
cd /mnt/sysfs01/users/cagatay/code/bayat-et-al
PY=/opt/conda/envs/ece/bin/python
echo "=== I0 four-variant Phase 2.1 start $(date +%H:%M:%S) ==="
for V in 0 1 2 3; do
  echo "--- variant $V : aref 0.10 (A + B + B') ---"
  $PY phase2_coupling.py 32 0.10 40 $V
  echo "--- variant $V : aref 0.90 (B + B') ---"
  $PY phase2_coupling.py 32 0.90 40 $V
done
echo "=== I0 four-variant Phase 2.1 done $(date +%H:%M:%S) ==="
