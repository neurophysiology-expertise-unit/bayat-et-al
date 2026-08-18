#!/usr/bin/env bash
# Two required re-runs at the canonical baseline I0_BASE=0.42.
#  (1) lagcheck, seeds 11-20, one process per seed -> per-seed npz, then combine.
#      Replaces a result that previously existed only in a logfile.
#  (2) phase2_coupling modes A/B/B' at i0_baseline=0.42.
#      The existing phase2v1 files ran at the undeclared default 0.2, and figdata_fig2.py
#      reads one of them, so Fig 2C currently spans a baseline mismatch.
set -u
PY=/opt/conda/envs/ece/bin/python
cd /mnt/sysfs01/users/cagatay/code/bayat-et-al
land() { echo "[LANDED] $(date -Iseconds) $*"; }
echo "[START] $(date -Iseconds) lagcheck 10 seeds + phase2 coupling at I0_BASE=0.42"

echo "--- (1) lagcheck seeds 11-20, parallel ---"
for s in 11 12 13 14 15 16 17 18 19 20; do
  $PY phase3_lagcheck.py $s > logs/lagcheck_seed$s.log 2>&1 &
done
wait
for s in 11 12 13 14 15 16 17 18 19 20; do land "lagcheck_seed$s.npz"; done
$PY figdata_lagcheck_combine.py && land "lagcheck.npz (ensemble)"

echo "--- (2) phase2 coupling, i0_form=1, i0_baseline=0.42 ---"
# aref 0.10 batch runs modes A, B, B'; aref 0.90 batch runs B, B' at the other reference.
$PY phase2_coupling.py 32 0.10 40 1 0.42 && land "phase2v1_bayat_b0.42 aref0.10 (A + B + Bprime)"
$PY phase2_coupling.py 32 0.90 40 1 0.42 && land "phase2v1_bayat_b0.42 aref0.90 (B + Bprime)"

echo "[COMPLETE] $(date -Iseconds)"
