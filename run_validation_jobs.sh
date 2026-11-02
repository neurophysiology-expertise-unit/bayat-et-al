#!/bin/bash
set -euo pipefail

PYTHON=/opt/conda/envs/ece/bin/python

echo "validation jobs start $(date --iso-8601=seconds) commit=$(git rev-parse HEAD)"

for seed in $(seq 11 20); do
    "$PYTHON" validation_jobs.py nucleation-seed "$seed" > "validation_nucleation_seed${seed}.log" 2>&1 &
done
wait
"$PYTHON" validation_jobs.py nucleation-combine
echo "Job A complete $(date --iso-8601=seconds)"

for seed in $(seq 11 20); do
    "$PYTHON" validation_jobs.py focal-seed "$seed" > "validation_focal_seed${seed}.log" 2>&1 &
done
wait
"$PYTHON" validation_jobs.py focal-combine
echo "Job B complete $(date --iso-8601=seconds)"
