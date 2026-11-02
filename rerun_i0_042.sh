#!/usr/bin/env bash
set -euo pipefail

PY=/opt/conda/envs/ece/bin/python
export MPLCONFIGDIR=/tmp/mpl-astro-i0-042

land() {
    printf '[LANDED] %s %s\n' "$(date --iso-8601=seconds)" "$1"
}

printf '[START] %s canonical I0_BASE=0.42 rerun\n' "$(date --iso-8601=seconds)"

"$PY" figdata_fig1.py
land processed_data/fig1_single_cell.npz

# Baseline-selection data displayed in Fig. 3E. Each seed lands independently.
for first in 11 16; do
    pids=()
    for seed in $(seq "$first" $((first + 4))); do
        (
            "$PY" phase3_spontaneous_fine.py "$seed"
            land "processed_data/i0fine_seed${seed}.npz"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done
"$PY" figdata_i0fine_combine.py
land processed_data/i0_fine.npz

"$PY" figdata_fig2.py
land processed_data/fig2_excitability.npz

"$PY" figdata_fig2d.py
land processed_data/fig2d_gamma_freeze.npz

"$PY" figdata_fig3.py
land processed_data/fig3_focal.npz

# Each seed is an independent, provenance-stamped checkpoint. Run five at a time.
for first in 11 16; do
    pids=()
    for seed in $(seq "$first" $((first + 4))); do
        (
            "$PY" figdata_fig4_seed.py "$seed"
            land "processed_data/fig4_seed${seed}.npz"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

"$PY" figdata_fig4_combine.py
land processed_data/fig4_mechanisms_ens.npz

# Persisted Appendix lag controls, one seed at a time, then aggregate.
for first in 11 16; do
    pids=()
    for seed in $(seq "$first" $((first + 4))); do
        (
            "$PY" phase3_lagcheck.py "$seed"
            land "processed_data/lagcheck_seed${seed}.npz"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done
"$PY" figdata_lagcheck_combine.py
land processed_data/lagcheck.npz

# Remaining Appendix diagnostics are logfile-only checks.
"$PY" phase3_frontshape.py
land logfile:phase3_frontshape

"$PY" fig_1_single_cell.py
land Figure_1.png
"$PY" fig_2_atp_excitability.py
land Figure_2.png
"$PY" fig_3_focal_front.py
land Figure_3.png
"$PY" fig_4_mechanisms.py
land Figure_4.png
"$PY" graphical_abstract.py
land graphical_abstract.png

printf '[COMPLETE] %s canonical I0_BASE=0.42 rerun\n' "$(date --iso-8601=seconds)"
