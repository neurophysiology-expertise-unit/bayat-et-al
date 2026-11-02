# Front termination and spontaneous nucleation in a noisy network model of astrocytes

Simulation and analysis code, and the processed data behind every figure and number, for

F. K. Bayat, F. Oktay and Ç. Aydın, *Front termination and spontaneous nucleation in a noisy network model of
astrocytes* (preprint; submitted to Physical Review E).

The model is a stochastic lattice of heterogeneous FitzHugh–Nagumo units with thresholded diffusive coupling, in
which extracellular ATP is a uniform control parameter that moves drive, baseline current, noise amplitude,
transmission threshold, recovery rate and coupling together. A refractory state and decremental release are added
where stated.

## Figures

| Figure | Script | Data (processed_data/) | Produced by |
|---|---|---|---|
| 1 Single unit | `fig_1_single_cell.py` | `fig1_single_cell.npz`, `fig1_isi_cv.npz` | `figdata_fig1.py`, `figdata_fig1_isi.py` |
| 2 Coordination (coupling × drive, ablations, front vs coupling) | `fig_coupling.py` | `constrain_*`, `ablate_*`, `front_vs_coupling.npz` | `phase4_constrain.py`, `phase4_ablate.py`, `phase4_front_coupling.py` |
| 3 Focal front | `fig_3_focal_front.py` | `fig3_focal.npz`, `i0_fine.npz` | `figdata_fig3.py` |
| 4 Refractory state and decremental release | `fig_4_mechanisms.py` | `fig4_mechanisms_ens.npz` | `figdata_fig4*.py` |
| 5 Ring-map range law | `fig_ringmap.py` | `extent_scaling_law.npz` | `phase4_scaling.py` |
| Appendix (Hopf analysis) | `fig_2_atp_excitability.py` | `fig2_excitability.npz`, `fig2d_gamma_freeze.npz` | `figdata_fig2.py`, `figdata_fig2d.py` |

Other results quoted in the text: `phase3_rhobar.py` (correlation vs ATP, coupling-only control),
`phase4_noisy_repaired.py` (repaired model under noise), `phase4_lattice_limit.py` (128 × 128 check),
`figdata_termination_time.py` (stopping times). Pre-registered tests carry their predictions in the script
docstring; the outcome, including failed predictions, is reported in the paper.

## Environment

`environment.yml` / `requirements.txt` (Python, NumPy, Numba, Matplotlib, SciPy). Model primitives are in
`core/model.py`; every result file is written by `core/provenance.py::save_result`, which stores the parameters
(`__params__`) and the git commit of the code that produced it (`__git_commit__`). See `PROVENANCE.md`.

## License

Code: MIT (see `LICENSE`). Processed data: CC BY 4.0.
