# Extracellular ATP Drives Spatial Fragmentation in Astrocyte Calcium Waves

Simulation and analysis code for Bayat, Oktay & Aydın, *Extracellular ATP Drives
Spatial Fragmentation in Astrocyte Calcium Waves* (submitted to *Chaos, Solitons
& Fractals*).

The model treats extracellular ATP as an externally controlled bifurcation
parameter in a stochastic FitzHugh–Nagumo-type network of astrocytes on a
periodic 2D lattice. ATP enters through three channels: intrinsic excitability
(`γα`), stochastic forcing (`σ_eff`), and a strongly nonlinear suppression of
intercellular coupling (`D_eff = D₀ / (1 + (κα)⁴)`).

## Figures

| Script | Figure | Produces |
|--------|--------|----------|
| `fig_1_single_cell.py` | Fig. 1 | Single-cell calcium traces at low / intermediate / high ATP |
| `fig_2_network_activity.py` | Fig. 2 | Network spatiotemporal activity + representative traces (low vs high ATP) |
| `fig_3_criticality.py` | Fig. 3 | Spatial heterogeneity `S_C`, susceptibility `χ`, coherence length `ξ` (healthy vs disease) + spatial snapshots (single realization) |
| `fig_3_criticality_ci.py` | Fig. 3 (with CIs) | Same observables as an **ensemble over 20 seeds**, drawn as mean ± 95% CI bands |
| `fig_4_lyapunov_robustness.py` | Fig. 4 | Largest Lyapunov exponent `λ`, disease shift `Δλ`, robustness `R_SC` across network sizes |
| `fig_5_phase_diagram.py` | Fig. 5 | 2D phase diagram of susceptibility `χ` over (ATP `α`, noise `σ`), healthy vs disease |

## Compute / cache / plot

`fig_3_criticality_ci.py` and `fig_5_phase_diagram.py` separate the heavy simulation from plotting.
The first run computes the ensemble (Numba-parallelised over CPU cores) and
caches results to `processed_data/` (`.npz` arrays + a tidy `.csv`); subsequent
runs load the cache and re-render instantly. Force a fresh run with
`--recompute`:

```bash
python fig_3_criticality_ci.py              # load cache if present, else compute, then plot
python fig_3_criticality_ci.py --recompute  # rerun the simulation and overwrite the cache
python fig_5_phase_diagram.py --recompute
```

Set the thread count with `NUMBA_NUM_THREADS=<n>`. Figures are saved as both
vector **PDF** (editable text, `pdf.fonttype=42`) and 300-dpi **PNG**; shared
publication styling lives in `plotstyle.py`.

## Reproducing the environment

With conda/mamba:

```bash
conda env create -f environment.yml
conda activate bayat-et-al
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Running

Each script is standalone and produces one manuscript figure:

```bash
python fig_1_single_cell.py
python fig_2_network_activity.py
python fig_3_criticality.py
python fig_3_criticality_ci.py
python fig_4_lyapunov_robustness.py
python fig_5_phase_diagram.py
```

`fig_3_criticality_ci.py`, `fig_4_lyapunov_robustness.py`, and `fig_5_phase_diagram.py` use
[Numba](https://numba.pydata.org/) JIT compilation and parallelism; the first
run incurs a one-time compilation cost. Publication figures use the Arial font;
if it is not installed, matplotlib falls back to DejaVu Sans (rendering is
unaffected).
