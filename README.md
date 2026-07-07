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
| `figfig1.py` | Fig. 1 | Single-cell calcium traces at low / intermediate / high ATP |
| `figfig2.py` | Fig. 2 | Network spatiotemporal activity + representative traces (low vs high ATP) |
| `figfig3.py` | Fig. 3 | Spatial heterogeneity `S_C`, susceptibility `χ`, coherence length `ξ` (healthy vs disease) + spatial snapshots |
| `figfig4norm2.py` | Fig. 4 | Largest Lyapunov exponent `λ`, disease shift `Δλ`, robustness `R_SC` across network sizes |

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
python figfig1.py
python figfig2.py
python figfig3.py
python figfig4norm2.py
```

`figfig4norm2.py` uses [Numba](https://numba.pydata.org/) JIT compilation; the
first run incurs a one-time compilation cost.
