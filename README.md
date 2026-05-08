# Physics-Informed Neural Networks for Parameter Estimation in a Two-Compartment Bioenergetic Model of Critical Power

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20076199.svg)](https://doi.org/10.5281/zenodo.20076199)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository contains the source code, synthetic data, and experiment scripts accompanying the manuscript:

> Gomez, T. (2026). *Physics-Informed Neural Networks for Parameter Estimation in a Two-Compartment Bioenergetic Model of Critical Power. Manuscript 
submitted for peer review..

---

## Overview

This study evaluates whether Physics-Informed Neural Networks (PINNs) can address the problem of estimating physiological parameters from noisy field-like data in a reduced two-compartment Margaria–Morton bioenergetic model of high-intensity exercise. Four PINN-based approaches are benchmarked against two classical optimization baselines on synthetic data from 50 virtual cyclists across three noise levels (SNR ≈ 40, 30, 20 dB).

The repository enables exact reproduction of all figures and quantitative results reported in the manuscript, including the structural identifiability analysis of Section 4.4.

## Methods compared

| Method | Description | Manuscript |
|---|---|---|
| **PINN (standard)** | Time-only network with composite data + ODE + IC loss | Section 2.3 |
| **PINN — Multi-task TL** | Pre-train shared weights with per-athlete θ, fine-tune with discriminative LR | Architecture A |
| **PINN — Conditional TL** | Network conditioned on (t, θ); θ-only fine-tuning with frozen weights | Architecture B |
| **Surrogate** | LHS-pretrained 4×128 network; θ-only optimization with ODE regularization | Section 3.7 |
| **Levenberg–Marquardt** | `scipy.optimize.least_squares` with Trust Region Reflective and 8 random restarts | Baseline |
| **Differential Evolution** | `scipy.optimize.differential_evolution` (best1bin, polish=True) | Baseline |

## Repository structure

```
.
├── README.md                       # This file
├── LICENSE                         # MIT (code)
├── CITATION.cff                    # Citation metadata
├── pyproject.toml                  # Package metadata and build configuration
├── requirements.txt                # Pinned Python dependencies
├── environment.yml                 # Conda environment specification
├── .gitignore
│
├── src/pinn_bioenergetic/          # Importable Python package
│   ├── __init__.py
│   ├── config.py                   # Centralized hyperparameters (single source of truth)
│   ├── model.py                    # ODE system + simulate()
│   ├── population.py               # Virtual athlete population + synthetic data
│   ├── pinn.py                     # Standard PINN architecture + train_pinn()
│   ├── baselines.py                # Levenberg–Marquardt + Differential Evolution
│   ├── transfer.py                 # MultiTaskPINN + ConditionalPINN + 6 routines
│   ├── surrogate.py                # NeuralSurrogate + supervised pre-training
│   ├── identifiability.py          # θ_id reparameterization for Section 4.4
│   └── plotting.py                 # Shared figure style and color palette
│
├── experiments/                    # Reproducible experiment scripts
│   ├── run_main_experiment.py            # Figures 1–5, Tables 1–3 (~25–35 min)
│   ├── run_transfer_multitask.py         # Figure 6 (~30 min)
│   ├── run_transfer_conditional.py       # Figure 7 (~30 min)
│   ├── run_surrogate.py                  # Surrogate column of Tables 1–2, Table 6
│   └── run_identifiability_analysis.py   # Section 4.4 results
│
├── docs/
│   └── reproducibility.md          # Step-by-step reproduction guide
│
├── figures/                        # Generated figures (PDF + PNG)
├── results/                        # Numerical outputs (JSON) + cached weights
└── data/synthetic/                 # Generated synthetic datasets
```

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/thomasgomez-univ/pinn-margaria-morton-2compartment.git
cd pinn-margaria-morton-2compartment

# Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate            # On Windows: venv\Scripts\activate

# Install the package in editable mode with all dependencies
pip install -e .
```

Alternatively with conda:
```bash
conda env create -f environment.yml
conda activate pinn-bioenergetic
pip install -e .
```

### 2. Verify the installation

```bash
python -c "import pinn_bioenergetic; print(pinn_bioenergetic.__version__)"
```

### 3. Reproduce a single experiment

```bash
python experiments/run_main_experiment.py
```

Figures will be written to `figures/` and numerical results to `results/`.

## Reproducing the manuscript

Each script reproduces a specific portion of the manuscript and can be run independently. All scripts use the centralized configuration in `src/pinn_bioenergetic/config.py`, which is set to the values used in the published study.

| Script | Reproduces | Approx. runtime (CPU) |
|---|---|---|
| `run_main_experiment.py` | Figures 1–5, Tables 1–3 | 25–35 min |
| `run_transfer_multitask.py` | Figure 6 | ~30 min |
| `run_transfer_conditional.py` | Figure 7 | ~30 min |
| `run_surrogate.py` | Surrogate column of Tables 1–2 + Table 6 | ~10 min |
| `run_identifiability_analysis.py` | Section 4.4 | ~10 min (loads cached surrogate) |

Total runtime for full reproduction: **roughly 2 hours on a 2-core CPU**. No GPU required.

For a quick smoke test, see `docs/reproducibility.md`.

## Reproducibility notes

- All random seeds are fixed; with the same Python + dependency versions, results are bit-identical.
- The virtual population is generated with `seed = 42` (multivariate-normal sampler with rank correlations defined in `src/pinn_bioenergetic/population.py`).
- PINN training uses deterministic seeds derived from `(athlete_id, noise_level)`.
- Transfer-learning subsampling repetitions use seeds 0–4 within each `(target, frac)` cell.
- Numerical integration uses SciPy `solve_ivp` with the adaptive Dormand–Prince method (RK45, `rtol=atol=1e-7`) and an exhaustion event at `A_P < 0.5 J`.

### Synthetic-data note

All four methods (PINN, LM, DE, surrogate) are evaluated on the same canonical virtual population (`gen_population` in `population.py`), so per-athlete error metrics are directly comparable across methods. The surrogate experiments use a dedicated synthetic-data generator (`gen_data_intermittent` in `surrogate.py`) that restricts evaluation to a single intermittent protocol (30 s / 30 s at 130 % / 50 % CP, fixed 600 s window), matched to the surrogate's pre-training distribution. The standard PINN, LM, and DE baselines use the two-protocol `gen_data` from `population.py` (Protocol A + Protocol B with exhaustion event).

## Software environment

| Package      | Version   |
|--------------|-----------|
| Python       | 3.11      |
| numpy        | 1.26.4    |
| scipy        | 1.17.1    |
| matplotlib   | 3.10.9    |
| torch        | 2.11.0    |

Full pinned versions: `requirements.txt`. These versions match the environment used to generate the figures in the manuscript.

## Citation

If you use this code or data, please cite the article:

```bibtex
@unpublished{Gomez2026PINN,
  author  = {Gomez, Thomas},
  title   = {Physics-Informed Neural Networks for Parameter Estimation in a
             Two-Compartment Bioenergetic Model of Critical Power},
  year    = {2026},
  note    = {Manuscript submitted for peer review}
}
```

A `CITATION.cff` is included for GitHub's citation generator.

## License

- **Source code**: MIT License (see `LICENSE`).
- **Synthetic data and generated figures**: Creative Commons Attribution 4.0 (CC BY 4.0).

## Author

**Thomas Gomez**
Université Paris-Saclay, Évry, France
Icam, Grand Paris Sud, France
[thomas.gomez@univ-evry.fr](mailto:thomas.gomez@univ-evry.fr)

## Acknowledgments

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Contributing

This repository accompanies a published study; active development is not planned. Reproducibility issues and bug reports are welcome via GitHub Issues. For methodological questions, please contact the corresponding author.
