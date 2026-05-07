# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] — 2026-05-06

Initial release accompanying the manuscript:

> Gomez, T. (2026). *Physics-Informed Neural Networks for Parameter Estimation
> in a Two-Compartment Bioenergetic Model of Critical Power*.
> **Computer Methods in Biomechanics and Biomedical Engineering**.

### Added
- Modular Python package `pinn_bioenergetic` with installable layout
  (`pip install -e .`), structured into nine modules: `config`, `model`,
  `population`, `pinn`, `baselines`, `transfer`, `surrogate`,
  `identifiability`, `plotting`.
- Five reproducible experiment scripts in `experiments/`:
  `run_main_experiment.py`, `run_transfer_multitask.py`,
  `run_transfer_conditional.py`, `run_surrogate.py`,
  `run_identifiability_analysis.py`.
- Full documentation: `README.md`, `docs/reproducibility.md`,
  `DEPLOYMENT_GUIDE.md`, `CITATION.cff`.
- Pinned dependencies (`requirements.txt`, `environment.yml`) matching
  the environment used to produce the manuscript figures.
- MIT license for code; CC BY 4.0 for synthetic data and figures.

### Methodological notes for transparency

During the modularization of the original development scripts, two
reproducibility issues were identified and resolved:

1. **Population sampler harmonized.** The original surrogate
   development scripts (`run_surrogate_v2.py`,
   `run_surrogate_v3b.py`) sampled the virtual athlete population from
   independent univariate Gaussians with a rank-correlation trick
   between :math:`M_O` and :math:`A_{O,\max}`. The remaining methods
   (PINN, LM, DE) sampled from a multivariate-normal distribution with
   physiologically motivated correlations. In this release, all four
   methods share the canonical multivariate-normal sampler
   (``pinn_bioenergetic.population.gen_population``), ensuring that
   per-athlete error metrics are directly comparable across methods.
   Empirically, this harmonization shifts the surrogate's reported CP
   error median by approximately +1 percentage point and tightens the
   W' error by approximately −7 percentage points at σ = 5 W
   (50 athletes); all qualitative conclusions of the manuscript
   (method orderings, structural non-identifiability of A_P,max,
   noise insensitivity) are preserved.

2. **Latent figure-rendering bugs in `run_main_experiment.py`.** Four
   bugs in the figure generators (hardcoded tick positions, hardcoded
   subplot grids, brittle SNR-label lookups, hardcoded sigma selection
   for the trajectory example) were corrected. The default canonical
   configuration (50 athletes × 3 noise levels) renders identically to
   the manuscript figures; reduced configurations (e.g., for smoke
   testing) now also render correctly.
