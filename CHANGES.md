# Changelog

All notable changes to this project will be documented in this file.

## [3.1.0] — 2026-10-09

Revision of the manuscript after review. The estimation campaign is unchanged;
this version adds the analyses of the revision and corrects the name of the
least-squares estimator.

### Added
- `campaign_v5/make_population_v5.py`: generator of the virtual population,
  reproducing `pop_recal.npy` exactly (`--check`), with options for other
  seeds, a CP–`A_P_max` correlation and another range of `1/M_R`.
- `campaign_v5/complements_v4/rev3_cp_pd.py`: Cramér–Rao bounds on the
  critical power and W' that a two-parameter power–duration test returns.
- `campaign_v5/complements_v4/rev3_sigma_fim.py`: Fisher information with the
  covariance of the `A_P` deviations induced by power-meter noise (Monte Carlo),
  with shrinkage and nugget regularization.
- `campaign_v5/complements_v4/rev3_tlim_div.py`: equilibria at constant power,
  Jacobian eigenvalues, logarithmic divergence of the time to exhaustion near
  critical power (Section S10 of the supplement).
- `campaign_v5/complements_v4/rev3_censoring.py`: censoring at the 3600 s horizon
  in the out-of-sample predictions (Table 5).
- `campaign_v5/complements_v4/rev3_binom.py`: exact (Clopper–Pearson) confidence
  intervals on every coverage reported.
- `campaign_v5/complements_v4/rev3_fgls.py`: feasible generalized least squares
  under AR(1) noise, `phi` estimated from the residuals (Cochrane–Orcutt);
  sweeps of `misspec_v4.py` in `phi` (0.3, 0.6, 0.95) and in the efficiency
  drift (2.5, 10 %), outputs `misspec_v4_*_phi*.jsonl`, `misspec_v4_*_d0*.jsonl`
  (Table S7).
- `campaign_v5/complements_v4/rev3_population.py`: Fisher information and
  least-squares fit on four alternative populations (Table S10, Section S12).
- `scripts/identifiability_global.log`: output of StructuralIdentifiability.jl
  (Julia 1.13.1) confirming global identifiability of the five parameters and
  both states, with the initial states unknown and known (Section S11).
- Raw outputs (`rev3_*.jsonl`, `rev3_*.json`) and logs of all the above.

### Changed
- The bound-constrained least-squares estimator is now named by the algorithm
  actually used, the trust-region reflective method of
  `scipy.optimize.least_squares` (`method="trf"`; Branch, Coleman & Li, 1999),
  instead of "Levenberg–Marquardt". The code is unchanged; the label `LM` in
  file and field names is kept for continuity. Figures 3, 4 and S1 relabelled
  (`make_figs_v4.py`).

## [3.0.0] — 2026-10-07

Release accompanying the manuscript submitted to PeerJ:

> Gomez, T. (2026). *Which quantities of a bioenergetic digital twin of an athlete can be individualized? Identifiability of a two-compartment model of critical power — a simulation study.*

Version 2.0.0 was tagged locally but never archived; 3.0.0 is the first
archived release since 1.0.0.

### Added
- `campaign_v5/`: complete estimation campaign (Levenberg–Marquardt,
  Differential Evolution, physics-informed network as submitted, two
  population baselines) and corrected physics-informed network, with raw
  outputs (`camp_v3_*.jsonl`, `camp_v4_*.jsonl`) and analysis scripts.
- `campaign_v5/complements_v4/`: Fisher information and least constrained
  direction; profile likelihood of `A_O_max` and `M_R`; practical
  identifiability under oxygen uptake and time to exhaustion; repeated
  realizations; seed variability of DE and of the network; inverse-crime
  control; residual-weight sweep; departures from the generating model
  (AR(1) noise, efficiency drift) with generalized least squares; protocol
  design (24 designs); power–duration relation; simulated oxygen kinetics.
- Figures of the submitted manuscript and supplement in `figures/`.

### Changed
- **Virtual population** recalibrated on critical power
  (N(300, 45) W truncated to 216–402 W) and on the steady oxidative fraction
  (U[0.78, 0.88]) instead of `M_O * eta` (`pop_recal.npy`, seed 7).
- **Noise calibration.** The standard deviation of the `A_P` observation
  propagated from the power-meter noise is computed on the exact power
  (switching instants preserved) and only on grid points before the earliest
  exhaustion of the noisy realizations. The earlier calibration inflated it
  about 2.5-fold for the intermittent protocol at 5 W.
- **Physics-informed network (corrected version).** Fitted to both protocols
  with a shared parameter vector; periodic input features; initial condition
  imposed by the ansatz; parameters frozen during the first 2000 iterations;
  300 collocation points per protocol.

## [2.0.0] — 2026-09-24

Release accompanying the revised manuscript:

> Gomez, T. (2026). *From identifiability to protocol design in a two-compartment
> bioenergetic model of critical power: a simulation study.*

This version **supersedes 1.0.0 and is not backward compatible with it**.
Results produced by 1.0.0 cannot be reproduced here and should not be used.

### Fixed
- **ODE residual of the physics-informed network.** The residual on the
  non-oxidative compartment divided the inter-compartment flow by
  `A_O_max` instead of `A_P_max`. The error affected both the training loss
  and the fine-tuning loss (`src/pinn_bioenergetic/pinn.py`, four sites in
  1.0.0). On the exact trajectory the erroneous residual has an RMS value of
  1.19 (20.9 % in relative terms) instead of zero, so the true parameter
  vector was penalized by the physics loss. The error factor
  `A_P_max / A_O_max` ranges over [0.78, 2.20] across the virtual population,
  median 1.31.

### Changed
- **Noise model.** In 1.0.0 the perturbation was applied to the input power
  while every method was fitted to noiseless observations of `A_P`. The
  reported robustness therefore described sensitivity to input error, not to
  measurement noise. Noise is now applied to the observation of `A_P`, with
  the standard deviation propagated from `sigma_P` by Monte Carlo.
- **Experimental controls.** Estimators now share one observation grid, one
  residual definition, one integrator and tolerance, and a computational
  budget counted in ODE solves. See the README.
- **Integration tolerance.** Raised from `1e-4` to `1e-6`, justified by a
  convergence study against a `1e-9` reference: at `1e-4` the estimates
  deviate from the reference by 2.64 %, which exceeds the estimation error
  itself.
- **Integration scheme.** Piecewise integration between the switching instants
  of the piecewise-constant input replaces a forced `max_step`.

### Added
- Two population baselines that ignore the observations.
- Fisher information matrix, eigenvalue spectrum, least constrained direction
  and Cramér–Rao lower bound.
- Global structural identifiability by differential elimination, with an
  explicit inversion of the parameters from the coefficients of the
  input–output relation.
- Protocol design: seventeen candidate designs compared at a fixed total
  observation budget.
- Sweep of the residual weight over two decades.

### Removed
- The learned trajectory surrogate, the multi-task and conditional transfer
  learning experiments, and the associated modules. They are not reported in
  the revised manuscript. Version 1.0.0 remains permanently archived at
  <https://doi.org/10.5281/zenodo.20076199> for the record.

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
