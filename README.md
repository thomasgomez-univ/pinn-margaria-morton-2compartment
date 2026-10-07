# Can a bioenergetic digital twin of an athlete be individualized? Identifiability of a two-compartment model of critical power — a simulation study

Code, synthetic data and analysis outputs accompanying the manuscript

> Gomez, T. (2026). *Can a bioenergetic digital twin of an athlete be individualized? Identifiability of a two-compartment model of critical power — a simulation study.* Submitted to PeerJ.

Archived on Zenodo: version 3.0.0, [10.5281/zenodo.23223356](https://doi.org/10.5281/zenodo.23223356); all versions, [10.5281/zenodo.20076198](https://doi.org/10.5281/zenodo.20076198).

> **Version 3.0.0 is the version used for the submitted manuscript.** It adds
> `campaign_v5/`, the complete estimation campaign and every complementary
> analysis reported in the article and its Supplemental Article S1, together
> with their raw outputs. Version 2.0.0 (`src/mm2c/`, `scripts/`, `data/`) is
> kept unchanged for the structural-identifiability scripts, which do not
> depend on the virtual population; its estimation results are superseded.
> Version 1.0.0 contained an incorrect ODE residual and must not be used.
> See `CHANGES.md`.

## The model

A reduced two-compartment member of the Margaria–Morton family. The
phosphagenic and glycolytic pathways are lumped into a single non-oxidative
reservoir $A_P$; the oxidative reservoir $A_O$ has finite capacity and
first-order replenishment. Five parameters
$\theta = (M_O, A_{O,\max}, A_{P,\max}, M_R, \eta)$ are estimated from
$A_P(t)$ alone.

## Experimental controls

Differences between estimators are meaningful only if nothing else differs.
Seven controls are enforced by `src/mm2c/pipeline.py`:

1. one observation grid, shared by every method (`N_OBS = 60` per trial);
2. one residual definition, shared by every method;
3. one integrator and one tolerance (`rtol = atol = 1e-6`, justified by a
   convergence study against a `1e-9` reference — see
   `scripts/convergence_tolerance.py`);
4. a common computational budget counted in ODE solves, not wall-clock time;
5. all random number generators seeded from the athlete index and noise level;
6. two population baselines that ignore the observations, so that an estimator
   can be shown to extract individual information;
7. the Cramér–Rao lower bound, so that an error becomes a statement about
   statistical efficiency rather than a ranking.

Noise is applied to the **observation** of $A_P$, not to the input power.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU PyTorch is sufficient: each fit is pinned to one thread and parallelism is
between processes.

## Reproducing the submitted manuscript (version 3.0.0)

All quantities of the submitted manuscript come from `campaign_v5/` (estimation,
Fisher information, prediction, departures from the model, protocol design,
oxygen-uptake observables) and from the population-independent structural
analyses of `scripts/`. Shipped `.jsonl`, `.json`, `.npz` and `.log` files are
the exact outputs used in the article, so the analysis scripts (`tables_v4.py`,
`make_figs_v4.py`, the `--analyse` modes and `analyse_*.py`) reproduce the
reported numbers and figures without re-running any estimation. Scripts are
resumable; file names keep their historical `v3`/`v4` prefixes. Code comments
and the two working notes (`LISEZMOI_v5.md`, `complements_v4/LISEZMOI_complements.md`)
are in French.

| Manuscript item | Script(s), run from `campaign_v5/` unless stated |
|---|---|
| Sec. 3.1, structural rank and closed-form inversion | `scripts/identifiability_rank_fispo.py`, `scripts/identifiability_global_elimination.py`, `scripts/identifiability_global.jl` (repository root) |
| Table 2, Fisher blocks (A_P, VO2, T_lim) | `complements_v4/fim_vo2_single_v5.py`, `complements_v4/vo2_tlim_v5.py` |
| Sec. 3.2, Fig. 2 | `complements_v4/fim_v3.py`, then `make_figs_v4.py` |
| Sec. 3.2, profile likelihood, Fig. S4 | `complements_v4/profile_v5.py 0/1/2`, `complements_v4/profile_rebuild_v5.py` |
| Sec. 3.3, Tables 3–4, Figs. 3–4, Table S4, Fig. S1 | `run_campagne.sh` (`pipeline_v3.py`), `run_v4.sh` (`pipeline_v4.py`, corrected network), then `tables_v4.py`, `make_figs_v4.py`, `cos_align_v5.py` |
| Sec. 3.3, repeated realizations | `complements_v4/rep_lm_v3.py` |
| Sec. 3.3, seed variability (DE, network) | `run_rev2.sh`; `complements_v4/de_seeds_v5.py --analyse`, `complements_v4/pinn_seeds_v5.py` |
| Sec. 2.5 and S4, inverse-crime control, residual-weight sweep | `complements_v4/inverse_crime_v3.py`, `complements_v4/wr_sweep_v4.py` |
| Table S2, network diagnostic | `pinn_diag.py` |
| Sec. 3.4, Table 5 | `prediction_v4.py` |
| Sec. 3.5, Tables S5–S6 | `complements_v4/misspec_v4.py`, `analyse_misspec.py`, `misspec_extra_v5.py`, `gls_ar1_v5.py --analyse` |
| Sec. 3.6, Table S3 | `complements_v4/protocoles_v3.py`, `protocoles_extra_v5.py`, `analyse_protocoles_v5.py`, `analyse_protocoles_v5b.py` |
| Section S6, Fig. S2 (power–duration relation) | `complements_v4/pd_curve_v4.py` |
| Section S7, Fig. S3 (simulated oxygen kinetics) | `complements_v4/vo2_kinetics_v5.py` |

The full campaign is chained by `run_v5_all.sh` (3–5 h on a laptop, CPU) and
the second-round complements by `run_rev2.sh` (about 3 h). The virtual
population is `pop_recal.npy` (seed 7). Figures are written to `figures/`.

## Version 2.0.0 material

### Reproducing the version 2.0.0 results

Every script writes to `data/` or `figures/` and is resumable: re-running the
same command skips work already recorded.

| Manuscript item | Command |
|---|---|
| Sec. 3.1, rank criterion | `python3 scripts/identifiability_rank_fispo.py` |
| Sec. 3.1, Eq. (inversion) | `python3 scripts/identifiability_global_elimination.py` |
| Sec. 3.1, tool cross-check | `julia scripts/identifiability_global.jl` |
| Sec. 3.2, Fig. 2a–b | `python3 scripts/fim_analysis.py` |
| Sec. 3.2, Fig. 2c | `python3 scripts/flat_direction.py` |
| Sec. 3.3, Tables 3–5 | `python3 -m mm2c.pipeline --athletes 50 --sigma 2 5 10 --budget 4000 --rtol 1e-6 --out data/campaign_main.jsonl` |
| Sec. 3.3, residual-weight sweep | `python3 scripts/run_sweep.py --param w_r --values 0.03 0.1 0.3 1 3 --athletes 0-14 --out data/sweep_residual_weight.jsonl` |
| Sec. 3.4, Table 6 | `python3 scripts/run_protocol_design.py 15` |
| Sec. 3.5, tolerance | `python3 scripts/convergence_tolerance.py --athletes 3` |
| Figs. 2–4 | `python3 scripts/make_figures.py` |

`scripts/analyse_sweep.py` prints the summary table and the paired Friedman and
Wilcoxon tests for any sweep file.

These commands reproduce the version 2.0.0 results, which the submitted
manuscript no longer uses.

## Layout

```
campaign_v5/           version 3.0.0: campaign and complements of the submitted manuscript
src/mm2c/pipeline.py   model, population, synthetic data, the three estimators,
                       the two baselines, the Fisher information and the CRLB
scripts/               one script per manuscript item (table above)
data/                  results of version 2.0.0 (superseded)
figures/               figures of the submitted manuscript and supplement (v3.0.0)
```

## Caveat

All data here are synthetic: the generating model is assumed to be the true
one. Nothing in this repository has been validated against measurements from
human participants.

## License

MIT, see `LICENSE`.
