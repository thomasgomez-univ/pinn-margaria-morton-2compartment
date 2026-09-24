# From identifiability to protocol design in a two-compartment bioenergetic model of critical power

Code and synthetic data accompanying the manuscript

> Gomez, T. (2026). *From identifiability to protocol design in a two-compartment
> bioenergetic model of critical power: a simulation study.*

> **Version 2.0.0 supersedes version 1.0.0.** The ODE residual used by the
> physics-informed network in v1.0.0 was incorrect, and the experimental design
> lacked the controls listed below. Results obtained with v1.0.0 are not
> reproducible with this version and should not be used. See `CHANGES.md`.

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

## Reproducing the manuscript

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

The shipped `data/` files are the exact outputs used in the manuscript, so
`make_figures.py` reproduces Figures 2–4 without re-running any estimation.

## Layout

```
src/mm2c/pipeline.py   model, population, synthetic data, the three estimators,
                       the two baselines, the Fisher information and the CRLB
scripts/               one script per manuscript item (table above)
data/                  results used in the manuscript
figures/               figures as they appear in the manuscript
```

## Caveat

All data here are synthetic: the generating model is assumed to be the true
one. Nothing in this repository has been validated against measurements from
human participants.

## License

MIT, see `LICENSE`.
