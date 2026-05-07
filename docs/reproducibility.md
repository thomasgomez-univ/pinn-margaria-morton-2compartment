# Reproducibility guide

This document provides detailed instructions for reproducing the figures and quantitative results of the manuscript:

> Gomez, T. (2026). *Physics-Informed Neural Networks for Parameter Estimation in a Two-Compartment Bioenergetic Model of Critical Power*. **Computer Methods in Biomechanics and Biomedical Engineering**.

---

## 1. Environment

### Tested environment
- macOS (Apple Silicon), Linux x86_64
- Python 3.11
- CPU only (no GPU required)

### Pinned dependencies
The exact versions used to produce the manuscript figures:

| Package      | Version   |
|--------------|-----------|
| numpy        | 1.26.4    |
| scipy        | 1.17.1    |
| matplotlib   | 3.10.9    |
| torch        | 2.11.0    |

Reproducibility is bit-identical with these versions; minor drift may occur with other versions (especially older NumPy or PyTorch).

### Installation

```bash
git clone https://github.com/thomasgomez-univ/pinn-margaria-morton-2compartment.git
cd pinn-margaria-morton-2compartment

python3.11 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Verify:
```bash
python -c "import pinn_bioenergetic; print(pinn_bioenergetic.__version__)"
# Expected: 1.0.0
```

---

## 2. Reproducing figures and tables

The default values in `src/pinn_bioenergetic/config.py` correspond exactly to the published configuration. **No modification is required** to reproduce the manuscript.

### 2.1 Main experiment — Figures 1–5, Tables 1–3

```bash
python experiments/run_main_experiment.py
```

**Runtime:** 25–35 minutes on a 2-core CPU.

**Outputs:**
- `figures/fig1_bland_altman.{pdf,png}` — Bland–Altman analysis (Fig. 1)
- `figures/fig2_robustness.{pdf,png}` — Box plots of CP / W' errors vs noise (Fig. 2)
- `figures/fig3_trajectories.{pdf,png}` — Dual-panel A_P(t) example (Fig. 3, combined)
- `figures/fig3_trajectories_abs.{pdf,png}` — Absolute panel (Fig. 3a, standalone)
- `figures/fig3_trajectories_norm.{pdf,png}` — Normalized panel (Fig. 3b, standalone)
- `figures/fig4_param_errors.{pdf,png}` — Per-parameter error bars (Fig. 4)
- `figures/fig5_architecture.{pdf,png}` — PINN architecture schematic (Fig. 5)
- `results/main_experiment_results.json` — Full numerical results

The script also prints Tables 1–3 to stdout and runs Friedman tests for statistical significance.

### 2.2 Transfer learning — Figure 6 (Architecture A)

```bash
python experiments/run_transfer_multitask.py
```

**Runtime:** ~30 minutes.
**Output:** `figures/fig6_transfer_learning.{pdf,png}` + `results/transfer_multitask_results.json`.

### 2.3 Transfer learning — Figure 7 (Architecture B)

```bash
python experiments/run_transfer_conditional.py
```

**Runtime:** ~30 minutes.
**Output:** `figures/fig7_transfer_conditional.{pdf,png}` + `results/transfer_conditional_results.json`.

### 2.4 Surrogate — Surrogate column of Tables 1–2, Table 6

```bash
python experiments/run_surrogate.py
```

**Runtime:** ~10 minutes (1 minute generation + 2 minutes pre-training + ~5 minutes per-athlete optimization).

**Outputs:**
- `results/surrogate_weights.pt` — Cached surrogate weights (used by the identifiability analysis to avoid retraining)
- `results/surrogate_results.json` — Full numerical results

### 2.5 Structural identifiability — Section 4.4

```bash
python experiments/run_identifiability_analysis.py
```

**Runtime:** ~10 minutes if `results/surrogate_weights.pt` exists (i.e., after running `run_surrogate.py`); otherwise the surrogate is retrained first.

**Output:** `results/identifiability_results.json`.

### 2.6 Full reproduction in one shot

```bash
# Approximately 2 hours total
python experiments/run_main_experiment.py
python experiments/run_transfer_multitask.py
python experiments/run_transfer_conditional.py
python experiments/run_surrogate.py
python experiments/run_identifiability_analysis.py
```

---

## 3. Quick smoke test (5 minutes)

To verify the installation without running the full experiments, edit `src/pinn_bioenergetic/config.py` to use a reduced configuration:

```python
N_ATHLETES = 3
NOISE_SIGMAS = [5.0]
PINN_ADAM = 200          # Instead of 8000
PINN_LBFGS = 0
LM_RESTARTS = 2
DE_MAXITER = 5
```

Then run:
```bash
python experiments/run_main_experiment.py
```

Expected outcome: all five figures generated within 5 minutes, with sensible (but noisy) numerical results. Restore the original values before any quantitative comparison with the manuscript.

---

## 4. Reproducibility internals

### 4.1 Random seeds

| Component | Seed | Source |
|---|---|---|
| Virtual population | 42 | `config.SEED` |
| Synthetic data noise (per-athlete) | `42 + i*100 + int(σ*10)` | computed in main script |
| PINN init (per athlete) | derived from athlete index | implicit through PyTorch's RNG state |
| Transfer-learning subsampling | `42 + i*100 + rep*10 + int(frac*100)` | derived in TL scripts |
| Differential Evolution | `i` (athlete index) | `run_DE(data, seed=i)` |
| LM random restarts | `i` | `run_LM(data, seed=i)` |
| Surrogate LHS sampling | 42 | passed to `qmc.LatinHypercube` |
| Surrogate batch sampling | `42 + 777` | dedicated RNG |

### 4.2 ODE integration

- Solver: `scipy.integrate.solve_ivp(method='RK45', max_step=1.0, rtol=1e-7, atol=1e-7)`
- Exhaustion event: `A_P(t) < 0.5 J` (terminal, direction = −1)
- Maximum simulation time: 900 s (Protocol A) / 1200 s (Protocol B intermittent)

### 4.3 Architecture

- **Standard PINN:** 3 hidden layers × 48 neurons, tanh activation, sigmoid output
- **Conditional PINN:** same hidden layers, input dim = 6 (1 time + 5 normalized parameters)
- **Surrogate:** 4 hidden layers × 128 neurons, tanh, sigmoid
- All parameter-bound constraints implemented via sigmoid reparameterization

### 4.4 Training

- Standard PINN: Adam (8000 iter, lr=5e-3, cosine annealing) → L-BFGS (80 closure calls, strong Wolfe)
- ODE residual weight ramps linearly from 0 to 0.3 over the first 15% of Adam iterations
- IC loss weight: 5.0 (constant)
- Gradient clipping: `clip_grad_norm_(..., 1.0)`

---

## 5. Synthetic-data note

The study uses a single canonical virtual-population sampler (`pinn_bioenergetic.population.gen_population`) — a multivariate normal with physiologically motivated correlations — across all four methods (PINN, LM, DE, surrogate). This guarantees that the per-athlete error metrics reported for each method are computed on the same set of synthetic athletes and are therefore directly comparable.

The surrogate experiments use a dedicated synthetic-data generator (`pinn_bioenergetic.surrogate.gen_data_intermittent`) that simulates a single intermittent-exercise protocol (30 s work at 130 % CP / 30 s rest at 50 % CP, fixed 600 s window). This matches the surrogate's pre-training distribution (also intermittent + constant) and is the protocol used to produce the surrogate numbers in Sections 3.7 and 4.4 of the manuscript. The PINN, LM, and DE baselines use the two-protocol `gen_data` from `population.py`, which combines Protocol A (constant 110 % CP) and Protocol B (intermittent) and uses an exhaustion event terminator.

---

## 6. Troubleshooting

### `ModuleNotFoundError: No module named 'pinn_bioenergetic'`

Make sure you ran `pip install -e .` from the repository root after activating your virtual environment.

### Figures look slightly different

- Check Python and dependency versions match those listed in Section 1.
- Confirm `config.py` has not been modified.
- Rare differences may arise from BLAS implementations (OpenBLAS vs MKL); they typically affect non-converged optimizations and not the final reported numbers.

### `RuntimeError: ... torch.autograd ...`

This usually indicates a NumPy/PyTorch version mismatch. Re-install the pinned versions:
```bash
pip install -r requirements.txt --force-reinstall
```

### Long runtime

The default settings target manuscript reproduction. For development, use the smoke-test configuration in Section 3.

---

## 7. Contact

For methodological questions or non-trivial reproducibility issues, contact:

**Thomas Gomez** — [thomas.gomez@univ-evry.fr](mailto:thomas.gomez@univ-evry.fr)

For bug reports, please use GitHub Issues.
