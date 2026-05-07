"""
Centralized configuration for the PINN Margaria-Morton study.

All hyperparameters, population sizes, training budgets, and seeds are
defined here so that every script and module imports a single source
of truth. Reproducing the manuscript results requires no modification
of these values.

Modify only if you intend to deviate from the published configuration.
"""

# ============================================================
# PINN architecture
# ============================================================
N_HIDDEN = 3        #: Number of hidden layers in the standard PINN.
N_NEURONS = 48      #: Neurons per hidden layer (uniform across all PINN scripts).

# ============================================================
# Main experiment (run_main_experiment.py)
# ============================================================
N_ATHLETES = 50                       #: Size of the virtual athlete population.
NOISE_SIGMAS = [2.0, 5.0, 10.0]       #: Power-output noise levels in W (SNR ≈ 40, 30, 20 dB).
PINN_ADAM = 8000                      #: Adam iterations during PINN training.
PINN_LBFGS = 80                       #: L-BFGS closure calls during PINN refinement.
LM_RESTARTS = 8                       #: Random restarts for the Levenberg–Marquardt baseline.
DE_MAXITER = 60                       #: Maximum generations for Differential Evolution.
DE_POPSIZE = 10                       #: DE population size (multiplier of parameter dim).

# ============================================================
# Transfer learning experiments
# ============================================================
N_POP_PRETRAIN = 25                   #: Athletes used for multi-task pre-training.
N_TARGET = 10                         #: Target athletes for transfer-learning evaluation.
N_REP_TL = 5                          #: Random subsampling repeats per data fraction.
PRETRAIN_EPOCHS = 4000                #: Pre-training epochs (multi-task and conditional).
FINETUNE_EPOCHS = 2000                #: Fine-tuning epochs on target athlete.
SCRATCH_EPOCHS = 2000                 #: From-scratch training epochs (matched budget).
TL_SIGMA = 5.0                        #: Noise level for TL experiments (W).
DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.10]  #: Fractions of individual data tested.

# ============================================================
# Shared
# ============================================================
SEED = 42                             #: Global seed for population generation.
N_COLLOCATION = 50                    #: Collocation points for ODE residual evaluation.
