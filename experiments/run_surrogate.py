#!/usr/bin/env python3
"""
Neural ODE surrogate experiment (Architecture C — Section 3.7 of the manuscript).

The surrogate is a 4×128 network mapping (t, θ_norm) → [a_O, a_P], pre-trained
in pure supervised mode on 1000 LHS-sampled θ vectors and two protocols
(intermittent + constant). Per-athlete parameter estimation freezes the
network and optimizes only θ via Adam with ODE residual regularization.

Reproduces the surrogate column of Tables 1-2 and Table 6 (computational
cost) of the manuscript.

The virtual population is the canonical multivariate-normal sampler
:func:`pinn_bioenergetic.population.gen_population` shared with the rest
of the study, so per-athlete error metrics are directly comparable across
PINN, LM, DE, and surrogate. The synthetic-data generator
:func:`pinn_bioenergetic.surrogate.gen_data_intermittent` restricts
evaluation to a single intermittent-exercise protocol matched to the
surrogate's pre-training distribution.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pinn_bioenergetic.config import SEED, N_ATHLETES, NOISE_SIGMAS
from pinn_bioenergetic.model import PARAM_KEYS
from pinn_bioenergetic.population import gen_population
from pinn_bioenergetic.surrogate import (
    gen_data_intermittent,
    generate_training_data, pretrain_supervised, estimate_params,
)


def main():
    t0_all = time.time()

    print("=" * 60)
    print("Neural ODE Surrogate — Pure Supervised Pre-training")
    print("=" * 60)

    # Phase 1: data generation + supervised pre-training
    print("\n--- Phase 1: Generate clean ODE solutions ---")
    t0 = time.time()
    T, AO, AP, TH = generate_training_data(n_samples=1000, seed=SEED)
    gen_time = time.time() - t0
    print(f"  Data generation: {gen_time:.1f}s")

    print("\n--- Phase 1b: Pure supervised training ---")
    t0 = time.time()
    surrogate = pretrain_supervised(T, AO, AP, TH, n_ep=8000, batch_size=4096)
    train_time = time.time() - t0
    print(f"  Training: {train_time:.1f}s")

    # Persist trained weights for the identifiability experiment
    weights_path = RESULTS_DIR / 'surrogate_weights.pt'
    torch.save(surrogate.state_dict(), weights_path)
    print(f"  Trained weights saved to {weights_path}")

    # Phase 2: per-athlete θ-only optimization
    print("\n--- Phase 2: Per-athlete parameter estimation ---")
    pop = gen_population(N_ATHLETES, seed=SEED)

    results = {sigma: {'cp_err': [], 'wp_err': [], 'param_err': [], 'time': []}
               for sigma in NOISE_SIGMAS}

    total = N_ATHLETES * len(NOISE_SIGMAS)
    count = 0

    for sigma in NOISE_SIGMAS:
        print(f"\n--- Noise σ_P = {sigma} W ---")
        for i in range(N_ATHLETES):
            theta_true = pop[i]
            CP_true = theta_true[0] * theta_true[4]
            Wp_true = theta_true[2]

            data = gen_data_intermittent(theta_true, sigma, seed=SEED + i * 100 + int(sigma * 10))

            t0_ath = time.time()
            theta_est = estimate_params(surrogate, data, n_ep=1500, seed=SEED + i)
            dt = time.time() - t0_ath

            CP_est = theta_est[0] * theta_est[4]
            Wp_est = theta_est[2]
            cp_err = abs(CP_est - CP_true) / CP_true * 100
            wp_err = abs(Wp_est - Wp_true) / Wp_true * 100
            param_errs = np.abs(theta_est - theta_true) / (theta_true + 1e-12) * 100

            results[sigma]['cp_err'].append(cp_err)
            results[sigma]['wp_err'].append(wp_err)
            results[sigma]['param_err'].append(param_errs)
            results[sigma]['time'].append(dt)

            count += 1
            if i % 10 == 0 or i == N_ATHLETES - 1:
                print(f"  [{count}/{total}] Ath {i} (CP={CP_true:.0f}W): "
                      f"CP={cp_err:.1f}%, W'={wp_err:.1f}% ({dt:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("TABLE: Median relative error (%) — Surrogate")
    print("=" * 70)
    print(f"{'σ_P':>6} {'CP med':>8} {'CP IQR':>8} {'Wp med':>8} {'Wp IQR':>8}")
    for sigma in NOISE_SIGMAS:
        cp = np.array(results[sigma]['cp_err'])
        wp = np.array(results[sigma]['wp_err'])
        print(f"{sigma:>6.1f} {np.median(cp):>8.1f} "
              f"{np.percentile(cp, 75) - np.percentile(cp, 25):>8.1f} "
              f"{np.median(wp):>8.1f} "
              f"{np.percentile(wp, 75) - np.percentile(wp, 25):>8.1f}")

    sigma_central = 5.0 if 5.0 in NOISE_SIGMAS else min(NOISE_SIGMAS, key=lambda s: abs(s - 5.0))
    print(f"\nPer-parameter errors at σ={sigma_central:.0f}W:")
    pe = np.array(results[sigma_central]['param_err'])
    for j, name in enumerate(PARAM_KEYS):
        print(f"  {name:<12} {np.median(pe[:, j]):>8.1f}%")

    times = [t for s in NOISE_SIGMAS for t in results[s]['time']]
    print(f"\nTime/athlete: {np.median(times):.1f}s")
    print(f"Pre-training (one-time): {gen_time + train_time:.1f}s")

    # Save
    save = {str(s): {'cp_err': results[s]['cp_err'], 'wp_err': results[s]['wp_err'],
                      'time': results[s]['time']} for s in NOISE_SIGMAS}
    save['pretrain_time'] = gen_time + train_time
    with open(RESULTS_DIR / 'surrogate_results.json', 'w') as f:
        json.dump(save, f, indent=2)

    print(f"\nTotal runtime: {(time.time() - t0_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
