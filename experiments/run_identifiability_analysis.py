#!/usr/bin/env python3
"""
Structural identifiability analysis (Section 4.4 of the manuscript).

This experiment evaluates whether the per-parameter recovery failure of
the surrogate (W' error ≈ 82%) is a structural property of the model
under partial observation rather than an optimization issue.

Method
------
The surrogate network from :mod:`run_surrogate` is reused (or retrained
if the weight file is missing). The 5-parameter optimization is
reparameterized to identifiable coordinates

    θ_id = [CP, φ_max, A_P_max, τ_rec, η]

instead of the raw coordinates. The optimization is then converted back
to θ_raw at each step to condition the surrogate network. Comparing the
errors in the two coordinate systems quantifies the structural
identifiability gap.

Reproduces the identifiability experiment of Section 4.4 of the
manuscript.
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
from pinn_bioenergetic.population import gen_population
from pinn_bioenergetic.surrogate import (
    NeuralSurrogate, gen_data_intermittent,
    generate_training_data, pretrain_supervised,
)
from pinn_bioenergetic.identifiability import (
    raw_to_id, id_to_raw, estimate_params_id, ID_NAMES,
)
from pinn_bioenergetic.model import PARAM_KEYS as RAW_NAMES


def load_or_train_surrogate(weights_path: Path) -> NeuralSurrogate:
    """Load surrogate weights if available, otherwise train from scratch."""
    surrogate = NeuralSurrogate()
    if weights_path.exists():
        print(f"  Loading pre-trained surrogate from {weights_path}")
        surrogate.load_state_dict(torch.load(weights_path))
        return surrogate, 0.0
    print("  No pre-trained weights found, training surrogate from scratch...")
    t0 = time.time()
    T, AO, AP, TH = generate_training_data(n_samples=1000, seed=SEED)
    surrogate = pretrain_supervised(T, AO, AP, TH, n_ep=8000, batch_size=4096)
    pretrain_time = time.time() - t0
    torch.save(surrogate.state_dict(), weights_path)
    print(f"  Surrogate weights saved to {weights_path}")
    return surrogate, pretrain_time


def main():
    t0_all = time.time()

    print("=" * 60)
    print("Structural Identifiability Analysis")
    print("  Reparameterization: θ_id = [CP, φ_max, A_P_max, τ_rec, η]")
    print("  Surrogate trained in θ_raw, optimization in θ_id")
    print("=" * 60)

    # Phase 1: load or train surrogate
    weights_path = RESULTS_DIR / 'surrogate_weights.pt'
    print("\n--- Phase 1: Surrogate ---")
    surrogate, pretrain_time = load_or_train_surrogate(weights_path)

    # Phase 2: per-athlete estimation in θ_id space
    print("\n--- Phase 2: Per-athlete estimation (θ_id optimization) ---")
    pop = gen_population(N_ATHLETES, seed=SEED)

    results = {sigma: {'cp_err': [], 'wp_err': [],
                        'id_err': [], 'raw_err': [], 'time': []}
               for sigma in NOISE_SIGMAS}

    total = N_ATHLETES * len(NOISE_SIGMAS)
    count = 0

    for sigma in NOISE_SIGMAS:
        print(f"\n--- Noise σ_P = {sigma} W ---")
        for i in range(N_ATHLETES):
            theta_raw_true = pop[i]
            theta_id_true = raw_to_id(theta_raw_true)
            CP_true = theta_id_true[0]
            Wp_true = theta_raw_true[2]

            data = gen_data_intermittent(theta_raw_true, sigma,
                                   seed=SEED + i * 100 + int(sigma * 10))

            t0_ath = time.time()
            id_est, raw_est = estimate_params_id(surrogate, data,
                                                  n_ep=1500, seed=SEED + i)
            dt = time.time() - t0_ath

            CP_est = id_est[0]
            Wp_est = raw_est[2]
            cp_err = abs(CP_est - CP_true) / CP_true * 100
            wp_err = abs(Wp_est - Wp_true) / Wp_true * 100

            id_errs = np.abs(id_est - theta_id_true) / (np.abs(theta_id_true) + 1e-12) * 100
            raw_errs = np.abs(raw_est - theta_raw_true) / (np.abs(theta_raw_true) + 1e-12) * 100

            results[sigma]['cp_err'].append(cp_err)
            results[sigma]['wp_err'].append(wp_err)
            results[sigma]['id_err'].append(id_errs)
            results[sigma]['raw_err'].append(raw_errs)
            results[sigma]['time'].append(dt)

            count += 1
            if i % 10 == 0 or i == N_ATHLETES - 1:
                print(f"  [{count}/{total}] Ath {i} (CP={CP_true:.0f}W): "
                      f"CP={cp_err:.1f}%, W'={wp_err:.1f}% ({dt:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("DERIVED QUANTITIES (CP, W')")
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

    print(f"\nIdentifiable parameter errors at σ={sigma_central:.0f}W:")
    ie = np.array(results[sigma_central]['id_err'])
    for j, name in enumerate(ID_NAMES):
        print(f"  {name:<12} {np.median(ie[:, j]):>8.1f}%")

    print(f"\nRaw parameter errors at σ={sigma_central:.0f}W:")
    re_ = np.array(results[sigma_central]['raw_err'])
    for j, name in enumerate(RAW_NAMES):
        print(f"  {name:<12} {np.median(re_[:, j]):>8.1f}%")

    times = [t for s in NOISE_SIGMAS for t in results[s]['time']]
    print(f"\nTime/athlete: {np.median(times):.1f}s")

    # Save
    save = {str(s): {
        'cp_err': results[s]['cp_err'], 'wp_err': results[s]['wp_err'],
        'time': results[s]['time'],
        'id_err': [e.tolist() for e in results[s]['id_err']],
        'raw_err': [e.tolist() for e in results[s]['raw_err']],
    } for s in NOISE_SIGMAS}
    save['pretrain_time'] = pretrain_time
    save['id_param_names'] = ID_NAMES
    save['raw_param_names'] = RAW_NAMES

    with open(RESULTS_DIR / 'identifiability_results.json', 'w') as f:
        json.dump(save, f, indent=2)

    print(f"\nTotal runtime: {(time.time() - t0_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
