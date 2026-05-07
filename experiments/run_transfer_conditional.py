#!/usr/bin/env python3
"""
Transfer learning experiment — Architecture B (conditional PINN).

Pre-trains a network that takes (t, θ_normalized) as input, learning
the family of ODE solutions f(t, θ) → [a_O, a_P]. Fine-tuning only
optimizes θ (5 parameters) with frozen network weights.

Reproduces Figure 7 and the conditional-architecture results in
Section 3.6 of the manuscript.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'figures'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pinn_bioenergetic.config import (
    SEED, N_HIDDEN, N_NEURONS,
    N_POP_PRETRAIN, N_TARGET, N_REP_TL,
    PRETRAIN_EPOCHS, FINETUNE_EPOCHS, SCRATCH_EPOCHS,
    TL_SIGMA, DATA_FRACTIONS,
)
from pinn_bioenergetic.population import gen_population, gen_data
from pinn_bioenergetic.transfer import (
    pretrain_conditional, finetune_conditional, scratch_conditional,
)
from pinn_bioenergetic.plotting import set_style


def main():
    set_style()
    t0_all = time.time()

    n_total = N_POP_PRETRAIN + N_TARGET
    all_pop = gen_population(n_total, seed=SEED + 500)
    pop_tr = all_pop[:N_POP_PRETRAIN]
    pop_te = all_pop[N_POP_PRETRAIN:]

    print("=" * 60)
    print("TRANSFER LEARNING — Conditional PINN f(t, θ) (Architecture B)")
    print(f"Pre-train: {len(pop_tr)} athletes, Target: {len(pop_te)} athletes")
    print(f"Architecture: {N_HIDDEN}×{N_NEURONS}, input dim=6 (t + θ_norm)")
    print("=" * 60)

    # Phase 1: pre-train conditional network
    print("\n--- Phase 1: Pre-training conditional PINN ---")
    t0 = time.time()
    net_pretrained = pretrain_conditional(pop_tr, sigma=TL_SIGMA, n_ep=PRETRAIN_EPOCHS)
    pt_time = time.time() - t0
    print(f"Pre-training completed in {pt_time:.1f}s")

    # Phase 2: θ-only fine-tune vs from-scratch
    print("\n--- Phase 2: Fine-tune (θ only) vs From-scratch ---")
    res_tl = {f: [] for f in DATA_FRACTIONS}
    res_sc = {f: [] for f in DATA_FRACTIONS}

    for i in range(len(pop_te)):
        theta = pop_te[i]
        CP_t = theta[0] * theta[4]
        data = gen_data(theta, TL_SIGMA, seed=SEED + 1000 + i)
        print(f"\nTarget {i} (CP={CP_t:.0f}W):")

        for frac in DATA_FRACTIONS:
            ft_errs, sc_errs = [], []
            for rep in range(N_REP_TL):
                sr = SEED + i * 100 + rep * 10 + int(frac * 100)
                est_ft = finetune_conditional(net_pretrained, data, frac=frac,
                                               n_ep=FINETUNE_EPOCHS, seed=sr)
                cp_e_ft = abs(est_ft[0] * est_ft[4] - CP_t) / CP_t * 100
                res_tl[frac].append(cp_e_ft)
                ft_errs.append(cp_e_ft)

                est_sc = scratch_conditional(data, frac=frac,
                                              n_ep=SCRATCH_EPOCHS, seed=sr)
                cp_e_sc = abs(est_sc[0] * est_sc[4] - CP_t) / CP_t * 100
                res_sc[frac].append(cp_e_sc)
                sc_errs.append(cp_e_sc)

            print(f"  f={frac:.0%}: TL={np.median(ft_errs):.1f}% "
                  f"Scratch={np.median(sc_errs):.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for f in DATA_FRACTIONS:
        tl_med = np.median(res_tl[f])
        sc_med = np.median(res_sc[f])
        advantage = sc_med - tl_med
        print(f"  {f:.0%}: TL={tl_med:.1f}% Scratch={sc_med:.1f}% "
              f"advantage={advantage:+.1f}%")

    # Save results
    save = {str(f): {'tl_cp': res_tl[f], 'sc_cp': res_sc[f]} for f in DATA_FRACTIONS}
    save['pretrain_time'] = pt_time
    save['method'] = 'conditional_pinn'
    with open(RESULTS_DIR / 'transfer_conditional_results.json', 'w') as fout:
        json.dump(save, fout, indent=2)

    # Figure 7
    fracs_pct = np.array(DATA_FRACTIONS) * 100
    ft_meds = [np.median(res_tl[f]) for f in DATA_FRACTIONS]
    ft_q25 = [np.percentile(res_tl[f], 25) for f in DATA_FRACTIONS]
    ft_q75 = [np.percentile(res_tl[f], 75) for f in DATA_FRACTIONS]
    sc_meds = [np.median(res_sc[f]) for f in DATA_FRACTIONS]
    sc_q25 = [np.percentile(res_sc[f], 25) for f in DATA_FRACTIONS]
    sc_q75 = [np.percentile(res_sc[f], 75) for f in DATA_FRACTIONS]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(fracs_pct, ft_meds, 'o-', color='#2196F3', lw=2.5, ms=9,
            label='Conditional PINN + TL (θ only)', zorder=5)
    ax.fill_between(fracs_pct, ft_q25, ft_q75, color='#2196F3', alpha=0.15)
    ax.plot(fracs_pct, sc_meds, 's--', color='#FF5722', lw=2.5, ms=9,
            label='Conditional PINN from scratch', zorder=4)
    ax.fill_between(fracs_pct, sc_q25, sc_q75, color='#FF5722', alpha=0.15)
    ax.set_xlabel('Individual data fraction (%)')
    ax.set_ylabel('CP estimation error (%)')
    ax.set_title('Transfer Learning — Conditional PINN f(t, θ)', fontweight='bold')
    ax.set_xticks(fracs_pct)
    ax.set_xticklabels([f'{f:.0f}%' for f in fracs_pct])
    ax.legend(fontsize=10, loc='upper right')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_transfer_conditional.pdf')
    plt.savefig(FIG_DIR / 'fig7_transfer_conditional.png')
    plt.close()
    print(f"\nFigure saved to {FIG_DIR}/fig7_transfer_conditional.pdf")
    print(f"Total time: {(time.time() - t0_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
