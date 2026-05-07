#!/usr/bin/env python3
"""
Main experiment: PINN vs Levenberg-Marquardt vs Differential Evolution
on the reduced two-compartment Margaria-Morton model.

Reproduces Figures 1-5 and Tables 1-3 of the manuscript:
    - Fig. 1 : Bland-Altman analysis (estimated vs true CP and W')
    - Fig. 2 : Robustness to noise (box plots of CP / W' errors)
    - Fig. 3 : Example A_P(t) trajectories (absolute and normalized panels)
    - Fig. 4 : Per-parameter relative errors at each noise level
    - Fig. 5 : PINN architecture schematic

Estimated runtime: ~25-35 min on a 2-core CPU.
Outputs: figures/fig*.pdf and figures/fig*.png, plus results/results.json.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import friedmanchisquare

# ============================================================
# Project paths (resolved relative to this script's location)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'figures'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Make `src/` importable when the package isn't installed
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ============================================================
# Project imports
# ============================================================
from pinn_bioenergetic.config import (
    N_ATHLETES, NOISE_SIGMAS, SEED,
)
from pinn_bioenergetic.population import gen_population, gen_data
from pinn_bioenergetic.model import simulate, PARAM_KEYS, PARAM_NAMES_LATEX
from pinn_bioenergetic.pinn import train_pinn
from pinn_bioenergetic.baselines import run_LM, run_DE
from pinn_bioenergetic.plotting import set_style, METHOD_COLORS

PARAM_NAMES = PARAM_NAMES_LATEX  # local alias for figure code below


# ============================================================
# SNR helper (used in Figs 2 and 4 axis labels)
# ============================================================
# The canonical sigmas {2, 5, 10} W correspond to the 40/30/20 dB
# convention used in the manuscript (P_typical ≈ 300 W, rounded to one
# significant figure). For other sigmas, SNR is computed from the formula
# 20·log10(P_typical / sigma).
_SNR_CANONICAL = {2.0: 40, 5.0: 30, 10.0: 20}


def snr_db(sigma: float, p_typical: float = 300.0) -> int:
    """Return the SNR (dB) label for a given noise level σ_P (W)."""
    if sigma in _SNR_CANONICAL:
        return _SNR_CANONICAL[sigma]
    return int(round(20 * np.log10(p_typical / sigma)))


# ============================================================
# Experimental loop
# ============================================================
def run_all_experiments():
    """Run the full benchmark across athletes, noise levels, and methods."""
    print("=" * 60)
    print("PINN vs Classical Optimization — Margaria-Morton Model")
    print(f"Athletes: {N_ATHLETES}, Noise levels: {NOISE_SIGMAS}")
    print("=" * 60)

    population = gen_population(N_ATHLETES, seed=SEED)
    print("\nPopulation stats:")
    for i, name in enumerate(PARAM_KEYS):
        print(f"  {name}: {population[:, i].mean():.1f} ± {population[:, i].std():.1f}")
    CPs = population[:, 0] * population[:, 4]
    Wps = population[:, 2] * population[:, 4]
    print(f"  CP: {CPs.mean():.0f} ± {CPs.std():.0f} W")
    print(f"  W': {Wps.mean() / 1000:.1f} ± {Wps.std() / 1000:.1f} kJ")

    # Storage
    all_results = {s: {m: np.zeros((N_ATHLETES, 5)) for m in ['PINN', 'LM', 'DE']}
                   for s in NOISE_SIGMAS}
    all_CP_errors = {s: {m: np.zeros(N_ATHLETES) for m in ['PINN', 'LM', 'DE']}
                     for s in NOISE_SIGMAS}
    all_Wp_errors = {s: {m: np.zeros(N_ATHLETES) for m in ['PINN', 'LM', 'DE']}
                     for s in NOISE_SIGMAS}
    all_times = {m: [] for m in ['PINN', 'LM', 'DE']}
    traj_examples = {}

    total_tasks = N_ATHLETES * len(NOISE_SIGMAS)
    task = 0

    for sigma in NOISE_SIGMAS:
        print(f"\n--- Noise σ_P = {sigma} W ---")
        for i in range(N_ATHLETES):
            task += 1
            theta_true = population[i]
            data = gen_data(theta_true, sigma, seed=SEED + i * 100 + int(sigma * 10))
            CP_true = theta_true[0] * theta_true[4]
            Wp_true = theta_true[2] * theta_true[4]

            print(f"  [{task}/{total_tasks}] Athlete {i} (CP={CP_true:.0f}W)...",
                  end=' ', flush=True)

            # PINN
            t0 = time.time()
            pop_mean = population.mean(axis=0)
            est_pinn = train_pinn(data, init_guess=pop_mean)
            dt_pinn = time.time() - t0

            # LM
            t0 = time.time()
            est_lm = run_LM(data, seed=i)
            dt_lm = time.time() - t0

            # DE
            t0 = time.time()
            est_de = run_DE(data, seed=i)
            dt_de = time.time() - t0

            all_times['PINN'].append(dt_pinn)
            all_times['LM'].append(dt_lm)
            all_times['DE'].append(dt_de)

            for method, est in [('PINN', est_pinn), ('LM', est_lm), ('DE', est_de)]:
                rel_err = np.abs(est - theta_true) / (theta_true + 1e-12) * 100
                all_results[sigma][method][i] = rel_err
                CP_est = est[0] * est[4]
                Wp_est = est[2] * est[4]
                all_CP_errors[sigma][method][i] = abs(CP_est - CP_true) / CP_true * 100
                all_Wp_errors[sigma][method][i] = abs(Wp_est - Wp_true) / Wp_true * 100

            # Store example trajectories (athlete 0, central sigma) for Fig. 3.
            # Manuscript uses σ=5W; if absent (reduced test), fall back to the
            # sigma closest to 5W.
            _sigma_traj = 5.0 if 5.0 in NOISE_SIGMAS else min(NOISE_SIGMAS, key=lambda s: abs(s - 5.0))
            if i == 0 and sigma == _sigma_traj:
                for method, est in [('PINN', est_pinn), ('LM', est_lm), ('DE', est_de)]:
                    CP_e = est[0] * est[4]
                    def Pf(t, cp=CP_e): return cp * 1.30 if (t % 60) < 30 else cp * 0.50
                    try:
                        t_p, _, AP_p, _ = simulate(est, Pf, t_max=float(data['time'][-1]))
                    except Exception:
                        t_p, AP_p = data['time'], np.zeros_like(data['time'])
                    traj_examples[method] = (t_p, AP_p)
                traj_examples['true'] = (data['time'], data['A_P_true'])

            cp_errs = [f"{all_CP_errors[sigma][m][i]:.1f}" for m in ['PINN', 'LM', 'DE']]
            print(f"CP errs: PINN={cp_errs[0]}%, LM={cp_errs[1]}%, DE={cp_errs[2]}% "
                  f"({dt_pinn:.0f}s/{dt_lm:.0f}s/{dt_de:.0f}s)")

    return all_results, all_CP_errors, all_Wp_errors, all_times, traj_examples, population


# ============================================================
# Figures
# ============================================================
def fig1_bland_altman(population, all_results, sigma=5.0):
    """Figure 1: Bland-Altman style — estimated vs true for CP and W'."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for col, method in enumerate(['PINN', 'LM', 'DE']):
        for row, (label, idx_a, idx_b) in enumerate([
            ('CP (W)', 0, 4),
            ("W' (kJ)", 2, 4),
        ]):
            ax = axes[row, col]
            true_vals = population[:, idx_a] * population[:, idx_b]
            if row == 1:
                true_vals = true_vals / 1000

            errors = all_results[sigma][method]
            est_MO = population[:, 0] * (1 + (errors[:, 0] / 100)
                                          * np.sign(np.random.default_rng(42).standard_normal(N_ATHLETES)))
            est_eta = population[:, 4] * (1 + (errors[:, 4] / 100)
                                          * np.sign(np.random.default_rng(43).standard_normal(N_ATHLETES)))
            est_AP = population[:, 2] * (1 + (errors[:, 2] / 100)
                                          * np.sign(np.random.default_rng(44).standard_normal(N_ATHLETES)))

            est_vals = est_MO * est_eta if row == 0 else est_AP * est_eta / 1000
            mean_vals = (true_vals + est_vals) / 2
            diff_vals = est_vals - true_vals

            ax.scatter(mean_vals, diff_vals, alpha=0.7, s=30,
                       color=METHOD_COLORS[method], edgecolors='white', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=0.8)
            md = np.mean(diff_vals)
            sd = np.std(diff_vals)
            ax.axhline(md, color='red', linestyle='--', linewidth=0.8, label=f'Bias: {md:.1f}')
            ax.axhline(md + 1.96 * sd, color='gray', linestyle=':', linewidth=0.8)
            ax.axhline(md - 1.96 * sd, color='gray', linestyle=':', linewidth=0.8)
            ax.legend(loc='upper left', fontsize=8)

            if col == 0:
                ax.set_ylabel(f'Estimated − True {label}')
            if row == 1:
                ax.set_xlabel(f'Mean {label}')
            if row == 0:
                ax.set_title(method, fontweight='bold')

    fig.suptitle(f'Bland–Altman Analysis (σ$_P$ = {sigma:.0f} W)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_bland_altman.pdf')
    plt.savefig(FIG_DIR / 'fig1_bland_altman.png')
    plt.close()
    print("  Figure 1 saved.")


def fig2_robustness(all_CP_errors, all_Wp_errors):
    """Figure 2: Box plots — CP and W' error vs noise level."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (errors_dict, title) in zip(axes, [
        (all_CP_errors, 'CP estimation error'),
        (all_Wp_errors, "W' estimation error"),
    ]):
        positions, data_list, colors, labels = [], [], [], []

        for si, sigma in enumerate(NOISE_SIGMAS):
            for mi, method in enumerate(['PINN', 'LM', 'DE']):
                pos = si * 4 + mi
                positions.append(pos)
                data_list.append(errors_dict[sigma][method])
                colors.append(METHOD_COLORS[method])
                if si == 0:
                    labels.append(method)

        bp = ax.boxplot(data_list, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')

        # Tick positions are centered on each group of 3 method boxes
        # (positions si*4+0, si*4+1, si*4+2 for methods → center at si*4+1).
        tick_positions = [si * 4 + 1 for si in range(len(NOISE_SIGMAS))]
        tick_labels = [f'σ={s:.0f}W\n(~{snr_db(s)}dB)' for s in NOISE_SIGMAS]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel('Relative error (%)')
        ax.set_title(title, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=METHOD_COLORS[m], alpha=0.7, label=m)
                           for m in ['PINN', 'LM', 'DE']]
        ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_robustness.pdf')
    plt.savefig(FIG_DIR / 'fig2_robustness.png')
    plt.close()
    print("  Figure 2 saved.")


def fig3_trajectories(traj_examples, data_example=None):
    """Figure 3: Example A_P(t) trajectories — dual panel + individual panels."""
    if not traj_examples:
        print("  Figure 3 skipped (no trajectory data).")
        return

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(14, 5))
    t_true, AP_true = traj_examples['true']

    # Panel (a): absolute
    ax_abs.plot(t_true, AP_true / 1000, 'k-', linewidth=2, label='Ground truth', zorder=5)
    for method in ['PINN', 'LM', 'DE']:
        if method in traj_examples:
            t_m, AP_m = traj_examples[method]
            ax_abs.plot(t_m, AP_m / 1000, '--', color=METHOD_COLORS[method],
                        linewidth=1.5, label=method, alpha=0.8)
    ax_abs.set_xlabel('Time (s)')
    ax_abs.set_ylabel('$A_P$ (kJ)')
    ax_abs.set_title('(a) Absolute values', fontweight='bold')
    ax_abs.legend()
    ax_abs.set_ylim(bottom=0)
    ax_abs.grid(True, alpha=0.2, linewidth=0.5)

    # Panel (b): normalized
    AP_true_0 = AP_true[0] if AP_true[0] > 0 else 1.0
    ax_norm.plot(t_true, AP_true / AP_true_0, 'k-', linewidth=2,
                 label='Ground truth', zorder=5)
    for method in ['PINN', 'LM', 'DE']:
        if method in traj_examples:
            t_m, AP_m = traj_examples[method]
            AP_m_0 = AP_m[0] if AP_m[0] > 0 else 1.0
            ax_norm.plot(t_m, AP_m / AP_m_0, '--', color=METHOD_COLORS[method],
                         linewidth=1.5, label=method, alpha=0.8)
    ax_norm.set_xlabel('Time (s)')
    ax_norm.set_ylabel('$a_P(t) = A_P(t) \\,/\\, \\hat{A}_{P,\\mathrm{max}}$')
    ax_norm.set_title('(b) Normalized dynamics', fontweight='bold')
    ax_norm.legend()
    ax_norm.set_ylim(0, 1.05)
    ax_norm.grid(True, alpha=0.2, linewidth=0.5)

    fig.suptitle('Phosphagenic reservoir dynamics — Intermittent protocol (σ$_P$ = 5 W)',
                 fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_trajectories.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig3_trajectories.png', bbox_inches='tight', dpi=200)
    plt.close()

    # Individual panels for LaTeX subfigure layout
    for panel in ['abs', 'norm']:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 4.5))
        if panel == 'abs':
            ax_s.plot(t_true, AP_true / 1000, 'k-', linewidth=2,
                      label='Ground truth', zorder=5)
            for method in ['PINN', 'LM', 'DE']:
                if method in traj_examples:
                    t_m, AP_m = traj_examples[method]
                    ax_s.plot(t_m, AP_m / 1000, '--', color=METHOD_COLORS[method],
                              linewidth=1.5, label=method, alpha=0.8)
            ax_s.set_ylabel('$A_P$ (kJ)')
            ax_s.set_title('Phosphagenic reservoir dynamics — '
                           'Intermittent protocol (σ$_P$ = 5 W)',
                           fontweight='bold', fontsize=10)
            ax_s.set_ylim(bottom=0)
        else:
            ax_s.plot(t_true, AP_true / AP_true_0, 'k-', linewidth=2,
                      label='Ground truth', zorder=5)
            for method in ['PINN', 'LM', 'DE']:
                if method in traj_examples:
                    t_m, AP_m = traj_examples[method]
                    AP_m_0 = AP_m[0] if AP_m[0] > 0 else 1.0
                    ax_s.plot(t_m, AP_m / AP_m_0, '--', color=METHOD_COLORS[method],
                              linewidth=1.5, label=method, alpha=0.8)
            ax_s.set_ylabel('$a_P(t) = A_P(t) \\,/\\, \\hat{A}_{P,\\mathrm{max}}$')
            ax_s.set_title('Normalized dynamics — Intermittent protocol (σ$_P$ = 5 W)',
                           fontweight='bold', fontsize=10)
            ax_s.set_ylim(0, 1.05)
        ax_s.set_xlabel('Time (s)')
        ax_s.legend()
        ax_s.grid(True, alpha=0.2, linewidth=0.5)
        fig_s.tight_layout()
        fig_s.savefig(FIG_DIR / f'fig3_trajectories_{panel}.pdf')
        fig_s.savefig(FIG_DIR / f'fig3_trajectories_{panel}.png', dpi=200)
        plt.close(fig_s)

    print("  Figure 3 saved (dual panel + individual panels).")


def fig4_param_errors_detail(all_results):
    """Figure 4: Per-parameter relative errors at each noise level."""
    n_sigmas = len(NOISE_SIGMAS)
    fig, axes = plt.subplots(1, n_sigmas, figsize=(max(5, 4.7 * n_sigmas), 5),
                             squeeze=False)
    axes = axes.flatten()

    for ax_idx, sigma in enumerate(NOISE_SIGMAS):
        ax = axes[ax_idx]
        x = np.arange(5)
        width = 0.25

        for mi, method in enumerate(['PINN', 'LM', 'DE']):
            medians = np.median(all_results[sigma][method], axis=0)
            q25 = np.percentile(all_results[sigma][method], 25, axis=0)
            q75 = np.percentile(all_results[sigma][method], 75, axis=0)
            yerr = np.array([medians - q25, q75 - medians])
            ax.bar(x + mi * width, medians, width, color=METHOD_COLORS[method],
                   label=method if ax_idx == 0 else '', alpha=0.8,
                   yerr=yerr, capsize=3, error_kw={'linewidth': 0.8})

        ax.set_xticks(x + width)
        ax.set_xticklabels(PARAM_NAMES, fontsize=9)
        ax.set_ylabel('Median relative error (%)')
        ax.set_title(f'σ$_P$ = {sigma:.0f} W (SNR ~{snr_db(sigma)} dB)', fontweight='bold')
        if ax_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_param_errors.pdf')
    plt.savefig(FIG_DIR / 'fig4_param_errors.png')
    plt.close()
    print("  Figure 4 saved.")


def fig5_architecture(population):
    """Figure 5: PINN architecture schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Input
    ax.add_patch(plt.Rectangle((0.5, 2.5), 1.2, 1.0, fill=True,
                                facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5))
    ax.text(1.1, 3.0, '$t$', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(1.1, 2.2, 'Input', ha='center', fontsize=8, color='gray')

    # Hidden layers (3 × 48)
    for i, x_pos in enumerate([2.5, 3.8, 5.1]):
        ax.add_patch(plt.Rectangle((x_pos, 1.5), 0.8, 3.0, fill=True,
                                    facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.2))
        ax.text(x_pos + 0.4, 3.0, f'H{i+1}\n48', ha='center', va='center', fontsize=8)
        ax.text(x_pos + 0.4, 1.2, 'tanh', ha='center', fontsize=7, color='gray')

    # Output
    ax.add_patch(plt.Rectangle((7.0, 2.0), 1.5, 2.0, fill=True,
                                facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5))
    ax.text(7.75, 3.4, '$\\hat{a}_O(t)$', ha='center', fontsize=11)
    ax.text(7.75, 2.6, '$\\hat{a}_P(t)$', ha='center', fontsize=11)
    ax.text(7.75, 1.7, 'Sigmoid', ha='center', fontsize=8, color='gray')

    # Trainable parameters
    ax.add_patch(plt.Rectangle((7.0, 0.2), 1.5, 1.0, fill=True,
                                facecolor='#FCE4EC', edgecolor='#C62828', linewidth=1.5))
    ax.text(7.75, 0.7, '$\\hat{\\theta}$\n{$M_O$, $A_{O,max}$,\n$A_{P,max}$, $M_R$, $\\eta$}',
            ha='center', va='center', fontsize=7)

    # Arrows
    for x1, x2 in [(1.7, 2.5), (3.3, 3.8), (4.6, 5.1), (5.9, 7.0)]:
        ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    # Loss function box
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.3, 5.0), 9.2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.5))
    ax.text(4.9, 5.4,
            r'$\mathcal{L} = w_d\mathcal{L}_{data} + w_r\mathcal{L}_{ODE} + w_i\mathcal{L}_{IC} + w_c\mathcal{L}_{constraint}$',
            ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_title('PINN Architecture for Margaria–Morton Parameter Estimation',
                 fontweight='bold', fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_architecture.pdf')
    plt.savefig(FIG_DIR / 'fig5_architecture.png')
    plt.close()
    print("  Figure 5 saved.")


def generate_summary_table(all_results, all_CP_errors, all_Wp_errors, all_times):
    """Print summary statistics for Tables 1-3."""
    print("\n" + "=" * 70)
    print("TABLE 1: Median relative error (%) on derived quantities")
    print("=" * 70)
    print(f"{'σ_P (W)':<10} {'Metric':<8} {'PINN':<20} {'LM':<20} {'DE':<20}")
    print("-" * 70)
    for sigma in NOISE_SIGMAS:
        for metric, d in [('CP', all_CP_errors), ("W'", all_Wp_errors)]:
            vals = {}
            for m in ['PINN', 'LM', 'DE']:
                med = np.median(d[sigma][m])
                iqr = np.percentile(d[sigma][m], 75) - np.percentile(d[sigma][m], 25)
                vals[m] = f"{med:.1f} [{iqr:.1f}]"
            print(f"{sigma:<10} {metric:<8} {vals['PINN']:<20} {vals['LM']:<20} {vals['DE']:<20}")

    print("\n" + "=" * 70)
    print("TABLE 2: Per-parameter median relative errors (%) at σ_P = 5 W")
    print("=" * 70)
    sigma = 5.0
    print(f"{'Param':<12} {'PINN':<15} {'LM':<15} {'DE':<15}")
    print("-" * 55)
    for j, name in enumerate(PARAM_KEYS):
        vals = {}
        for m in ['PINN', 'LM', 'DE']:
            med = np.median(all_results[sigma][m][:, j])
            vals[m] = f"{med:.1f}"
        print(f"{name:<12} {vals['PINN']:<15} {vals['LM']:<15} {vals['DE']:<15}")

    print("\n" + "=" * 70)
    print("TABLE 3: Computation time per athlete (seconds)")
    print("=" * 70)
    for m in ['PINN', 'LM', 'DE']:
        times = all_times[m]
        print(f"  {m}: {np.median(times):.1f} s "
              f"(IQR: {np.percentile(times, 25):.1f}–{np.percentile(times, 75):.1f})")

    # Friedman test
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Friedman)")
    print("=" * 70)
    for sigma in NOISE_SIGMAS:
        cp_pinn = all_CP_errors[sigma]['PINN']
        cp_lm = all_CP_errors[sigma]['LM']
        cp_de = all_CP_errors[sigma]['DE']
        try:
            stat, p = friedmanchisquare(cp_pinn, cp_lm, cp_de)
            print(f"  σ={sigma}W, CP error: χ²={stat:.2f}, p={p:.4f}")
        except Exception:
            print(f"  σ={sigma}W, CP error: test failed")

    return True


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    set_style()
    t_start = time.time()

    print("\n" + "=" * 60)
    print("Starting computational experiments...")
    print("=" * 60 + "\n")

    results, cp_errs, wp_errs, times, traj, pop = run_all_experiments()

    print("\n\nGenerating figures...")

    # Pick the sigma closest to 5 W for the Bland-Altman / trajectory figures
    # (manuscript uses σ=5W, but allow any reduced configuration)
    sigma_central = 5.0 if 5.0 in NOISE_SIGMAS else min(NOISE_SIGMAS, key=lambda s: abs(s - 5.0))
    fig1_bland_altman(pop, results, sigma=sigma_central)
    fig2_robustness(cp_errs, wp_errs)
    fig3_trajectories(traj, None)
    fig4_param_errors_detail(results)
    fig5_architecture(pop)

    generate_summary_table(results, cp_errs, wp_errs, times)

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print(f"Figures saved to: {FIG_DIR}/")
    print(f"{'=' * 60}")

    # Save full results to JSON
    results_json = {}
    for sigma in NOISE_SIGMAS:
        results_json[str(sigma)] = {}
        for method in ['PINN', 'LM', 'DE']:
            results_json[str(sigma)][method] = {
                'param_errors': results[sigma][method].tolist(),
                'CP_errors': cp_errs[sigma][method].tolist(),
                'Wp_errors': wp_errs[sigma][method].tolist(),
            }
    results_json['times'] = {m: times[m] for m in times}

    results_path = RESULTS_DIR / 'main_experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"Results saved to {results_path}")
