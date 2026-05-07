"""
Shared plotting style and color palette.

This module centralizes the visual identity of the figures so that all
scripts produce consistent typography, colors, and resolution.
"""

import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt


# ============================================================
# Method color palette (used across all figures)
# ============================================================
METHOD_COLORS = {
    'PINN': '#2196F3',   # blue
    'LM':   '#FF9800',   # orange
    'DE':   '#4CAF50',   # green
}


def set_style() -> None:
    """Apply the publication style used in all manuscript figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
