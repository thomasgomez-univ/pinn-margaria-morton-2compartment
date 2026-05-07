"""
pinn_bioenergetic
=================

Physics-Informed Neural Networks for Parameter Estimation
in a Reduced Two-Compartment Margaria-Morton Bioenergetic Model.

This package accompanies the manuscript:
    Gomez, T. (2026). Physics-Informed Neural Networks for Parameter
    Estimation in a Two-Compartment Bioenergetic Model of Critical Power.
    Computer Methods in Biomechanics and Biomedical Engineering.

Public API
----------
- model:        ODE system and forward simulation
- population:   Virtual athlete population and synthetic data generation
- pinn:         PINN architecture and training
- baselines:    Levenberg-Marquardt and Differential Evolution baselines
- identifiability: Reparameterization for the structural identifiability analysis
- plotting:     Shared figure style
- config:       Centralized hyperparameters
"""

from . import config
from . import model
from . import population
from . import pinn
from . import baselines
from . import transfer
from . import surrogate
from . import identifiability
from . import plotting

__version__ = "1.0.0"
__author__ = "Thomas Gomez"
__email__ = "thomas.gomez@univ-evry.fr"

__all__ = [
    "config",
    "model",
    "population",
    "pinn",
    "baselines",
    "transfer",
    "surrogate",
    "identifiability",
    "plotting",
]
