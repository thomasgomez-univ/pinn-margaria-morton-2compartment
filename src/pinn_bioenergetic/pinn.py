"""
Physics-Informed Neural Network (PINN) for parameter estimation.

This module implements the standard PINN architecture used in the main
experiment of the manuscript (Section 2.3): a feedforward network mapping
normalized time :math:`t \\in [0, 1]` to normalized compartment states
:math:`[\\hat{a}_O, \\hat{a}_P] \\in [0, 1]`, with the five model parameters
represented as trainable scalars constrained to physiological bounds via
a sigmoid reparameterization.

Training proceeds in two phases:
    1. Adam (8000 iterations) with cosine annealing and gradient clipping.
    2. L-BFGS (80 closure calls) with strong Wolfe line search.

The composite loss combines data fidelity, ODE residual, and an initial
condition term, with an ODE weight ramp following Section 2.3.4.
"""

import numpy as np
import torch
import torch.nn as nn

from .config import N_HIDDEN, N_NEURONS, N_COLLOCATION, PINN_ADAM, PINN_LBFGS
from .model import PARAM_LB, PARAM_UB


class PINN(nn.Module):
    """Standard PINN architecture for the two-compartment Margaria-Morton model.

    Architecture
    ------------
        - Input: scalar normalized time :math:`t \\in [0, 1]`.
        - Hidden: ``n_hidden`` layers of ``n_neurons`` units, tanh activation.
        - Output: 2 sigmoid units producing :math:`[\\hat{a}_O, \\hat{a}_P]`.
        - Trainable parameters: network weights :math:`\\mathbf{W}` and a
          5-vector :math:`\\mathbf{z}` mapped to the physiological parameter
          vector via :math:`\\theta = \\mathrm{lb} + (\\mathrm{ub} - \\mathrm{lb})
          \\sigma(\\mathbf{z})`.

    Parameters
    ----------
    n_hidden : int, optional
        Number of hidden layers (default from ``config.N_HIDDEN``).
    n_neurons : int, optional
        Neurons per hidden layer (default from ``config.N_NEURONS``).
    """

    def __init__(self, n_hidden: int = N_HIDDEN, n_neurons: int = N_NEURONS):
        super().__init__()
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        self.z = nn.Parameter(torch.zeros(5))
        self.register_buffer('lb', torch.tensor(PARAM_LB, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(PARAM_UB, dtype=torch.float32))

    def params(self) -> torch.Tensor:
        """Return the physiologically constrained parameter vector :math:`\\theta`."""
        return self.lb + (self.ub - self.lb) * torch.sigmoid(self.z)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def train_pinn(data: dict,
               n_adam: int = PINN_ADAM,
               n_lbfgs: int = PINN_LBFGS,
               init_guess: np.ndarray = None) -> np.ndarray:
    """Train the PINN on a single athlete's noisy data and return the parameter estimate.

    Parameters
    ----------
    data : dict
        Dataset produced by :func:`pinn_bioenergetic.population.gen_data`.
    n_adam : int, optional
        Number of Adam iterations (default from config).
    n_lbfgs : int, optional
        Number of L-BFGS closure calls (default from config).
    init_guess : array_like, shape (5,), optional
        Initial guess for :math:`\\theta` (e.g., population mean).
        If ``None``, the network's parameter vector is initialized at the
        center of the physiological range.

    Returns
    -------
    ndarray, shape (5,)
        Estimated parameter vector ``[M_O, A_O_max, A_P_max, M_R, eta]``.
    """
    pinn = PINN()

    if init_guess is not None:
        with torch.no_grad():
            g = torch.tensor(init_guess, dtype=torch.float32)
            f = (g - pinn.lb) / (pinn.ub - pinn.lb)
            f = torch.clamp(f, 0.02, 0.98)
            pinn.z.copy_(torch.log(f / (1 - f)))

    t_raw = data['time']
    AP = data['A_P_true']
    P_ext = data['P_ext_noisy']
    T = float(t_raw[-1])

    # Subsample observations
    step = max(1, len(t_raw) // 60)
    idx = np.arange(0, len(t_raw), step)

    t_d = torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1)
    AP_obs = torch.tensor(AP[idx], dtype=torch.float32).unsqueeze(1)
    AP_scale = float(AP.max()) + 1.0
    AP_norm = AP_obs / AP_scale

    # Collocation grid for ODE residual
    nc = N_COLLOCATION
    t_c = torch.linspace(0, 1, nc).unsqueeze(1).requires_grad_(True)
    P_c = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, P_ext),
                       dtype=torch.float32).unsqueeze(1)

    # ===== Phase 1: Adam =====
    opt = torch.optim.Adam(pinn.parameters(), lr=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_adam)

    for ep in range(n_adam):
        opt.zero_grad()
        p = pinn.params()
        M_O, AO_mx, AP_mx, MR, eta = p

        # Data loss (on A_P observations)
        pred = pinn(t_d)
        loss_d = torch.mean((pred[:, 1:2] * AP_mx / AP_scale - AP_norm) ** 2)

        # ODE residual (autodiff)
        tc = t_c.clone().requires_grad_(True)
        pc = pinn(tc)
        aO, aP = pc[:, 0:1], pc[:, 1:2]
        daO = torch.autograd.grad(aO, tc, torch.ones_like(aO), True, True)[0]
        daP = torch.autograd.grad(aP, tc, torch.ones_like(aP), True, True)[0]

        rO = T * (MR * (1 - aO) - (M_O / AO_mx) * aO * (1 - aP))
        rP = T * ((M_O / AO_mx) * aO * (1 - aP) - P_c / (eta * AP_mx))
        loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

        # Initial condition (full reservoirs)
        p0 = pinn(torch.tensor([[0.0]]))
        loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

        # ODE weight ramp (linear over first 15% of training)
        w_r = 0.3 * min(1.0, ep / (n_adam * 0.15))
        loss = loss_d + w_r * loss_r + 5.0 * loss_i

        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        opt.step()
        sch.step()

    # ===== Phase 2: L-BFGS =====
    opt2 = torch.optim.LBFGS(pinn.parameters(), lr=0.05, max_iter=10,
                              history_size=20, line_search_fn='strong_wolfe')
    for _ in range(n_lbfgs):
        def closure():
            opt2.zero_grad()
            p = pinn.params()
            M_O, AO_mx, AP_mx, MR, eta = p
            pred = pinn(t_d)
            ld = torch.mean((pred[:, 1:2] * AP_mx / AP_scale - AP_norm) ** 2)
            tc2 = t_c.clone().requires_grad_(True)
            pc2 = pinn(tc2)
            daO2 = torch.autograd.grad(pc2[:, 0:1], tc2, torch.ones(nc, 1), True, True)[0]
            daP2 = torch.autograd.grad(pc2[:, 1:2], tc2, torch.ones(nc, 1), True, True)[0]
            rO2 = T * (MR * (1 - pc2[:, 0:1]) - (M_O / AO_mx) * pc2[:, 0:1] * (1 - pc2[:, 1:2]))
            rP2 = T * ((M_O / AO_mx) * pc2[:, 0:1] * (1 - pc2[:, 1:2]) - P_c / (eta * AP_mx))
            lr2 = torch.mean((daO2 - rO2) ** 2 + (daP2 - rP2) ** 2)
            p0 = pinn(torch.tensor([[0.0]]))
            li = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2
            l = ld + 0.3 * lr2 + 5.0 * li
            l.backward()
            return l
        try:
            opt2.step(closure)
        except Exception:
            break

    return pinn.params().detach().numpy()
