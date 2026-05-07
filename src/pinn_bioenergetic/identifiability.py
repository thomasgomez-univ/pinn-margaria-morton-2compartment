"""
Structural identifiability analysis: reparameterization to identifiable coordinates.

This module supports the structural identifiability experiment described
in Section 4.4 of the manuscript. The original parameter set
:math:`\\theta_{\\text{raw}} = (M_O, A_{O,\\max}, A_{P,\\max}, M_R, \\eta)`
is non-identifiable from :math:`A_P(t)` alone: products like
:math:`M_O \\cdot \\eta` (which equals critical power CP) are well
constrained by the data, while individual factors :math:`M_O` and
:math:`\\eta` are not.

The reparameterization to identifiable coordinates is

.. math::

    \\theta_{\\text{id}} = (CP, \\, \\varphi_{\\max}, \\, A_{P,\\max}, \\, \\tau_{\\text{rec}}, \\, \\eta)

where

.. math::

    CP = M_O \\cdot \\eta, \\qquad
    \\varphi_{\\max} = M_O / A_{O,\\max}, \\qquad
    \\tau_{\\text{rec}} = 1 / M_R.

Optimizing in :math:`\\theta_{\\text{id}}` space and converting back to
:math:`\\theta_{\\text{raw}}` (to condition the surrogate network) gives a
better-conditioned optimization landscape. The remaining error after this
reparameterization is interpreted as the structural identifiability gap
(detailed in Section 4.4).

Bounds on :math:`\\theta_{\\text{id}}` are chosen to span the same physical
range as the raw bounds.
"""

from typing import Tuple

import numpy as np
import torch


# ============================================================
# Identifiable parameter bounds
# ============================================================
ID_LB = np.array([150.0, 0.005, 30_000.0, 12.5, 0.18])
ID_UB = np.array([450.0, 0.050, 160_000.0, 200.0, 0.32])
ID_NAMES = ['CP', 'phi_max', 'A_P_max', 'tau_rec', 'eta']


# ============================================================
# Numpy conversions
# ============================================================
def raw_to_id(theta_raw: np.ndarray) -> np.ndarray:
    """Convert :math:`\\theta_{\\text{raw}}` to :math:`\\theta_{\\text{id}}` (numpy)."""
    M_O, A_O_max, A_P_max, M_R, eta = theta_raw
    return np.array([M_O * eta, M_O / A_O_max, A_P_max, 1.0 / M_R, eta])


def id_to_raw(theta_id: np.ndarray) -> np.ndarray:
    """Convert :math:`\\theta_{\\text{id}}` to :math:`\\theta_{\\text{raw}}` (numpy)."""
    CP, phi_max, A_P_max, tau_rec, eta = theta_id
    M_O = CP / eta
    A_O_max = M_O / phi_max
    M_R = 1.0 / tau_rec
    return np.array([M_O, A_O_max, A_P_max, M_R, eta])


# ============================================================
# Differentiable conversions for PyTorch optimization
# ============================================================
def id_to_raw_torch(theta_id_bounded: torch.Tensor) -> torch.Tensor:
    """Differentiable :math:`\\theta_{\\text{id}} \\to \\theta_{\\text{raw}}` for autograd."""
    CP, phi_max, A_P_max, tau_rec, eta = theta_id_bounded
    M_O = CP / eta
    A_O_max = M_O / phi_max
    M_R = 1.0 / tau_rec
    return torch.stack([M_O, A_O_max, A_P_max, M_R, eta])


def make_theta_id_raw(guess_id: np.ndarray) -> torch.nn.Parameter:
    """Initialize a trainable raw-θ_id parameter from a physical guess in id space."""
    lb = torch.tensor(ID_LB, dtype=torch.float32)
    ub = torch.tensor(ID_UB, dtype=torch.float32)
    g = torch.tensor(np.array(guess_id), dtype=torch.float32)
    f = torch.clamp((g - lb) / (ub - lb), 0.02, 0.98)
    return torch.nn.Parameter(torch.log(f / (1 - f)))


# ============================================================
# Hybrid estimation: θ_id optimization with surrogate trained on θ_raw
# ============================================================
def estimate_params_id(net,
                       data: dict,
                       n_ep: int = 1500,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """θ_id optimization with frozen surrogate trained on θ_raw (the v3b hybrid).

    The surrogate network has been trained on parameters drawn from the
    raw bounds (good coverage of the trajectory manifold). Optimization
    operates in the better-conditioned θ_id space and converts back to
    θ_raw at each step to condition the network.

    Parameters
    ----------
    net : pinn_bioenergetic.surrogate.NeuralSurrogate
        Pre-trained surrogate (must expose ``forward_raw``).
    data : dict
        Target athlete's dataset.
    n_ep : int
        θ-optimization epochs.
    seed : int
        Random seed.

    Returns
    -------
    theta_id : ndarray, shape (5,)
        Estimated parameter vector in identifiable coordinates.
    theta_raw : ndarray, shape (5,)
        Same estimate converted back to raw coordinates.
    """
    from .config import N_COLLOCATION
    from .model import PARAM_LB, PARAM_UB

    for p in net.parameters():
        p.requires_grad = False

    lb_id = torch.tensor(ID_LB, dtype=torch.float32)
    ub_id = torch.tensor(ID_UB, dtype=torch.float32)
    lb_raw = torch.tensor(PARAM_LB, dtype=torch.float32)
    ub_raw = torch.tensor(PARAM_UB, dtype=torch.float32)

    # Initialize at center of θ_id space
    guess_id = (ID_LB + ID_UB) / 2
    theta_id_z = make_theta_id_raw(guess_id)

    t_raw_arr, AP_true, Pn = data['time'], data['A_P_true'], data['P_ext_noisy']
    T = float(t_raw_arr[-1])
    step = max(1, len(t_raw_arr) // 80)
    idx = np.arange(0, len(t_raw_arr), step)

    sc = float(AP_true.max()) + 1.0
    t_d = torch.tensor(t_raw_arr[idx] / T, dtype=torch.float32).unsqueeze(1)
    ap_n = torch.tensor(AP_true[idx] / sc, dtype=torch.float32).unsqueeze(1)

    nc = N_COLLOCATION
    tc = torch.linspace(0, 1, nc).unsqueeze(1)
    Pc = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw_arr, Pn),
                      dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam([theta_id_z], lr=2e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

    for ep in range(n_ep):
        opt.zero_grad()

        # θ_id_z → bounded θ_id → θ_raw (differentiable chain)
        theta_id = lb_id + (ub_id - lb_id) * torch.sigmoid(theta_id_z)
        theta_raw = id_to_raw_torch(theta_id)
        theta_raw_clamped = torch.max(torch.min(theta_raw, ub_raw), lb_raw)
        M_O, AO, AP_mx, MR, eta = theta_raw_clamped

        pred = net.forward_raw(t_d, theta_raw_clamped)
        loss_d = torch.mean((pred[:, 1:2] * AP_mx / sc - ap_n) ** 2)

        tc2 = tc.clone().requires_grad_(True)
        pc = net.forward_raw(tc2, theta_raw_clamped)
        daO = torch.autograd.grad(pc[:, 0:1], tc2, torch.ones(nc, 1),
                                   create_graph=True, retain_graph=True)[0]
        daP = torch.autograd.grad(pc[:, 1:2], tc2, torch.ones(nc, 1),
                                   create_graph=True, retain_graph=True)[0]
        rO = T * (MR * (1 - pc[:, 0:1]) - (M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]))
        rP = T * ((M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]) - Pc / (eta * AP_mx))
        loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

        p0 = net.forward_raw(torch.tensor([[0.0]]), theta_raw_clamped)
        loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

        w = 0.3 * min(1.0, ep / (n_ep * 0.1))
        loss = loss_d + w * loss_r + 5.0 * loss_i
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta_id_z], 1.0)
        opt.step()
        sch.step()

    for p in net.parameters():
        p.requires_grad = True

    with torch.no_grad():
        theta_id_final = lb_id + (ub_id - lb_id) * torch.sigmoid(theta_id_z)
        theta_raw_final = id_to_raw_torch(theta_id_final)

    return theta_id_final.numpy(), theta_raw_final.numpy()
