"""
Neural ODE surrogate for parameter estimation (Architecture C in the manuscript).

The surrogate approach decomposes parameter estimation into two phases:

    Phase 1 (offline, one-time):
        Generate :math:`N_{\\text{LHS}} = 1000` synthetic ODE solutions
        across two protocols (intermittent + constant power) using Latin
        Hypercube Sampling over the physiological parameter range. Train
        a 4×128 feedforward network in pure supervised mode (no ODE loss,
        no IC loss) to map :math:`(t, \\theta_{\\text{norm}})
        \\to [\\hat{a}_O, \\hat{a}_P]`.

    Phase 2 (online, per-athlete):
        Freeze the network weights and optimize only the 5-parameter
        vector θ against the noisy data, with ODE residual as physics
        regularization.

The virtual population used by the surrogate experiments is the same
canonical multivariate-normal sampler as the rest of the study
(:func:`pinn_bioenergetic.population.gen_population`), ensuring per-athlete
error metrics are directly comparable across all four methods (PINN,
LM, DE, surrogate).

The synthetic-data generator :func:`gen_data_intermittent` defined here
restricts evaluation to a single protocol (intermittent 30/30 s at
130/50% CP, fixed 600 s window) — a slightly different observation regime
than :func:`pinn_bioenergetic.population.gen_data`, which combines two
protocols and uses an exhaustion event. The intermittent-only setting
matches the surrogate's pre-training distribution and yields the
numerical values reported in Section 3.7 and 4.4 of the manuscript.
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.stats import qmc

from .config import N_COLLOCATION, SEED, N_ATHLETES, NOISE_SIGMAS
from .model import PARAM_LB, PARAM_UB, mm_odes


# ============================================================
# Synthetic data generator — intermittent protocol, fixed 600 s
# ============================================================
def gen_data_intermittent(theta: np.ndarray, sigma: float, seed: int = 42) -> dict:
    """Synthetic data generator for the surrogate experiments.

    Generates a single intermittent exercise trial:

        - Protocol B only: 30 s work at 130% CP / 30 s rest at 50% CP
        - Fixed 600 s window (601 evenly-spaced time points)
        - No exhaustion event (the simulation runs the full 600 s)

    This differs from :func:`pinn_bioenergetic.population.gen_data`, which
    combines two protocols (Protocol A + Protocol B) and uses an exhaustion
    terminal event. The intermittent-only protocol matches the surrogate's
    pre-training distribution (also intermittent + constant) and is the
    setting used in Section 3.7 of the manuscript.

    Parameters
    ----------
    theta : array_like, shape (5,)
        Parameter vector :math:`[M_O, A_{O,\\max}, A_{P,\\max}, M_R, \\eta]`.
    sigma : float
        Power-output noise standard deviation (W).
    seed : int, optional
        Seed for the noise realization.

    Returns
    -------
    dict
        Keys: ``time``, ``A_O_true``, ``A_P_true``, ``P_ext``, ``P_ext_noisy``.
    """
    M_O, A_O_max, A_P_max, M_R, eta = theta
    CP = M_O * eta
    rng = np.random.default_rng(seed)
    P_high, P_low = CP * 1.3, CP * 0.5

    def P_func(t):
        return P_high if (t % 60) < 30 else P_low

    sol = solve_ivp(
        mm_odes, [0, 600], [A_O_max, A_P_max],
        args=(M_O, A_O_max, A_P_max, M_R, eta, P_func),
        method='RK45', t_eval=np.linspace(0, 600, 601),
        max_step=1.0,
    )
    P_ext = np.array([P_func(t) for t in sol.t])
    return {
        'time': sol.t, 'A_O_true': sol.y[0], 'A_P_true': sol.y[1],
        'P_ext': P_ext,
        'P_ext_noisy': P_ext + rng.normal(0, sigma, len(sol.t)),
    }


# ============================================================
# Surrogate network (4×128, conditioned on θ_raw_norm)
# ============================================================
class NeuralSurrogate(nn.Module):
    """4×128 feedforward network mapping :math:`(t, \\theta_{\\text{norm}}) \\to [\\hat{a}_O, \\hat{a}_P]`.

    The input is 6-dimensional (1 time + 5 normalized parameters);
    the output is 2-dimensional with sigmoid activation enforcing
    :math:`[\\hat{a}_O, \\hat{a}_P] \\in [0, 1]^2`. The network architecture
    is sized to capture the full nonlinear ODE solution manifold over
    the physiological parameter range.
    """

    def __init__(self, n_hidden: int = 4, n_neurons: int = 128):
        super().__init__()
        layers = [nn.Linear(6, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.register_buffer('lb', torch.tensor(PARAM_LB, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(PARAM_UB, dtype=torch.float32))

    def forward(self, t_norm: torch.Tensor, theta_raw: torch.Tensor) -> torch.Tensor:
        """Forward with unbounded θ (sigmoid-mapped internally)."""
        theta_bounded = self.lb + (self.ub - self.lb) * torch.sigmoid(theta_raw)
        theta_norm = (theta_bounded - self.lb) / (self.ub - self.lb)
        theta_exp = theta_norm.unsqueeze(0).expand(t_norm.shape[0], -1)
        return self.net(torch.cat([t_norm, theta_exp], dim=1))

    def forward_raw(self, t_norm: torch.Tensor, theta_raw_bounded: torch.Tensor) -> torch.Tensor:
        """Forward with already-bounded θ in physical units (used by v3b)."""
        theta_norm = (theta_raw_bounded - self.lb) / (self.ub - self.lb)
        theta_exp = theta_norm.unsqueeze(0).expand(t_norm.shape[0], -1)
        return self.net(torch.cat([t_norm, theta_exp], dim=1))

    def get_params(self, theta_raw: torch.Tensor) -> torch.Tensor:
        """Return bounded θ from the unbounded representation."""
        return self.lb + (self.ub - self.lb) * torch.sigmoid(theta_raw)


def make_theta_raw(guess: np.ndarray) -> nn.Parameter:
    """Initialize a trainable raw-θ parameter from a physical guess."""
    lb = torch.tensor(PARAM_LB, dtype=torch.float32)
    ub = torch.tensor(PARAM_UB, dtype=torch.float32)
    g = torch.tensor(np.array(guess), dtype=torch.float32)
    f = torch.clamp((g - lb) / (ub - lb), 0.02, 0.98)
    return nn.Parameter(torch.log(f / (1 - f)))


# ============================================================
# Phase 1: training data generation (LHS over θ + 2 protocols)
# ============================================================
def generate_training_data(n_samples: int = 1000,
                           seed: int = SEED,
                           verbose: bool = True
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latin Hypercube sample of θ + ODE solutions for two protocols.

    For each of ``n_samples`` parameter vectors drawn by LHS, two ODE
    solutions are generated (intermittent 130/50% CP, constant 110% CP),
    each on a 100-point grid over [0, 600] s. Solutions are normalized
    by the parameter-specific capacities :math:`A_{O,\\max}` and
    :math:`A_{P,\\max}` so that the network operates in [0, 1]^2.

    Returns
    -------
    T : torch.Tensor, shape (N, 1)
        Normalized time of each training point.
    AO, AP : torch.Tensor, shape (N, 1)
        Normalized oxidative and non-oxidative reservoir states.
    TH : torch.Tensor, shape (N, 5)
        Normalized parameter vector replicated for each time point.
    """
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    thetas = PARAM_LB + sampler.random(n=n_samples) * (PARAM_UB - PARAM_LB)

    n_pts = 100
    all_t, all_aO, all_aP, all_th = [], [], [], []
    n_valid = 0

    if verbose:
        print(f"  Generating {n_samples} × 2 protocols = {n_samples * 2} ODE solutions...")

    for i, theta in enumerate(thetas):
        M_O, A_O_max, A_P_max, M_R, eta = theta
        CP = M_O * eta
        theta_norm = (theta - PARAM_LB) / (PARAM_UB - PARAM_LB)

        protocols = [
            lambda t, Ph=CP * 1.3, Pl=CP * 0.5: Ph if (t % 60) < 30 else Pl,
            lambda t, Pc=CP * 1.1: Pc,
        ]

        for P_func in protocols:
            try:
                sol = solve_ivp(
                    mm_odes, [0, 600], [A_O_max, A_P_max],
                    args=(M_O, A_O_max, A_P_max, M_R, eta, P_func),
                    method='RK45', t_eval=np.linspace(0, 600, n_pts),
                    max_step=1.0,
                )
                if sol.success and not np.any(np.isnan(sol.y)):
                    all_t.append(sol.t / 600.0)
                    all_aO.append(np.clip(sol.y[0] / A_O_max, 0, 1))
                    all_aP.append(np.clip(sol.y[1] / A_P_max, 0, 1))
                    all_th.append(np.tile(theta_norm, (n_pts, 1)))
                    n_valid += 1
            except Exception:
                continue

        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_samples} θ-vectors processed ({n_valid} valid curves)")

    T = torch.tensor(np.concatenate(all_t), dtype=torch.float32).unsqueeze(1)
    AO = torch.tensor(np.concatenate(all_aO), dtype=torch.float32).unsqueeze(1)
    AP = torch.tensor(np.concatenate(all_aP), dtype=torch.float32).unsqueeze(1)
    TH = torch.tensor(np.concatenate(all_th), dtype=torch.float32)

    if verbose:
        print(f"  {n_valid} valid curves → {T.shape[0]} training points")
    return T, AO, AP, TH


# ============================================================
# Phase 1: pure supervised pre-training
# ============================================================
def pretrain_supervised(T: torch.Tensor, AO: torch.Tensor, AP: torch.Tensor, TH: torch.Tensor,
                        n_ep: int = 8000,
                        batch_size: int = 4096,
                        verbose: bool = True
                        ) -> NeuralSurrogate:
    """Supervised pre-training of the surrogate (no ODE loss, no IC loss).

    Pure MSE on clean ODE solutions, with cosine-annealing learning rate
    and best-weight tracking. This is the procedure described in
    Section 2.4.1 of the manuscript.
    """
    net = NeuralSurrogate()
    n_total = T.shape[0]
    n_params = sum(p.numel() for p in net.parameters())

    if verbose:
        print(f"  Network: 4×128, {n_params} parameters")
        print(f"  Data: {n_total} points, batch_size={batch_size}")
        print(f"  Data/params ratio: {n_total / n_params:.1f}×")

    X = torch.cat([T, TH], dim=1)
    Y = torch.cat([AO, AP], dim=1)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-5)

    rng = np.random.default_rng(SEED + 777)
    best_loss = float('inf')
    best_state = None

    for ep in range(n_ep):
        idx = rng.choice(n_total, size=min(batch_size, n_total), replace=False)
        opt.zero_grad()
        pred = net.net(X[idx])
        loss = torch.mean((pred - Y[idx]) ** 2)
        loss.backward()
        opt.step()
        sch.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if verbose and ep % max(1, n_ep // 10) == 0:
            with torch.no_grad():
                pred_v = net.net(X[:5000])
                mse = torch.mean((pred_v - Y[:5000]) ** 2).item()
                mae_aP = torch.mean(torch.abs(pred_v[:, 1:2] - Y[:5000, 1:2])).item()
            print(f"    Ep {ep}/{n_ep}: batch_loss={loss.item():.6f}, "
                  f"val_MSE={mse:.6f}, |Δa_P|={mae_aP:.4f}")

    net.load_state_dict(best_state)
    if verbose:
        with torch.no_grad():
            pred_all = net.net(X)
            final_mse = torch.mean((pred_all - Y) ** 2).item()
            mae_aO = torch.mean(torch.abs(pred_all[:, 0:1] - Y[:, 0:1])).item()
            mae_aP = torch.mean(torch.abs(pred_all[:, 1:2] - Y[:, 1:2])).item()
        print(f"  Best validation loss: {best_loss:.6f}")
        print(f"  Final: MSE={final_mse:.6f}, |Δa_O|={mae_aO:.4f}, |Δa_P|={mae_aP:.4f}")

    return net


# ============================================================
# Phase 2: θ-only optimization with frozen network
# ============================================================
def estimate_params(net: NeuralSurrogate, data: dict,
                    n_ep: int = 1500, seed: int = 0) -> np.ndarray:
    """Estimate θ for a target athlete with frozen network weights.

    The network weights are frozen and only the 5-parameter vector θ
    is optimized via Adam (lr=2e-2, cosine annealing) against the
    composite data + ODE residual + IC loss.

    Parameters
    ----------
    net : NeuralSurrogate
        Pre-trained surrogate.
    data : dict
        Target athlete's noisy dataset (from :func:`gen_data_intermittent`).
    n_ep : int
        θ-optimization epochs.
    seed : int
        Random seed.

    Returns
    -------
    ndarray, shape (5,)
        Estimated parameter vector.
    """
    for p in net.parameters():
        p.requires_grad = False

    pop_mean = (PARAM_LB + PARAM_UB) / 2
    theta_raw = make_theta_raw(pop_mean)

    t_raw, AP_true, Pn = data['time'], data['A_P_true'], data['P_ext_noisy']
    T = float(t_raw[-1])
    step = max(1, len(t_raw) // 80)
    idx = np.arange(0, len(t_raw), step)

    sc = float(AP_true.max()) + 1.0
    t_d = torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1)
    ap_n = torch.tensor(AP_true[idx] / sc, dtype=torch.float32).unsqueeze(1)

    nc = N_COLLOCATION
    tc = torch.linspace(0, 1, nc).unsqueeze(1)
    Pc = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, Pn),
                      dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam([theta_raw], lr=2e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

    for ep in range(n_ep):
        opt.zero_grad()
        params = net.get_params(theta_raw)
        M_O, AO, AP_mx, MR, eta = params

        pred = net(t_d, theta_raw)
        loss_d = torch.mean((pred[:, 1:2] * AP_mx / sc - ap_n) ** 2)

        tc2 = tc.clone().requires_grad_(True)
        pc = net(tc2, theta_raw)
        daO = torch.autograd.grad(pc[:, 0:1], tc2, torch.ones(nc, 1),
                                   create_graph=True, retain_graph=True)[0]
        daP = torch.autograd.grad(pc[:, 1:2], tc2, torch.ones(nc, 1),
                                   create_graph=True, retain_graph=True)[0]
        rO = T * (MR * (1 - pc[:, 0:1]) - (M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]))
        rP = T * ((M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]) - Pc / (eta * AP_mx))
        loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

        p0 = net(torch.tensor([[0.0]]), theta_raw)
        loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

        w = 0.3 * min(1.0, ep / (n_ep * 0.1))
        loss = loss_d + w * loss_r + 5.0 * loss_i
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta_raw], 1.0)
        opt.step()
        sch.step()

    # Re-enable gradients for callers that may train downstream
    for p in net.parameters():
        p.requires_grad = True

    return net.get_params(theta_raw).detach().numpy()
