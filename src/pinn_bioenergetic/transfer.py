"""
Transfer learning architectures and training routines.

This module implements the two PINN transfer-learning approaches compared
in Sections 3.6 and 3.7 of the manuscript:

    Architecture A (multi-task):
        Pre-train a standard time-only network on N_pre virtual athletes,
        each with their own trainable parameter vector θ_i but sharing
        the network weights W. Fine-tune on a target athlete with all
        layers trainable using discriminative learning rates.

    Architecture B (conditional):
        Pre-train a network that takes (t, θ_normalized) as input, learning
        the *family* of ODE solutions f(t, θ) → [a_O, a_P]. Fine-tune by
        FREEZING the network weights and optimizing only the 5-parameter
        vector θ (θ-only optimization).

The module exposes:
    - ``MultiTaskPINN`` class  +  ``pretrain_multitask``, ``finetune_multitask``,
      ``scratch_multitask``  (Architecture A)
    - ``ConditionalPINN`` class  +  ``pretrain_conditional``,
      ``finetune_conditional``, ``scratch_conditional``  (Architecture B)
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import (
    N_HIDDEN, N_NEURONS, N_COLLOCATION,
    PRETRAIN_EPOCHS, FINETUNE_EPOCHS, SCRATCH_EPOCHS,
    TL_SIGMA, SEED,
)
from .model import PARAM_LB, PARAM_UB
from .pinn import PINN
from .population import gen_data


# ============================================================
# Architecture A: Multi-task PINN
# ============================================================
class MultiTaskPINN(nn.Module):
    """Multi-task PINN: shared time-only network, per-athlete trainable θ.

    The network maps :math:`t \\mapsto [\\hat{a}_O(t), \\hat{a}_P(t)]` and is
    shared across all athletes during pre-training. Each athlete is given
    its own trainable 5-parameter vector :math:`\\theta_i`. This separates
    the learning of generic ODE features (shared :math:`\\mathbf{W}`) from
    the fitting of individual physiology (per-athlete :math:`\\theta_i`).

    Parameters
    ----------
    n_athletes : int
        Number of athletes in the pre-training population.
    n_hidden, n_neurons : int
        Network architecture (defaults from config).
    """

    def __init__(self, n_athletes: int,
                 n_hidden: int = N_HIDDEN, n_neurons: int = N_NEURONS):
        super().__init__()

        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        self.athlete_z = nn.ParameterList([
            nn.Parameter(torch.zeros(5)) for _ in range(n_athletes)
        ])

        self.register_buffer('lb', torch.tensor(PARAM_LB, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(PARAM_UB, dtype=torch.float32))

    def get_params(self, athlete_idx: int) -> torch.Tensor:
        """Return bounded :math:`\\theta_i` for athlete ``athlete_idx``."""
        return self.lb + (self.ub - self.lb) * torch.sigmoid(self.athlete_z[athlete_idx])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def init_athlete_params(self, athlete_idx: int, guess: np.ndarray) -> None:
        """Initialize :math:`\\theta_i` from a physical guess in the pre-image space."""
        with torch.no_grad():
            g = torch.tensor(guess, dtype=torch.float32)
            f = (g - self.lb) / (self.ub - self.lb)
            f = torch.clamp(f, 0.02, 0.98)
            self.athlete_z[athlete_idx].copy_(torch.log(f / (1 - f)))

    def extract_shared_weights(self) -> Dict[str, torch.Tensor]:
        """Extract a clone of the shared network state dict (for transfer)."""
        return {k: v.clone() for k, v in self.net.state_dict().items()}


def _build_dataset_dict(theta: np.ndarray, sigma: float, seed: int) -> dict:
    """Helper: produce the torch tensors needed for one athlete's training."""
    d = gen_data(theta, sigma, seed=seed)
    t_raw, AP, Pn = d['time'], d['A_P_true'], d['P_ext_noisy']
    T = float(t_raw[-1])
    step = max(1, len(t_raw) // 60)
    idx = np.arange(0, len(t_raw), step)
    sc = float(AP.max()) + 1.0
    nc = N_COLLOCATION
    return {
        't': torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1),
        'ap': torch.tensor(AP[idx] / sc, dtype=torch.float32).unsqueeze(1),
        'sc': sc, 'T': T, 'nc': nc,
        'tc': torch.linspace(0, 1, nc).unsqueeze(1),
        'Pc': torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, Pn),
                           dtype=torch.float32).unsqueeze(1),
    }


def pretrain_multitask(pop: np.ndarray,
                       sigma: float = TL_SIGMA,
                       n_ep: int = PRETRAIN_EPOCHS,
                       verbose: bool = True) -> Dict[str, torch.Tensor]:
    """Multi-task pre-training: shared :math:`\\mathbf{W}`, per-athlete :math:`\\theta_i`.

    Returns the shared network state dict, ready to be loaded into a
    standard ``PINN`` for fine-tuning.

    Parameters
    ----------
    pop : ndarray, shape (n_pre, 5)
        Pre-training population.
    sigma : float
        Noise level (W) used for synthetic data.
    n_ep : int
        Pre-training epochs.
    verbose : bool
        Print progress every n_ep/5 epochs.

    Returns
    -------
    dict
        State dict of the shared network weights only (no athlete params).
    """
    n_ath = len(pop)
    pinn = MultiTaskPINN(n_athletes=n_ath)

    datasets = []
    for i in range(n_ath):
        datasets.append(_build_dataset_dict(pop[i], sigma, SEED + i * 200))
        # Initialize θ_i near the true values with 10% noise
        rng_i = np.random.default_rng(SEED + i)
        noisy_guess = pop[i] * (1 + rng_i.normal(0, 0.1, 5))
        noisy_guess = np.clip(noisy_guess, PARAM_LB, PARAM_UB)
        pinn.init_athlete_params(i, noisy_guess)

    opt = torch.optim.Adam(pinn.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

    if verbose:
        print(f"  Multi-task pre-training: {n_ath} athletes, {n_ep} epochs")

    for ep in range(n_ep):
        opt.zero_grad()
        total_loss = torch.tensor(0.0)

        for i in range(n_ath):
            d = datasets[i]
            params = pinn.get_params(i)
            M_O, AO, AP_mx, MR, eta = params

            pred = pinn(d['t'])
            loss_d = torch.mean((pred[:, 1:2] * AP_mx / d['sc'] - d['ap']) ** 2)

            tc = d['tc'].clone().requires_grad_(True)
            pc = pinn(tc)
            daO = torch.autograd.grad(pc[:, 0:1], tc, torch.ones(d['nc'], 1),
                                       True, True)[0]
            daP = torch.autograd.grad(pc[:, 1:2], tc, torch.ones(d['nc'], 1),
                                       True, True)[0]
            T_v = d['T']
            rO = T_v * (MR * (1 - pc[:, 0:1]) - (M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]))
            rP = T_v * ((M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]) - d['Pc'] / (eta * AP_mx))
            loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

            p0 = pinn(torch.tensor([[0.0]]))
            loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

            w_ode = 0.3 * min(1.0, ep / (n_ep * 0.15))
            total_loss = total_loss + loss_d + w_ode * loss_r + 5.0 * loss_i

        total_loss = total_loss / n_ath
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        opt.step()
        sch.step()

        if verbose and ep % max(1, n_ep // 5) == 0:
            errs = []
            for i in range(n_ath):
                est = pinn.get_params(i).detach().numpy()
                CP_t = pop[i, 0] * pop[i, 4]
                CP_e = est[0] * est[4]
                errs.append(abs(CP_e - CP_t) / CP_t * 100)
            print(f"    Ep {ep}/{n_ep}: loss={total_loss.item():.4e}, "
                  f"median CP err={np.median(errs):.1f}%")

    return pinn.extract_shared_weights()


def finetune_multitask(shared_weights: Dict[str, torch.Tensor],
                       data: dict,
                       frac: float = 1.0,
                       n_ep: int = FINETUNE_EPOCHS,
                       seed: int = 0) -> np.ndarray:
    """Fine-tune from multi-task pre-trained weights with discriminative LR.

    All layers are trainable, but with three learning-rate groups:
        - Early hidden layers   : lr × 0.1
        - Last hidden + output  : lr × 1.0
        - Parameter vector θ    : lr × 3.0

    Parameters
    ----------
    shared_weights : dict
        State dict from :func:`pretrain_multitask`.
    data : dict
        Target athlete's dataset.
    frac : float
        Fraction of individual data used (1.0 = all available subsamples).
    n_ep : int
        Fine-tuning epochs.
    seed : int
        Seed for the random subsampling.

    Returns
    -------
    ndarray, shape (5,)
        Fine-tuned parameter estimate.
    """
    rng = np.random.default_rng(seed)

    pinn = PINN()
    pinn.net.load_state_dict(shared_weights)

    # Initialize θ near population mean (pre-training centroid)
    pop_mean = np.array([1120.0, 60_000.0, 80_000.0, 0.030, 0.25])
    with torch.no_grad():
        g = torch.tensor(pop_mean, dtype=torch.float32)
        f = (g - pinn.lb) / (pinn.ub - pinn.lb)
        f = torch.clamp(f, 0.02, 0.98)
        pinn.z.copy_(torch.log(f / (1 - f)))

    t_raw, AP, Pn = data['time'], data['A_P_true'], data['P_ext_noisy']
    T = float(t_raw[-1])
    step = max(1, len(t_raw) // 60)
    all_idx = np.arange(0, len(t_raw), step)
    if frac < 1.0:
        n_use = max(3, int(len(all_idx) * frac))
        chosen = np.sort(rng.choice(len(all_idx), size=n_use, replace=False))
        idx = all_idx[chosen]
    else:
        idx = all_idx

    sc = float(AP.max()) + 1.0
    t_d = torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1)
    ap_n = torch.tensor(AP[idx] / sc, dtype=torch.float32).unsqueeze(1)
    nc = N_COLLOCATION
    tc = torch.linspace(0, 1, nc).unsqueeze(1)
    Pc = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, Pn),
                      dtype=torch.float32).unsqueeze(1)

    base_lr = 3e-3
    # Linear layers in the Sequential are at even indices (0, 2, 4, ...)
    linear_indices = list(range(0, 2 * N_HIDDEN + 2, 2))
    early_params, late_params = [], []
    for li in linear_indices[:-2]:
        early_params.extend(pinn.net[li].parameters())
    for li in linear_indices[-2:]:
        late_params.extend(pinn.net[li].parameters())

    opt = torch.optim.Adam([
        {'params': early_params, 'lr': base_lr * 0.1},
        {'params': late_params,  'lr': base_lr * 1.0},
        {'params': [pinn.z],     'lr': base_lr * 3.0},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

    for ep in range(n_ep):
        opt.zero_grad()
        p = pinn.params()
        M_O, AO, AP_mx, MR, eta = p

        pred = pinn(t_d)
        loss_d = torch.mean((pred[:, 1:2] * AP_mx / sc - ap_n) ** 2)

        tc2 = tc.clone().requires_grad_(True)
        pc = pinn(tc2)
        daO = torch.autograd.grad(pc[:, 0:1], tc2, torch.ones(nc, 1), True, True)[0]
        daP = torch.autograd.grad(pc[:, 1:2], tc2, torch.ones(nc, 1), True, True)[0]
        rO = T * (MR * (1 - pc[:, 0:1]) - (M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]))
        rP = T * ((M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]) - Pc / (eta * AP_mx))
        loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

        p0 = pinn(torch.tensor([[0.0]]))
        loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

        w = 0.3 * min(1.0, ep / (n_ep * 0.1))
        loss = loss_d + w * loss_r + 5.0 * loss_i
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        opt.step()
        sch.step()

    return pinn.params().detach().numpy()


def scratch_multitask(data: dict,
                      frac: float = 1.0,
                      pop_mean: np.ndarray = None,
                      n_ep: int = SCRATCH_EPOCHS,
                      seed: int = 0) -> np.ndarray:
    """From-scratch baseline matching the multi-task fine-tuning architecture."""
    rng = np.random.default_rng(seed)
    pinn = PINN()

    if pop_mean is None:
        pop_mean = np.array([1120.0, 60_000.0, 80_000.0, 0.030, 0.25])
    with torch.no_grad():
        g = torch.tensor(pop_mean, dtype=torch.float32)
        f = (g - pinn.lb) / (pinn.ub - pinn.lb)
        f = torch.clamp(f, 0.02, 0.98)
        pinn.z.copy_(torch.log(f / (1 - f)))

    t_raw, AP, Pn = data['time'], data['A_P_true'], data['P_ext_noisy']
    T = float(t_raw[-1])
    step = max(1, len(t_raw) // 60)
    all_idx = np.arange(0, len(t_raw), step)
    if frac < 1.0:
        n_use = max(3, int(len(all_idx) * frac))
        chosen = np.sort(rng.choice(len(all_idx), size=n_use, replace=False))
        idx = all_idx[chosen]
    else:
        idx = all_idx

    sc = float(AP.max()) + 1.0
    t_d = torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1)
    ap_n = torch.tensor(AP[idx] / sc, dtype=torch.float32).unsqueeze(1)
    nc = N_COLLOCATION
    tc = torch.linspace(0, 1, nc).unsqueeze(1)
    Pc = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, Pn),
                      dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam(pinn.parameters(), lr=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

    for ep in range(n_ep):
        opt.zero_grad()
        p = pinn.params()
        M_O, AO, AP_mx, MR, eta = p

        pred = pinn(t_d)
        loss_d = torch.mean((pred[:, 1:2] * AP_mx / sc - ap_n) ** 2)

        tc2 = tc.clone().requires_grad_(True)
        pc = pinn(tc2)
        daO = torch.autograd.grad(pc[:, 0:1], tc2, torch.ones(nc, 1), True, True)[0]
        daP = torch.autograd.grad(pc[:, 1:2], tc2, torch.ones(nc, 1), True, True)[0]
        rO = T * (MR * (1 - pc[:, 0:1]) - (M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]))
        rP = T * ((M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]) - Pc / (eta * AP_mx))
        loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

        p0 = pinn(torch.tensor([[0.0]]))
        loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

        w = 0.3 * min(1.0, ep / (n_ep * 0.15))
        loss = loss_d + w * loss_r + 5.0 * loss_i
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        opt.step()
        sch.step()

    return pinn.params().detach().numpy()


# ============================================================
# Architecture B: Conditional PINN
# ============================================================
class ConditionalPINN(nn.Module):
    """Conditional PINN: input is :math:`(t, \\theta_{\\text{norm}})`, output is :math:`[\\hat{a}_O, \\hat{a}_P]`.

    The network learns the *family* of ODE solutions parametrized by θ.
    During fine-tuning, only the 5-parameter vector θ is optimized while
    the network weights remain frozen — the network has already learned
    how the solution varies with θ.
    """

    def __init__(self, n_hidden: int = N_HIDDEN, n_neurons: int = N_NEURONS):
        super().__init__()
        # Input dim: 1 (time) + 5 (θ_norm) = 6
        layers = [nn.Linear(6, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        self.register_buffer('lb', torch.tensor(PARAM_LB, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(PARAM_UB, dtype=torch.float32))

    def forward(self, t: torch.Tensor, theta_raw: torch.Tensor) -> torch.Tensor:
        """Forward pass at :math:`(t, \\theta)`.

        Parameters
        ----------
        t : tensor, shape (batch, 1)
            Normalized time.
        theta_raw : tensor, shape (5,)
            Unbounded parameter vector (sigmoid-mapped to physical units inside).
        """
        theta_bounded = self.lb + (self.ub - self.lb) * torch.sigmoid(theta_raw)
        theta_norm = (theta_bounded - self.lb) / (self.ub - self.lb)
        theta_expanded = theta_norm.unsqueeze(0).expand(t.shape[0], -1)
        x = torch.cat([t, theta_expanded], dim=1)
        return self.net(x)

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


def pretrain_conditional(pop: np.ndarray,
                         sigma: float = TL_SIGMA,
                         n_ep: int = PRETRAIN_EPOCHS,
                         verbose: bool = True) -> ConditionalPINN:
    """Pre-train the conditional PINN over a population.

    Each athlete has its own trainable θ_i; the network weights learn
    the mapping :math:`(t, \\theta) \\to [a_O, a_P]`.
    """
    n_ath = len(pop)
    net = ConditionalPINN()

    theta_raws = []
    datasets = []
    for i in range(n_ath):
        rng_i = np.random.default_rng(SEED + i)
        noisy = pop[i] * (1 + rng_i.normal(0, 0.1, 5))
        noisy = np.clip(noisy, PARAM_LB, PARAM_UB)
        theta_raws.append(make_theta_raw(noisy))
        datasets.append(_build_dataset_dict(pop[i], sigma, SEED + i * 200))

    all_params = list(net.parameters()) + theta_raws
    opt = torch.optim.Adam(all_params, lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)

    if verbose:
        print(f"  Conditional PINN pre-training: {n_ath} athletes, {n_ep} epochs")
        print(f"  Network input dim: 6 (t + θ_norm), architecture: {N_HIDDEN}×{N_NEURONS}")

    for ep in range(n_ep):
        opt.zero_grad()
        total_loss = torch.tensor(0.0)

        for i in range(n_ath):
            d = datasets[i]
            params = net.get_params(theta_raws[i])
            M_O, AO, AP_mx, MR, eta = params

            pred = net(d['t'], theta_raws[i])
            loss_d = torch.mean((pred[:, 1:2] * AP_mx / d['sc'] - d['ap']) ** 2)

            tc = d['tc'].clone().requires_grad_(True)
            pc = net(tc, theta_raws[i])
            daO = torch.autograd.grad(pc[:, 0:1], tc, torch.ones(d['nc'], 1),
                                       create_graph=True, retain_graph=True)[0]
            daP = torch.autograd.grad(pc[:, 1:2], tc, torch.ones(d['nc'], 1),
                                       create_graph=True, retain_graph=True)[0]
            T_v = d['T']
            rO = T_v * (MR * (1 - pc[:, 0:1]) - (M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]))
            rP = T_v * ((M_O / AO) * pc[:, 0:1] * (1 - pc[:, 1:2]) - d['Pc'] / (eta * AP_mx))
            loss_r = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)

            p0 = net(torch.tensor([[0.0]]), theta_raws[i])
            loss_i = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2

            w_ode = 0.3 * min(1.0, ep / (n_ep * 0.15))
            total_loss = total_loss + loss_d + w_ode * loss_r + 5.0 * loss_i

        total_loss = total_loss / n_ath
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sch.step()

        if verbose and ep % max(1, n_ep // 5) == 0:
            errs = []
            for i in range(n_ath):
                est = net.get_params(theta_raws[i]).detach().numpy()
                CP_t = pop[i, 0] * pop[i, 4]
                CP_e = est[0] * est[4]
                errs.append(abs(CP_e - CP_t) / CP_t * 100)
            print(f"    Ep {ep}/{n_ep}: loss={total_loss.item():.4e}, "
                  f"median CP err={np.median(errs):.1f}%")

    return net


def finetune_conditional(net_pretrained: ConditionalPINN,
                         data: dict,
                         frac: float = 1.0,
                         n_ep: int = FINETUNE_EPOCHS,
                         seed: int = 0) -> np.ndarray:
    """θ-only fine-tuning with frozen network weights.

    The network is deep-copied to avoid mutating the pre-trained instance,
    its weights are frozen, and a fresh θ is optimized for the target.
    """
    rng = np.random.default_rng(seed)
    net = copy.deepcopy(net_pretrained)

    for p in net.net.parameters():
        p.requires_grad = False

    pop_mean = (PARAM_LB + PARAM_UB) / 2
    theta_raw = make_theta_raw(pop_mean)

    t_raw, AP, Pn = data['time'], data['A_P_true'], data['P_ext_noisy']
    T = float(t_raw[-1])
    step = max(1, len(t_raw) // 60)
    all_idx = np.arange(0, len(t_raw), step)
    if frac < 1.0:
        n_use = max(3, int(len(all_idx) * frac))
        chosen = np.sort(rng.choice(len(all_idx), size=n_use, replace=False))
        idx = all_idx[chosen]
    else:
        idx = all_idx

    sc = float(AP.max()) + 1.0
    t_d = torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1)
    ap_n = torch.tensor(AP[idx] / sc, dtype=torch.float32).unsqueeze(1)
    nc = N_COLLOCATION
    tc = torch.linspace(0, 1, nc).unsqueeze(1)
    Pc = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, Pn),
                      dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam([theta_raw], lr=1e-2)
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

    return net.get_params(theta_raw).detach().numpy()


def scratch_conditional(data: dict,
                        frac: float = 1.0,
                        n_ep: int = SCRATCH_EPOCHS,
                        seed: int = 0) -> np.ndarray:
    """From-scratch baseline matching the conditional-PINN architecture."""
    rng = np.random.default_rng(seed)
    net = ConditionalPINN()
    pop_mean = (PARAM_LB + PARAM_UB) / 2
    theta_raw = make_theta_raw(pop_mean)

    t_raw, AP, Pn = data['time'], data['A_P_true'], data['P_ext_noisy']
    T = float(t_raw[-1])
    step = max(1, len(t_raw) // 60)
    all_idx = np.arange(0, len(t_raw), step)
    if frac < 1.0:
        n_use = max(3, int(len(all_idx) * frac))
        chosen = np.sort(rng.choice(len(all_idx), size=n_use, replace=False))
        idx = all_idx[chosen]
    else:
        idx = all_idx

    sc = float(AP.max()) + 1.0
    t_d = torch.tensor(t_raw[idx] / T, dtype=torch.float32).unsqueeze(1)
    ap_n = torch.tensor(AP[idx] / sc, dtype=torch.float32).unsqueeze(1)
    nc = N_COLLOCATION
    tc = torch.linspace(0, 1, nc).unsqueeze(1)
    Pc = torch.tensor(np.interp(np.linspace(0, T, nc), t_raw, Pn),
                      dtype=torch.float32).unsqueeze(1)

    all_params = list(net.parameters()) + [theta_raw]
    opt = torch.optim.Adam(all_params, lr=5e-3)
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

        w = 0.3 * min(1.0, ep / (n_ep * 0.15))
        loss = loss_d + w * loss_r + 5.0 * loss_i
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sch.step()

    return net.get_params(theta_raw).detach().numpy()
