"""
Reduced two-compartment Margaria-Morton ODE model.

This module defines the bioenergetic ODE system used throughout the study
and a forward-simulation utility based on SciPy's adaptive Dormand-Prince
solver. The model is described in detail in Section 2.1 of the manuscript.

Notation
--------
The state vector is :math:`[A_O, A_P]`:

    - :math:`A_O(t)` : oxidative reservoir energy content (J)
    - :math:`A_P(t)` : non-oxidative (phosphagenic + glycolytic) reservoir
      energy content (J)

The five model parameters (vector :math:`\\theta`) are:

    1. ``M_O``     : maximal oxidative metabolic power (W)
    2. ``A_O_max`` : oxidative reservoir capacity (J)
    3. ``A_P_max`` : non-oxidative reservoir capacity (J)
    4. ``M_R``     : oxidative replenishment rate (s\\ :sup:`-1`)
    5. ``eta``     : mechanical efficiency (dimensionless)

Derived quantities follow:

    - Critical power :math:`CP = M_O \\cdot \\eta`
    - W-prime :math:`W' = A_{P,\\max} \\cdot \\eta`
"""

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# Physiological parameter bounds (used throughout the study)
# ============================================================
PARAM_LB = np.array([600.0, 25_000.0, 30_000.0, 0.005, 0.18])
PARAM_UB = np.array([1800.0, 120_000.0, 160_000.0, 0.080, 0.32])

PARAM_KEYS = ['M_O', 'A_O_max', 'A_P_max', 'M_R', 'eta']
PARAM_NAMES_LATEX = ['$M_O$', '$A_{O,max}$', '$A_{P,max}$', '$M_R$', '$\\eta$']


# ============================================================
# ODE right-hand side
# ============================================================
def mm_odes(t: float,
            state: np.ndarray,
            M_O: float, A_O_max: float, A_P_max: float,
            M_R: float, eta: float,
            P_func: Callable[[float], float]) -> list:
    """Right-hand side of the reduced two-compartment Margaria-Morton ODE.

    Parameters
    ----------
    t : float
        Time (s).
    state : array_like, shape (2,)
        Current state ``[A_O, A_P]`` in joules.
    M_O, A_O_max, A_P_max, M_R, eta : float
        Model parameters (see module docstring).
    P_func : callable
        External mechanical power profile, signature ``P_func(t) -> float`` (W).

    Returns
    -------
    list of float
        ``[dA_O/dt, dA_P/dt]`` (W).
    """
    A_O, A_P = state
    A_O = np.clip(A_O, 0, A_O_max)
    A_P = np.clip(A_P, 0, A_P_max)

    R = M_R * (A_O_max - A_O)                              # oxidative replenishment
    phi1 = M_O * (A_O / A_O_max) * (1.0 - A_P / A_P_max)   # O → P inter-compartment flow
    phi2 = P_func(t) / eta                                 # P → mechanical work

    return [R - phi1, phi1 - phi2]


# ============================================================
# Forward simulation
# ============================================================
def simulate(theta: np.ndarray,
             P_func: Callable[[float], float],
             t_max: float = 600.0,
             dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate the ODE forward from full reservoirs until exhaustion or t_max.

    The simulation uses an adaptive Dormand-Prince (RK45) integrator with
    tight tolerances (rtol=1e-7, atol=1e-7). Exhaustion is detected via a
    terminal event ``A_P < 0.5 J``.

    Parameters
    ----------
    theta : array_like, shape (5,)
        Parameter vector ``[M_O, A_O_max, A_P_max, M_R, eta]``.
    P_func : callable
        External power profile.
    t_max : float, optional
        Maximum simulation time (s). Default 600 s.
    dt : float, optional
        Output time step (s). Default 1 s.

    Returns
    -------
    t_eval : ndarray
        Time grid (s).
    A_O : ndarray
        Oxidative reservoir trajectory (J).
    A_P : ndarray
        Non-oxidative reservoir trajectory (J).
    t_exhaustion : float
        Time of exhaustion (s); equals ``t_max`` if the athlete did not exhaust.
    """
    M_O, A_O_max, A_P_max, M_R, eta = theta
    y0 = [A_O_max, A_P_max]

    def exhaustion_event(t, y, *args):
        return y[1] - 0.5
    exhaustion_event.terminal = True
    exhaustion_event.direction = -1

    sol = solve_ivp(
        mm_odes, [0, t_max], y0,
        method='RK45',
        args=(M_O, A_O_max, A_P_max, M_R, eta, P_func),
        events=exhaustion_event,
        max_step=1.0, rtol=1e-7, atol=1e-7,
        dense_output=True,
    )

    t_exhaustion = sol.t_events[0][0] if sol.t_events[0].size > 0 else t_max
    t_eval = np.arange(0, min(t_exhaustion, t_max), dt)
    states = sol.sol(t_eval)
    return t_eval, states[0], states[1], t_exhaustion


# ============================================================
# Helper: simulate-and-extract-A_P (used by baselines for residuals)
# ============================================================
def simulate_AP_subsampled(theta: np.ndarray,
                           t_sub: np.ndarray,
                           P_arr: np.ndarray,
                           t_raw: np.ndarray,
                           T: float) -> np.ndarray:
    """Integrate the ODE for a candidate ``theta`` and return :math:`A_P` at ``t_sub``.

    Used by the LM and DE baselines to evaluate residuals against subsampled
    observations. Failures (e.g., stiff configurations during optimization)
    return zeros to signal a bad parameter vector to the optimizer.

    Parameters
    ----------
    theta : array_like, shape (5,)
        Candidate parameter vector.
    t_sub : ndarray
        Subsampled times at which :math:`A_P` is requested.
    P_arr : ndarray
        Power-output array aligned on ``t_raw``.
    t_raw : ndarray
        Original time grid.
    T : float
        Total duration.

    Returns
    -------
    ndarray
        :math:`A_P(t_{sub})`; zero array on integration failure.
    """
    M_O, A_O_max, A_P_max, M_R, eta = theta
    P_f = lambda t: float(np.interp(t, t_raw, P_arr))

    def ev(t, y, *a):
        return y[1] - 0.5
    ev.terminal = True
    ev.direction = -1

    try:
        sol = solve_ivp(
            mm_odes, [0, T], [A_O_max, A_P_max],
            method='RK23',
            args=(M_O, A_O_max, A_P_max, M_R, eta, P_f),
            events=ev,
            max_step=5.0, rtol=1e-4, atol=1e-4,
            dense_output=True,
        )
        return sol.sol(t_sub)[1]
    except Exception:
        return np.zeros_like(t_sub)
