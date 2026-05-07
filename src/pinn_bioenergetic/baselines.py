"""
Classical optimization baselines: Levenberg-Marquardt and Differential Evolution.

Both methods minimize the squared residual between the noisy observed
:math:`A_P(t)` trajectory and the trajectory predicted by the ODE for a
candidate parameter vector. They serve as reference points for the PINN
benchmark in Section 3 of the manuscript.
"""

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from .config import LM_RESTARTS, DE_MAXITER, DE_POPSIZE
from .model import simulate_AP_subsampled, PARAM_LB, PARAM_UB


def run_LM(data: dict, seed: int = 0) -> np.ndarray:
    """Run Levenberg-Marquardt with random restarts.

    Implementation uses ``scipy.optimize.least_squares`` with the Trust
    Region Reflective (``trf``) method to support parameter bounds. Eight
    random initial parameter vectors (uniform within physiological bounds)
    are tried and the best (lowest residual) is returned.

    Parameters
    ----------
    data : dict
        Dataset from :func:`pinn_bioenergetic.population.gen_data`.
    seed : int, optional
        Seed for the random restarts.

    Returns
    -------
    ndarray, shape (5,)
        Best parameter vector found across restarts. Falls back to the
        midpoint of the physiological range if all restarts fail.
    """
    rng = np.random.default_rng(seed)
    t_raw = data['time']
    AP = data['A_P_true']
    P_n = data['P_ext_noisy']
    T = float(t_raw[-1])

    # Subsample residuals (every ~40th point)
    s = max(1, len(t_raw) // 40)
    ts, APs = t_raw[::s], AP[::s]
    sc = np.std(APs) + 1.0

    best_cost, best_x = np.inf, None
    for _ in range(LM_RESTARTS):
        x0 = rng.uniform(PARAM_LB, PARAM_UB)
        try:
            r = least_squares(
                lambda th: (simulate_AP_subsampled(th, ts, P_n, t_raw, T) - APs) / sc,
                x0,
                bounds=(PARAM_LB, PARAM_UB),
                method='trf',
                max_nfev=150,
            )
            if r.cost < best_cost:
                best_cost, best_x = r.cost, r.x.copy()
        except Exception:
            continue

    return best_x if best_x is not None else (PARAM_LB + PARAM_UB) / 2.0


def run_DE(data: dict, seed: int = 0) -> np.ndarray:
    """Run Differential Evolution (DE).

    Uses ``scipy.optimize.differential_evolution`` with the ``best1bin``
    strategy and L-BFGS-B polishing. DE is a global optimizer that does
    not require an initial guess and is robust to local minima.

    Parameters
    ----------
    data : dict
        Dataset from :func:`pinn_bioenergetic.population.gen_data`.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    ndarray, shape (5,)
        Best parameter vector found by DE.
    """
    t_raw = data['time']
    AP = data['A_P_true']
    P_n = data['P_ext_noisy']
    T = float(t_raw[-1])

    s = max(1, len(t_raw) // 30)
    ts, APs = t_raw[::s], AP[::s]
    bounds = list(zip(PARAM_LB.tolist(), PARAM_UB.tolist()))

    r = differential_evolution(
        lambda th: np.mean((simulate_AP_subsampled(th, ts, P_n, t_raw, T) - APs) ** 2),
        bounds,
        seed=seed,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=1e-6,
        polish=True,
    )
    return r.x
