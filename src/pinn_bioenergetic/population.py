"""
Virtual athlete population and synthetic data generation.

The population is sampled from a 5-dimensional truncated multivariate
normal distribution with physiologically motivated correlations
(detailed in Section 2.2 of the manuscript). Two exercise protocols
are simulated for each athlete (Protocol A: constant 110% CP;
Protocol B: 30s/30s intermittent at 130%/50% CP), with Gaussian
noise added to the power-output signal.
"""

import numpy as np

from .model import simulate, PARAM_LB, PARAM_UB


# ============================================================
# Distribution parameters (Section 2.2, Table 2 of manuscript)
# ============================================================
_POP_MEANS = np.array([1120.0, 60_000.0, 80_000.0, 0.030, 0.25])
_POP_SDS = np.array([150.0, 10_000.0, 18_000.0, 0.012, 0.02])
_POP_CORR = np.array([
    [1.00, 0.60, 0.30, 0.20, 0.10],   # M_O
    [0.60, 1.00, 0.40, 0.15, 0.05],   # A_O_max
    [0.30, 0.40, 1.00, 0.10, 0.05],   # A_P_max
    [0.20, 0.15, 0.10, 1.00, 0.05],   # M_R
    [0.10, 0.05, 0.05, 0.05, 1.00],   # eta
])
# Uniform fallback for M_R (least characterized in the literature)
_MR_LOW, _MR_HIGH = 0.010, 0.055


def gen_population(n: int, seed: int = 42) -> np.ndarray:
    """Generate a virtual athlete population.

    Parameters are sampled from a multivariate normal distribution truncated
    to physiological bounds, with M_R re-sampled from a uniform distribution
    given its weaker empirical characterization.

    Parameters
    ----------
    n : int
        Number of virtual athletes.
    seed : int, optional
        Random seed (default 42, matching the manuscript).

    Returns
    -------
    ndarray, shape (n, 5)
        Parameter matrix with columns ``[M_O, A_O_max, A_P_max, M_R, eta]``.
    """
    rng = np.random.default_rng(seed)
    cov = np.outer(_POP_SDS, _POP_SDS) * _POP_CORR
    samples = rng.multivariate_normal(_POP_MEANS, cov, size=n)

    # Re-sample M_R uniformly (weaker empirical anchor)
    samples[:, 3] = rng.uniform(_MR_LOW, _MR_HIGH, n)

    # Enforce physiological bounds on remaining parameters
    for i in range(n):
        for j in range(5):
            while samples[i, j] < PARAM_LB[j] or samples[i, j] > PARAM_UB[j]:
                samples[i, j] = rng.normal(_POP_MEANS[j], _POP_SDS[j])

    return samples


def gen_data(theta: np.ndarray, sigma_P: float, seed: int = 0) -> dict:
    """Generate synthetic measurement data for a single virtual athlete.

    Two protocols are simulated:
        - Protocol A: constant 110% of CP for up to 900 s.
        - Protocol B: intermittent 30 s work at 130% CP / 30 s rest at 50% CP
          for up to 1200 s. Used as the primary fitting target.

    Gaussian noise of standard deviation ``sigma_P`` (in W) is added to the
    Protocol B power-output time series.

    Parameters
    ----------
    theta : array_like, shape (5,)
        True parameter vector for the athlete.
    sigma_P : float
        Power-output noise standard deviation (W).
    seed : int, optional
        Random seed for the noise realization.

    Returns
    -------
    dict
        Dictionary containing time grids, true and noisy observables for
        both protocols, plus times to exhaustion. See source for the
        complete list of keys.
    """
    rng = np.random.default_rng(seed)
    M_O, A_O_max, A_P_max, M_R, eta = theta
    CP = M_O * eta

    # Protocol A: constant 110% CP
    P_A = CP * 1.10
    t_A, AO_A, AP_A, tte_A = simulate(theta, lambda t: P_A, t_max=900.0)

    # Protocol B: 30s/30s intermittent at 130% / 50% CP
    def P_B(t):
        return CP * 1.30 if (t % 60) < 30 else CP * 0.50
    t_B, AO_B, AP_B, tte_B = simulate(theta, P_B, t_max=1200.0)

    P_ext_B = np.array([P_B(t) for t in t_B])
    P_ext_A = np.ones_like(t_A) * P_A

    # Additive Gaussian noise on Protocol B (used for fitting)
    P_noisy_B = P_ext_B + rng.normal(0, sigma_P, len(t_B))

    return {
        # Protocol B (used for fitting)
        'time': t_B,
        'A_P_true': AP_B,
        'A_O_true': AO_B,
        'P_ext_clean': P_ext_B,
        'P_ext_noisy': P_noisy_B,
        'TTE_true': tte_B,
        # Protocol A (held out for validation)
        'time_A': t_A,
        'A_P_true_A': AP_A,
        'P_ext_A': P_ext_A,
        'TTE_A': tte_A,
    }
