"""Posterior predictive checks for prior validation and selection.

Implements spectral goodness-of-fit tests comparing observed data against
the prior predictive p(x_T | phi) = N(m_phi, Q_rho^{-1}).

The key statistic is the per-mode chi-squared ratio:
    r_j = q_j * S_j / (n * d)

where S_j is the observed spectral energy on mode j and n*d/q_j is the
expected energy under the predictive. Under correct model specification,
r_j should be approximately 1.

We define several aggregate scores:
1. Mean absolute log-ratio: mean(|log r_j|)  -- 0 = perfect calibration
2. Chi-squared statistic: sum((r_j - 1)^2)  -- tests uniform calibration
3. KS statistic on r_j vs expected distribution
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


def predictive_check_statistics(
    S: np.ndarray,      # (k,) empirical spectral energies
    q: np.ndarray,      # (k,) Q_rho eigenvalues
    n: int,
    d: int,
) -> Dict[str, float]:
    """Compute posterior predictive check statistics.

    Returns dict with several goodness-of-fit measures.
    """
    k = len(S)
    q_safe = np.maximum(q, 1e-12)
    S_safe = np.maximum(S, 1e-12)

    # Per-mode ratio: observed / expected energy
    r = q_safe * S_safe / (n * d)

    # 1. Mean absolute log-ratio (MALR): 0 = perfect
    log_r = np.log(np.maximum(r, 1e-12))
    malr = np.mean(np.abs(log_r))

    # 2. Calibration chi-squared: sum((r_j - 1)^2 / 1)
    chi2 = np.sum((r - 1.0) ** 2)

    # 3. Mean ratio (should be ~1)
    mean_r = np.mean(r)

    # 4. Fraction of well-calibrated modes (0.5 < r_j < 2.0)
    frac_ok = np.mean((r > 0.5) & (r < 2.0))

    # 5. Score spread: std of log(r_j) -- measures heterogeneity of miscalibration
    spread = np.std(log_r)

    return {
        "malr": float(malr),
        "chi2": float(chi2),
        "mean_r": float(mean_r),
        "frac_calibrated": float(frac_ok),
        "log_r_spread": float(spread),
    }


def predictive_check_score(
    S: np.ndarray,
    q: np.ndarray,
    n: int, d: int,
) -> float:
    """Single scalar goodness-of-fit score (lower = better calibration).

    Uses negative mean absolute log-ratio (MALR), so higher = better calibrated.
    Can be used as prior selection criterion: argmax_phi (-MALR).
    """
    stats = predictive_check_statistics(S, q, n, d)
    return -stats["malr"]  # higher = better


def select_prior_by_predictive_check(
    candidates: list,  # list of (phi_dict, S_array, q_array) tuples
    n: int, d: int,
) -> Tuple[int, Dict]:
    """Select the prior with the best posterior predictive check score.

    Returns (best_index, best_stats).
    """
    best_idx = 0
    best_score = -np.inf
    best_stats = {}

    for i, (phi, S, q) in enumerate(candidates):
        score = predictive_check_score(S, q, n, d)
        if score > best_score:
            best_score = score
            best_idx = i
            best_stats = predictive_check_statistics(S, q, n, d)
            best_stats["phi"] = phi

    return best_idx, best_stats
