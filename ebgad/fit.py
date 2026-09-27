"""Empirical-Bayes fits: (rho, kappa) per gamma, gamma by the x-likelihood,
and optionally the horizon Gamma.

Stationary fit (Algorithm 1 of the paper): for each template gamma and each
kappa on the fixed grid, rho is solved by clamped Newton (concave in rho,
Remark 1), the boundaries rho in {0, 1} are compared, and the best (rho,
kappa) by marginal likelihood is kept.

Outer gamma: the paper profiles the residual-space likelihood and omits the
Jacobian of x -> delta = T_gamma x. With the unit-gain template the
Jacobian log|det T_gamma| = d sum_j log(lam_j / (gamma^2 + lam_j)) is
finite for every gamma, and adding it makes gamma an EB-fitted quantity by
the marginal likelihood of the observed features. ``log_jacobian`` handles
both template families.

Horizon: with the template start, delta does not depend on
Gamma and the finite-horizon law is N(0, Sigma_Gamma) with precision
q_j / (1 - e^{-2 Gamma q_j}). ``fit_horizon`` maximizes that likelihood
jointly over (rho, kappa, Gamma) and reports the gain over Gamma = inf.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from soc.prior_optimizer import (
    gamma_jacobian_correction,
    marginal_log_likelihood_stationary,
    solve_rho_newton,
)

PAPER_KAPPAS: List[float] = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
HORIZON_GRID: List[float] = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def q_eigs(lam: np.ndarray, rho: float, kappa: float) -> np.ndarray:
    """q_j = rho (kappa^2 + lam_j) + (1 - rho), nu = 1."""
    return rho * (kappa ** 2 + lam) + (1.0 - rho)


def horizon_precision(q: np.ndarray, Gamma: float) -> np.ndarray:
    q_safe = np.maximum(q, 1e-12)
    if np.isinf(Gamma):
        return q_safe
    return q_safe / np.maximum(-np.expm1(-2.0 * float(Gamma) * q_safe), 1e-300)


def log_jacobian(prep) -> float:
    """log|det T_gamma| restricted to the modeled modes (times d)."""
    lam = prep.lam[prep.lam > 1e-12]
    g2 = prep.gamma ** 2
    if prep.template.endswith("_unit"):
        return float(prep.d * np.sum(np.log(lam / (g2 + lam))))
    return float(gamma_jacobian_correction(prep.lam, prep.gamma, prep.d, nu=1.0))


@dataclass
class Fit:
    rho: float
    kappa: float
    ml: float          # stationary residual-space marginal log-likelihood
    ml_jac: float      # ml + log|det T_gamma|: marginal log-likelihood of x
    gamma: float
    q: np.ndarray      # (k,) fitted precision eigenvalues

    def as_dict(self) -> Dict:
        return {"rho": self.rho, "kappa": self.kappa, "ml": self.ml,
                "ml_jac": self.ml_jac, "gamma": self.gamma}


def fit_rho_kappa(
    S: np.ndarray, lam: np.ndarray, n: int, d: int,
    kappas: Sequence[float] = PAPER_KAPPAS,
) -> Dict:
    best = {"ml": -np.inf}
    for kappa in kappas:
        rho_newton = solve_rho_newton(S, lam, float(kappa), d, nu=1.0)
        for rho in (0.0, 1.0, rho_newton):
            q = q_eigs(lam, rho, float(kappa))
            if np.any(q <= 0):
                continue
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best["ml"]:
                best = {"ml": float(ml), "rho": float(rho), "kappa": float(kappa), "q": q}
    return best


def fit_prepared(prep, kappas: Sequence[float] = PAPER_KAPPAS) -> Fit:
    best = fit_rho_kappa(prep.S, prep.lam, prep.n, prep.d, kappas)
    return Fit(rho=best["rho"], kappa=best["kappa"], ml=best["ml"],
               ml_jac=best["ml"] + log_jacobian(prep), gamma=prep.gamma, q=best["q"])


def select_gamma(fits: Sequence[Fit], use_jacobian: bool = False) -> Fit:
    key = (lambda f: f.ml_jac) if use_jacobian else (lambda f: f.ml)
    return max(fits, key=key)


def _ml_horizon(S: np.ndarray, q: np.ndarray, Gamma: float, n: int, d: int) -> float:
    qt = horizon_precision(q, Gamma)
    return float(-0.5 * np.sum(S * qt) + 0.5 * d * np.sum(np.log(qt)) - 0.5 * n * d * np.log(2 * np.pi))


def fit_horizon(prep, kappas: Sequence[float] = PAPER_KAPPAS,
                horizons: Sequence[float] = HORIZON_GRID) -> Dict:
    """Joint EB over (rho, kappa, Gamma) with the template start.

    Returns the best finite-horizon fit, the best Gamma = inf fit (which is
    the stationary fit), and their log-likelihood difference. A positive
    ``gain`` means the likelihood prefers a finite horizon.
    """
    S, lam, n, d = prep.S, prep.lam, prep.n, prep.d
    best = {"ml": -np.inf}
    best_inf = {"ml": -np.inf}
    for Gamma in list(horizons) + [np.inf]:
        for kappa in kappas:
            a = (float(kappa) ** 2 + lam) - 1.0

            def neg(rho):
                q = rho * a + 1.0
                if np.any(q <= 0):
                    return np.inf
                return -_ml_horizon(S, q, Gamma, n, d)

            cands = [0.0, 1.0]
            res = minimize_scalar(neg, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
            if res.x is not None and np.isfinite(res.x):
                cands.append(float(res.x))
            for rho in cands:
                val = -neg(rho)
                rec = {"ml": val, "rho": float(rho), "kappa": float(kappa), "Gamma": float(Gamma)}
                if np.isinf(Gamma):
                    if val > best_inf["ml"]:
                        best_inf = rec
                elif val > best["ml"]:
                    best = rec
    return {"finite": best, "inf": best_inf, "gain": best["ml"] - best_inf["ml"]}
