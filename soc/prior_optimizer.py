"""Closed-form empirical Bayes prior optimization via marginal likelihood.

Implements coordinate descent on the marginal likelihood (Theorem 2):
    log p(D|φ) = -Σ_j S_j/(2D_j) - nd/(2k) Σ_j log(D_j) + const

where D_j = λ⁻¹ + (1-exp(-2αTq_j))/q_j and S_j is the empirical spectral
energy on eigenmode j.

Usage:
    optimizer = PriorOptimizer(data, device='cuda')
    result = optimizer.optimize()
    print(result.best_config, result.best_marglik)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar


@dataclass
class PriorOptResult:
    best_config: Dict
    best_marglik: float
    best_ce_auc: float
    best_cr_auc: float
    all_results: List[Dict] = field(default_factory=list)


def compute_spectral_energy(
    delta: np.ndarray,  # (n, d) template residuals
    V: np.ndarray,      # (n, k) eigenvectors
) -> np.ndarray:
    """Compute empirical spectral energy S_j = Σ_i Σ_l (v_j^T δ_i[:,l])².

    Returns (k,) array.
    """
    # V^T @ delta: (k, d)
    proj = V.T @ delta  # (k, d)
    S = (proj ** 2).sum(axis=1)  # (k,)
    return S


def marginal_log_likelihood(
    S: np.ndarray,   # (k,) spectral energies
    D: np.ndarray,   # (k,) predictive variances per mode
    n: int,
    d: int,
) -> float:
    """Closed-form marginal log-likelihood (Theorem 2)."""
    k = len(S)
    ll = -0.5 * np.sum(S / D) - (d / 2.0) * np.sum(np.log(D)) - (n * d / 2.0) * np.log(2 * np.pi)
    return float(ll)


def predictive_variance(
    q: np.ndarray,       # (k,) Q_rho eigenvalues
    lam: float,          # terminal penalty
    alpha_T: float,      # α * T product
) -> np.ndarray:
    """D_j = λ⁻¹ + (1 - exp(-2αTq_j))/q_j."""
    q_safe = np.maximum(q, 1e-8)
    sigma = (1.0 - np.exp(-2.0 * alpha_T * q_safe)) / q_safe
    return 1.0 / lam + sigma


def _canonicalize_two_scale_params(
    kappa: float,
    kappa2: Optional[float],
    eta: float,
) -> Tuple[float, float, float]:
    """Canonicalise the two-scale family so eta always weights the smaller cutoff."""
    k1 = float(kappa)
    k2 = float(kappa if kappa2 is None else kappa2)
    mix = float(eta)
    if k2 < k1:
        return k2, k1, 1.0 - mix
    return k1, k2, mix


def compute_prior_base_eigenvalues(
    lam_L: np.ndarray,
    kappa: float,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> np.ndarray:
    """Base spectral precision before rho-blending."""
    if precision_family in ("single_scale", "matern", "legacy"):
        return (kappa ** 2 + lam_L) ** nu

    if precision_family in ("two_scale", "two_band", "mixture"):
        k1, k2, mix = _canonicalize_two_scale_params(kappa, kappa2, eta)
        q1 = (k1 ** 2 + lam_L) ** nu
        q2 = (k2 ** 2 + lam_L) ** nu
        return mix * q1 + (1.0 - mix) * q2

    raise ValueError(f"Unknown precision_family: {precision_family!r}")


def Q_rho_eigenvalues(
    lam_L: np.ndarray,   # (k,) Laplacian eigenvalues
    rho: float,
    kappa: float,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> np.ndarray:
    """q_j = rho * q_prior,j + (1-rho) for a label-free spectral prior family."""
    q_prior = compute_prior_base_eigenvalues(
        lam_L,
        kappa=kappa,
        nu=nu,
        precision_family=precision_family,
        eta=eta,
        kappa2=kappa2,
    )
    return rho * q_prior + (1.0 - rho)


def compute_modeled_subspace_mask(
    lam_L: np.ndarray,
    mode: str = "legacy",
    tol: float = 1e-8,
) -> np.ndarray:
    """Boolean mask for the modeled eigenspace."""
    if mode in ("legacy", "full", "all"):
        return np.ones_like(lam_L, dtype=bool)

    if mode in ("drop_nullspace", "exclude_nullspace", "project_nullspace"):
        mask = lam_L > tol
        if not np.any(mask):
            return np.ones_like(lam_L, dtype=bool)
        return mask

    raise ValueError(f"Unknown modeled_subspace mode: {mode!r}")


def apply_modeled_subspace(
    S: np.ndarray,
    lam_L: np.ndarray,
    mode: str = "legacy",
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict spectral statistics to the modeled subspace."""
    mask = compute_modeled_subspace_mask(lam_L, mode=mode, tol=tol)
    return S[mask], lam_L[mask], mask


def optimize_lambda(
    S: np.ndarray, q: np.ndarray, alpha_T: float, n: int, d: int,
    lam_range: Tuple[float, float] = (0.01, 10000.0),
) -> Tuple[float, float]:
    """Optimize λ by 1D bisection on the marginal likelihood."""
    def neg_marglik(log_lam):
        lam = np.exp(log_lam)
        D = predictive_variance(q, lam, alpha_T)
        return -marginal_log_likelihood(S, D, n, d)

    result = minimize_scalar(neg_marglik,
                             bounds=(np.log(lam_range[0]), np.log(lam_range[1])),
                             method='bounded')
    lam_opt = np.exp(result.x)
    return lam_opt, -result.fun


def optimize_alpha_T(
    S: np.ndarray, q: np.ndarray, lam: float, n: int, d: int,
    aT_range: Tuple[float, float] = (0.01, 20.0),
) -> Tuple[float, float]:
    """Optimize αT by 1D bisection."""
    def neg_marglik(aT):
        D = predictive_variance(q, lam, aT)
        return -marginal_log_likelihood(S, D, n, d)

    result = minimize_scalar(neg_marglik, bounds=aT_range, method='bounded')
    return result.x, -result.fun


def optimize_rho(
    S: np.ndarray, lam_L: np.ndarray, kappa: float, nu: float,
    lam: float, alpha_T: float, n: int, d: int,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> Tuple[float, float]:
    """Optimize ρ ∈ [0,1] by 1D bisection."""
    def neg_marglik(rho):
        q = Q_rho_eigenvalues(
            lam_L, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        D = predictive_variance(q, lam, alpha_T)
        return -marginal_log_likelihood(S, D, n, d)

    result = minimize_scalar(neg_marglik, bounds=(0.0, 1.0), method='bounded')
    return result.x, -result.fun


def optimize_kappa(
    S: np.ndarray, lam_L: np.ndarray, rho: float, nu: float,
    lam: float, alpha_T: float, n: int, d: int,
    kappa_range: Tuple[float, float] = (0.01, 50.0),
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> Tuple[float, float]:
    """Optimize κ by 1D bisection."""
    def neg_marglik(log_kappa):
        kappa = np.exp(log_kappa)
        q = Q_rho_eigenvalues(
            lam_L, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        D = predictive_variance(q, lam, alpha_T)
        return -marginal_log_likelihood(S, D, n, d)

    result = minimize_scalar(neg_marglik,
                             bounds=(np.log(kappa_range[0]), np.log(kappa_range[1])),
                             method='bounded')
    return np.exp(result.x), -result.fun


def time_dependent_precision(
    q: np.ndarray,       # (k,) Q_rho eigenvalues
    Gamma: float,        # effective time horizon
) -> np.ndarray:
    """Time-dependent precision: q_tilde_j = q_j / (1 - exp(-2*Gamma*q_j)).

    At Gamma -> inf: q_tilde_j -> q_j (stationary limit).
    At Gamma -> 0:   q_tilde_j -> 1/(2*Gamma) (isotropic).
    """
    q_safe = np.maximum(q, 1e-8)
    exponent = -2.0 * Gamma * q_safe
    # For large negative exponents, exp -> 0 and q_tilde -> q
    # For near-zero exponents, use Taylor: 1 - exp(-x) ~ x
    denom = 1.0 - np.exp(exponent)
    denom = np.maximum(denom, 1e-12)
    return q_safe / denom


def marginal_log_likelihood_dynamic(
    S: np.ndarray,   # (k,) spectral energies
    q: np.ndarray,   # (k,) Q_rho eigenvalues
    Gamma: float,    # effective time horizon
    n: int,
    d: int,
) -> float:
    """Marginal log-likelihood under finite-time GOU generative model.

    log p(D|phi,Gamma) = -1/2 sum q_tilde_j S_j + d/2 sum log(q_tilde_j) - nd/2 log(2pi)

    where q_tilde_j = q_j / (1 - exp(-2*Gamma*q_j)).
    """
    qt = time_dependent_precision(q, Gamma)
    ll = -0.5 * np.sum(S * qt) + (d / 2.0) * np.sum(np.log(qt)) - (n * d / 2.0) * np.log(2 * np.pi)
    return float(ll)


def solve_rho_dynamic(
    S: np.ndarray,
    lam_L: np.ndarray,
    kappa: float,
    Gamma: float,
    d: int,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
    rho_init: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> float:
    """Solve for optimal rho under finite-time ML via bounded optimization.

    Unlike the stationary case, the ML is not guaranteed concave in rho
    (because q_tilde is nonlinear in q). We use bounded scalar search.
    """
    q_prior = compute_prior_base_eigenvalues(
        lam_L,
        kappa=kappa,
        nu=nu,
        precision_family=precision_family,
        eta=eta,
        kappa2=kappa2,
    )
    a = q_prior - 1.0

    def neg_ml(rho):
        q = rho * a + 1.0
        qt = time_dependent_precision(q, Gamma)
        return 0.5 * np.sum(S * qt) - (d / 2.0) * np.sum(np.log(qt))

    result = minimize_scalar(neg_ml, bounds=(0.0, 1.0), method='bounded')
    return float(np.clip(result.x, 0.0, 1.0))


def gamma_jacobian_correction(
    lam_L: np.ndarray,   # (k,) Laplacian eigenvalues
    gamma: float,
    d: int,
    nu: float = 1.0,
) -> float:
    """Log-Jacobian of the residual operator T_{gamma,nu} = I - S_{gamma,nu}.

    Returns d * sum_{j: lam_j > 0} log|1 - (gamma^2 + lam_j)^{-nu}|.
    This correction is needed when comparing likelihoods across different gamma.
    """
    mask = lam_L > 1e-12  # exclude zero mode
    s_j = 1.0 / (gamma ** 2 + lam_L[mask]) ** nu
    t_j = 1.0 - s_j  # eigenvalues of T_{gamma,nu}
    t_j = np.maximum(t_j, 1e-12)
    return float(d * np.sum(np.log(t_j)))


def per_mode_diagnostic(
    S: np.ndarray, D: np.ndarray, n: int, d: int,
) -> np.ndarray:
    """Per-mode calibration ratio r_j = D_j / (S_j/d).

    r_j ≈ 1: well-calibrated. r_j >> 1: too loose. r_j << 1: too tight.
    """
    target = S / d
    target = np.maximum(target, 1e-12)
    return D / target


def marginal_log_likelihood_stationary(
    S: np.ndarray,   # (k,) spectral energies
    q: np.ndarray,   # (k,) Q_rho eigenvalues = precision per mode
    n: int,
    d: int,
) -> float:
    """Marginal log-likelihood under stationary prior N(m, Q_rho^{-1}).

    log p(D|φ) = -Σ S_j q_j / 2 + (d/2) Σ log(q_j) - nd/2 log(2π)
    """
    q_safe = np.maximum(q, 1e-12)
    ll = -0.5 * np.sum(S * q_safe) + (d / 2.0) * np.sum(np.log(q_safe)) - (n * d / 2.0) * np.log(2 * np.pi)
    return float(ll)


def optimize_rho_stationary(
    S: np.ndarray, lam_L: np.ndarray, kappa: float, nu: float,
    n: int, d: int,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> Tuple[float, float]:
    """Optimize ρ ∈ [0,1] under stationary ML."""
    def neg_ml(rho):
        q = Q_rho_eigenvalues(
            lam_L, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        return -marginal_log_likelihood_stationary(S, q, n, d)

    result = minimize_scalar(neg_ml, bounds=(0.0, 1.0), method='bounded')
    return result.x, -result.fun


def optimize_kappa_stationary(
    S: np.ndarray, lam_L: np.ndarray, rho: float, nu: float,
    n: int, d: int,
    kappa_range: Tuple[float, float] = (0.01, 50.0),
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> Tuple[float, float]:
    """Optimize κ under stationary ML."""
    def neg_ml(log_kappa):
        kappa = np.exp(log_kappa)
        q = Q_rho_eigenvalues(
            lam_L, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        return -marginal_log_likelihood_stationary(S, q, n, d)

    result = minimize_scalar(neg_ml,
                             bounds=(np.log(kappa_range[0]), np.log(kappa_range[1])),
                             method='bounded')
    return np.exp(result.x), -result.fun


def optimize_kappa2_stationary(
    S: np.ndarray, lam_L: np.ndarray, rho: float, kappa: float, nu: float,
    n: int, d: int,
    eta: float = 0.5,
    kappa2_range: Tuple[float, float] = (0.01, 50.0),
) -> Tuple[float, float]:
    """Optimize the second cutoff under the two-scale stationary ML."""
    def neg_ml(log_kappa2):
        kappa2 = np.exp(log_kappa2)
        q = Q_rho_eigenvalues(
            lam_L, rho, kappa, nu,
            precision_family="two_scale", eta=eta, kappa2=kappa2,
        )
        return -marginal_log_likelihood_stationary(S, q, n, d)

    result = minimize_scalar(
        neg_ml,
        bounds=(np.log(kappa2_range[0]), np.log(kappa2_range[1])),
        method='bounded',
    )
    return np.exp(result.x), -result.fun


def optimize_eta_stationary(
    S: np.ndarray, lam_L: np.ndarray, rho: float, kappa: float, nu: float,
    n: int, d: int,
    kappa2: float,
) -> Tuple[float, float]:
    """Optimize the mixture weight under the two-scale stationary ML."""
    def neg_ml(eta):
        q = Q_rho_eigenvalues(
            lam_L, rho, kappa, nu,
            precision_family="two_scale", eta=eta, kappa2=kappa2,
        )
        return -marginal_log_likelihood_stationary(S, q, n, d)

    result = minimize_scalar(neg_ml, bounds=(0.0, 1.0), method='bounded')
    return result.x, -result.fun


def coordinate_descent_stationary(
    S: np.ndarray,
    lam_L: np.ndarray,
    n: int, d: int,
    rho_init: float = 0.5,
    kappa_init: float = 1.0,
    kappa2_init: Optional[float] = None,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta_init: float = 0.5,
    optimize_eta: bool = False,
    optimize_kappa2: bool = False,
    modeled_subspace: str = "legacy",
    nullspace_tol: float = 1e-8,
    n_iters: int = 5,
    kappa_range: Optional[Tuple[float, float]] = None,
    kappa2_range: Optional[Tuple[float, float]] = None,
    use_data_driven_bounds: bool = False,
    bound_percentile: float = 50.0,
) -> Dict:
    """Coordinate descent under stationary prior N(m, Q_rho^{-1}).

    Only optimizes (ρ, κ) — λ and αT are irrelevant since the predictive
    variance is 1/q_j independent of the bridge dynamics.
    """
    S_eff, lam_L_eff, modeled_mask = apply_modeled_subspace(
        S, lam_L, mode=modeled_subspace, tol=nullspace_tol,
    )
    rho, kappa = rho_init, kappa_init
    eta = eta_init
    kappa2 = kappa_init if kappa2_init is None else kappa2_init
    if precision_family in ("two_scale", "two_band", "mixture"):
        kappa, kappa2, eta = _canonicalize_two_scale_params(kappa, kappa2, eta)

    for iteration in range(n_iters):
        # Optimize ρ
        rho, ml = optimize_rho_stationary(
            S_eff, lam_L_eff, kappa, nu, n, d,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )

        # Compute κ bounds
        if kappa_range is not None:
            kr = kappa_range
        elif use_data_driven_bounds:
            kr = compute_kappa_bounds(S_eff, lam_L_eff, rho, n, d, nu, bound_percentile)
        else:
            kr = (0.01, 50.0)

        # Optimize κ
        kappa, ml = optimize_kappa_stationary(
            S_eff, lam_L_eff, rho, nu, n, d,
            kappa_range=kr,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )

        if precision_family in ("two_scale", "two_band", "mixture"):
            if optimize_kappa2:
                kr2 = kappa2_range if kappa2_range is not None else kr
                kappa2, ml = optimize_kappa2_stationary(
                    S_eff, lam_L_eff, rho, kappa, nu, n, d,
                    eta=eta, kappa2_range=kr2,
                )
            if optimize_eta:
                eta, ml = optimize_eta_stationary(
                    S_eff, lam_L_eff, rho, kappa, nu, n, d, kappa2=kappa2,
                )
            kappa, kappa2, eta = _canonicalize_two_scale_params(kappa, kappa2, eta)

    q = Q_rho_eigenvalues(
        lam_L_eff, rho, kappa, nu,
        precision_family=precision_family, eta=eta, kappa2=kappa2,
    )
    ml = marginal_log_likelihood_stationary(S_eff, q, n, d)

    if kappa_range is not None:
        final_kr = kappa_range
    elif use_data_driven_bounds:
        final_kr = compute_kappa_bounds(S_eff, lam_L_eff, rho, n, d, nu, bound_percentile)
    else:
        final_kr = (0.01, 50.0)

    return {
        "rho": float(rho),
        "kappa": float(kappa),
        "kappa2": float(kappa2),
        "eta": float(eta),
        "lam_penalty": 50.0,  # placeholder, not used
        "alpha_T": 2.0,       # placeholder, not used
        "marginal_likelihood": float(ml),
        "kappa_bounds": (float(final_kr[0]), float(final_kr[1])),
        "kappa2_bounds": (
            float((kappa2_range or final_kr)[0]),
            float((kappa2_range or final_kr)[1]),
        ),
        "precision_family": precision_family,
        "modeled_subspace": modeled_subspace,
        "modeled_rank": int(lam_L_eff.shape[0]),
        "modeled_mask": modeled_mask,
    }


def solve_rho_newton(
    S: np.ndarray,
    lam_L: np.ndarray,
    kappa: float,
    d: int,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
    rho_init: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> float:
    """Solve for optimal rho via Newton's method (guaranteed by concavity).

    The ML is strictly concave in rho, so Newton converges quadratically
    from any initialization to the unique global optimum.
    """
    q_prior = compute_prior_base_eigenvalues(
        lam_L,
        kappa=kappa,
        nu=nu,
        precision_family=precision_family,
        eta=eta,
        kappa2=kappa2,
    )
    a = q_prior - 1.0

    rho = np.clip(rho_init, 1e-6, 1.0 - 1e-6)
    for _ in range(max_iter):
        q = rho * a + 1.0
        q_safe = np.maximum(q, 1e-12)

        # Gradient: Σ_j a_j * (-S_j/2 + d/(2q_j))
        grad = np.sum(a * (-S / 2.0 + d / (2.0 * q_safe)))

        # Hessian: -Σ_j d * a_j² / (2 * q_j²)  (always negative)
        hess = -np.sum(d * a ** 2 / (2.0 * q_safe ** 2))

        if abs(hess) < 1e-15:
            break

        step = -grad / hess
        rho_new = np.clip(rho + step, 0.0, 1.0)

        if abs(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new

    return float(np.clip(rho, 0.0, 1.0))


def profile_newton_optimize(
    S: np.ndarray,
    lam_L: np.ndarray,
    n: int, d: int,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
    modeled_subspace: str = "legacy",
    nullspace_tol: float = 1e-8,
    kappa_range: Optional[Tuple[float, float]] = None,
    use_data_driven_bounds: bool = True,
    bound_percentile: float = 50.0,
    n_kappa_grid: int = 50,
) -> Dict:
    """Profile-Newton optimizer: solve rho exactly (Newton), search kappa (1D).

    For each candidate kappa, solves rho*(kappa) via Newton (guaranteed
    global optimum by concavity), then picks the kappa that maximizes
    the profile ML(kappa) = max_rho ML(rho, kappa).
    """
    S_eff, lam_L_eff, _ = apply_modeled_subspace(
        S, lam_L, mode=modeled_subspace, tol=nullspace_tol,
    )

    # Determine kappa bounds
    if kappa_range is not None:
        kr = kappa_range
    elif use_data_driven_bounds:
        kr = compute_kappa_bounds(S_eff, lam_L_eff, 0.5, n, d, nu, bound_percentile)
    else:
        kr = (0.01, 50.0)

    kappa_grid = np.logspace(np.log10(max(kr[0], 0.01)),
                              np.log10(max(kr[1], 0.1)),
                              n_kappa_grid)

    best_ml = -np.inf
    best_rho = 0.5
    best_kappa = 1.0

    for kappa in kappa_grid:
        # Solve rho exactly via Newton
        rho = solve_rho_newton(
            S_eff, lam_L_eff, kappa, d, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )

        # Evaluate ML
        q = Q_rho_eigenvalues(
            lam_L_eff, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        ml = marginal_log_likelihood_stationary(S_eff, q, n, d)

        if ml > best_ml:
            best_ml = ml
            best_rho = rho
            best_kappa = kappa

    # Refine kappa with bounded scalar search around best
    from scipy.optimize import minimize_scalar
    def neg_profile_ml(log_kappa):
        kappa = np.exp(log_kappa)
        rho = solve_rho_newton(
            S_eff, lam_L_eff, kappa, d, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        q = Q_rho_eigenvalues(
            lam_L_eff, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        return -marginal_log_likelihood_stationary(S_eff, q, n, d)

    result = minimize_scalar(neg_profile_ml,
                             bounds=(np.log(max(kr[0], 0.01)),
                                     np.log(max(kr[1], 0.1))),
                             method='bounded')
    kappa_opt = np.exp(result.x)
    rho_opt = solve_rho_newton(
        S_eff, lam_L_eff, kappa_opt, d, nu,
        precision_family=precision_family, eta=eta, kappa2=kappa2,
    )
    q_opt = Q_rho_eigenvalues(
        lam_L_eff, rho_opt, kappa_opt, nu,
        precision_family=precision_family, eta=eta, kappa2=kappa2,
    )
    ml_opt = marginal_log_likelihood_stationary(S_eff, q_opt, n, d)

    if ml_opt > best_ml:
        best_ml = ml_opt
        best_rho = rho_opt
        best_kappa = kappa_opt

    return {
        "rho": float(best_rho),
        "kappa": float(best_kappa),
        "lam_penalty": 50.0,
        "alpha_T": 2.0,
        "marginal_likelihood": float(best_ml),
        "kappa_bounds": (float(kr[0]), float(kr[1])),
        "optimizer": "profile_newton",
        "precision_family": precision_family,
        "eta": float(eta),
        "kappa2": float(kappa if kappa2 is None else kappa2),
        "modeled_subspace": modeled_subspace,
    }


def optimize_joint_lbfgsb_stationary(
    S: np.ndarray,
    lam_L: np.ndarray,
    n: int, d: int,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
    modeled_subspace: str = "legacy",
    nullspace_tol: float = 1e-8,
    kappa_range: Tuple[float, float] = (0.01, 50.0),
    n_restarts: int = 10,
) -> Dict:
    """Joint (ρ, κ) optimization via L-BFGS-B under stationary prior.

    Optimizes log ML(ρ, κ) = -Σ S_j q_j/2 + (d/2) Σ log(q_j) with
    analytical gradients. Operates on (ρ, log κ) for better conditioning.
    """
    from scipy.optimize import minimize

    S_eff, lam_L_eff, _ = apply_modeled_subspace(
        S, lam_L, mode=modeled_subspace, tol=nullspace_tol,
    )
    log_kappa_bounds = (np.log(kappa_range[0]), np.log(kappa_range[1]))

    def neg_ml_and_grad(params):
        rho, log_kappa = params
        kappa = np.exp(log_kappa)
        q_prior = compute_prior_base_eigenvalues(
            lam_L_eff,
            kappa=kappa,
            nu=nu,
            precision_family=precision_family,
            eta=eta,
            kappa2=kappa2,
        )
        q = rho * q_prior + (1.0 - rho)
        q_safe = np.maximum(q, 1e-12)

        # ML
        ml = -0.5 * np.sum(S_eff * q_safe) + (d / 2.0) * np.sum(np.log(q_safe))

        # Gradients
        common = -0.5 * S_eff + (d / 2.0) / q_safe  # ∂ML/∂q_j

        # ∂q_j/∂ρ = q_prior - 1
        dq_drho = q_prior - 1.0
        grad_rho = np.sum(common * dq_drho)

        # ∂q_j/∂κ = ρ * ν * (κ² + λ_j)^(ν-1) * 2κ
        if precision_family in ("single_scale", "matern", "legacy"):
            if nu == 1.0:
                dq_dkappa = rho * 2.0 * kappa
            else:
                dq_dkappa = rho * nu * (kappa ** 2 + lam_L_eff) ** (nu - 1.0) * 2.0 * kappa
        else:
            k1, k2, mix = _canonicalize_two_scale_params(kappa, kappa2, eta)
            if abs(k1 - kappa) < 1e-12:
                dq_base = mix * nu * (kappa ** 2 + lam_L_eff) ** (nu - 1.0) * 2.0 * kappa
            else:
                dq_base = (1.0 - mix) * nu * (kappa ** 2 + lam_L_eff) ** (nu - 1.0) * 2.0 * kappa
            dq_dkappa = rho * dq_base
        # Chain rule for log κ: ∂ML/∂(log κ) = ∂ML/∂κ * κ
        grad_log_kappa = np.sum(common * dq_dkappa) * kappa

        return -ml, np.array([-grad_rho, -grad_log_kappa])

    best_result = None
    best_ml = -np.inf

    # Multi-start
    rng = np.random.RandomState(42)
    starts = []
    for rho_init in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for lk_init in np.linspace(log_kappa_bounds[0], log_kappa_bounds[1], max(1, n_restarts // 5)):
            starts.append([rho_init, lk_init])

    for x0 in starts:
        try:
            res = minimize(neg_ml_and_grad, x0, method='L-BFGS-B', jac=True,
                          bounds=[(0.0, 1.0), log_kappa_bounds],
                          options={'maxiter': 200, 'ftol': 1e-12})
            if -res.fun > best_ml:
                best_ml = -res.fun
                best_result = res
        except:
            pass

    if best_result is None:
        return {"rho": 0.5, "kappa": 1.0, "lam_penalty": 50.0, "alpha_T": 2.0,
                "marginal_likelihood": -np.inf, "kappa_bounds": kappa_range,
                "optimizer": "lbfgsb_failed", "precision_family": precision_family,
                "eta": float(eta), "kappa2": float(kappa if kappa2 is None else kappa2),
                "modeled_subspace": modeled_subspace}

    rho_opt = best_result.x[0]
    kappa_opt = np.exp(best_result.x[1])

    return {
        "rho": float(rho_opt),
        "kappa": float(kappa_opt),
        "lam_penalty": 50.0,
        "alpha_T": 2.0,
        "marginal_likelihood": float(best_ml),
        "kappa_bounds": (float(kappa_range[0]), float(kappa_range[1])),
        "optimizer": "lbfgsb",
        "precision_family": precision_family,
        "eta": float(eta),
        "kappa2": float(kappa_opt if kappa2 is None else kappa2),
        "modeled_subspace": modeled_subspace,
    }


def compute_kappa_bounds(
    S: np.ndarray,
    lam_L: np.ndarray,
    rho: float,
    n: int, d: int,
    nu: float = 1.0,
    percentile: float = 50.0,
) -> Tuple[float, float]:
    """Data-driven κ bounds from variance matching (Corollary 1).

    The variance matching target q_j* = d/S_j gives the ideal precision
    per spectral mode. We bound κ so the median mode's precision doesn't
    exceed the percentile-th target, preventing over-tightening.

    For ν=1: q_j = ρ(κ² + λ_j) + (1-ρ), so
        κ_max = sqrt(max(0, (q̃* - 1)/ρ + 1 - λ̃))
    where q̃* = percentile(d/S_j), λ̃ = median(lam_L).

    Returns (kappa_min, kappa_max).
    """
    if rho < 1e-8:
        return (0.01, 50.0)  # κ irrelevant when ρ=0

    # Variance matching targets (avoid division by zero)
    S_safe = np.maximum(S, 1e-12)
    # Use n*d/S_j for practical bounds (covers the parametric range needed
    # when a single kappa must serve all eigenmodes simultaneously).
    # The per-mode optimum is d/S_j, but the parametric family q_j(rho,kappa)
    # cannot match all modes independently, so bounds must be looser.
    q_targets = n * d / S_safe

    q_ref = np.percentile(q_targets, percentile)
    lam_med = np.median(lam_L)

    # κ² = (q_ref - 1)/ρ + 1 - λ_med  (for ν=1)
    if nu == 1.0:
        kappa_sq = (q_ref - 1.0) / rho + 1.0 - lam_med
    else:
        # General ν: q_ref = ρ(κ² + λ_med)^ν + (1-ρ)
        # κ² = ((q_ref - (1-ρ))/ρ)^{1/ν} - λ_med
        inner = (q_ref - (1.0 - rho)) / rho
        if inner < 0:
            kappa_sq = 0.0
        else:
            kappa_sq = inner ** (1.0 / nu) - lam_med

    kappa_max = np.sqrt(max(0.0, kappa_sq))
    kappa_max = max(kappa_max, 0.1)  # floor at 0.1

    return (0.01, float(kappa_max))


def coordinate_descent(
    S: np.ndarray,
    lam_L: np.ndarray,
    n: int, d: int,
    rho_init: float = 0.5,
    kappa_init: float = 1.0,
    lam_init: float = 50.0,
    alpha_T_init: float = 2.0,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
    modeled_subspace: str = "legacy",
    nullspace_tol: float = 1e-8,
    n_iters: int = 5,
    kappa_range: Optional[Tuple[float, float]] = None,
    use_data_driven_bounds: bool = False,
    bound_percentile: float = 50.0,
) -> Dict:
    """Coordinate descent on continuous parameters (λ, αT, ρ, κ).

    Args:
        kappa_range: explicit (min, max) for κ. Overrides data-driven bounds.
        use_data_driven_bounds: if True and kappa_range is None, compute
            κ bounds from the variance matching condition each iteration.
        bound_percentile: percentile of d/S_j for data-driven κ_max.

    Returns dict with optimized parameters and marginal likelihood.
    """
    S_eff, lam_L_eff, _ = apply_modeled_subspace(
        S, lam_L, mode=modeled_subspace, tol=nullspace_tol,
    )
    rho, kappa, lam, alpha_T = rho_init, kappa_init, lam_init, alpha_T_init

    for iteration in range(n_iters):
        # Optimize λ
        q = Q_rho_eigenvalues(
            lam_L_eff, rho, kappa, nu,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )
        lam, ml = optimize_lambda(S_eff, q, alpha_T, n, d)

        # Optimize αT
        alpha_T, ml = optimize_alpha_T(S_eff, q, lam, n, d)

        # Optimize ρ
        rho, ml = optimize_rho(
            S_eff, lam_L_eff, kappa, nu, lam, alpha_T, n, d,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )

        # Compute κ bounds (recomputed each iteration since ρ changes)
        if kappa_range is not None:
            kr = kappa_range
        elif use_data_driven_bounds:
            kr = compute_kappa_bounds(S_eff, lam_L_eff, rho, n, d, nu, bound_percentile)
        else:
            kr = (0.01, 50.0)

        # Optimize κ
        kappa, ml = optimize_kappa(
            S_eff, lam_L_eff, rho, nu, lam, alpha_T, n, d,
            kappa_range=kr,
            precision_family=precision_family, eta=eta, kappa2=kappa2,
        )

    q = Q_rho_eigenvalues(
        lam_L_eff, rho, kappa, nu,
        precision_family=precision_family, eta=eta, kappa2=kappa2,
    )
    D = predictive_variance(q, lam, alpha_T)
    ml = marginal_log_likelihood(S_eff, D, n, d)
    diag = per_mode_diagnostic(S_eff, D, n, d)

    # Record the final κ bounds used
    if kappa_range is not None:
        final_kr = kappa_range
    elif use_data_driven_bounds:
        final_kr = compute_kappa_bounds(S_eff, lam_L_eff, rho, n, d, nu, bound_percentile)
    else:
        final_kr = (0.01, 50.0)

    return {
        "rho": float(rho),
        "kappa": float(kappa),
        "lam_penalty": float(lam),
        "alpha_T": float(alpha_T),
        "marginal_likelihood": float(ml),
        "diagnostic_mean_ratio": float(np.mean(diag)),
        "diagnostic_std_ratio": float(np.std(np.log(diag))),
        "kappa_bounds": (float(final_kr[0]), float(final_kr[1])),
        "precision_family": precision_family,
        "eta": float(eta),
        "kappa2": float(kappa if kappa2 is None else kappa2),
        "modeled_subspace": modeled_subspace,
    }
