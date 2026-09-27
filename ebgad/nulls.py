"""Exact per-node null distributions under the fitted prior.

Under the fitted equilibrium law the mode coefficients xi_j = V_j^T delta are
independent N(0, q_j^{-1} I_D). For node i and a bank member with weights c,
each of the D coordinates of (u_i, delta_i) is bivariate normal with
    var(u) = sigma2_i,  var(delta) = tau2_i,  cov = cov_i
(all computed in scores.compute_bank). Hence

    energy_i = ||u_i||^2                ~ sigma2_i * chi^2_D      (central)
             (origin start)              ~ sigma2_i * chi^2_D(ncp = mu2_i / sigma2_i)
    ratio_i  = ||u_i||^2 / ||delta_i||^2:  P(ratio <= r) = F_{D,D}( |l_-| / l_+ )
             where l_+ > 0 > l_- are the eigenvalues of Sigma_i diag(1, -r).
    P_i      = ||(Q delta)_i||^2 / Q_ii ~ chi^2_D

Monte Carlo helpers draw the residual field from the fitted prior to verify
these closed forms (the self-check in run_ebgad.py --null-check).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2, f as f_dist, kstest, ncx2

from .scores import BankMember, bank_configs, compute_bank


Tails = Tuple[np.ndarray, np.ndarray]  # (upper tail P(stat >= obs), lower tail P(stat <= obs))


def energy_tails(score: np.ndarray, sigma2: np.ndarray, d: int,
                 mu2: Optional[np.ndarray] = None) -> Tails:
    x = np.maximum(score, 0.0) / np.maximum(sigma2, 1e-300)
    if mu2 is None:
        return _clean(chi2.sf(x, df=d)), _clean(chi2.cdf(x, df=d))
    nc = np.maximum(mu2, 0.0) / np.maximum(sigma2, 1e-300)
    return _clean(ncx2.sf(x, df=d, nc=nc)), _clean(ncx2.cdf(x, df=d, nc=nc))


def ratio_tails(score: np.ndarray, sigma2: np.ndarray, tau2: np.ndarray,
                cov: np.ndarray, d: int) -> Tails:
    r = np.maximum(score, 0.0)
    a, b, c = sigma2, tau2, cov
    t = a - r * b
    det = -r * np.maximum(a * b - c * c, 0.0)
    disc = np.sqrt(np.maximum(t * t - 4.0 * det, 0.0))
    l_plus = 0.5 * (t + disc)
    l_minus = 0.5 * (t - disc)
    x = np.abs(l_minus) / np.maximum(l_plus, 1e-300)
    return _clean(f_dist.sf(x, d, d)), _clean(f_dist.cdf(x, d, d))


def cond_tails(score: np.ndarray, d: int) -> Tails:
    x = np.maximum(score, 0.0)
    return _clean(chi2.sf(x, df=d)), _clean(chi2.cdf(x, df=d))


def member_tails(m: BankMember, d: int) -> Tails:
    """Upper and lower tail probabilities, each accurate in its own small regime."""
    if m.kind == "energy":
        return energy_tails(m.score, m.sigma2, d, m.mu2)
    if m.kind == "ratio":
        if m.mu2 is not None:
            raise NotImplementedError("ratio null with origin start: use Monte Carlo")
        return ratio_tails(m.score, m.sigma2, m.tau2, m.cov, d)
    if m.kind == "cond":
        return cond_tails(m.score, d)
    raise ValueError(m.kind)


def member_pvalues(m: BankMember, d: int) -> np.ndarray:
    return member_tails(m, d)[0]


def _clean(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(p, 1e-300, 1.0)


# --------------------------------------------------------------------------
# Monte Carlo self-check: sample the residual field from the fitted prior,
# recompute every statistic and its closed-form p-value, test uniformity.
# --------------------------------------------------------------------------

def simulate_residual(prep, q: np.ndarray, rng: np.random.Generator,
                      rough: str = "isotropic") -> np.ndarray:
    k, d = prep.Xi.shape
    Xi = rng.standard_normal((k, d)) / np.sqrt(np.maximum(q, 1e-12))[:, None]
    delta = prep.V @ Xi
    if rough == "isotropic":
        from .scores import rough_variance
        rv = rough_variance(prep)
        # Unretained component: isotropic N(0, rv I) projected onto the orthogonal
        # complement of V, so node i gets variance rv (1 - h_i), independent of Xi.
        E = rng.standard_normal((prep.n, d)) * np.sqrt(max(rv, 0.0))
        E = E - prep.V @ (prep.V.T @ E)
        delta = delta + E
    return delta


def null_check(prep, q: np.ndarray, specs: Sequence = None, B: int = 50,
               seed: int = 0, start: str = "template", rough: str = "isotropic",
               include_cond: bool = True, rho: float = None, kappa: float = None,
               n_sub: int = 1000) -> Dict[str, Dict[str, float]]:
    """Per-node calibration of the closed-form nulls on data simulated from the fitted prior.

    Draws B residual fields, evaluates every bank statistic on a random
    subset of n_sub nodes, and pools the (node, draw) p-values. Within one
    draw the node statistics are correlated (on a truncated spectrum all
    nodes share the same k d Gaussians), so the check needs many draws and
    few nodes rather than one draw and all nodes: with B = 5 and every node
    the pooled KS read 0.04 to 0.27 on Questions although each node's mean
    matched its null to 0.3 percent (2026-09-17).

    Returns per statistic: KS distance of pooled p-values from U(0,1), and
    mean_ratio = average of statistic / null mean (1.0 when exact).
    Only template-start members are checked (origin-start ratios have no
    closed form). P is checked only if include_cond (full-spectrum graphs).
    """
    from .scores import bank_specs, conditional_residual, rough_variance
    if specs is None:
        specs = bank_specs(q, bank_configs())
    rng = np.random.default_rng(seed)
    n, d, k = prep.n, prep.d, prep.k
    idx = np.sort(rng.choice(n, size=min(n_sub, n), replace=False))
    V = prep.V
    Vs = V[idx]
    V2s = Vs * Vs
    q_safe = np.maximum(q, 1e-12)
    inv_q = 1.0 / q_safe
    rv = rough_variance(prep) if rough == "isotropic" else 0.0
    lev_s = prep.lev[idx]
    tau2 = V2s @ inv_q + rv * (1.0 - lev_s)
    weights = {}
    for spec in specs:
        c = spec.c
        weights[(spec.ename, spec.rname)] = (c, V2s @ (c * inv_q), V2s @ (np.sqrt(c) * inv_q))
    pooled: Dict[str, List[np.ndarray]] = {}
    ratios: Dict[str, List[float]] = {}
    for _ in range(B):
        Xi = rng.standard_normal((k, d)) / np.sqrt(q_safe)[:, None]
        delta_s = Vs @ Xi
        if rough == "isotropic" and rv > 0:
            E = rng.standard_normal((n, d)) * np.sqrt(rv)
            delta_s = delta_s + (E[idx] - Vs @ (V.T @ E))
        den = np.maximum((delta_s * delta_s).sum(axis=1), 1e-12)
        for (ename, rname), (c, sigma2, cov) in weights.items():
            u = Vs @ (np.sqrt(c)[:, None] * Xi)
            energy = (u * u).sum(axis=1)
            pooled.setdefault(ename, []).append(energy_tails(energy, sigma2, d)[0])
            ratios.setdefault(ename, []).append(float(np.mean(energy / (d * sigma2))))
            r = energy / den
            pooled.setdefault(rname, []).append(ratio_tails(r, sigma2, tau2, cov, d)[0])
            ratios.setdefault(rname, []).append(float(np.mean(r / (sigma2 / tau2))))
        if include_cond and rho is not None and not prep.truncated:
            from copy import copy
            sim = copy(prep)
            sim.delta = V @ Xi
            if rough == "isotropic" and rv > 0:
                sim.delta = sim.delta + (E - V @ (V.T @ E))
            m = conditional_residual(sim, rho, kappa)
            pooled.setdefault("P", []).append(cond_tails(m.score[idx], d)[0])
            ratios.setdefault("P", []).append(float(np.mean(m.score[idx] / d)))
    out = {}
    for name, ps in pooled.items():
        p = np.concatenate(ps)
        out[name] = {"ks": float(kstest(p, "uniform").statistic),
                     "mean_ratio": float(np.mean(ratios[name]))}
    return out
