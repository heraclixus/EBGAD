"""Two-groups empirical-Bayes aggregation over a channel's profile members.

Per node i and member m, the exact null (nulls.py) gives a tail probability;
its probit z is calibrated by Efron's empirical null (central matching per
member, select.py) so the bulk is N(0,1), giving p_im. A member whose z is
degenerate (IQR below the numerical guard) is dropped.

Model (Efron's two-groups with a "which member" latent):
    with probability 1 - pi the node is null: every p_im ~ Uniform(0,1);
    with probability pi it is non-null at one member m ~ w: p_im ~ Beta(a_m, 1)
    (density a_m p^{a_m - 1}, enriched near 0 for a_m < 1), other members
    uniform.
EM fits pi, w, a from the unlabeled p-values. Output: the posterior
probability of being non-null, r_i, and the responsible member.

This is the only aggregation in v2 and it has no dataset-specific
constants: pi is bounded above by 1/2 (identifiability of a two-groups
model), a_m by (0, 1] (definition of an enriched alternative).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .select import MIN_LOG_IQR, binned_density, efron_central_matching


MAX_TIE_FRACTION = 0.10  # numerical guard: a member whose modal value covers more nodes is unusable


def calibrated_pvalues(z: np.ndarray) -> Optional[np.ndarray]:
    """Empirical-null calibrated one-sided p-values from probit z's; None if degenerate.

    Degenerate means a bulk without spread (IQR below the numerical guard)
    or massive ties: on YelpChi's R-U-R graph thousands of nodes sit in
    two-node components whose ratio statistic is identical by construction,
    and a Beta alternative fitted to a column of tied p-values drives its
    enrichment to zero and the mixing fraction to its cap (2026-09-18).
    """
    zc = np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0, posinf=40.0, neginf=-40.0)
    iqr = float(np.subtract(*np.percentile(zc, [75, 25])))
    if zc.size < 10 or iqr < MIN_LOG_IQR:
        return None
    _vals, counts = np.unique(np.round(zc, 6), return_counts=True)
    if counts.max() > MAX_TIE_FRACTION * zc.size:
        return None
    grid, dens, _ = binned_density(zc)
    d0, s0, _pi0 = efron_central_matching(zc, grid, dens)
    p = norm.sf((zc - d0) / s0)
    return np.clip(p, 1e-300, 1.0)


def fit_two_groups(P: np.ndarray, pi_init: float = 0.05, pi_max: float = 0.5,
                   a_init: float = 0.5, max_iter: int = 300, tol: float = 1e-8) -> Dict:
    """EM for the two-groups model on an (n, M) matrix of calibrated p-values."""
    n, M = P.shape
    logp = np.log(np.clip(P, 1e-300, 1.0))
    pi = float(np.clip(pi_init, 1e-4, pi_max))
    w = np.full(M, 1.0 / M)
    a = np.full(M, float(a_init))
    prev = None  # (a -inf sentinel made |obj - prev| = inf <= inf true at the first iteration: one-step EM, 2026-09-18)
    for it in range(max_iter):
        log_f1 = np.log(a)[None, :] + (a[None, :] - 1.0) * logp          # (n, M)
        log_comp = np.log(np.maximum(w, 1e-300))[None, :] + log_f1
        log_A = logsumexp(log_comp, axis=1)                                # log sum_m w_m f1_m
        log_anom = np.log(pi) + log_A
        log_null = np.log(1.0 - pi)
        log_marg = np.logaddexp(log_null, log_anom)
        r = np.exp(log_anom - log_marg)                                    # posterior non-null
        c = np.exp(log_comp - log_A[:, None])                              # member responsibility
        rc = r[:, None] * c
        # M-step
        pi = float(np.clip(r.mean(), 1e-4, pi_max))
        wsum = rc.sum(axis=0)
        w = np.maximum(wsum / max(wsum.sum(), 1e-300), 1e-6)
        w = w / w.sum()
        denom = (rc * logp).sum(axis=0)                                    # sum r c log p (negative)
        a = np.clip(-wsum / np.minimum(denom, -1e-12), 1e-3, 1.0)
        obj = float(log_marg.sum())
        if prev is not None and np.isfinite(obj) and abs(obj - prev) <= tol * max(1.0, abs(prev)):
            prev = obj
            break
        prev = obj
    log_f1 = np.log(a)[None, :] + (a[None, :] - 1.0) * logp
    log_comp = np.log(np.maximum(w, 1e-300))[None, :] + log_f1
    log_A = logsumexp(log_comp, axis=1)
    log_anom = np.log(pi) + log_A
    log_marg = np.logaddexp(log_null, log_anom)
    r = np.exp(log_anom - log_marg)
    resp_member = np.argmax(log_comp, axis=1)
    return {"posterior": r, "log_bayes_factor": log_A, "pi": pi, "w": w, "a": a,
            "loglik": prev, "iterations": it + 1, "responsible": resp_member}


def two_groups_channel(z_by_member: Dict[str, np.ndarray]) -> Optional[Dict]:
    """Calibrate each member, drop degenerate ones, fit the two-groups model."""
    names, cols = [], []
    for name, z in z_by_member.items():
        p = calibrated_pvalues(z)
        if p is not None:
            names.append(name)
            cols.append(p)
    if not cols:
        return None
    P = np.column_stack(cols)
    fit = fit_two_groups(P)
    fit["members"] = names
    fit["member_weights"] = {nm: float(wt) for nm, wt in zip(names, fit["w"])}
    fit["member_a"] = {nm: float(av) for nm, av in zip(names, fit["a"])}
    return fit
