"""Hierarchical node-scale prior.

Model M: delta_i = sqrt(s_i) z_i with z the GOU stationary field and
s_i iid log-normal(mu_s, sigma_s^2), fitted by empirical Bayes from the
population of scale estimates. The base model is sigma_s -> 0.

Per node, with tau_i^2 the base-model variance of a coordinate of delta_i:
    s_hat_i = ||delta_i||^2 / (d tau_i^2)            ~ s_i chi^2_d / d
so log s_hat_i = log s_i + log(chi^2_d / d), whose null under M is
    mean mu_s + m_d,  variance sigma_s^2 + v_d,
with m_d = psi(d/2) - log(d/2), v_d = psi'(d/2) (log chi-squared moments).
EB fit: mu_s = median(log s_hat) - median(log chi^2_d/d);
        sigma_s^2 = max(MAD^2(log s_hat) - v_d, 0)  (robust moments).

Channels under M:
- magnitude: z_mag = (log s_hat - mu_s - m_d) / sqrt(sigma_s^2 + v_d).
- energy profile members E_i(t): log(E / null mean) = log s_i + log(chi^2_d/d)
  under M, so every energy member shares the one fitted null above. This
  is the single calibrated null the energy channel lacked (bulk z at -13
  under the base model on Weibo).
- shape (ratio) members: scale-free by construction; exact F nulls.

Scores exported: MAG, HSCAN-E (max over M-calibrated energy members),
HSCAN-R (max over exact-null ratio members), HSCAN (max over all),
H-SUM (z_mag + best shape z: the log-product combination, i.e. the
calibrated analogue of the best energy member), and H-CHAN (channel
chosen label-free by upper-tail excess mass, then that channel's scan).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import chi2, norm

from .nulls import member_tails
from .scores import BankMember
from .select import binned_density, efron_central_matching, upper_excess_mass, z_from_tails


def log_chi2_moments(d: int):
    m_d = float(digamma(d / 2.0) - np.log(d / 2.0))
    v_d = float(polygamma(1, d / 2.0))
    med = float(np.log(chi2.ppf(0.5, d) / d))
    return m_d, v_d, med


def fit_scale_prior(s_hat: np.ndarray, d: int):
    ls = np.log(np.maximum(s_hat, 1e-300))
    lo = np.percentile(ls, 0.5)
    ls = np.maximum(ls, lo)
    m_d, v_d, med = log_chi2_moments(d)
    mu_s = float(np.median(ls) - med)
    mad2 = float((1.4826 * np.median(np.abs(ls - np.median(ls)))) ** 2)
    sigma_s2 = max(mad2 - v_d, 1e-6)
    return mu_s, sigma_s2, m_d, v_d


def excess_mass(z: np.ndarray) -> float:
    zf = z[np.isfinite(z)]
    if zf.size < 10 or np.std(zf) < 1e-9:
        return 0.0
    grid, dens, _ = binned_density(zf)
    d0, s0, pi0 = efron_central_matching(zf, grid, dens)
    return upper_excess_mass(grid, dens, d0, s0, pi0)


def camouflage_scores(prep, q: np.ndarray, rough: str = "isotropic",
                      eval_mask: Optional[np.ndarray] = None) -> Dict:
    """Camouflage surprise: the lower tail of the node's template displacement.

    Under the fitted hierarchical GOU a normal node's endpoint fluctuates
    around its graph template with displacement D_i = ||x_i - m_i||^2 ~
    s_i d tau_i^2 chi^2_d / d, s_i from the fitted scale prior. A
    camouflaged node (transaction fraud: fraudsters conform to their
    neighborhood, Dou et al. 2020) has completed its relaxation: D_i lies
    in the LOWER tail of the fitted prior. The statistic is the same
    surprise the framework computes for deviant nodes, read from the
    opposite tail; the declared prior is that one sign. The null is
    fitted on the evaluated population (transductive convention).
    """
    from scipy.stats import norm
    from .scores import node_tau2
    d = prep.d
    tau2 = node_tau2(prep, q, rough=rough)
    s_hat = (prep.delta * prep.delta).sum(axis=1) / np.maximum(d * tau2, 1e-300)
    ref = s_hat[eval_mask] if eval_mask is not None else s_hat
    mu_s, sigma_s2, m_d, v_d = fit_scale_prior(ref, d)
    z = (np.log(np.maximum(s_hat, 1e-300)) - mu_s - m_d) / np.sqrt(sigma_s2 + v_d)
    lower = norm.logcdf(z)          # log P(scale <= observed) under the fitted prior
    upper = norm.logsf(z)
    return {"scores": {"CAM": -lower, "DEV": -upper, "TWO-SIDED": -np.log(2.0) - np.minimum(lower, upper)},
            "diag": {"mu_s": mu_s, "sigma_s": float(np.sqrt(sigma_s2)), "null_population": "eval" if eval_mask is not None else "all"}}


def hierarchical_scores(prep, q: np.ndarray, members: Sequence[BankMember], rough: str = "isotropic") -> Dict:
    from .scores import rough_variance
    d = prep.d
    V2 = prep.V * prep.V
    inv_q = 1.0 / np.maximum(q, 1e-12)
    rv = rough_variance(prep) if rough == "isotropic" else 0.0
    tau2 = V2 @ inv_q + rv * (1.0 - prep.lev)
    row_sq = (prep.delta * prep.delta).sum(axis=1)
    s_hat = row_sq / np.maximum(d * tau2, 1e-300)
    mu_s, sigma_s2, m_d, v_d = fit_scale_prior(s_hat, d)
    z_mag = (np.log(np.maximum(s_hat, 1e-300)) - mu_s - m_d) / np.sqrt(sigma_s2 + v_d)

    energy_z: Dict[str, np.ndarray] = {}
    ratio_z: Dict[str, np.ndarray] = {}
    for m in members:
        if m.kind in ("energy", "cond"):
            w = np.log(np.maximum(m.score, 1e-300) / np.maximum(m.null_mean(d), 1e-300))
            energy_z[m.name] = (w - mu_s - m_d) / np.sqrt(sigma_s2 + v_d)
        elif m.kind == "ratio":
            try:
                sf, cdf = member_tails(m, d)
            except NotImplementedError:
                continue
            ratio_z[m.name] = z_from_tails(sf, cdf)

    def clean(a):
        return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=40.0, neginf=-40.0)

    def standardize(z):
        zc = clean(z)
        med = float(np.median(zc))
        mad = float(1.4826 * np.median(np.abs(zc - med)))
        return (zc - med) / max(mad, 1e-6)

    # Shape channel: the exact F null describes the bulk poorly when rows are
    # non-Gaussian across features (Weibo: bulk median +2.3, spread 4.9), so
    # ratio members are standardized by their empirical bulk (median, MAD).
    ratio_raw_bulk = {k: (float(np.median(clean(v))), float(1.4826 * np.median(np.abs(clean(v) - np.median(clean(v))))))
                      for k, v in ratio_z.items()}

    def usable(v):
        # Only the numerical spread guard here: ties are harmless for a max of
        # standardized z's (tied nodes tie); the tie guard belongs to the
        # two-groups Beta fit (twogroups.py), not to the scan.
        vc = clean(v)
        return float(np.subtract(*np.percentile(vc, [75, 25]))) >= 1e-3

    ratio_z = {k: standardize(v) for k, v in ratio_z.items() if usable(v)}

    E = np.vstack([clean(v) for v in energy_z.values()]) if energy_z else None
    R = np.vstack([clean(v) for v in ratio_z.values()]) if ratio_z else None
    scores: Dict[str, np.ndarray] = {"MAG": clean(z_mag)}
    if E is not None:
        scores["HSCAN-E"] = E.max(axis=0)
    if R is not None:
        scores["HSCAN-R"] = R.max(axis=0)
        scores["H-SUM"] = clean(z_mag) + R.max(axis=0)
    allz = [clean(z_mag)] + ([E] if E is not None else []) + ([R] if R is not None else [])
    scores["HSCAN"] = np.vstack(allz).max(axis=0)

    # label-free channel choice: upper-tail excess mass of each channel's members
    pi_E = {k: excess_mass(v) for k, v in energy_z.items()}
    pi_R = {k: excess_mass(v) for k, v in ratio_z.items()}
    pi_mag = excess_mass(z_mag)
    best_E = max(pi_E.values()) if pi_E else 0.0
    best_R = max(pi_R.values()) if pi_R else 0.0
    chan = "energy" if max(best_E, pi_mag) >= best_R else "ratio"
    scores["H-CHAN"] = scores["HSCAN-E"] if (chan == "energy" and "HSCAN-E" in scores) else scores.get("HSCAN-R", scores["HSCAN"])

    def bulk(z):
        zf = z[np.isfinite(z)]
        return float(np.median(zf)), float(1.4826 * np.median(np.abs(zf - np.median(zf))))

    diag = {
        "mu_s": mu_s, "sigma_s": float(np.sqrt(sigma_s2)), "m_d": m_d, "v_d": v_d,
        "bulk_z_mag": bulk(z_mag),
        "bulk_z_energy_Jstar": bulk(energy_z["J*"]) if "J*" in energy_z else None,
        "bulk_z_ratio_R": ratio_raw_bulk.get("R"),
        "pi_mag": pi_mag, "pi_E_best": best_E, "pi_R_best": best_R,
        "pi_E": pi_E, "pi_R": pi_R, "channel": chan,
    }
    return {"scores": scores, "diag": diag}
