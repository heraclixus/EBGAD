"""v3 score: marginal surprise of the relaxation profile under the hierarchical GOU.

Bands. The retained modes are split by relaxation rate q_j into T contiguous
bands (equal mode counts, slowest first). Band energies
    E_ia = || sum_{j in a} sqrt(q_j) V_ij xi_j ||^2
involve disjoint modes, so under the null they are independent across
bands with E_ia ~ sigma_ia^2 chi^2_d exactly, sigma_ia^2 = sum_{j in a} V_ij^2
(the node's leverage on the band). On truncated spectra the unretained
remainder is one more band with the isotropic scale rv (1 - h_i).

Hierarchical null. With a node scale s_i, the standardized energies
    e_ia = E_ia / sigma_ia^2  ~  s_i chi^2_{d_eff}, iid over bands,
where d_eff <= d is an effective dimension fitted by EB from within-node
band contrasts log e_ia - log e_ib (which cancel s_i), and s_i has an
EB-fitted log-normal prior (mu_s, sigma_s^2).

Score. S_i = -log int prod_a Gamma(e_ia; d_eff/2, 2 s) p(s) ds. Given s the
likelihood depends on the e_ia through the total e_i = sum_a e_ia and the
term (d_eff/2 - 1) sum_a log e_ia, so
    S_i = S_scale(e_i) + S_shape(e_i.)
with S_scale the marginal surprise of the total energy (two-sided in scale
through the prior) and S_shape = -(d_eff/2 - 1) [sum_a log e_ia - T log(e_i/T)]
>= 0 the surprise of an uneven distribution of energy over bands (zero
when all bands carry equal standardized energy). Everything is exact
under the model, calibrated by the fitted prior, and has no channel, scan
or selection step. The null (mu_s, sigma_s, d_eff) is fitted on the
evaluated population when a mask is given (transductive convention).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, logsumexp, polygamma


LEVERAGE_EPS = 1e-10   # a node has a band only where its leverage on the band is positive


def relaxation_bands(q: np.ndarray, n_bands: int) -> List[np.ndarray]:
    order = np.argsort(q)
    return [b for b in np.array_split(order, n_bands) if b.size > 0]


def band_energies(prep, q: np.ndarray, n_bands: int = 4, rough: str = "isotropic",
                  block: int = 200_000):
    """(E, sig2, names): (n, T') energies and their null scales; remainder band appended on truncated spectra."""
    from .scores import rough_variance
    V, Xi = prep.V, prep.Xi
    n, d = prep.n, prep.d
    # Modes carrying no residual energy (the null space of L under a unit-gain
    # template: one mode per connected component, 7,308 of 23,831 on YelpChi)
    # are not modes of the residual field and are excluded from the bands;
    # a band made of them has zero energy for every node.
    mode_energy = (Xi * Xi).sum(axis=1)
    active = mode_energy > 1e-12 * max(float(mode_energy.max()), 1e-300)
    q_act = np.where(active, q, np.inf)
    bands = [b[active[b]] for b in relaxation_bands(q_act, n_bands)]
    bands = [b for b in bands if b.size > 0]
    T = len(bands)
    E = np.zeros((n, T))
    sig2 = np.zeros((n, T))
    q_safe = np.maximum(q, 1e-12)
    for a, idx in enumerate(bands):
        W = np.sqrt(q_safe[idx])[:, None] * Xi[idx]          # (|a|, d)
        Vb = V[:, idx]
        sig2[:, a] = (Vb * Vb).sum(axis=1)
        for s in range(0, n, block):
            u = Vb[s:s + block] @ W
            E[s:s + block, a] = (u * u).sum(axis=1)
    names = ["band%d" % (a + 1) for a in range(T)]
    if prep.truncated and rough == "isotropic":
        rv = rough_variance(prep)
        if rv > 0:
            E = np.column_stack([E, prep.delta_perp_sq])
            sig2 = np.column_stack([sig2, rv * (1.0 - prep.lev)])
            names.append("remainder")
    return E, np.maximum(sig2, 1e-300), names


def band_variances(E: np.ndarray, sig2: np.ndarray, d: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """EB spectral prior per band: the marginal-likelihood variance multiplier.

    Under the Gaussian null E_ia ~ v_a sigma_ia^2 chi^2_d, the maximum
    marginal likelihood of v_a is the mean of E_ia / (d sigma_ia^2) over
    nodes. The three-parameter Matern family cannot express the residual's
    spectrum after a low-pass template (Elliptic: the slowest band carries
    400x less standardized energy than the others under the flat fit), so
    the prior's spectral density is refined per relaxation band, one EB
    parameter per band, closed form. A robust (median-based) estimator is
    used so that a contaminating group does not set the band scale.
    """
    e = E / sig2
    active = sig2 > LEVERAGE_EPS
    if mask is not None:
        e, active = e[mask], active[mask]
    # median of chi^2_d / d as the location; v_a = median(e_a) / median(chi^2_d/d),
    # over the nodes that have leverage on band a (on a graph of small components
    # a node only touches its component's modes; YelpChi: 7,308 components).
    from scipy.stats import chi2
    med_null = chi2.ppf(0.5, d) / d
    v = np.array([np.median(e[active[:, a], a]) if active[:, a].sum() >= 10 else 1.0 for a in range(e.shape[1])])
    return np.maximum(v / med_null, 1e-300)


def _solve_trigamma(v: float) -> float:
    """x with psi'(x) = v, x > 0."""
    v = max(float(v), 1e-8)
    return brentq(lambda x: polygamma(1, x) - v, 1e-4, 1e6)


def fit_hierarchical_null(e: np.ndarray, d: int, mask: Optional[np.ndarray] = None) -> Dict:
    """EB fit of (d_eff, mu_s, sigma_s2) from standardized band energies e (n, T)."""
    le = np.log(np.maximum(e, 1e-300))          # NaN where the node has no leverage on the band
    if mask is not None:
        le = le[mask]
    T = le.shape[1]
    T_i = np.isfinite(le).sum(axis=1)
    # within-node contrasts cancel the node scale: Var(log e_a - log e_b) = 2 psi'(d_eff/2)
    contrasts = np.empty(0)
    if T >= 2:
        contrasts = (le[:, :, None] - le[:, None, :])[:, np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]].ravel()
        contrasts = contrasts[np.isfinite(contrasts)]
    if contrasts.size >= 100:
        v_pair = (1.4826 * np.median(np.abs(contrasts - np.median(contrasts)))) ** 2
        d_eff = float(np.clip(2.0 * _solve_trigamma(max(v_pair / 2.0, 1e-6)), 2.0, float(d)))
    else:
        d_eff = float(d)          # too few nodes with two or more bands: no within-node evidence
    m_d = float(digamma(d_eff / 2.0) - np.log(d_eff / 2.0))
    v_d = float(polygamma(1, d_eff / 2.0))
    ok = T_i >= 1
    if ok.sum() < 10:
        return {"d_eff": d_eff, "mu_s": 0.0, "sigma_s": 1.0, "m_d": m_d, "v_d": v_d, "degenerate": True}
    node_mean = np.nanmean(le[ok], axis=1)            # log s + mean_a log(chi2/d)-ish
    mu_s = float(np.median(node_mean) - m_d)
    mad2 = float((1.4826 * np.median(np.abs(node_mean - np.median(node_mean)))) ** 2)
    sigma_s2 = max(mad2 - v_d / max(float(np.median(T_i[ok])), 1.0), 1e-4)
    return {"d_eff": d_eff, "mu_s": mu_s, "sigma_s": float(np.sqrt(sigma_s2)), "m_d": m_d, "v_d": v_d}


def marginal_surprise(e: np.ndarray, d_eff: float, mu_s: float, sigma_s: float, n_grid: int = 241) -> Dict[str, np.ndarray]:
    """S = -log int prod_a Gamma(e_a; d_eff/2, 2 s) LogNormal(s) ds, plus its scale and shape parts."""
    n, T = e.shape
    act = np.isfinite(e)
    T_i = np.maximum(act.sum(axis=1), 1)              # bands the node has leverage on
    e = np.where(act, np.maximum(e, 1e-300), np.nan)
    k = d_eff / 2.0
    ls = np.linspace(mu_s - 6 * sigma_s, mu_s + 6 * sigma_s, n_grid)
    dls = ls[1] - ls[0]
    log_prior = -0.5 * ((ls - mu_s) / sigma_s) ** 2 - np.log(sigma_s * np.sqrt(2 * np.pi)) + np.log(dls)
    e_tot = np.nansum(e, axis=1)
    sum_log_e = np.nansum(np.log(e), axis=1)
    Tk = T_i * k
    # total energy given s: Gamma(T_i k, 2 s); log-density in e_tot
    log_scale_lik = ((Tk - 1.0) * np.log(e_tot))[:, None] - e_tot[:, None] / (2.0 * np.exp(ls))[None, :] \
        - Tk[:, None] * (np.log(2.0) + ls)[None, :] - gammaln(Tk)[:, None]
    S_scale = -logsumexp(log_scale_lik + log_prior[None, :], axis=1)
    # shape: product of T_i Gammas vs one Gamma of the total (Dirichlet-type term), independent of s
    # log prod_a Gamma(e_a; k, 2s) - log Gamma(e_tot; T_i k, 2s)
    #   = (k-1) sum_a log e_a - (T_i k - 1) log e_tot + gammaln(T_i k) - T_i gammaln(k)
    log_shape = (k - 1.0) * sum_log_e - (Tk - 1.0) * np.log(e_tot) + gammaln(Tk) - T_i * gammaln(k)
    S_shape = -log_shape
    return {"S": S_scale + S_shape, "S_scale": S_scale, "S_shape": S_shape,
            "log_e_tot": np.log(e_tot), "z_scale": (np.log(e_tot / (T_i * d_eff)) - mu_s) / max(sigma_s, 1e-6)}


def empirical_two_sided_surprise(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """-log of the two-sided empirical tail probability of x on the evaluated population.

    The Gamma-density version of the scale surprise is one-sided in
    practice: a chi-squared with T d_eff degrees of freedom vanishes so fast
    below its mean that low-activity nodes dominate (Weibo: 18.7 AUROC).
    The empirical null on the evaluated population is calibrated by
    construction and symmetric in probability mass: p = 2 min(F(x), 1 - F(x)).
    """
    ref = np.sort(x[mask] if mask is not None else x)
    n = ref.size
    F = (np.searchsorted(ref, x, side="right") + 0.5) / (n + 1.0)
    p = 2.0 * np.minimum(F, 1.0 - F)
    return -np.log(np.clip(p, 1.0 / (n + 1.0), 1.0))


def surprise_scores(prep, q: np.ndarray, rough: str = "isotropic", n_bands: int = 4,
                    eval_mask: Optional[np.ndarray] = None) -> Dict:
    E, sig2, names = band_energies(prep, q, n_bands=n_bands, rough=rough)
    return surprise_from_bands(E, sig2, names, prep.d, eval_mask)


def surprise_from_bands(E: np.ndarray, sig2: np.ndarray, names: List[str], d: int,
                        eval_mask: Optional[np.ndarray] = None) -> Dict:
    """Surprise scores from band energies E (n, T) and their null scales sig2 (n, T)
    (spectral bands from prep, or matrix-free Chebyshev bands from ebsmooth)."""
    v_band = band_variances(E, sig2, d, mask=eval_mask)
    e = E / (sig2 * v_band[None, :])
    e = np.where(sig2 > LEVERAGE_EPS, e, np.nan)      # a node has a band only where it has leverage
    fit = fit_hierarchical_null(e, d, mask=eval_mask)
    fit["band_variance"] = [float(v) for v in v_band]
    out = marginal_surprise(e, fit["d_eff"], fit["mu_s"], fit["sigma_s"])
    s_scale_emp = empirical_two_sided_surprise(out["log_e_tot"], eval_mask)
    # Empirically calibrated shape surprise: log band shares standardized on the
    # evaluated population, combined quadratically. The Gamma model's shape term
    # is powerless when d_eff is ~3 (fraud graphs), although the ratio profile
    # shows strong shape signal there; this keeps the same object (how energy is
    # distributed over relaxation bands) but calibrates it empirically.
    shares = np.log(np.maximum(e, 1e-300) / np.maximum(np.nansum(e, axis=1, keepdims=True), 1e-300))
    ref = shares[eval_mask] if eval_mask is not None else shares
    med = np.nanmedian(ref, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(ref - med), axis=0) + 1e-9
    zsh = np.clip((shares - med) / mad, -8, 8)
    s_shape_emp = np.nansum(zsh ** 2, axis=1)
    s_shape_emp_max = np.nanmax(np.where(np.isfinite(zsh), zsh, -np.inf), axis=1)
    scores = {"SURPRISE": out["S"], "S-scale": out["S_scale"], "S-shape": out["S_shape"],
              "S-scale-up": out["z_scale"], "S-scale-down": -out["z_scale"],
              "SURPRISE-emp": s_scale_emp + out["S_shape"], "S-scale-emp": s_scale_emp,
              "S-shape-emp": s_shape_emp, "S-shape-emp-max": s_shape_emp_max,
              "SURPRISE-emp2": s_scale_emp + s_shape_emp}
    diag = {**fit, "bands": names, "n_bands": len(names),
            "band_energy_share_median": [float(np.nanmedian(e[:, a] / np.maximum(np.nansum(e, axis=1), 1e-300))) for a in range(e.shape[1])],
            "bands_per_node_median": float(np.median(np.isfinite(e).sum(axis=1)))}
    return {"scores": scores, "diag": diag}
