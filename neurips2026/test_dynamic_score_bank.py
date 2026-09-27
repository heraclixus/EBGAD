"""Prototype dynamic GOU score banks.

This script tests a stronger multiscale use of the GOU dynamics than the
product-of-experts prototype.  Instead of collapsing horizons into a single
effective precision, it keeps scale-specific anomaly scores and aggregates
them in label-free rank space.

Implemented banks:

1. Horizon bank:
       J_i(Gamma_m) = ||Sigma_Gamma_m^{-1/2} delta||_i^2

2. GOU band-pass bank:
       H_m(Q) = exp(-Gamma_{m-1} Q) - exp(-Gamma_m Q)
       B_i(m) = ||Q^{1/2} H_m(Q) delta||_i^2

3. Optional scale-refit bank:
       Fit EB geometry separately for each Gamma_m, then aggregate.

4. Optional graph bank:
       Repeat the banks on graph_type=original and graph_type=affinity.

All aggregation and model selection rules in this file are label-free.  AUROC
is reported only as diagnostics after scores are produced.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, f as f_dist
from sklearn.metrics import roc_auc_score

from data_utils import load_data
from soc.prior_optimizer import Q_rho_eigenvalues, compute_spectral_energy, gamma_jacobian_correction
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from test_multiscale_gou_poe import (
    BASELINES,
    DEFAULT_HORIZONS,
    KAPPAS,
    REPORTED,
    compute_graph_stats,
    density_adjusted_template_gammas,
    format_horizon,
    gou_precisions,
    ks_null_deviation,
    optimize_rho_kappa_poe,
    parse_grid,
    percentile_rank,
    score_nodes,
)


def parse_names(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def canonical_horizons(horizons: Sequence[float], include_inf: bool = True) -> List[float]:
    finite = sorted({float(h) for h in horizons if np.isfinite(h) and float(h) > 0.0})
    out = finite
    if include_inf and not any(np.isinf(h) for h in out):
        out = out + [np.inf]
    elif any(np.isinf(h) for h in horizons):
        out = out + [np.inf]
    return out


def band_edges_from_horizons(horizons: Sequence[float]) -> List[float]:
    finite = sorted({float(h) for h in horizons if np.isfinite(h) and float(h) > 0.0})
    return [0.0] + finite + [np.inf]


def edge_label(left: float, right: float) -> str:
    return "%s:%s" % (format_horizon(left), format_horizon(right))


def build_trainer_config(
    data,
    gamma: float,
    pca_dim: int | None,
    truncated_k: int | None,
    modeled_subspace: str,
    graph_type: str,
    template_type: str,
) -> SOCTrainerConfig:
    cfg = dict(
        kappa=1.0,
        nu=1.0,
        rho=0.5,
        lam_penalty=50.0,
        alpha=2.0,
        T=1.0,
        normalize_mode="zscore",
        prior_mean_mode="stationary",
        laplacian_variant="sym",
        d_hidden=64,
        n_hidden_layers=2,
        epochs=0,
        normalize_features=True,
        template_type=template_type,
        graph_type=graph_type,
        gamma=gamma,
        verbose=False,
        truncated_k=truncated_k,
        modeled_subspace=modeled_subspace,
    )
    if pca_dim is not None and data.x.shape[1] > pca_dim:
        cfg.update(dict(
            use_encoder=True,
            encoder_type="pca",
            encoder_hid_dim=pca_dim,
            encoder_num_layers=1,
            encoder_dropout=0.0,
            encoder_lr=0.0,
            encoder_epochs=0,
            encoder_alpha=1.0,
            encoder_weight_decay=0.0,
        ))
    return SOCTrainerConfig(**cfg)


def prepare_model_entries(
    data,
    graph_types: Iterable[str],
    template_gammas: Iterable[float],
    pca_dim: int | None,
    truncated_k: int | None,
    modeled_subspace: str,
    template_type: str,
    device: str,
) -> List[Dict]:
    entries: List[Dict] = []
    for graph_type in graph_types:
        for gamma in template_gammas:
            print("    preparing graph=%s template gamma=%g" % (graph_type, gamma), flush=True)
            trainer = SOCTrainer(build_trainer_config(
                data,
                gamma=gamma,
                pca_dim=pca_dim,
                truncated_k=truncated_k,
                modeled_subspace=modeled_subspace,
                graph_type=graph_type,
                template_type=template_type,
            ))
            trainer.train(data, device=device)
            V = trainer.V.detach().cpu().numpy()
            lam_L = trainer.lam_L_model.detach().cpu().numpy()
            x = trainer.x_T.detach().cpu().numpy()
            template = trainer.template.detach().cpu().numpy()
            delta = x - template
            S = compute_spectral_energy(delta, V)
            n, d = x.shape
            entries.append({
                "key": "%s/gamma=%g" % (graph_type, gamma),
                "graph_type": graph_type,
                "template_gamma": float(gamma),
                "V": V,
                "lam_L": lam_L,
                "delta": delta,
                "S": S,
                "n": int(n),
                "d": int(d),
            })
    return entries


def fit_best_entry(
    entries: Sequence[Dict],
    horizons: Sequence[float],
    kappas: Sequence[float],
    use_gamma_jacobian: bool,
) -> Dict:
    best: Dict = {"ml": -np.inf}
    for entry in entries:
        fit = optimize_rho_kappa_poe(
            entry["S"], entry["lam_L"], horizons, entry["d"], kappas,
        )
        ml = fit["ml"]
        if use_gamma_jacobian:
            ml += gamma_jacobian_correction(entry["lam_L"], entry["template_gamma"], entry["d"])
        if ml > best["ml"]:
            best = {
                **fit,
                "ml": float(ml),
                "entry_key": entry["key"],
                "graph_type": entry["graph_type"],
                "template_gamma": entry["template_gamma"],
                "horizons": list(horizons),
            }
    return best


def entry_for_fit(entries: Sequence[Dict], fit: Dict) -> Dict:
    for entry in entries:
        if entry["key"] == fit["entry_key"]:
            return entry
    raise KeyError("No entry for fit key %r" % fit["entry_key"])


def q_for_fit(entry: Dict, fit: Dict) -> np.ndarray:
    return Q_rho_eigenvalues(
        entry["lam_L"],
        rho=float(fit["rho"]),
        kappa=float(fit["kappa"]),
        nu=1.0,
    )


def horizon_score_bank(entry: Dict, q: np.ndarray, horizons: Sequence[float]) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    tau = gou_precisions(q, horizons)
    for horizon, precision in zip(horizons, tau):
        je, jr = score_nodes(entry["delta"], entry["V"], precision)
        label = format_horizon(horizon)
        scores["J@%s" % label] = je.astype(np.float32, copy=False)
        scores["R@%s" % label] = jr.astype(np.float32, copy=False)
    return scores


def leverage_terms(entry: Dict, q: np.ndarray, precision: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v2 = entry["V"] * entry["V"]
    q_safe = np.maximum(q, 1e-12)
    p_safe = np.maximum(precision, 0.0)
    num_scale = v2 @ (p_safe / q_safe)
    den_scale = v2 @ (1.0 / q_safe)
    cross_scale = v2 @ (np.sqrt(p_safe) / q_safe)
    return (
        np.maximum(num_scale, 1e-12),
        np.maximum(den_scale, 1e-12),
        cross_scale,
    )


def energy_pvalues(scores: np.ndarray, num_scale: np.ndarray, d: int) -> np.ndarray:
    x = np.maximum(scores, 0.0) / np.maximum(num_scale, 1e-12)
    p = chi2.sf(x, df=max(int(d), 1))
    return np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 1e-300, 1.0)


def ratio_pvalues(
    scores: np.ndarray,
    num_scale: np.ndarray,
    den_scale: np.ndarray,
    cross_scale: np.ndarray,
    d: int,
) -> np.ndarray:
    ratio_scale = num_scale / np.maximum(den_scale, 1e-12)
    corr2 = (cross_scale * cross_scale) / np.maximum(num_scale * den_scale, 1e-12)
    corr2 = np.clip(corr2, 0.0, 0.999)
    df_eff = np.clip(float(max(int(d), 1)) / np.maximum(1.0 - corr2, 1e-3), float(max(int(d), 1)), 1e4)
    x = np.maximum(scores, 0.0) / np.maximum(ratio_scale, 1e-12)
    p = f_dist.sf(x, df_eff, df_eff)
    return np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 1e-300, 1.0)


def sanitize_pvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(p, 1e-300, 1.0)


def median_calibrated_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """One-parameter empirical-null calibration under majority-normality.

    The graph-Gaussian model can be globally over- or under-dispersed for a
    given filter.  A power calibration preserves the node ordering while
    matching the median p-value to the Uniform(0, 1) median, so sparse-tail
    tests respond to localized excess rather than global scale drift.
    """
    p = sanitize_pvalues(pvalues)
    med = float(np.median(p))
    if not np.isfinite(med) or med <= 1e-12 or med >= 1.0 - 1e-12:
        return p
    alpha = np.log(0.5) / np.log(med)
    alpha = float(np.clip(alpha, 0.05, 20.0))
    return sanitize_pvalues(p ** alpha)


def berk_jones_stat(pvalues: np.ndarray, max_tail_frac: float = 0.10) -> float:
    """Sparse-mixture tail evidence for calibrated null p-values.

    If the fitted graph-Gaussian null is adequate for a score, its p-values are
    approximately Uniform(0, 1).  Berk-Jones is the one-sided likelihood-ratio
    statistic for an excess of small p-values, restricted to the sparse tail.
    """
    p = np.sort(sanitize_pvalues(pvalues))
    n = len(p)
    if n < 20:
        return 0.0
    m = max(1, min(n, int(np.floor(max_tail_frac * n))))
    p = p[:m]
    frac = np.arange(1, m + 1, dtype=np.float64) / float(n)
    mask = frac > p
    if not np.any(mask):
        return 0.0
    frac = frac[mask]
    p = p[mask]
    one_minus_p = np.maximum(1.0 - p, 1e-300)
    one_minus_frac = np.maximum(1.0 - frac, 1e-300)
    bj = n * (
        frac * np.log(frac / p)
        + one_minus_frac * np.log(one_minus_frac / one_minus_p)
    )
    return float(np.nanmax(bj))


def higher_criticism_stat(pvalues: np.ndarray, max_tail_frac: float = 0.10) -> float:
    p = np.sort(sanitize_pvalues(pvalues))
    n = len(p)
    if n < 20:
        return 0.0
    m = max(1, min(n, int(np.floor(max_tail_frac * n))))
    p = p[:m]
    frac = np.arange(1, m + 1, dtype=np.float64) / float(n)
    denom = np.sqrt(np.maximum(p * (1.0 - p), 1e-300))
    hc = np.sqrt(float(n)) * (frac - p) / denom
    hc = hc[frac > p]
    if len(hc) == 0:
        return 0.0
    return float(np.nanmax(hc))


def pvalue_aggregate_scores(
    pvalue_bank: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    primitive = {
        name: sanitize_pvalues(pvalues)
        for name, pvalues in pvalue_bank.items()
    }
    primitive = {
        name: pvalues for name, pvalues in primitive.items()
        if pvalues.ndim == 1 and len(pvalues) > 0
    }
    if not primitive:
        return {}, {}

    names = sorted(primitive)
    raw = np.vstack([primitive[name] for name in names])
    cal = np.vstack([median_calibrated_pvalues(primitive[name]) for name in names])

    def bonferroni_minp(pmat: np.ndarray) -> np.ndarray:
        return sanitize_pvalues(np.minimum(1.0, pmat.shape[0] * np.min(pmat, axis=0)))

    def acat(pmat: np.ndarray) -> np.ndarray:
        clipped = np.clip(pmat, 1e-15, 1.0 - 1e-15)
        stat = np.mean(np.tan((0.5 - clipped) * np.pi), axis=0)
        p = 0.5 - np.arctan(stat) / np.pi
        return sanitize_pvalues(p)

    pvalues = {
        "model_minp": bonferroni_minp(raw),
        "model_acat": acat(raw),
        "cal_minp": bonferroni_minp(cal),
        "cal_acat": acat(cal),
    }
    scores = {
        name: (-np.log(p)).astype(np.float32, copy=False)
        for name, p in pvalues.items()
    }
    return scores, pvalues


def horizon_pvalue_bank(
    entry: Dict,
    q: np.ndarray,
    horizons: Sequence[float],
    scores: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    pvalues: Dict[str, np.ndarray] = {}
    tau = gou_precisions(q, horizons)
    for horizon, precision in zip(horizons, tau):
        label = format_horizon(horizon)
        num_scale, den_scale, cross_scale = leverage_terms(entry, q, precision)
        j_name = "J@%s" % label
        r_name = "R@%s" % label
        if j_name in scores:
            pvalues[j_name] = energy_pvalues(scores[j_name], num_scale, entry["d"])
        if r_name in scores:
            pvalues[r_name] = ratio_pvalues(scores[r_name], num_scale, den_scale, cross_scale, entry["d"])
    return pvalues


def bandpass_precisions(q: np.ndarray, edges: Sequence[float]) -> List[Tuple[str, np.ndarray]]:
    q_safe = np.maximum(q, 1e-12)
    out: List[Tuple[str, np.ndarray]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        prev = np.ones_like(q_safe) if left == 0.0 else np.exp(-float(left) * q_safe)
        nxt = np.zeros_like(q_safe) if np.isinf(right) else np.exp(-float(right) * q_safe)
        h = np.maximum(prev - nxt, 0.0)
        precision = q_safe * h * h
        out.append((edge_label(left, right), precision))
    return out


def bandpass_score_bank(entry: Dict, q: np.ndarray, horizons: Sequence[float]) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    total = np.maximum(np.sum(entry["delta"] ** 2, axis=1), 1e-12)
    for label, precision in bandpass_precisions(q, band_edges_from_horizons(horizons)):
        je, _ = score_nodes(entry["delta"], entry["V"], precision)
        scores["B@%s" % label] = je.astype(np.float32, copy=False)
        scores["BR@%s" % label] = (je / total).astype(np.float32, copy=False)
    return scores


def bandpass_pvalue_bank(
    entry: Dict,
    q: np.ndarray,
    horizons: Sequence[float],
    scores: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    pvalues: Dict[str, np.ndarray] = {}
    for label, precision in bandpass_precisions(q, band_edges_from_horizons(horizons)):
        num_scale, den_scale, cross_scale = leverage_terms(entry, q, precision)
        b_name = "B@%s" % label
        br_name = "BR@%s" % label
        if b_name in scores:
            pvalues[b_name] = energy_pvalues(scores[b_name], num_scale, entry["d"])
        if br_name in scores:
            pvalues[br_name] = ratio_pvalues(scores[br_name], num_scale, den_scale, cross_scale, entry["d"])
    return pvalues


def scale_refit_horizon_bank(
    entries: Sequence[Dict],
    horizons: Sequence[float],
    kappas: Sequence[float],
    use_gamma_jacobian: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict]]:
    scores: Dict[str, np.ndarray] = {}
    pvalues: Dict[str, np.ndarray] = {}
    fits: List[Dict] = []
    for horizon in horizons:
        print("  scale-refit horizon=%s" % format_horizon(horizon), flush=True)
        fit = fit_best_entry(entries, [horizon], kappas, use_gamma_jacobian)
        entry = entry_for_fit(entries, fit)
        precision = fit["precision"]
        je, jr = score_nodes(entry["delta"], entry["V"], precision)
        label = format_horizon(horizon)
        j_name = "refit:J@%s" % label
        r_name = "refit:R@%s" % label
        scores[j_name] = je.astype(np.float32, copy=False)
        scores[r_name] = jr.astype(np.float32, copy=False)
        q = q_for_fit(entry, fit)
        num_scale, den_scale, cross_scale = leverage_terms(entry, q, precision)
        pvalues[j_name] = energy_pvalues(je, num_scale, entry["d"])
        pvalues[r_name] = ratio_pvalues(jr, num_scale, den_scale, cross_scale, entry["d"])
        fits.append({
            "horizon": label,
            "ml": float(fit["ml"]),
            "graph_type": fit["graph_type"],
            "template_gamma": float(fit["template_gamma"]),
            "rho": float(fit["rho"]),
            "kappa": float(fit["kappa"]),
        })
    return scores, pvalues, fits


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def rank_corr_from_ranks(a: np.ndarray, b: np.ndarray) -> float:
    za = zscore(a)
    zb = zscore(b)
    denom = np.sqrt(np.dot(za, za) * np.dot(zb, zb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(za, zb) / denom)


def top_indices(scores: np.ndarray, frac: float) -> np.ndarray:
    n = len(scores)
    k = max(10, int(np.ceil(frac * n)))
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=np.int64)
    return np.argpartition(scores, n - k)[n - k:]


def tail_lift(scores: np.ndarray, reference: np.ndarray, frac: float) -> float:
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(np.asarray(reference, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(s) != len(r) or len(s) < 20:
        return 0.0
    top = top_indices(s, frac)
    if len(top) == 0 or len(top) == len(s):
        return 0.0
    mask = np.ones(len(s), dtype=bool)
    mask[top] = False
    sd = np.std(r)
    if sd < 1e-12:
        return 0.0
    return float((np.mean(r[top]) - np.mean(r[mask])) / sd)


def node_unsupervised_references(data, residual: np.ndarray) -> Dict[str, np.ndarray]:
    """Node-level label-free references used to select among score-bank members."""
    x = data.x.detach().cpu().numpy().astype(np.float64)
    ei = data.edge_index.detach().cpu().numpy()
    n = x.shape[0]
    src, dst = ei[0], ei[1]

    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    edge_cos = np.sum(x_norm[src] * x_norm[dst], axis=1)
    aff_sum = np.zeros(n, dtype=np.float64)
    deg = np.zeros(n, dtype=np.float64)
    np.add.at(aff_sum, src, edge_cos)
    np.add.at(deg, src, 1.0)
    local_affinity = aff_sum / np.maximum(deg, 1.0)
    local_discord = -local_affinity

    xz = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    feature_magnitude = np.sum(xz * xz, axis=1)
    degree_anomaly = np.abs(zscore(np.log1p(deg)))
    residual_magnitude = np.sum(residual * residual, axis=1)

    return {
        "local_discord": local_discord,
        "feature_magnitude": feature_magnitude,
        "degree_anomaly": degree_anomaly,
        "residual_magnitude": residual_magnitude,
    }


def aggregate_rank_scores(score_bank: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    valid = {
        name: np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        for name, score in score_bank.items()
        if np.nanmax(score) > np.nanmin(score)
    }
    if not valid:
        return {}
    rank_stack = np.vstack([
        percentile_rank(score).astype(np.float32, copy=False)
        for score in valid.values()
    ])
    k_top = min(2, rank_stack.shape[0])
    topk = np.partition(rank_stack, rank_stack.shape[0] - k_top, axis=0)[-k_top:]
    tau = 0.10
    return {
        "rank_mean": rank_stack.mean(axis=0),
        "rank_max": rank_stack.max(axis=0),
        "rank_top2": topk.mean(axis=0),
        "rank_lse": (tau * logsumexp(rank_stack / tau, axis=0)).astype(np.float32, copy=False),
    }


def selector_diagnostics(
    candidates: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray] | None,
) -> Dict[str, Dict[str, float]]:
    names = list(candidates.keys())
    if not names:
        return {}
    scores = {
        name: np.nan_to_num(np.asarray(candidates[name], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        for name in names
    }
    ranks = {name: percentile_rank(score) for name, score in scores.items()}
    rank_stack = np.vstack([ranks[name] for name in names])
    consensus = rank_stack.mean(axis=0)

    diagnostics: Dict[str, Dict[str, float]] = {}
    top_frac = 0.02
    top_sets = {name: set(top_indices(scores[name], top_frac).tolist()) for name in names}

    ref_ranks = {}
    if references:
        ref_ranks = {
            key: percentile_rank(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
            for key, value in references.items()
        }

    for idx, name in enumerate(names):
        if len(names) > 1:
            consensus_minus = (rank_stack.sum(axis=0) - ranks[name]) / float(len(names) - 1)
        else:
            consensus_minus = consensus
        stability_corr = rank_corr_from_ranks(ranks[name], consensus_minus)
        consensus_top = set(top_indices(consensus_minus, top_frac).tolist())
        union = len(top_sets[name] | consensus_top)
        top_jaccard = float(len(top_sets[name] & consensus_top) / union) if union else 0.0

        if references:
            local_tail = np.mean([
                tail_lift(scores[name], references["local_discord"], frac)
                for frac in (0.01, 0.02, 0.05)
            ])
            residual_tail = np.mean([
                tail_lift(scores[name], references["residual_magnitude"], frac)
                for frac in (0.01, 0.02, 0.05)
            ])
            feature_tail = np.mean([
                tail_lift(scores[name], references["feature_magnitude"], frac)
                for frac in (0.01, 0.02, 0.05)
            ])
            degree_tail = np.mean([
                tail_lift(scores[name], references["degree_anomaly"], frac)
                for frac in (0.01, 0.02, 0.05)
            ])
            graph_tail = local_tail + 0.25 * degree_tail
            residual_graph_tail = 0.50 * local_tail + 0.35 * residual_tail + 0.15 * degree_tail
            ref_corr = np.mean([
                rank_corr_from_ranks(ranks[name], ref_rank)
                for ref_rank in ref_ranks.values()
            ])
        else:
            local_tail = residual_tail = feature_tail = degree_tail = 0.0
            graph_tail = residual_graph_tail = ref_corr = 0.0

        diagnostics[name] = {
            "stability_corr": float(stability_corr),
            "top_jaccard": float(top_jaccard),
            "local_tail": float(local_tail),
            "residual_tail": float(residual_tail),
            "feature_tail": float(feature_tail),
            "degree_tail": float(degree_tail),
            "graph_tail": float(graph_tail),
            "residual_graph_tail": float(residual_graph_tail),
            "ref_corr": float(ref_corr),
        }

    for field in [
        "stability_corr", "top_jaccard", "graph_tail",
        "residual_graph_tail", "ref_corr",
    ]:
        vals = np.array([diagnostics[name][field] for name in names], dtype=np.float64)
        vals_z = zscore(vals)
        for name, value in zip(names, vals_z):
            diagnostics[name][field + "_z"] = float(value)

    for name in names:
        d = diagnostics[name]
        d["tail_stable"] = (
            0.65 * d["residual_graph_tail_z"]
            + 0.25 * d["stability_corr_z"]
            + 0.10 * d["top_jaccard_z"]
        )
        d["graph_tail_stable"] = (
            0.70 * d["graph_tail_z"]
            + 0.20 * d["stability_corr_z"]
            + 0.10 * d["top_jaccard_z"]
        )
        d["corr_stable"] = (
            0.55 * d["ref_corr_z"]
            + 0.35 * d["stability_corr_z"]
            + 0.10 * d["top_jaccard_z"]
        )

    return diagnostics


def evaluate_score_bank(
    y: np.ndarray,
    mask: np.ndarray,
    score_bank: Dict[str, np.ndarray],
    add_aggregates: bool = True,
    references: Dict[str, np.ndarray] | None = None,
    pvalue_bank: Dict[str, np.ndarray] | None = None,
) -> Dict:
    candidates = dict(score_bank)
    candidate_pvalues = dict(pvalue_bank) if pvalue_bank else {}
    pvalue_score_names: set[str] = set()
    if add_aggregates and pvalue_bank:
        pvalue_scores, aggregate_pvalues = pvalue_aggregate_scores(pvalue_bank)
        candidates.update(pvalue_scores)
        candidate_pvalues.update(aggregate_pvalues)
        pvalue_score_names = set(pvalue_scores)
    rank_score_names: set[str] = set()
    if add_aggregates:
        rank_scores = aggregate_rank_scores(score_bank)
        candidates.update(rank_scores)
        rank_score_names = set(rank_scores)

    diagnostics = selector_diagnostics(candidates, references)
    model_tail: Dict[str, Dict[str, float]] = {}
    if candidate_pvalues:
        for name, pvalues in candidate_pvalues.items():
            if name not in candidates:
                continue
            pv = sanitize_pvalues(np.asarray(pvalues)[mask])
            cal_pv = median_calibrated_pvalues(pv)
            model_tail[name] = {
                "model_bj": berk_jones_stat(pv),
                "model_hc": higher_criticism_stat(pv),
                "cal_bj": berk_jones_stat(cal_pv),
                "cal_hc": higher_criticism_stat(cal_pv),
            }
    per_score = []
    y_eval = y[mask]
    for name, score in candidates.items():
        s = np.nan_to_num(np.asarray(score), nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmax(s) <= np.nanmin(s):
            continue
        auc = safe_auc(y_eval, s[mask])
        ks = ks_null_deviation(s[mask])
        per_score.append({
            "name": name,
            "auc": float(auc),
            "ks": float(ks),
            **diagnostics.get(name, {}),
            **model_tail.get(name, {}),
        })

    if not per_score:
        return {
            "n_scores": 0,
            "auc_ks_selected": float("nan"),
            "score_ks": None,
            "auc_model_bj_selected": float("nan"),
            "score_model_bj": None,
            "auc_cal_bj_selected": float("nan"),
            "score_cal_bj": None,
            "auc_oracle": float("nan"),
            "oracle_score": None,
            "top_auc": [],
            "top_ks": [],
            "top_model_bj": [],
            "top_cal_bj": [],
        }

    valid_auc = [r for r in per_score if np.isfinite(r["auc"])]
    oracle = max(valid_auc, key=lambda r: r["auc"]) if valid_auc else per_score[0]
    by_name_all = {r["name"]: r for r in per_score}
    classic_names = set(score_bank) | rank_score_names
    model_names = (set(pvalue_bank or {}) | pvalue_score_names) & set(by_name_all)

    def pool_records(names: set[str]) -> List[Dict]:
        records = [by_name_all[name] for name in names if name in by_name_all]
        return records if records else per_score

    classic_pool = pool_records(classic_names)
    model_pool = pool_records(model_names)
    ks_pick = max(classic_pool, key=lambda r: r["ks"])
    selector_fields = {
        "graph_tail": "graph_tail",
        "tail_stable": "tail_stable",
        "graph_tail_stable": "graph_tail_stable",
        "corr_stable": "corr_stable",
        "stability": "stability_corr",
        "residual_graph_tail": "residual_graph_tail",
        "model_bj": "model_bj",
        "model_hc": "model_hc",
        "cal_bj": "cal_bj",
        "cal_hc": "cal_hc",
    }
    selector_results = {}
    for selector, field in selector_fields.items():
        base_pool = model_pool if field.startswith("model_") or field.startswith("cal_") else classic_pool
        eligible = [r for r in base_pool if np.isfinite(r.get(field, -np.inf))]
        pick = max(eligible, key=lambda r: r.get(field, -np.inf)) if eligible else ks_pick
        selector_results[selector] = {
            "score": pick["name"],
            "auc": float(pick["auc"]),
            "value": float(pick.get(field, 0.0)),
        }
    top_auc = sorted(valid_auc, key=lambda r: r["auc"], reverse=True)[:10]
    top_ks = sorted(per_score, key=lambda r: r["ks"], reverse=True)[:10]
    top_tail_stable = sorted(
        per_score, key=lambda r: r.get("tail_stable", -np.inf), reverse=True,
    )[:10]
    top_graph_tail = sorted(
        per_score, key=lambda r: r.get("graph_tail", -np.inf), reverse=True,
    )[:10]
    top_model_bj = sorted(
        [r for r in per_score if np.isfinite(r.get("model_bj", -np.inf))],
        key=lambda r: r.get("model_bj", -np.inf),
        reverse=True,
    )[:10]
    top_cal_bj = sorted(
        [r for r in per_score if np.isfinite(r.get("cal_bj", -np.inf))],
        key=lambda r: r.get("cal_bj", -np.inf),
        reverse=True,
    )[:10]
    compact_names = [
        "J@inf", "R@inf", "rank_mean", "rank_max", "rank_top2", "rank_lse",
        "model_minp", "model_acat", "cal_minp", "cal_acat",
        "dp_profile_acat", "dp_profile_minp", "dp_profile_mean_logp",
        "dp_ebmix_posterior", "dp_ebmix_logbf", "dp_eb_weighted_logp",
        "dp_eb_weighted_rank", "dp_eb_stouffer", "dp_eb_acat",
        "h8_static", "h8_sparse", "h8_shape",
        "h8_static_sparse_maxrank", "h8_static_shape_maxrank",
        "h8_three_way_maxrank", "h8_soft_gate",
        "h8_regime_v1", "h8_regime_v2",
        "h9_anchor", "h9_sparse", "h9_entropy", "h9_dense_shape",
        "h9_anchor_sparse_maxrank", "h9_anchor_entropy_maxrank",
        "h9_anchor_shape_maxrank", "h9_four_way_maxrank",
        "h9_override",
        "h10_base_h8", "h10_anchor_rescue", "h10_entropy",
        "h10_dense_soft", "h10_base_anchor_maxrank",
        "h10_base_dynamic_maxrank", "h10_override",
        "h10_override_soft",
    ]
    compact = {name: by_name_all[name] for name in compact_names if name in by_name_all}
    return {
        "n_scores": len(per_score),
        "auc_ks_selected": float(ks_pick["auc"]),
        "score_ks": ks_pick["name"],
        "ks_value": float(ks_pick["ks"]),
        "selectors": selector_results,
        "auc_tail_stable_selected": float(selector_results["tail_stable"]["auc"]),
        "score_tail_stable": selector_results["tail_stable"]["score"],
        "auc_graph_tail_selected": float(selector_results["graph_tail"]["auc"]),
        "score_graph_tail": selector_results["graph_tail"]["score"],
        "auc_corr_stable_selected": float(selector_results["corr_stable"]["auc"]),
        "score_corr_stable": selector_results["corr_stable"]["score"],
        "auc_model_bj_selected": float(selector_results["model_bj"]["auc"]),
        "score_model_bj": selector_results["model_bj"]["score"],
        "model_bj_value": float(selector_results["model_bj"]["value"]),
        "auc_cal_bj_selected": float(selector_results["cal_bj"]["auc"]),
        "score_cal_bj": selector_results["cal_bj"]["score"],
        "cal_bj_value": float(selector_results["cal_bj"]["value"]),
        "auc_oracle": float(oracle["auc"]),
        "oracle_score": oracle["name"],
        "compact": compact,
        "top_auc": top_auc,
        "top_ks": top_ks,
        "top_tail_stable": top_tail_stable,
        "top_graph_tail": top_graph_tail,
        "top_model_bj": top_model_bj,
        "top_cal_bj": top_cal_bj,
    }


def summarize_fit(fit: Dict) -> Dict:
    return {
        "ml": float(fit["ml"]),
        "entry_key": fit["entry_key"],
        "graph_type": fit["graph_type"],
        "template_gamma": float(fit["template_gamma"]),
        "rho": float(fit["rho"]),
        "kappa": float(fit["kappa"]),
    }


def print_bank_summary(label: str, summary: Dict) -> None:
    print("  %-26s KS(%s)=%.2f%% CBJ(%s)=%.2f%% TS(%s)=%.2f%% GT(%s)=%.2f%% oracle(%s)=%.2f%% n=%d" % (
        label,
        summary.get("score_ks"),
        100.0 * summary.get("auc_ks_selected", float("nan")),
        summary.get("score_cal_bj"),
        100.0 * summary.get("auc_cal_bj_selected", float("nan")),
        summary.get("score_tail_stable"),
        100.0 * summary.get("auc_tail_stable_selected", float("nan")),
        summary.get("score_graph_tail"),
        100.0 * summary.get("auc_graph_tail_selected", float("nan")),
        summary.get("oracle_score"),
        100.0 * summary.get("auc_oracle", float("nan")),
        summary.get("n_scores", 0),
    ), flush=True)
    if summary.get("top_auc"):
        top = ", ".join(
            "%s %.1f" % (r["name"], 100.0 * r["auc"])
            for r in summary["top_auc"][:4]
        )
        print("    top AUC: %s" % top, flush=True)


def run_dataset(args, ds: str) -> Dict:
    load_name = ("YelpChi" if ds.lower() == "yelpchi" else
                 "Facebook" if ds.lower() == "facebook" else ds)
    print("\n=== %s ===" % ds, flush=True)
    data = load_data(load_name)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    pca_dim = args.pca_dim if data.x.shape[1] > args.pca_threshold else None

    h, density = compute_graph_stats(data)
    if args.template_gammas == "density":
        template_gammas = density_adjusted_template_gammas(h, density)
    else:
        template_gammas = parse_grid(args.template_gammas)
    graph_types = parse_names(args.graph_types)
    horizons = canonical_horizons(parse_grid(args.horizons), include_inf=True)
    kappas = parse_grid(args.kappas)

    print("  nodes=%d dim=%d edges=%d h=%.3f density=%.1f graphs=%s templates=%s pca=%s" % (
        data.x.shape[0], data.x.shape[1], data.edge_index.shape[1],
        h, density, graph_types, template_gammas, pca_dim,
    ), flush=True)

    entries = prepare_model_entries(
        data,
        graph_types=graph_types,
        template_gammas=template_gammas,
        pca_dim=pca_dim,
        truncated_k=args.truncated_k,
        modeled_subspace=args.modeled_subspace,
        template_type=args.template_type,
        device=args.device,
    )

    print("  fitting best stationary geometry", flush=True)
    stationary_fit = fit_best_entry(entries, [np.inf], kappas, args.gamma_jacobian)
    stationary_entry = entry_for_fit(entries, stationary_fit)
    stationary_q = q_for_fit(stationary_entry, stationary_fit)
    references = node_unsupervised_references(data, stationary_entry["delta"])

    stationary_scores = horizon_score_bank(stationary_entry, stationary_q, [np.inf])
    stationary_pvalues = horizon_pvalue_bank(
        stationary_entry, stationary_q, [np.inf], stationary_scores,
    )
    stationary_summary = evaluate_score_bank(
        y, mask, stationary_scores, references=references, pvalue_bank=stationary_pvalues,
    )

    horizon_scores = horizon_score_bank(stationary_entry, stationary_q, horizons)
    horizon_pvalues = horizon_pvalue_bank(
        stationary_entry, stationary_q, horizons, horizon_scores,
    )
    horizon_summary = evaluate_score_bank(
        y, mask, horizon_scores, references=references, pvalue_bank=horizon_pvalues,
    )

    band_scores = bandpass_score_bank(stationary_entry, stationary_q, horizons)
    band_pvalues = bandpass_pvalue_bank(
        stationary_entry, stationary_q, horizons, band_scores,
    )
    band_summary = evaluate_score_bank(
        y, mask, band_scores, references=references, pvalue_bank=band_pvalues,
    )

    combined_scores = {}
    combined_scores.update(horizon_scores)
    combined_scores.update(band_scores)
    combined_pvalues = {}
    combined_pvalues.update(horizon_pvalues)
    combined_pvalues.update(band_pvalues)
    combined_summary = evaluate_score_bank(
        y, mask, combined_scores, references=references, pvalue_bank=combined_pvalues,
    )

    graph_combined_scores = {}
    graph_combined_pvalues = {}
    graph_fits: List[Dict] = []
    for graph_type in graph_types:
        graph_entries = [entry for entry in entries if entry["graph_type"] == graph_type]
        graph_fit = fit_best_entry(graph_entries, [np.inf], kappas, args.gamma_jacobian)
        graph_entry = entry_for_fit(graph_entries, graph_fit)
        graph_q = q_for_fit(graph_entry, graph_fit)
        graph_fits.append(summarize_fit(graph_fit))
        local_scores = {}
        local_horizon_scores = horizon_score_bank(graph_entry, graph_q, horizons)
        local_band_scores = bandpass_score_bank(graph_entry, graph_q, horizons)
        local_scores.update(local_horizon_scores)
        local_scores.update(local_band_scores)
        local_pvalues = {}
        local_pvalues.update(horizon_pvalue_bank(
            graph_entry, graph_q, horizons, local_horizon_scores,
        ))
        local_pvalues.update(bandpass_pvalue_bank(
            graph_entry, graph_q, horizons, local_band_scores,
        ))
        for name, score in local_scores.items():
            graph_combined_scores["%s:%s" % (graph_type, name)] = score
        for name, pvalues in local_pvalues.items():
            graph_combined_pvalues["%s:%s" % (graph_type, name)] = pvalues
    graph_combined_summary = evaluate_score_bank(
        y, mask, graph_combined_scores, references=references,
        pvalue_bank=graph_combined_pvalues,
    )

    scale_refit_summary = None
    scale_refit_fits: List[Dict] = []
    if args.scale_refit:
        refit_scores, refit_pvalues, scale_refit_fits = scale_refit_horizon_bank(
            entries, horizons, kappas, args.gamma_jacobian,
        )
        scale_refit_summary = evaluate_score_bank(
            y, mask, refit_scores, references=references, pvalue_bank=refit_pvalues,
        )

    print_bank_summary("stationary", stationary_summary)
    print_bank_summary("horizon bank", horizon_summary)
    print_bank_summary("band-pass bank", band_summary)
    print_bank_summary("combined bank", combined_summary)
    print_bank_summary("graph combined bank", graph_combined_summary)
    if scale_refit_summary is not None:
        print_bank_summary("scale-refit bank", scale_refit_summary)

    reported = REPORTED.get(ds.lower(), 0.0)
    base_name, base_auc = BASELINES.get(ds.lower(), ("?", 0.0))
    if reported or base_auc:
        print("  reference: reported=%.1f%%  %s=%.1f%%" % (
            reported, base_name, base_auc,
        ), flush=True)

    result = {
        "dataset": ds,
        "graph": {
            "nodes": int(data.x.shape[0]),
            "dim": int(data.x.shape[1]),
            "edges": int(data.edge_index.shape[1]),
            "homophily_cos": h,
            "density": density,
            "graph_types": graph_types,
            "template_gammas": list(template_gammas),
            "template_type": args.template_type,
            "pca_dim": pca_dim,
            "modeled_subspace": args.modeled_subspace,
            "gamma_jacobian": args.gamma_jacobian,
            "horizons": [format_horizon(x) for x in horizons],
            "kappas": kappas,
        },
        "stationary_fit": summarize_fit(stationary_fit),
        "graph_fits": graph_fits,
        "stationary": stationary_summary,
        "horizon_bank": horizon_summary,
        "bandpass_bank": band_summary,
        "combined_bank": combined_summary,
        "graph_combined_bank": graph_combined_summary,
        "scale_refit_bank": scale_refit_summary,
        "scale_refit_fits": scale_refit_fits,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name or comma-separated list.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--template-gammas", default="density",
                        help="'density' or comma grid, e.g. 0.2,0.5,1.0")
    parser.add_argument("--template-type", default="low",
                        choices=["low", "affinity", "affinity_high", "affinity_mix"])
    parser.add_argument("--graph-types", default="original",
                        help="Comma-separated graph types: original,affinity")
    parser.add_argument("--horizons", default=",".join(format_horizon(g) for g in DEFAULT_HORIZONS))
    parser.add_argument("--kappas", default=",".join("%g" % k for k in KAPPAS))
    parser.add_argument("--pca-threshold", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--truncated-k", type=int, default=None)
    parser.add_argument("--modeled-subspace", default="legacy",
                        choices=["legacy", "drop_nullspace"])
    parser.add_argument("--gamma-jacobian", action="store_true")
    parser.add_argument("--scale-refit", action="store_true",
                        help="Fit a separate EB geometry for each horizon before aggregating.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(args, ds)

    os.makedirs("results", exist_ok=True)
    outfile = args.out
    if outfile is None:
        safe_name = "_".join(datasets)
        outfile = "results/dynamic_score_bank_%s.json" % safe_name
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved %s" % outfile, flush=True)


if __name__ == "__main__":
    main()
