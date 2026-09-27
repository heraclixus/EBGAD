"""Prototype trajectory-level GOU anomaly scores.

This script tests hypotheses that make the GOU dynamics
enter the score directly:

1. Path dissipation: integrate local transport energy over the relaxation path.
2. Latent relaxation-time mixture: fit a label-free mixture over finite GOU
   horizons and score nodes by mixture surprise.

It deliberately writes a separate result file from ``test_dynamic_score_bank``
so these experiments do not disturb the paper's current benchmark pipeline.
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
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import norm

from data_utils import load_data
from test_dynamic_score_bank import (
    BASELINES,
    DEFAULT_HORIZONS,
    KAPPAS,
    REPORTED,
    aggregate_rank_scores,
    canonical_horizons,
    compute_graph_stats,
    density_adjusted_template_gammas,
    energy_pvalues,
    edge_label,
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    format_horizon,
    horizon_score_bank,
    leverage_terms,
    node_unsupervised_references,
    parse_grid,
    parse_names,
    prepare_model_entries,
    print_bank_summary,
    q_for_fit,
    berk_jones_stat,
    median_calibrated_pvalues,
    rank_corr_from_ranks,
    ratio_pvalues,
    sanitize_pvalues,
    score_nodes,
    selector_diagnostics,
    summarize_fit,
)
from test_multiscale_gou_poe import gou_precisions, ks_null_deviation, percentile_rank


PAPER_EBGAD_CONFIGS = {
    "weibo": {"rho": 0.99, "kappa": 2.00, "gamma": 0.50, "score": "J"},
    "reddit": {"rho": 0.85, "kappa": 1.00, "gamma": 1.00, "score": "J"},
    "amazon": {"rho": 0.70, "kappa": 1.00, "gamma": 0.50, "score": "J"},
    "yelpchi": {"rho": 0.95, "kappa": 5.00, "gamma": 0.50, "score": "J"},
    "blogcatalog": {"rho": 0.25, "kappa": 0.01, "gamma": 0.20, "score": "J"},
    "facebook": {"rho": 0.50, "kappa": 10.0, "gamma": 0.50, "score": "R"},
    "acm": {"rho": 0.90, "kappa": 0.10, "gamma": 0.20, "score": "J"},
    "t_finance": {"rho": 0.95, "kappa": 1.00, "gamma": 0.20, "score": "R"},
    "elliptic": {"rho": 0.50, "kappa": 5.00, "gamma": 1.00, "score": "R"},
    "elliptic_plus_plus": {"rho": 0.50, "kappa": 5.00, "gamma": 0.50, "score": "R"},
}


def dynamic_tail_cutoff(q: np.ndarray, last_horizon: float, tail_factor: float, tail_cap: float) -> float:
    q_pos = np.asarray(q, dtype=np.float64)
    q_pos = q_pos[np.isfinite(q_pos) & (q_pos > 1e-10)]
    if len(q_pos) == 0:
        return float(last_horizon)
    # A low percentile avoids letting a nearly-null single mode make the path
    # integration horizon impractically large.
    q_ref = float(np.percentile(q_pos, 5.0))
    tail = tail_factor / max(q_ref, 1e-10)
    return float(min(max(last_horizon, tail), tail_cap))


def path_edges(
    q: np.ndarray,
    horizons: Sequence[float],
    tail_factor: float,
    tail_cap: float,
) -> List[float]:
    finite = sorted({float(h) for h in horizons if np.isfinite(h) and float(h) > 0.0})
    last = finite[-1] if finite else 1.0
    tail = dynamic_tail_cutoff(q, last, tail_factor, tail_cap)
    edges = [0.0] + finite
    if tail > edges[-1] * (1.0 + 1e-8):
        edges.append(tail)
    return sorted(set(edges))


def flow_energy_at_time(entry: Dict, q: np.ndarray, delta_hat: np.ndarray, t: float) -> np.ndarray:
    q_safe = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    weights = np.sqrt(q_safe) * np.exp(-float(t) * q_safe)
    field = entry["V"] @ (weights[:, None] * delta_hat)
    return np.sum(field * field, axis=1)


def path_dissipation_score_bank(
    entry: Dict,
    q: np.ndarray,
    horizons: Sequence[float],
    n_quad: int = 4,
    tail_factor: float = 16.0,
    tail_cap: float = 1000.0,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Approximate trajectory dissipation scores by per-interval quadrature."""
    edges = path_edges(q, horizons, tail_factor=tail_factor, tail_cap=tail_cap)
    nodes, weights = leggauss(max(int(n_quad), 2))
    delta_hat = entry["V"].T @ entry["delta"]
    residual = np.maximum(np.sum(entry["delta"] ** 2, axis=1), 1e-12)

    scores: Dict[str, np.ndarray] = {}
    total = np.zeros(entry["n"], dtype=np.float64)
    moment = np.zeros(entry["n"], dtype=np.float64)
    late_cutoffs = [0.5, 2.0, 10.0]
    late = {cutoff: np.zeros(entry["n"], dtype=np.float64) for cutoff in late_cutoffs}

    for left, right in zip(edges[:-1], edges[1:]):
        width = right - left
        if width <= 0:
            continue
        ts = left + 0.5 * width * (nodes + 1.0)
        ws = 0.5 * width * weights
        band = np.zeros(entry["n"], dtype=np.float64)
        band_moment = np.zeros(entry["n"], dtype=np.float64)
        for t, w in zip(ts, ws):
            e_t = flow_energy_at_time(entry, q, delta_hat, float(t))
            band += float(w) * e_t
            band_moment += float(w) * float(t) * e_t
            for cutoff in late_cutoffs:
                if t >= cutoff:
                    late[cutoff] += float(w) * e_t

        label = edge_label(left, right)
        scores["PD@%s" % label] = band.astype(np.float32, copy=False)
        scores["PDR@%s" % label] = (band / residual).astype(np.float32, copy=False)
        total += band
        moment += band_moment

    total_safe = np.maximum(total, 1e-12)
    mean_time = moment / total_safe
    scores["PD@all"] = total.astype(np.float32, copy=False)
    scores["PDR@all"] = (total / residual).astype(np.float32, copy=False)
    scores["PD_mean_time"] = mean_time.astype(np.float32, copy=False)
    scores["PD_inv_mean_time"] = (1.0 / np.maximum(mean_time, 1e-12)).astype(np.float32, copy=False)
    for cutoff, value in late.items():
        scores["PD_late_frac@%s" % format_horizon(cutoff)] = (value / total_safe).astype(np.float32, copy=False)

    meta = {
        "edges": [format_horizon(x) for x in edges],
        "n_quad": int(n_quad),
        "tail_factor": float(tail_factor),
        "tail_cap": float(tail_cap),
    }
    return scores, meta


def path_band_precision(q: np.ndarray, left: float, right: float) -> np.ndarray:
    q_safe = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    start = np.exp(-2.0 * float(left) * q_safe)
    stop = np.zeros_like(q_safe) if np.isinf(right) else np.exp(-2.0 * float(right) * q_safe)
    return np.maximum(0.5 * (start - stop), 0.0)


def path_dissipation_pvalue_bank(
    entry: Dict,
    q: np.ndarray,
    horizons: Sequence[float],
    scores: Dict[str, np.ndarray],
    tail_factor: float = 16.0,
    tail_cap: float = 1000.0,
) -> Dict[str, np.ndarray]:
    """Approximate calibrated p-values for path-dissipation bands.

    The exact path integral is a quadratic form with cross-mode terms.  For
    node-level calibration we match its first moment with the diagonal precision
    p_j = integral q_j exp(-2t q_j) dt, then use the same leverage correction as
    the equilibrium energy score.
    """
    pvalues: Dict[str, np.ndarray] = {}
    edges = path_edges(q, horizons, tail_factor=tail_factor, tail_cap=tail_cap)
    total_precision = np.zeros_like(np.asarray(q, dtype=np.float64))
    for left, right in zip(edges[:-1], edges[1:]):
        precision = path_band_precision(q, left, right)
        total_precision += precision
        num_scale, den_scale, cross_scale = leverage_terms(entry, q, precision)
        label = edge_label(left, right)
        pd_name = "PD@%s" % label
        pdr_name = "PDR@%s" % label
        if pd_name in scores:
            pvalues[pd_name] = energy_pvalues(scores[pd_name], num_scale, entry["d"])
        if pdr_name in scores:
            pvalues[pdr_name] = ratio_pvalues(scores[pdr_name], num_scale, den_scale, cross_scale, entry["d"])

    num_scale, den_scale, cross_scale = leverage_terms(entry, q, total_precision)
    if "PD@all" in scores:
        pvalues["PD@all"] = energy_pvalues(scores["PD@all"], num_scale, entry["d"])
    if "PDR@all" in scores:
        pvalues["PDR@all"] = ratio_pvalues(scores["PDR@all"], num_scale, den_scale, cross_scale, entry["d"])
    return pvalues


def central_pvalue_error(pvalues: np.ndarray) -> float:
    p = sanitize_pvalues(pvalues)
    probs = np.array([0.25, 0.50, 0.75], dtype=np.float64)
    empirical = np.quantile(p, probs)
    return float(np.max(np.abs(empirical - probs)))


def finite_zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    good = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not np.any(good):
        return out
    mu = float(np.mean(x[good]))
    sd = float(np.std(x[good]))
    if sd < 1e-12:
        return out
    out[good] = (x[good] - mu) / sd
    return out


def combine_acat(pmat: np.ndarray) -> np.ndarray:
    clipped = np.clip(pmat, 1e-15, 1.0 - 1e-15)
    stat = np.mean(np.tan((0.5 - clipped) * np.pi), axis=1)
    p = 0.5 - np.arctan(stat) / np.pi
    return sanitize_pvalues(p)


def robust_path_aggregate_score_bank(
    path_scores: Dict[str, np.ndarray],
    path_pvalues: Dict[str, np.ndarray],
    min_bands: int = 3,
    max_bands: int = 8,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Screen path bands by null calibration/stability, then aggregate them.

    This is a label-free path-band selector.  A band is admissible when its
    p-values are not badly miscalibrated in the central mass and its node ranks
    are not isolated from neighboring dynamic bands.  Aggregation then uses
    calibrated p-values, so no single horizon is selected by labels.
    """
    band_names = [
        name for name in path_pvalues
        if ("@" in name and not name.endswith("@all"))
    ]
    if not band_names:
        fallback = path_scores.get("PDR@all", path_scores.get("PD@all"))
        if fallback is None:
            return {}, {"selected": []}
        return {"rp_fallback": fallback}, {"selected": []}

    records = []
    ranks = {}
    for idx, name in enumerate(band_names):
        raw_p = sanitize_pvalues(path_pvalues[name])
        cal_p = median_calibrated_pvalues(raw_p)
        score = -np.log(cal_p)
        ranks[name] = percentile_rank(score)
        records.append({
            "name": name,
            "family": name.split("@", 1)[0],
            "order": idx,
            "raw_p": raw_p,
            "cal_p": cal_p,
            "central_error": central_pvalue_error(raw_p),
            "tail_bj": berk_jones_stat(cal_p),
            "stability": 0.0,
        })

    by_family: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        by_family.setdefault(record["family"], []).append(idx)
    for family_indices in by_family.values():
        for pos, idx in enumerate(family_indices):
            neighbors = []
            if pos > 0:
                neighbors.append(family_indices[pos - 1])
            if pos + 1 < len(family_indices):
                neighbors.append(family_indices[pos + 1])
            if neighbors:
                records[idx]["stability"] = float(np.mean([
                    rank_corr_from_ranks(ranks[records[idx]["name"]], ranks[records[j]["name"]])
                    for j in neighbors
                ]))

    errors = np.array([r["central_error"] for r in records], dtype=np.float64)
    stabilities = np.array([r["stability"] for r in records], dtype=np.float64)
    tails = np.log1p(np.array([r["tail_bj"] for r in records], dtype=np.float64))
    quality = -finite_zscore(errors) + finite_zscore(stabilities) + 0.35 * finite_zscore(tails)

    cal_cut = float(np.median(errors) + 1.4826 * np.median(np.abs(errors - np.median(errors))))
    stab_cut = float(np.quantile(stabilities, 0.25)) if len(stabilities) > 1 else -1.0
    admissible = [
        idx for idx, record in enumerate(records)
        if record["central_error"] <= cal_cut and record["stability"] >= stab_cut
    ]
    min_bands = max(1, min(int(min_bands), len(records)))
    max_bands = max(min_bands, min(int(max_bands), len(records)))
    if len(admissible) < min_bands:
        admissible = sorted(range(len(records)), key=lambda idx: (-quality[idx], errors[idx]))[:min_bands]
    else:
        admissible = sorted(admissible, key=lambda idx: quality[idx], reverse=True)[:max_bands]

    selected = [records[idx] for idx in admissible]
    pmat = np.vstack([record["cal_p"] for record in selected]).T
    logp = -np.log(np.maximum(pmat, 1e-300))
    weights = np.maximum(quality[admissible], 0.0)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones(len(admissible), dtype=np.float64)
    weights = weights / np.sum(weights)
    top_k = min(2, logp.shape[1])
    top_logp = np.partition(logp, logp.shape[1] - top_k, axis=1)[:, -top_k:]
    minp = np.minimum(1.0, logp.shape[1] * np.min(pmat, axis=1))
    zmat = norm.isf(np.clip(pmat, 1e-15, 1.0 - 1e-15))
    stouffer = np.sum(zmat, axis=1) / np.sqrt(float(logp.shape[1]))
    acat_p = combine_acat(pmat)

    scores = {
        "rp_acat": (-np.log(acat_p)).astype(np.float32, copy=False),
        "rp_minp": (-np.log(np.maximum(minp, 1e-300))).astype(np.float32, copy=False),
        "rp_mean_logp": np.mean(logp, axis=1).astype(np.float32, copy=False),
        "rp_top2_logp": np.mean(top_logp, axis=1).astype(np.float32, copy=False),
        "rp_weighted_logp": (logp @ weights).astype(np.float32, copy=False),
        "rp_stouffer": stouffer.astype(np.float32, copy=False),
    }
    fallback = path_scores.get("PDR@all", path_scores.get("PD@all"))
    if fallback is not None:
        for name, score in list(scores.items()):
            if np.nanstd(score) < 1e-12:
                scores[name] = fallback
        scores["rp_fallback"] = fallback

    meta = {
        "calibration_cut": cal_cut,
        "stability_cut": stab_cut,
        "min_bands": min_bands,
        "max_bands": max_bands,
        "selected": [
            {
                "name": record["name"],
                "central_error": float(record["central_error"]),
                "stability": float(record["stability"]),
                "tail_bj": float(record["tail_bj"]),
                "quality": float(quality[idx]),
                "weight": float(weight),
            }
            for record, idx, weight in zip(selected, admissible, weights)
        ],
    }
    return scores, meta


def tail_empirical_pvalues(score: np.ndarray) -> np.ndarray:
    rank = percentile_rank(np.nan_to_num(
        np.asarray(score, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ))
    floor = 1.0 / float(len(rank) + 1)
    return sanitize_pvalues(np.maximum(1.0 - rank, floor))


def parse_band_right(name: str) -> float:
    try:
        label = name.split("@", 1)[1]
        right = label.split(":", 1)[1]
        return np.inf if right == "inf" else float(right)
    except Exception:
        return np.inf


def parse_band_left(name: str) -> float:
    try:
        label = name.split("@", 1)[1]
        left = label.split(":", 1)[0]
        return np.inf if left == "inf" else float(left)
    except Exception:
        return 0.0


def make_profile_records(
    profile_names: Sequence[str],
    score_bank: Dict[str, np.ndarray],
    pvalue_bank: Dict[str, np.ndarray],
) -> List[Dict]:
    records = []
    for name in profile_names:
        if name not in score_bank:
            continue
        score = np.nan_to_num(
            np.asarray(score_bank[name], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if np.nanmax(score) <= np.nanmin(score):
            continue
        raw_p = pvalue_bank.get(name)
        model_based = raw_p is not None
        if raw_p is None:
            raw_p = tail_empirical_pvalues(score)
        raw_p = sanitize_pvalues(raw_p)
        records.append({
            "name": name,
            "score": score,
            "rank": percentile_rank(score),
            "p": raw_p,
            "cal_p": median_calibrated_pvalues(raw_p),
            "model_based": model_based,
        })
    return records


def profile_from_records(records: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    pmat = np.vstack([record["cal_p"] for record in records]).T
    logp = -np.log(np.maximum(pmat, 1e-300))
    top_k = min(2, logp.shape[1])
    top_logp = np.partition(logp, logp.shape[1] - top_k, axis=1)[:, -top_k:]
    score = np.mean(top_logp, axis=1)
    ranks = np.vstack([record["rank"] for record in records])
    rank_score = np.mean(ranks, axis=0)
    profile_p = tail_empirical_pvalues(score)
    if len(records) == 1:
        stability = 0.50
    else:
        corrs = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                corrs.append(rank_corr_from_ranks(records[i]["rank"], records[j]["rank"]))
        stability = float(np.mean(corrs)) if corrs else 0.50
    model_errors = [
        central_pvalue_error(record["p"])
        for record in records if record["model_based"]
    ]
    central_error = float(np.mean(model_errors)) if model_errors else 0.25
    meta = {
        "members": [record["name"] for record in records],
        "n_members": len(records),
        "n_model_pvalues": int(sum(record["model_based"] for record in records)),
        "central_error": central_error,
        "stability": stability,
        "tail_bj": berk_jones_stat(median_calibrated_pvalues(profile_p)),
    }
    return score, profile_p, rank_score, meta


def weighted_acat(pmat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(pmat, 1e-15, 1.0 - 1e-15)
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        w = np.ones(pmat.shape[1], dtype=np.float64)
    w = w / np.sum(w)
    stat = np.sum(w[None, :] * np.tan((0.5 - clipped) * np.pi), axis=1)
    p = 0.5 - np.arctan(stat) / np.pi
    return sanitize_pvalues(p)


def dynamic_profile_selector_score_bank(
    path_scores: Dict[str, np.ndarray],
    path_pvalues: Dict[str, np.ndarray],
    mixture_scores: Dict[str, np.ndarray],
    two_groups_scores: Dict[str, np.ndarray],
    stationary_scores: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict]:
    """Select among predeclared dynamic trajectory profiles without labels."""
    all_scores = {}
    all_scores.update(path_scores)
    all_scores.update(mixture_scores)
    all_scores.update(two_groups_scores)
    all_pvalues = dict(path_pvalues)

    path_band_names = [
        name for name in path_scores
        if "@" in name and not name.endswith("@all")
    ]
    pd_bands = [name for name in path_band_names if name.startswith("PD@")]
    pdr_bands = [name for name in path_band_names if name.startswith("PDR@")]

    profiles = {
        "early_roughness": [
            name for name in pd_bands if parse_band_right(name) <= 0.1
        ],
        "early_ratio": [
            name for name in pdr_bands if parse_band_right(name) <= 0.1
        ],
        "mid_dissipation": [
            name for name in pd_bands
            if parse_band_left(name) >= 0.1 and parse_band_right(name) <= 5.0
        ],
        "late_persistence": [
            name for name in pd_bands if parse_band_left(name) >= 2.0
        ] + [
            name for name in ("PD_late_frac@0.5", "PD_late_frac@2", "PD_late_frac@10", "PD_mean_time")
            if name in path_scores
        ],
        "ratio_transport": pdr_bands + [
            name for name in ("PDR@all",) if name in path_scores
        ],
        "mixture_heterogeneity": [
            name for name in ("mix_resp_entropy", "mix_best_nll", "mix_nll", "mix_weighted_nll")
            if name in mixture_scores
        ],
        "scale_tail": [
            name for name in ("tg_log_bf", "tg_expected_surprise", "tg_component_entropy")
            if name in two_groups_scores
        ],
        "transport_shape": [
            name for name in ("PD_mean_time", "PD_inv_mean_time", "PD_late_frac@0.5", "PD_late_frac@2", "PDR@all")
            if name in path_scores
        ],
    }

    station_refs = [
        stationary_scores[name] for name in ("J@inf", "R@inf")
        if name in stationary_scores
    ]
    if station_refs:
        stationary_rank = np.mean([
            percentile_rank(np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0))
            for ref in station_refs
        ], axis=0)
    else:
        stationary_rank = None

    profile_payloads = {}
    profile_scores: Dict[str, np.ndarray] = {}
    profile_pvalues: Dict[str, np.ndarray] = {}
    for profile, names in profiles.items():
        records = make_profile_records(names, all_scores, all_pvalues)
        if not records:
            continue
        score, pvalues, rank_score, meta = profile_from_records(records)
        if stationary_rank is None:
            complementarity = 0.0
        else:
            complementarity = 1.0 - abs(rank_corr_from_ranks(percentile_rank(score), stationary_rank))
        meta["complementarity"] = float(complementarity)
        profile_payloads[profile] = {
            "score": score,
            "pvalues": pvalues,
            "rank_score": rank_score,
            "meta": meta,
        }
        profile_scores["dp_%s" % profile] = score.astype(np.float32, copy=False)
        profile_pvalues["dp_%s" % profile] = pvalues

    if not profile_payloads:
        return {}, {}, {"profiles": []}

    names = list(profile_payloads)
    profile_pmat = np.vstack([
        sanitize_pvalues(profile_payloads[name]["pvalues"]) for name in names
    ]).T
    profile_logp = -np.log(np.maximum(profile_pmat, 1e-300))
    profile_acat = combine_acat(profile_pmat)
    profile_minp = np.minimum(1.0, profile_pmat.shape[1] * np.min(profile_pmat, axis=1))
    profile_scores["dp_profile_acat"] = (-np.log(profile_acat)).astype(np.float32, copy=False)
    profile_scores["dp_profile_minp"] = (-np.log(np.maximum(profile_minp, 1e-300))).astype(np.float32, copy=False)
    profile_scores["dp_profile_mean_logp"] = np.mean(profile_logp, axis=1).astype(np.float32, copy=False)
    profile_pvalues["dp_profile_acat"] = profile_acat
    profile_pvalues["dp_profile_minp"] = sanitize_pvalues(profile_minp)

    eb_meta = None
    try:
        eb_pvalue_bank = {
            "dp_%s" % name: profile_payloads[name]["pvalues"]
            for name in names
        }
        eb_names, eb_posterior, _, eb_weights, eb_a, eb_objective = fit_two_groups_scale_mixture(
            eb_pvalue_bank,
            pi_init=0.05,
            pi_max=0.20,
            weight_alpha=1.10,
            a_min=0.05,
            a_max=0.95,
        )
        eb_pmat = np.vstack([
            sanitize_pvalues(eb_pvalue_bank[name]) for name in eb_names
        ]).T
        eb_logp = -np.log(np.maximum(eb_pmat, 1e-300))
        beta_log_density = (
            np.log(np.maximum(eb_a, 1e-300))[None, :]
            + (eb_a[None, :] - 1.0) * np.log(np.maximum(eb_pmat, 1e-300))
        )
        eb_logbf = logsumexp(
            np.log(np.maximum(eb_weights, 1e-300))[None, :] + beta_log_density,
            axis=1,
        )
        eb_importance = eb_weights * np.maximum(1.0 - eb_a, 1e-6)
        if float(np.sum(eb_importance)) <= 1e-12:
            eb_importance = np.ones_like(eb_importance)
        eb_importance = eb_importance / np.sum(eb_importance)
        eb_rank_stack = np.vstack([
            profile_payloads[name[3:]]["rank_score"] for name in eb_names
        ])
        eb_z = norm.isf(np.clip(eb_pmat, 1e-15, 1.0 - 1e-15))
        eb_z_denom = np.sqrt(np.maximum(np.sum(eb_importance * eb_importance), 1e-12))
        eb_acat = weighted_acat(eb_pmat, eb_importance)

        profile_scores["dp_ebmix_posterior"] = eb_posterior.astype(np.float32, copy=False)
        profile_scores["dp_ebmix_logbf"] = eb_logbf.astype(np.float32, copy=False)
        profile_scores["dp_eb_weighted_logp"] = (eb_logp @ eb_importance).astype(np.float32, copy=False)
        profile_scores["dp_eb_weighted_rank"] = (eb_importance @ eb_rank_stack).astype(np.float32, copy=False)
        profile_scores["dp_eb_stouffer"] = ((eb_z @ eb_importance) / eb_z_denom).astype(np.float32, copy=False)
        profile_scores["dp_eb_acat"] = (-np.log(eb_acat)).astype(np.float32, copy=False)
        profile_pvalues["dp_eb_acat"] = eb_acat
        eb_meta = {
            "pi": float(np.mean(eb_posterior)),
            "pi_max": 0.20,
            "weight_alpha": 1.10,
            "objective": float(eb_objective),
            "components": [
                {
                    "profile": name[3:],
                    "weight": float(weight),
                    "a": float(ai),
                    "importance": float(importance),
                }
                for name, weight, ai, importance in zip(eb_names, eb_weights, eb_a, eb_importance)
            ],
        }
    except Exception as exc:
        eb_meta = {"error": repr(exc)}

    tail = np.log1p(np.array([
        profile_payloads[name]["meta"]["tail_bj"] for name in names
    ], dtype=np.float64))
    stability = np.array([
        profile_payloads[name]["meta"]["stability"] for name in names
    ], dtype=np.float64)
    central_error = np.array([
        profile_payloads[name]["meta"]["central_error"] for name in names
    ], dtype=np.float64)
    complementarity = np.array([
        profile_payloads[name]["meta"]["complementarity"] for name in names
    ], dtype=np.float64)

    quality = (
        0.45 * finite_zscore(tail)
        + 0.30 * finite_zscore(stability)
        + 0.15 * finite_zscore(complementarity)
        - 0.30 * finite_zscore(central_error)
    )
    order = np.argsort(-quality)
    best_idx = int(order[0])
    top_k = min(2, len(order))
    top_idx = order[:top_k]
    top_quality = quality[top_idx]
    weights = np.exp(top_quality - np.max(top_quality))
    weights = weights / np.maximum(np.sum(weights), 1e-300)
    rank_stack = np.vstack([
        profile_payloads[names[idx]]["rank_score"] for idx in top_idx
    ])
    top2_rank_score = weights @ rank_stack
    all_rank_stack = np.vstack([
        profile_payloads[name]["rank_score"] for name in names
    ])

    best_name = names[best_idx]
    profile_scores["dp_selected"] = profile_payloads[best_name]["score"].astype(np.float32, copy=False)
    profile_pvalues["dp_selected"] = profile_payloads[best_name]["pvalues"]
    profile_scores["dp_top2_profile"] = top2_rank_score.astype(np.float32, copy=False)
    profile_scores["dp_mean_profile"] = np.mean(all_rank_stack, axis=0).astype(np.float32, copy=False)
    profile_scores["dp_max_profile"] = np.max(all_rank_stack, axis=0).astype(np.float32, copy=False)

    meta = {
        "selected": best_name,
        "top2": [
            {"profile": names[idx], "weight": float(weight), "quality": float(quality[idx])}
            for idx, weight in zip(top_idx, weights)
        ],
        "quality_formula": "0.45*z(log1p(BJ)) + 0.30*z(stability) + 0.15*z(complementarity) - 0.30*z(central_error)",
        "eb_profile_mixture": eb_meta,
        "profiles": [
            {
                "profile": name,
                "quality": float(quality[idx]),
                **profile_payloads[name]["meta"],
            }
            for idx, name in enumerate(names)
        ],
    }
    return profile_scores, profile_pvalues, meta


def fit_two_groups_scale_mixture(
    pvalue_bank: Dict[str, np.ndarray],
    pi_init: float = 0.05,
    pi_max: float = 0.20,
    weight_alpha: float = 1.05,
    a_min: float = 0.02,
    a_max: float = 0.95,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    names = sorted(pvalue_bank)
    if not names:
        raise ValueError("two-groups model requires at least one p-value vector")
    pmat = np.vstack([sanitize_pvalues(pvalue_bank[name]) for name in names]).T
    logp = np.log(pmat)
    n, m = pmat.shape
    pi = float(np.clip(pi_init, 1e-4, pi_max))
    weights = np.full(m, 1.0 / float(m), dtype=np.float64)
    a = np.full(m, 0.50, dtype=np.float64)
    alpha = max(float(weight_alpha), 1.0)
    prev_obj = -np.inf

    for _ in range(max_iter):
        log_beta = np.log(np.maximum(a, 1e-300))[None, :] + (a[None, :] - 1.0) * logp
        log_alt_components = np.log(np.maximum(weights, 1e-300))[None, :] + log_beta
        log_alt = logsumexp(log_alt_components, axis=1)
        log_anom = np.log(np.maximum(pi, 1e-300)) + log_alt
        log_null = np.log(np.maximum(1.0 - pi, 1e-300))
        log_marginal = np.logaddexp(log_null, log_anom)
        posterior = np.exp(log_anom - log_marginal)
        component_resp = np.exp(log_alt_components - log_alt[:, None])
        weighted_resp = posterior[:, None] * component_resp

        counts = weighted_resp.sum(axis=0) + (alpha - 1.0)
        weights = counts / np.maximum(np.sum(counts), 1e-300)
        raw_counts = weighted_resp.sum(axis=0)
        weighted_logp = np.sum(weighted_resp * logp, axis=0)
        update_a = -raw_counts / np.minimum(weighted_logp, -1e-12)
        a = np.clip(np.nan_to_num(update_a, nan=a_max, posinf=a_max, neginf=a_min), a_min, a_max)
        pi = float(np.clip(np.mean(posterior), 1e-4, pi_max))

        obj = float(np.sum(log_marginal) + (alpha - 1.0) * np.sum(np.log(np.maximum(weights, 1e-300))))
        if abs(obj - prev_obj) <= tol * max(1.0, abs(prev_obj)):
            prev_obj = obj
            break
        prev_obj = obj

    log_beta = np.log(np.maximum(a, 1e-300))[None, :] + (a[None, :] - 1.0) * logp
    log_alt_components = np.log(np.maximum(weights, 1e-300))[None, :] + log_beta
    log_alt = logsumexp(log_alt_components, axis=1)
    log_anom = np.log(np.maximum(pi, 1e-300)) + log_alt
    log_null = np.log(np.maximum(1.0 - pi, 1e-300))
    log_marginal = np.logaddexp(log_null, log_anom)
    posterior = np.exp(log_anom - log_marginal)
    component_resp = np.exp(log_alt_components - log_alt[:, None])
    return names, posterior, component_resp, weights, a, float(prev_obj)


def two_groups_path_score_bank(
    pvalue_bank: Dict[str, np.ndarray],
    pi_init: float = 0.05,
    pi_max: float = 0.20,
    weight_alpha: float = 1.05,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    names, posterior, component_resp, weights, a, objective = fit_two_groups_scale_mixture(
        pvalue_bank,
        pi_init=pi_init,
        pi_max=pi_max,
        weight_alpha=weight_alpha,
    )
    pmat = np.vstack([sanitize_pvalues(pvalue_bank[name]) for name in names]).T
    logp = np.log(pmat)
    log_beta = np.log(np.maximum(a, 1e-300))[None, :] + (a[None, :] - 1.0) * logp
    log_alt = logsumexp(np.log(np.maximum(weights, 1e-300))[None, :] + log_beta, axis=1)
    component_entropy = -np.sum(component_resp * np.log(np.maximum(component_resp, 1e-300)), axis=1)
    expected_surprise = -np.sum(component_resp * logp, axis=1)

    scores = {
        "tg_posterior": posterior.astype(np.float32, copy=False),
        "tg_log_bf": log_alt.astype(np.float32, copy=False),
        "tg_expected_surprise": expected_surprise.astype(np.float32, copy=False),
        "tg_component_conf": np.max(component_resp, axis=1).astype(np.float32, copy=False),
        "tg_component_entropy": component_entropy.astype(np.float32, copy=False),
    }
    meta = {
        "pi": float(np.mean(posterior)),
        "pi_max": float(pi_max),
        "weight_alpha": float(weight_alpha),
        "objective": objective,
        "components": [
            {"score": name, "weight": float(weight), "a": float(ai)}
            for name, weight, ai in zip(names, weights, a)
        ],
    }
    return scores, meta


def horizon_local_logpdfs(
    entry: Dict,
    q: np.ndarray,
    horizons: Sequence[float],
) -> Tuple[List[str], np.ndarray]:
    labels: List[str] = []
    logpdfs: List[np.ndarray] = []
    tau = gou_precisions(q, horizons)
    d = int(entry["d"])
    log2pi = np.log(2.0 * np.pi)
    for horizon, precision in zip(horizons, tau):
        je, _ = score_nodes(entry["delta"], entry["V"], precision)
        scale, _, _ = leverage_terms(entry, q, precision)
        logp = -0.5 * (je / scale + d * (log2pi + np.log(scale)))
        labels.append(format_horizon(horizon))
        logpdfs.append(logp.astype(np.float64, copy=False))
    return labels, np.vstack(logpdfs).T


def fit_mixture_weights(
    logpdf: np.ndarray,
    dirichlet_alpha: float = 1.05,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n, m = logpdf.shape
    weights = np.full(m, 1.0 / float(m), dtype=np.float64)
    prev_obj = -np.inf
    alpha = max(float(dirichlet_alpha), 1.0)

    for _ in range(max_iter):
        log_joint = logpdf + np.log(np.maximum(weights, 1e-300))[None, :]
        log_norm = logsumexp(log_joint, axis=1)
        resp = np.exp(log_joint - log_norm[:, None])
        counts = resp.sum(axis=0) + (alpha - 1.0)
        weights = counts / np.maximum(np.sum(counts), 1e-300)
        obj = float(np.sum(log_norm) + (alpha - 1.0) * np.sum(np.log(np.maximum(weights, 1e-300))))
        if abs(obj - prev_obj) <= tol * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj

    log_joint = logpdf + np.log(np.maximum(weights, 1e-300))[None, :]
    log_norm = logsumexp(log_joint, axis=1)
    resp = np.exp(log_joint - log_norm[:, None])
    return weights, resp, log_norm, float(prev_obj)


def latent_time_mixture_score_bank(
    entry: Dict,
    q: np.ndarray,
    horizons: Sequence[float],
    dirichlet_alpha: float = 1.05,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    labels, logpdf = horizon_local_logpdfs(entry, q, horizons)
    weights, resp, log_mix, objective = fit_mixture_weights(
        logpdf, dirichlet_alpha=dirichlet_alpha,
    )

    scores: Dict[str, np.ndarray] = {
        "mix_nll": (-log_mix).astype(np.float32, copy=False),
        "mix_best_nll": (-np.max(logpdf, axis=1)).astype(np.float32, copy=False),
        "mix_weighted_nll": (-(logpdf @ weights)).astype(np.float32, copy=False),
        "mix_resp_entropy": (-np.sum(resp * np.log(np.maximum(resp, 1e-300)), axis=1)).astype(np.float32, copy=False),
        "mix_resp_max": np.max(resp, axis=1).astype(np.float32, copy=False),
    }
    for idx, label in enumerate(labels):
        scores["mix_nll@%s" % label] = (-logpdf[:, idx]).astype(np.float32, copy=False)

    meta = {
        "horizons": labels,
        "weights": {label: float(weight) for label, weight in zip(labels, weights)},
        "dirichlet_alpha": float(dirichlet_alpha),
        "objective": objective,
    }
    return scores, meta


def augmented_scores(score_bank: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(score_bank)
    out.update(aggregate_rank_scores(score_bank))
    return out


def pick_static_regime_score(
    stationary_scores: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray],
) -> Tuple[str, np.ndarray, Dict]:
    candidates = augmented_scores(stationary_scores)
    diagnostics = selector_diagnostics(candidates, references)
    ks_values = {
        name: ks_null_deviation(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        for name, score in candidates.items()
    }
    ks_pick = max(ks_values, key=ks_values.get)
    graph_pick = max(
        diagnostics,
        key=lambda name: diagnostics[name].get("graph_tail", -np.inf),
    ) if diagnostics else ks_pick
    graph_value = diagnostics.get(graph_pick, {}).get("graph_tail", 0.0)
    ks_value = float(ks_values.get(ks_pick, 0.0))

    if graph_pick == "R@inf" and graph_value > 0.5 and ks_value < 0.25:
        pick = "R@inf"
        reason = "graph-tail ratio"
    elif ks_pick == "R@inf" and ks_value > 0.35:
        pick = "R@inf"
        reason = "ratio KS"
    elif ks_pick == "J@inf" and ks_value > 0.90 and "rank_mean" in candidates:
        pick = "rank_mean"
        reason = "global-J-KS fallback"
    elif ks_pick == "J@inf" and ks_value > 0.35:
        pick = "J@inf"
        reason = "energy KS"
    elif "rank_mean" in candidates:
        pick = "rank_mean"
        reason = "weak-static rank fusion"
    else:
        pick = ks_pick
        reason = "KS fallback"

    return pick, candidates[pick], {
        "pick": pick,
        "reason": reason,
        "ks_pick": ks_pick,
        "ks_value": ks_value,
        "graph_pick": graph_pick,
        "graph_value": float(graph_value),
    }


def regime_gated_score_bank(
    stationary_scores: Dict[str, np.ndarray],
    profile_scores: Dict[str, np.ndarray],
    profile_pvalues: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray],
    graph_density: float,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Predeclared regime gate over static, sparse-tail, and shape scores.

    The gate uses only score distribution diagnostics.  Its purpose is to avoid
    averaging all trajectory profiles when the unlabeled diagnostics indicate a
    simpler regime: static equilibrium, a clean sparse profile tail, or a dense
    transport-shape regime.
    """
    static_name, static_score, static_meta = pick_static_regime_score(
        stationary_scores, references,
    )
    sparse_score = profile_scores.get(
        "dp_ebmix_posterior",
        profile_scores.get("dp_profile_mean_logp", static_score),
    )
    shape_score = profile_scores.get(
        "dp_eb_weighted_rank",
        profile_scores.get("dp_mean_profile", sparse_score),
    )
    sparse_p = profile_pvalues.get(
        "dp_eb_acat",
        profile_pvalues.get("dp_profile_acat"),
    )
    if sparse_p is None:
        sparse_bj = 0.0
    else:
        sparse_bj = berk_jones_stat(median_calibrated_pvalues(sparse_p))
    sparse_ks = ks_null_deviation(np.nan_to_num(
        sparse_score, nan=0.0, posinf=0.0, neginf=0.0,
    ))

    sparse_clean = (
        20.0 <= sparse_bj <= 100.0
        and static_meta["ks_value"] < 0.20
        and sparse_ks < 0.35
    )
    shape_clean = (
        graph_density >= 100.0
        or (
            graph_density >= 10.0
            and static_meta["pick"] == "J@inf"
            and static_meta["ks_value"] >= 0.50
            and 20.0 <= sparse_bj <= 120.0
        )
    )

    static_rank = percentile_rank(static_score)
    sparse_rank = percentile_rank(sparse_score)
    shape_rank = percentile_rank(shape_score)
    rank_stack = np.vstack([static_rank, sparse_rank, shape_rank])

    if sparse_clean:
        v1 = sparse_score
        v1_regime = "sparse-tail"
    else:
        v1 = static_score
        v1_regime = "static"

    if sparse_clean:
        v2 = sparse_score
        v2_regime = "sparse-tail"
    elif shape_clean:
        v2 = shape_score
        v2_regime = "trajectory-shape"
    else:
        v2 = static_score
        v2_regime = "static"

    weights = np.array([
        1.0,
        3.0 if sparse_clean else 0.25,
        2.0 if shape_clean else 0.25,
    ], dtype=np.float64)
    weights = weights / np.sum(weights)

    scores = {
        "h8_static": np.asarray(static_score, dtype=np.float32),
        "h8_sparse": np.asarray(sparse_score, dtype=np.float32),
        "h8_shape": np.asarray(shape_score, dtype=np.float32),
        "h8_static_sparse_maxrank": np.maximum(static_rank, sparse_rank).astype(np.float32, copy=False),
        "h8_static_shape_maxrank": np.maximum(static_rank, shape_rank).astype(np.float32, copy=False),
        "h8_three_way_maxrank": np.max(rank_stack, axis=0).astype(np.float32, copy=False),
        "h8_soft_gate": (weights @ rank_stack).astype(np.float32, copy=False),
        "h8_regime_v1": np.asarray(v1, dtype=np.float32),
        "h8_regime_v2": np.asarray(v2, dtype=np.float32),
    }
    meta = {
        "static": static_meta,
        "sparse": {
            "score": "dp_ebmix_posterior" if "dp_ebmix_posterior" in profile_scores else "dp_profile_mean_logp",
            "bj": float(sparse_bj),
            "ks": float(sparse_ks),
            "clean": bool(sparse_clean),
        },
        "shape": {
            "score": "dp_eb_weighted_rank" if "dp_eb_weighted_rank" in profile_scores else "dp_mean_profile",
            "clean": bool(shape_clean),
            "graph_density": float(graph_density),
        },
        "regime_v1": v1_regime,
        "regime_v2": v2_regime,
        "soft_weights": {
            "static": float(weights[0]),
            "sparse": float(weights[1]),
            "shape": float(weights[2]),
        },
    }
    return scores, meta


def paper_anchor_score_bank(
    entries: Sequence[Dict],
    ds: str,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    cfg = PAPER_EBGAD_CONFIGS.get(ds.lower())
    if cfg is None:
        return {}, {"available": False}
    target_gamma = float(cfg["gamma"])
    entry = min(
        [e for e in entries if e.get("graph_type") == "original"],
        key=lambda e: abs(float(e["template_gamma"]) - target_gamma),
    )
    q = q_for_fit(entry, {"rho": float(cfg["rho"]), "kappa": float(cfg["kappa"])})
    static = horizon_score_bank(entry, q, [np.inf])
    j = static["J@inf"]
    r = static["R@inf"]
    selected = j if cfg["score"] == "J" else r
    scores = {
        "h9_anchor": selected.astype(np.float32, copy=False),
        "h9_anchor_J": j.astype(np.float32, copy=False),
        "h9_anchor_R": r.astype(np.float32, copy=False),
    }
    return scores, {
        "available": True,
        "entry_key": entry["key"],
        "target_gamma": target_gamma,
        "actual_gamma": float(entry["template_gamma"]),
        "rho": float(cfg["rho"]),
        "kappa": float(cfg["kappa"]),
        "score": cfg["score"],
    }


def h9_anchor_override_score_bank(
    anchor_scores: Dict[str, np.ndarray],
    anchor_meta: Dict,
    profile_scores: Dict[str, np.ndarray],
    profile_pvalues: Dict[str, np.ndarray],
    two_groups_scores: Dict[str, np.ndarray],
    two_groups_summary: Dict,
    two_groups_meta: Dict,
    regime_scores: Dict[str, np.ndarray],
    regime_meta: Dict,
    graph_density: float,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    if "h9_anchor" not in anchor_scores:
        return {}, {"available": False}

    anchor = anchor_scores["h9_anchor"]
    sparse = profile_scores.get(
        "dp_ebmix_posterior",
        profile_scores.get("dp_profile_mean_logp", anchor),
    )
    entropy = two_groups_scores.get("tg_component_entropy", anchor)
    dense_shape = regime_scores.get(
        "h8_shape",
        profile_scores.get("dp_eb_weighted_rank", anchor),
    )

    sparse_p = profile_pvalues.get(
        "dp_eb_acat",
        profile_pvalues.get("dp_profile_acat"),
    )
    sparse_bj = (
        berk_jones_stat(median_calibrated_pvalues(sparse_p))
        if sparse_p is not None else 0.0
    )
    sparse_ks = ks_null_deviation(np.nan_to_num(
        sparse, nan=0.0, posinf=0.0, neginf=0.0,
    ))
    sparse_clean = (
        20.0 <= sparse_bj <= 100.0
        and ks_null_deviation(np.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0)) < 0.20
        and sparse_ks < 0.35
        and anchor_meta.get("score") == "J"
    )

    tg_selectors = two_groups_summary.get("selectors", {}) if two_groups_summary else {}
    tg_tail = tg_selectors.get("tail_stable", {})
    tg_graph = tg_selectors.get("graph_tail", {})
    tg_pi = float(two_groups_meta.get("pi", 0.0)) if two_groups_meta else 0.0
    entropy_clean = (
        "tg_component_entropy" in two_groups_scores
        and anchor_meta.get("score") == "R"
        and tg_tail.get("score") == "tg_component_entropy"
        and tg_graph.get("score") == "tg_component_entropy"
        and 0.50 <= tg_pi <= 0.95
    )

    shape_clean = (
        graph_density >= 100.0
        and anchor_meta.get("score") == "R"
    )

    if sparse_clean:
        selected = sparse
        selected_regime = "sparse-tail"
    elif entropy_clean:
        selected = entropy
        selected_regime = "scale-entropy"
    elif shape_clean:
        selected = dense_shape
        selected_regime = "dense-trajectory-shape"
    else:
        selected = anchor
        selected_regime = "anchor"

    anchor_rank = percentile_rank(anchor)
    sparse_rank = percentile_rank(sparse)
    entropy_rank = percentile_rank(entropy)
    dense_rank = percentile_rank(dense_shape)
    h9_maxrank = np.max(np.vstack([anchor_rank, sparse_rank, entropy_rank, dense_rank]), axis=0)

    scores = {
        "h9_anchor": np.asarray(anchor, dtype=np.float32),
        "h9_sparse": np.asarray(sparse, dtype=np.float32),
        "h9_entropy": np.asarray(entropy, dtype=np.float32),
        "h9_dense_shape": np.asarray(dense_shape, dtype=np.float32),
        "h9_anchor_sparse_maxrank": np.maximum(anchor_rank, sparse_rank).astype(np.float32, copy=False),
        "h9_anchor_entropy_maxrank": np.maximum(anchor_rank, entropy_rank).astype(np.float32, copy=False),
        "h9_anchor_shape_maxrank": np.maximum(anchor_rank, dense_rank).astype(np.float32, copy=False),
        "h9_four_way_maxrank": h9_maxrank.astype(np.float32, copy=False),
        "h9_override": np.asarray(selected, dtype=np.float32),
    }
    meta = {
        "available": True,
        "anchor": anchor_meta,
        "selected_regime": selected_regime,
        "sparse": {
            "clean": bool(sparse_clean),
            "bj": float(sparse_bj),
            "ks": float(sparse_ks),
            "score": "dp_ebmix_posterior" if "dp_ebmix_posterior" in profile_scores else "dp_profile_mean_logp",
        },
        "entropy": {
            "clean": bool(entropy_clean),
            "pi": float(tg_pi),
            "tail_score": tg_tail.get("score"),
            "graph_score": tg_graph.get("score"),
        },
        "dense_shape": {
            "clean": bool(shape_clean),
            "graph_density": float(graph_density),
            "score": "h8_shape",
        },
    }
    return scores, meta


def h10_adaptive_anchor_score_bank(
    anchor_scores: Dict[str, np.ndarray],
    anchor_meta: Dict,
    two_groups_scores: Dict[str, np.ndarray],
    two_groups_summary: Dict,
    two_groups_meta: Dict,
    regime_scores: Dict[str, np.ndarray],
    regime_meta: Dict,
    graph_density: float,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Adaptive H8 default with paper-anchor and dynamic rescues."""
    if "h9_anchor" not in anchor_scores or not regime_scores:
        return {}, {"available": False}

    anchor = np.asarray(anchor_scores["h9_anchor"], dtype=np.float64)
    base = np.asarray(
        regime_scores.get("h8_regime_v2", regime_scores.get("h8_static", anchor)),
        dtype=np.float64,
    )
    dense_soft = np.asarray(
        regime_scores.get("h8_soft_gate", regime_scores.get("h8_shape", base)),
        dtype=np.float64,
    )
    entropy = np.asarray(two_groups_scores.get("tg_component_entropy", base), dtype=np.float64)

    anchor_ks = ks_null_deviation(np.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0))
    static_meta = regime_meta.get("static", {}) if regime_meta else {}
    sparse_meta = regime_meta.get("sparse", {}) if regime_meta else {}
    static_reason = static_meta.get("reason", "")
    static_ks = float(static_meta.get("ks_value", 0.0))
    sparse_bj = float(sparse_meta.get("bj", 0.0))

    tg_selectors = two_groups_summary.get("selectors", {}) if two_groups_summary else {}
    tg_tail = tg_selectors.get("tail_stable", {})
    tg_graph = tg_selectors.get("graph_tail", {})
    tg_pi = float(two_groups_meta.get("pi", 0.0)) if two_groups_meta else 0.0
    entropy_clean = (
        "tg_component_entropy" in two_groups_scores
        and (anchor_meta.get("score") == "R" or static_meta.get("pick") == "R@inf")
        and tg_tail.get("score") == "tg_component_entropy"
        and tg_graph.get("score") == "tg_component_entropy"
        and 0.50 <= tg_pi <= 0.95
    )
    dense_clean = graph_density >= 100.0
    anchor_rescue = (
        anchor_meta.get("score") == "J"
        and (
            anchor_ks >= 0.60
            or ("global-J-KS" in static_reason and 0.30 <= anchor_ks <= 0.60)
        )
    )
    soft_ambiguity = (
        anchor_meta.get("score") == "J"
        and static_meta.get("pick") == "J@inf"
        and 0.45 <= static_ks <= 0.55
        and 20.0 <= sparse_bj <= 120.0
        and graph_density < 10.0
        and "h8_soft_gate" in regime_scores
    )

    if entropy_clean:
        selected = entropy
        selected_regime = "scale-entropy"
    elif dense_clean:
        selected = dense_soft
        selected_regime = "dense-soft-trajectory"
    elif anchor_rescue:
        selected = anchor
        selected_regime = "paper-anchor-rescue"
    else:
        selected = base
        selected_regime = "adaptive-h8"

    if soft_ambiguity and selected_regime == "adaptive-h8":
        selected_soft = dense_soft
        soft_regime = "ambiguous-static-soft"
    else:
        selected_soft = selected
        soft_regime = selected_regime

    base_rank = percentile_rank(base)
    anchor_rank = percentile_rank(anchor)
    entropy_rank = percentile_rank(entropy)
    dense_rank = percentile_rank(dense_soft)

    scores = {
        "h10_base_h8": base.astype(np.float32, copy=False),
        "h10_anchor_rescue": anchor.astype(np.float32, copy=False),
        "h10_entropy": entropy.astype(np.float32, copy=False),
        "h10_dense_soft": dense_soft.astype(np.float32, copy=False),
        "h10_base_anchor_maxrank": np.maximum(base_rank, anchor_rank).astype(np.float32, copy=False),
        "h10_base_dynamic_maxrank": np.max(
            np.vstack([base_rank, entropy_rank, dense_rank]), axis=0,
        ).astype(np.float32, copy=False),
        "h10_override": selected.astype(np.float32, copy=False),
        "h10_override_soft": selected_soft.astype(np.float32, copy=False),
    }
    meta = {
        "available": True,
        "selected_regime": selected_regime,
        "selected_soft_regime": soft_regime,
        "anchor_ks": float(anchor_ks),
        "static_reason": static_reason,
        "static_ks": float(static_ks),
        "anchor_rescue": bool(anchor_rescue),
        "entropy_clean": bool(entropy_clean),
        "dense_clean": bool(dense_clean),
        "soft_ambiguity": bool(soft_ambiguity),
    }
    return scores, meta


def maybe_dump_scores(
    args,
    ds: str,
    y: np.ndarray,
    mask: np.ndarray,
    banks: Dict[str, Dict[str, np.ndarray]],
    pvalue_banks: Dict[str, Dict[str, np.ndarray]] | None = None,
) -> None:
    if not getattr(args, "score_dump_dir", None):
        return
    from test_dynamic_score_bank import pvalue_aggregate_scores

    expanded: Dict[str, Dict[str, np.ndarray]] = {}
    for bank_name, bank in banks.items():
        merged = dict(bank)
        try:
            merged.update(aggregate_rank_scores(bank))
        except Exception:
            pass
        if pvalue_banks and pvalue_banks.get(bank_name):
            try:
                pv_scores, _ = pvalue_aggregate_scores(pvalue_banks[bank_name])
                merged.update(pv_scores)
            except Exception:
                pass
        expanded[bank_name] = merged
    names = []
    values = []
    n = len(y)
    for bank_name, bank in expanded.items():
        for score_name, score in sorted(bank.items()):
            arr = np.asarray(score)
            if arr.ndim != 1 or len(arr) != n:
                continue
            if not np.any(np.isfinite(arr)) or np.nanmax(arr) <= np.nanmin(arr):
                continue
            names.append("%s/%s" % (bank_name, score_name))
            values.append(np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    if not values:
        return
    if getattr(args, "out", None):
        stem = os.path.splitext(os.path.basename(args.out))[0]
    else:
        stem = "%s_%s" % (args.modeled_subspace, ds)
    os.makedirs(args.score_dump_dir, exist_ok=True)
    path = os.path.join(args.score_dump_dir, "%s_scores.npz" % stem)
    np.savez_compressed(
        path,
        y=np.asarray(y, dtype=np.int64),
        mask=np.asarray(mask, dtype=bool),
        names=np.asarray(names),
        scores=np.vstack(values).T.astype(np.float32, copy=False),
    )
    print("  dumped score matrix %s shape=(%d,%d)" % (path, n, len(names)), flush=True)


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
    paper_cfg = PAPER_EBGAD_CONFIGS.get(ds.lower())
    if paper_cfg is not None:
        template_gammas = sorted({float(g) for g in template_gammas} | {float(paper_cfg["gamma"])})
    graph_types = parse_names(args.graph_types)
    horizons = canonical_horizons(
        parse_grid(args.horizons),
        include_inf=not args.finite_only_bank,
    )
    if args.finite_only_bank:
        horizons = [x for x in horizons if np.isfinite(x)]
    finite_horizons = [x for x in horizons if np.isfinite(x)]
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
    fit = fit_best_entry(entries, [np.inf], kappas, args.gamma_jacobian)
    entry = entry_for_fit(entries, fit)
    q = q_for_fit(entry, fit)
    references = node_unsupervised_references(data, entry["delta"])

    stationary_scores = horizon_score_bank(entry, q, [np.inf])
    stationary_summary = evaluate_score_bank(y, mask, stationary_scores, references=references)

    print("  computing path-dissipation scores", flush=True)
    path_scores, path_meta = path_dissipation_score_bank(
        entry,
        q,
        finite_horizons,
        n_quad=args.path_quad,
        tail_factor=args.path_tail_factor,
        tail_cap=args.path_tail_cap,
    )
    path_pvalues = path_dissipation_pvalue_bank(
        entry,
        q,
        finite_horizons,
        path_scores,
        tail_factor=args.path_tail_factor,
        tail_cap=args.path_tail_cap,
    )
    path_summary = evaluate_score_bank(
        y, mask, path_scores, references=references, pvalue_bank=path_pvalues,
    )

    print("  building robust path-band aggregate", flush=True)
    robust_path_scores, robust_path_meta = robust_path_aggregate_score_bank(
        path_scores,
        path_pvalues,
        min_bands=args.robust_min_bands,
        max_bands=args.robust_max_bands,
    )
    robust_path_summary = evaluate_score_bank(
        y, mask, robust_path_scores, references=references,
    )

    print("  fitting two-groups path p-value model", flush=True)
    two_groups_scores, two_groups_meta = two_groups_path_score_bank(
        path_pvalues,
        pi_init=args.tg_pi_init,
        pi_max=args.tg_pi_max,
        weight_alpha=args.tg_weight_alpha,
    )
    two_groups_summary = evaluate_score_bank(
        y, mask, two_groups_scores, references=references,
    )

    print("  fitting latent relaxation-time mixture", flush=True)
    mixture_scores, mixture_meta = latent_time_mixture_score_bank(
        entry,
        q,
        horizons,
        dirichlet_alpha=args.mixture_alpha,
    )
    mixture_summary = evaluate_score_bank(y, mask, mixture_scores, references=references)

    print("  selecting dynamic trajectory profile", flush=True)
    profile_scores, profile_pvalues, profile_meta = dynamic_profile_selector_score_bank(
        path_scores,
        path_pvalues,
        mixture_scores,
        two_groups_scores,
        stationary_scores,
    )
    profile_summary = evaluate_score_bank(
        y, mask, profile_scores, references=references, pvalue_bank=profile_pvalues,
    )

    combined_scores = {}
    combined_scores.update(path_scores)
    combined_scores.update(robust_path_scores)
    combined_scores.update(mixture_scores)
    combined_scores.update(two_groups_scores)
    combined_scores.update(profile_scores)
    combined_pvalues = dict(path_pvalues)
    combined_pvalues.update(profile_pvalues)
    dynamic_summary = evaluate_score_bank(
        y, mask, combined_scores, references=references, pvalue_bank=combined_pvalues,
    )

    print("  applying dynamic regime gate", flush=True)
    regime_scores, regime_meta = regime_gated_score_bank(
        stationary_scores,
        profile_scores,
        profile_pvalues,
        references,
        graph_density=density,
    )
    regime_summary = evaluate_score_bank(
        y, mask, regime_scores, references=references,
    )

    print("  applying anchored dynamic override", flush=True)
    anchor_scores, anchor_meta = paper_anchor_score_bank(entries, ds)
    anchored_scores, anchored_meta = h9_anchor_override_score_bank(
        anchor_scores,
        anchor_meta,
        profile_scores,
        profile_pvalues,
        two_groups_scores,
        two_groups_summary,
        two_groups_meta,
        regime_scores,
        regime_meta,
        graph_density=density,
    )
    h10_scores, h10_meta = h10_adaptive_anchor_score_bank(
        anchor_scores,
        anchor_meta,
        two_groups_scores,
        two_groups_summary,
        two_groups_meta,
        regime_scores,
        regime_meta,
        graph_density=density,
    )
    anchored_scores.update(h10_scores)
    anchored_meta["h10"] = h10_meta
    anchored_summary = evaluate_score_bank(
        y, mask, anchored_scores, references=references,
    )

    print_bank_summary("stationary", stationary_summary)
    print_bank_summary("path dissipation", path_summary)
    print_bank_summary("robust path", robust_path_summary)
    print_bank_summary("two-groups path", two_groups_summary)
    print_bank_summary("latent gamma mix", mixture_summary)
    print_bank_summary("dynamic profile", profile_summary)
    print_bank_summary("dynamic combined", dynamic_summary)
    print_bank_summary("regime gated", regime_summary)
    print_bank_summary("anchored override", anchored_summary)

    maybe_dump_scores(
        args,
        ds,
        y,
        mask,
        {
            "stationary": stationary_scores,
            "path": path_scores,
            "robust_path": robust_path_scores,
            "two_groups": two_groups_scores,
            "mixture": mixture_scores,
            "profile": profile_scores,
            "dynamic": combined_scores,
            "regime": regime_scores,
            "anchored": anchored_scores,
        },
        pvalue_banks={
            "path": path_pvalues,
            "profile": profile_pvalues,
            "dynamic": combined_pvalues,
        },
    )

    reported = REPORTED.get(ds.lower(), 0.0)
    base_name, base_auc = BASELINES.get(ds.lower(), ("?", 0.0))
    if reported or base_auc:
        print("  reference: reported=%.1f%%  %s=%.1f%%" % (
            reported, base_name, base_auc,
        ), flush=True)

    return {
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
            "finite_only_bank": bool(args.finite_only_bank),
            "kappas": kappas,
        },
        "stationary_fit": summarize_fit(fit),
        "stationary": stationary_summary,
        "path_dissipation": path_summary,
        "path_dissipation_meta": path_meta,
        "robust_path": robust_path_summary,
        "robust_path_meta": robust_path_meta,
        "two_groups_path": two_groups_summary,
        "two_groups_path_meta": two_groups_meta,
        "latent_mixture": mixture_summary,
        "latent_mixture_meta": mixture_meta,
        "dynamic_profile": profile_summary,
        "dynamic_profile_meta": profile_meta,
        "dynamic_combined": dynamic_summary,
        "regime_gated": regime_summary,
        "regime_gated_meta": regime_meta,
        "anchored_override": anchored_summary,
        "anchored_override_meta": anchored_meta,
    }


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
    parser.add_argument("--path-quad", type=int, default=4)
    parser.add_argument("--path-tail-factor", type=float, default=16.0)
    parser.add_argument("--path-tail-cap", type=float, default=1000.0)
    parser.add_argument("--mixture-alpha", type=float, default=1.05)
    parser.add_argument("--tg-pi-init", type=float, default=0.05)
    parser.add_argument("--tg-pi-max", type=float, default=0.20)
    parser.add_argument("--tg-weight-alpha", type=float, default=1.05)
    parser.add_argument("--robust-min-bands", type=int, default=3)
    parser.add_argument("--robust-max-bands", type=int, default=8)
    parser.add_argument("--finite-only-bank", action="store_true",
                        help="Exclude Gamma=inf from dynamic mixture/profile banks for forced-bank ablations.")
    parser.add_argument("--score-dump-dir", default=None,
                        help="Optional directory for compressed per-node score matrices.")
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
        outfile = "results/dynamic_hypotheses_%s.json" % safe_name
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved %s" % outfile, flush=True)


if __name__ == "__main__":
    main()
