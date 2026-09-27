"""Prototype multiscale GOU horizon aggregation.

This script tests a multiscale product-of-experts aggregation over GOU horizons:

    p^M(delta | phi, w) ∝ prod_m p(delta | phi, Gamma_m)^{w_m}

For GOU horizon Gamma_m, the per-mode precision is

    tau_j(Gamma_m) = q_j / (1 - exp(-2 Gamma_m q_j)),

with tau_j(inf) = q_j.  The normalized Gaussian product-of-experts has
effective precision

    tau_eff,j = sum_m w_m tau_j(Gamma_m),  w ∈ simplex.

For fixed (template gamma, rho, kappa), EB weight selection is a concave
optimization over w:

    max_w -1/2 sum_j S_j tau_eff,j + d/2 sum_j log tau_eff,j.
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
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import kstest
from sklearn.metrics import roc_auc_score

from data_utils import load_data
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    compute_spectral_energy,
    gamma_jacobian_correction,
)
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig


KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
TEMPLATE_GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]
DEFAULT_HORIZONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, np.inf]

REPORTED = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1, "yelpchi": 72.0,
    "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}
BASELINES = {
    "weibo": ("ANOM.", 95.0), "reddit": ("CoLA", 60.3), "amazon": ("DIF", 78.1),
    "yelpchi": ("LOF", 57.2), "blogcatalog": ("TAM", 82.5), "facebook": ("TAM", 91.4),
    "acm": ("TAM", 88.8), "elliptic": ("CoLA", 65.3),
    "elliptic_plus_plus": ("CoLA", 62.9), "t_finance": ("ECOD", 83.5),
}


def parse_grid(text: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        values.append(np.inf if token in {"inf", "infty", "infinity"} else float(token))
    return values


def format_horizon(gamma: float) -> str:
    return "inf" if np.isinf(gamma) else ("%g" % gamma)


def ks_null_deviation(scores: np.ndarray) -> float:
    s = scores[scores > 0]
    if len(s) < 10:
        return 0.0
    mu, var = float(s.mean()), float(s.var())
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k = max(0.01, 2.0 * mu**2 / var)
    a = var / (2.0 * mu)
    try:
        stat, _ = kstest(s / a, "chi2", args=(k,))
        return float(stat)
    except Exception:
        return 0.0


def compute_graph_stats(data) -> Tuple[float, float]:
    edge_index = data.edge_index.cpu().numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = edge_index.shape[1] // 2 / max(n, 1)
    n_sample = min(100000, edge_index.shape[1])
    rng = np.random.RandomState(42)
    idx = rng.choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(
        x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1,
    )
    return float(cos.mean()), float(density)


def density_adjusted_template_gammas(h: float, density: float) -> List[float]:
    density_factor = np.sqrt(max(density, 10.0) / 10.0)
    gamma_center = np.clip((0.5 + h) / density_factor, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    candidates = [g for g in TEMPLATE_GAMMAS if g <= max_gamma]
    if not candidates:
        return [TEMPLATE_GAMMAS[0]]
    if len(candidates) > 3:
        candidates.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        candidates = sorted(candidates[:3])
    return candidates


def gou_precisions(q: np.ndarray, horizons: Sequence[float]) -> np.ndarray:
    q_safe = np.maximum(q, 1e-12)
    rows = []
    for horizon in horizons:
        if np.isinf(horizon):
            rows.append(q_safe)
            continue
        denom = -np.expm1(-2.0 * float(horizon) * q_safe)
        rows.append(q_safe / np.maximum(denom, 1e-12))
    return np.vstack(rows)


def poe_marginal_log_likelihood(S: np.ndarray, precision: np.ndarray, d: int) -> float:
    p = np.maximum(precision, 1e-12)
    return float(-0.5 * np.dot(S, p) + (d / 2.0) * np.sum(np.log(p)))


def optimize_poe_weights(S: np.ndarray, tau: np.ndarray, d: int) -> Tuple[np.ndarray, float]:
    """Optimize simplex weights for a fixed bank of horizon precisions."""
    n_horizons = tau.shape[0]
    if n_horizons == 1:
        w = np.ones(1)
        return w, poe_marginal_log_likelihood(S, tau[0], d)

    def objective(w: np.ndarray) -> float:
        p = np.maximum(w @ tau, 1e-12)
        return float(0.5 * np.dot(S, p) - (d / 2.0) * np.sum(np.log(p)))

    def gradient(w: np.ndarray) -> np.ndarray:
        p = np.maximum(w @ tau, 1e-12)
        per_mode = 0.5 * S - d / (2.0 * p)
        return tau @ per_mode

    starts = [np.full(n_horizons, 1.0 / n_horizons)]
    starts.extend(np.eye(n_horizons))
    bounds = [(0.0, 1.0)] * n_horizons
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones_like(w)}]

    best_w = starts[0]
    best_obj = objective(best_w)
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10, "disp": False},
        )
        candidate = result.x if result.x is not None else start
        candidate = np.maximum(candidate, 0.0)
        if candidate.sum() <= 1e-12:
            candidate = start
        candidate = candidate / candidate.sum()
        obj = objective(candidate)
        if obj < best_obj:
            best_obj = obj
            best_w = candidate

    p_eff = best_w @ tau
    return best_w, poe_marginal_log_likelihood(S, p_eff, d)


def optimize_rho_kappa_poe(
    S: np.ndarray,
    lam_L: np.ndarray,
    horizons: Sequence[float],
    d: int,
    kappas: Sequence[float],
    nu: float = 1.0,
) -> Dict:
    best: Dict = {"ml": -np.inf}

    for kappa in kappas:

        def fit_rho(rho: float) -> Tuple[float, np.ndarray, np.ndarray]:
            q = Q_rho_eigenvalues(lam_L, float(rho), float(kappa), nu=nu)
            tau = gou_precisions(q, horizons)
            weights, ml = optimize_poe_weights(S, tau, d)
            return ml, weights, weights @ tau

        def negative_profile_ml(rho: float) -> float:
            ml, _, _ = fit_rho(rho)
            return -ml

        result = minimize_scalar(
            negative_profile_ml,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 80},
        )

        rho_candidates = [0.0, 1.0]
        if result.x is not None and np.isfinite(result.x):
            rho_candidates.append(float(result.x))
        for rho in rho_candidates:
            ml, weights, precision = fit_rho(rho)
            if ml > best["ml"]:
                best = {
                    "ml": float(ml),
                    "rho": float(np.clip(rho, 0.0, 1.0)),
                    "kappa": float(kappa),
                    "weights": weights,
                    "precision": precision,
                }

    return best


def score_nodes(delta: np.ndarray, V: np.ndarray, precision: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta_hat = V.T @ delta
    weighted = np.sqrt(np.maximum(precision, 1e-12))[:, None] * delta_hat
    je = np.sum((V @ weighted) ** 2, axis=1)
    total = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    return je, je / total


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Return deterministic percentile ranks in [0, 1] for unsupervised fusion."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    if len(order) <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64) / float(len(order) - 1)
    return ranks


def select_metric(metrics: Dict, selector: str) -> Tuple[str, float]:
    if selector == "ks":
        return metrics["score_ks"], metrics["auc_ks_selected"]
    if selector == "rank_avg":
        return "rank_avg", metrics["auc_rank_avg"]
    if selector == "rank_max":
        return "rank_max", metrics["auc_rank_max"]
    if selector == "j":
        return "J*", metrics["auc_je"]
    if selector == "r":
        return "R", metrics["auc_jr"]
    raise ValueError(f"Unknown score selector: {selector!r}")


def evaluate_scores(
    y: np.ndarray,
    mask: np.ndarray,
    delta: np.ndarray,
    V: np.ndarray,
    precision: np.ndarray,
    selector: str,
) -> Dict:
    je, jr = score_nodes(delta, V, precision)
    auc_je = roc_auc_score(y[mask], je[mask])
    auc_jr = roc_auc_score(y[mask], jr[mask])
    ks_je = ks_null_deviation(je[mask])
    ks_jr = ks_null_deviation(jr[mask])
    if ks_je > ks_jr:
        score_ks, auc_ks_selected, ks_value = "J*", auc_je, ks_je
    else:
        score_ks, auc_ks_selected, ks_value = "R", auc_jr, ks_jr

    rank_je = percentile_rank(je)
    rank_jr = percentile_rank(jr)
    rank_avg = 0.5 * (rank_je + rank_jr)
    rank_max = np.maximum(rank_je, rank_jr)
    auc_rank_avg = roc_auc_score(y[mask], rank_avg[mask])
    auc_rank_max = roc_auc_score(y[mask], rank_max[mask])
    auc_oracle = max(auc_je, auc_jr, auc_rank_avg, auc_rank_max)
    oracle_score = max(
        [("J*", auc_je), ("R", auc_jr), ("rank_avg", auc_rank_avg), ("rank_max", auc_rank_max)],
        key=lambda item: item[1],
    )[0]
    selected_score, auc_selected = select_metric(
        {
            "score_ks": score_ks,
            "auc_ks_selected": auc_ks_selected,
            "auc_rank_avg": auc_rank_avg,
            "auc_rank_max": auc_rank_max,
            "auc_je": auc_je,
            "auc_jr": auc_jr,
        },
        selector,
    )

    return {
        "auc_selected": float(auc_selected),
        "score": selected_score,
        "selector": selector,
        "auc_ks_selected": float(auc_ks_selected),
        "score_ks": score_ks,
        "auc_je": float(auc_je),
        "auc_jr": float(auc_jr),
        "auc_rank_avg": float(auc_rank_avg),
        "auc_rank_max": float(auc_rank_max),
        "auc_oracle": float(auc_oracle),
        "oracle_score": oracle_score,
        "ks": float(ks_value),
        "ks_je": float(ks_je),
        "ks_jr": float(ks_jr),
    }


def build_trainer_config(
    data,
    gamma: float,
    pca_dim: int | None,
    truncated_k: int | None,
    modeled_subspace: str,
):
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
        template_type="low",
        graph_type="original",
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


def prepare_template_cache(
    data,
    template_gammas: Iterable[float],
    pca_dim: int | None,
    truncated_k: int | None,
    modeled_subspace: str,
    device: str,
) -> Dict[float, Dict]:
    cache: Dict[float, Dict] = {}
    for gamma in template_gammas:
        print("    preparing template gamma=%g" % gamma, flush=True)
        trainer = SOCTrainer(build_trainer_config(
            data, gamma, pca_dim, truncated_k, modeled_subspace,
        ))
        trainer.train(data, device=device)
        V = trainer.V.detach().cpu().numpy()
        lam_L = trainer.lam_L_model.detach().cpu().numpy()
        x = trainer.x_T.detach().cpu().numpy()
        template = trainer.template.detach().cpu().numpy()
        delta = x - template
        S = compute_spectral_energy(delta, V)
        n, d = x.shape
        cache[gamma] = {
            "V": V,
            "lam_L": lam_L,
            "delta": delta,
            "S": S,
            "n": int(n),
            "d": int(d),
        }
    return cache


def fit_across_template_gammas(
    cache: Dict[float, Dict],
    horizons: Sequence[float],
    kappas: Sequence[float],
    use_gamma_jacobian: bool,
) -> Dict:
    best: Dict = {"ml": -np.inf}
    for template_gamma, gd in cache.items():
        fit = optimize_rho_kappa_poe(
            gd["S"], gd["lam_L"], horizons, gd["d"], kappas,
        )
        ml = fit["ml"]
        if use_gamma_jacobian:
            ml += gamma_jacobian_correction(gd["lam_L"], template_gamma, gd["d"])
        if ml > best["ml"]:
            best = {
                **fit,
                "ml": float(ml),
                "template_gamma": float(template_gamma),
                "horizons": list(horizons),
            }
    return best


def summarize_weights(horizons: Sequence[float], weights: np.ndarray, cutoff: float = 1e-3) -> Dict[str, float]:
    return {
        format_horizon(h): float(w)
        for h, w in zip(horizons, weights)
        if float(w) >= cutoff
    }


def attach_evaluation(
    result: Dict,
    cache: Dict[float, Dict],
    y: np.ndarray,
    mask: np.ndarray,
    selector: str,
) -> Dict:
    gd = cache[result["template_gamma"]]
    metrics = evaluate_scores(y, mask, gd["delta"], gd["V"], result["precision"], selector)
    out = dict(result)
    out.update(metrics)
    out["weights_summary"] = summarize_weights(result["horizons"], result["weights"])
    out["weights"] = [float(w) for w in result["weights"]]
    out["horizons"] = [format_horizon(h) for h in result["horizons"]]
    out.pop("precision", None)
    return out


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
    print("  nodes=%d dim=%d edges=%d h=%.3f density=%.1f templates=%s pca=%s" % (
        data.x.shape[0], data.x.shape[1], data.edge_index.shape[1],
        h, density, template_gammas, pca_dim,
    ), flush=True)

    cache = prepare_template_cache(
        data,
        template_gammas,
        pca_dim=pca_dim,
        truncated_k=args.truncated_k,
        modeled_subspace=args.modeled_subspace,
        device=args.device,
    )
    kappas = parse_grid(args.kappas)
    horizons = parse_grid(args.horizons)
    finite_horizons = [h for h in horizons if not np.isinf(h)]

    print("  fitting stationary Gamma=inf", flush=True)
    stationary = fit_across_template_gammas(
        cache, [np.inf], kappas, use_gamma_jacobian=args.gamma_jacobian,
    )
    stationary_eval = attach_evaluation(stationary, cache, y, mask, args.selector)

    print("  fitting best single finite Gamma", flush=True)
    single_results = []
    for horizon in finite_horizons:
        fit = fit_across_template_gammas(
            cache, [horizon], kappas, use_gamma_jacobian=args.gamma_jacobian,
        )
        single_results.append(attach_evaluation(fit, cache, y, mask, args.selector))
    best_single = max(single_results, key=lambda r: r["ml"]) if single_results else stationary_eval

    print("  fitting multiscale GOU product-of-experts", flush=True)
    multiscale = fit_across_template_gammas(
        cache, horizons, kappas, use_gamma_jacobian=args.gamma_jacobian,
    )
    multiscale_eval = attach_evaluation(multiscale, cache, y, mask, args.selector)

    for name, r in [
        ("stationary", stationary_eval),
        ("single finite", best_single),
        ("multiscale", multiscale_eval),
    ]:
        print("  %-13s %s=%.1f%% KS(%s)=%.1f%% J*=%.1f%% R=%.1f%% avg=%.1f%% max=%.1f%% tmpl=%s rho=%.3f kap=%.2f w=%s" % (
            name,
            r["score"],
            r["auc_selected"] * 100,
            r["score_ks"],
            r["auc_ks_selected"] * 100,
            r["auc_je"] * 100,
            r["auc_jr"] * 100,
            r["auc_rank_avg"] * 100,
            r["auc_rank_max"] * 100,
            "%g" % r["template_gamma"],
            r["rho"],
            r["kappa"],
            r["weights_summary"],
        ), flush=True)

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
            "template_gammas": list(template_gammas),
            "pca_dim": pca_dim,
            "modeled_subspace": args.modeled_subspace,
            "gamma_jacobian": args.gamma_jacobian,
            "selector": args.selector,
        },
        "stationary": stationary_eval,
        "best_single_finite": best_single,
        "multiscale_poe": multiscale_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name or comma-separated list.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--template-gammas", default="density",
                        help="'density' or comma grid, e.g. 0.2,0.5,1.0")
    parser.add_argument("--horizons", default=",".join(format_horizon(g) for g in DEFAULT_HORIZONS))
    parser.add_argument("--kappas", default=",".join("%g" % k for k in KAPPAS))
    parser.add_argument("--pca-threshold", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--truncated-k", type=int, default=None)
    parser.add_argument("--modeled-subspace", default="legacy",
                        choices=["legacy", "drop_nullspace"])
    parser.add_argument("--selector", default="ks",
                        choices=["ks", "rank_avg", "rank_max", "j", "r"])
    parser.add_argument("--gamma-jacobian", action="store_true",
                        help="Add residual-transform Jacobian when comparing template gamma values.")
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
        outfile = "results/multiscale_gou_poe_%s.json" % safe_name
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved %s" % outfile, flush=True)


if __name__ == "__main__":
    main()
