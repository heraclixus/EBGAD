"""DGraph cached-eigenpair H6 dynamic-profile prototype.

The generic H6 runner recomputes graph eigenpairs, which is not appropriate for
DGraph.  This script mirrors the H6 score construction using precomputed
labeled-subgraph eigenpairs from ``cache/dgraph_eigen``.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
from typing import Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_utils import load_data
from soc.prior_optimizer import compute_spectral_energy
from test_dynamic_score_bank import (
    KAPPAS,
    canonical_horizons,
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    format_horizon,
    horizon_score_bank,
    parse_grid,
    print_bank_summary,
    q_for_fit,
    summarize_fit,
)
from test_dynamic_hypotheses import (
    dynamic_profile_selector_score_bank,
    latent_time_mixture_score_bank,
    maybe_dump_scores,
    path_dissipation_pvalue_bank,
    path_dissipation_score_bank,
    robust_path_aggregate_score_bank,
    two_groups_path_score_bank,
)


def normalize_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zscore":
        return StandardScaler().fit_transform(x).astype(np.float32)
    if mode == "minmax":
        return MinMaxScaler().fit_transform(x).astype(np.float32)
    if mode == "raw":
        return x.astype(np.float32)
    raise ValueError(f"Unknown normalization mode {mode!r}")


def compute_template(x: np.ndarray, V: np.ndarray, lam_L: np.ndarray, gamma: float) -> np.ndarray:
    smoother = 1.0 / np.maximum(float(gamma) ** 2 + lam_L, 1e-12)
    return V @ (smoother[:, None] * (V.T @ x))


def build_entries(
    x_sub: np.ndarray,
    V: np.ndarray,
    lam_L: np.ndarray,
    gammas: List[float],
    cache_label: str,
) -> List[Dict]:
    entries = []
    n, d = x_sub.shape
    for gamma in gammas:
        print(f"  building residuals gamma={gamma:g}", flush=True)
        template = compute_template(x_sub, V, lam_L, gamma)
        delta = x_sub - template
        S = compute_spectral_energy(delta, V)
        entries.append({
            "key": "%s/gamma=%g" % (cache_label, gamma),
            "graph_type": "cached_labeled",
            "template_gamma": float(gamma),
            "V": V,
            "lam_L": lam_L,
            "delta": delta,
            "S": S,
            "n": int(n),
            "d": int(d),
        })
    return entries


def run(args) -> Dict:
    data = load_data("dgraph")
    x_raw = data.x.detach().cpu().float().numpy()
    y = data.y.detach().cpu().numpy()
    labeled = (y >= 0) & (y <= 1)
    idx = np.where(labeled)[0]
    y_sub = y[idx]
    mask = (y_sub >= 0) & (y_sub <= 1)
    print("DGraph full n=%d d=%d labeled=%d anom=%.2f%%" % (
        x_raw.shape[0], x_raw.shape[1], len(idx), 100.0 * float(np.mean(y_sub == 1)),
    ), flush=True)

    x = normalize_features(x_raw, args.normalize)
    x_sub = x[idx]
    del x_raw, x

    cache_path = os.path.join(args.cache_dir, args.cache_file)
    print("Loading cache %s" % cache_path, flush=True)
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    V = cache["V"].detach().cpu().numpy().astype(np.float32, copy=False)
    lam_L = cache["lam_L"].detach().cpu().numpy().astype(np.float64, copy=False)
    print("  V=%s lam=%s lam_range=[%.3g, %.3g] method=%s" % (
        tuple(V.shape), tuple(lam_L.shape), float(np.min(lam_L)), float(np.max(lam_L)),
        cache.get("method", "unknown"),
    ), flush=True)
    if V.shape[0] != x_sub.shape[0]:
        raise ValueError("Cache row count %d does not match labeled nodes %d" % (V.shape[0], x_sub.shape[0]))

    gammas = parse_grid(args.template_gammas)
    horizons = canonical_horizons(
        parse_grid(args.horizons),
        include_inf=not args.finite_only_bank,
    )
    if args.finite_only_bank:
        horizons = [x for x in horizons if np.isfinite(x)]
    finite_horizons = [x for x in horizons if np.isfinite(x)]
    kappas = parse_grid(args.kappas)

    cache_label = os.path.splitext(os.path.basename(args.cache_file))[0]
    entries = build_entries(x_sub, V, lam_L, gammas, cache_label)

    print("Fitting stationary EB geometry", flush=True)
    fit = fit_best_entry(entries, [np.inf], kappas, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    q = q_for_fit(entry, fit)

    print("Computing H6 banks", flush=True)
    stationary_scores = horizon_score_bank(entry, q, [np.inf])
    stationary_summary = evaluate_score_bank(y_sub, mask, stationary_scores, references=None)

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
        y_sub, mask, path_scores, references=None, pvalue_bank=path_pvalues,
    )

    robust_path_scores, robust_path_meta = robust_path_aggregate_score_bank(
        path_scores, path_pvalues, min_bands=args.robust_min_bands, max_bands=args.robust_max_bands,
    )
    robust_path_summary = evaluate_score_bank(y_sub, mask, robust_path_scores, references=None)

    two_groups_scores, two_groups_meta = two_groups_path_score_bank(
        path_pvalues,
        pi_init=args.tg_pi_init,
        pi_max=args.tg_pi_max,
        weight_alpha=args.tg_weight_alpha,
    )
    two_groups_summary = evaluate_score_bank(y_sub, mask, two_groups_scores, references=None)

    mixture_scores, mixture_meta = latent_time_mixture_score_bank(
        entry, q, horizons, dirichlet_alpha=args.mixture_alpha,
    )
    mixture_summary = evaluate_score_bank(y_sub, mask, mixture_scores, references=None)

    profile_scores, profile_pvalues, profile_meta = dynamic_profile_selector_score_bank(
        path_scores, path_pvalues, mixture_scores, two_groups_scores, stationary_scores,
    )
    profile_summary = evaluate_score_bank(
        y_sub, mask, profile_scores, references=None, pvalue_bank=profile_pvalues,
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
        y_sub, mask, combined_scores, references=None, pvalue_bank=combined_pvalues,
    )

    print_bank_summary("stationary", stationary_summary)
    print_bank_summary("path dissipation", path_summary)
    print_bank_summary("robust path", robust_path_summary)
    print_bank_summary("two-groups path", two_groups_summary)
    print_bank_summary("latent gamma mix", mixture_summary)
    print_bank_summary("dynamic profile", profile_summary)
    print_bank_summary("dynamic combined", dynamic_summary)

    maybe_dump_scores(
        args,
        "dgraph",
        y_sub,
        mask,
        {
            "stationary": stationary_scores,
            "path": path_scores,
            "robust_path": robust_path_scores,
            "two_groups": two_groups_scores,
            "mixture": mixture_scores,
            "profile": profile_scores,
            "dynamic": combined_scores,
        },
    )

    return {
        "dataset": "dgraph",
        "graph": {
            "nodes_full": int(data.x.shape[0]),
            "nodes_labeled": int(len(idx)),
            "dim": int(data.x.shape[1]),
            "cache_dir": args.cache_dir,
            "cache_file": args.cache_file,
            "normalize": args.normalize,
            "template_gammas": gammas,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/dgraph_eigen")
    parser.add_argument("--cache-file", default="labeled_original_k128.pt")
    parser.add_argument("--normalize", default="zscore", choices=["zscore", "minmax", "raw"])
    parser.add_argument("--template-gammas", default="0.5")
    parser.add_argument("--horizons", default="0.1,1,10,inf")
    parser.add_argument("--kappas", default=",".join("%g" % k for k in KAPPAS))
    parser.add_argument("--path-quad", type=int, default=2)
    parser.add_argument("--path-tail-factor", type=float, default=8.0)
    parser.add_argument("--path-tail-cap", type=float, default=100.0)
    parser.add_argument("--mixture-alpha", type=float, default=1.05)
    parser.add_argument("--tg-pi-init", type=float, default=0.05)
    parser.add_argument("--tg-pi-max", type=float, default=0.20)
    parser.add_argument("--tg-weight-alpha", type=float, default=1.05)
    parser.add_argument("--robust-min-bands", type=int, default=2)
    parser.add_argument("--robust-max-bands", type=int, default=6)
    parser.add_argument("--finite-only-bank", action="store_true",
                        help="Exclude Gamma=inf from dynamic mixture/profile banks for forced-bank ablations.")
    parser.add_argument("--out", default="results/dynamic_hypotheses_h6_dgraph.json")
    parser.add_argument("--score-dump-dir", default=None,
                        help="Dump all bank score vectors as npz for AP computation.")
    args = parser.parse_args()

    result = run(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dgraph": result}, f, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
