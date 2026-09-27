"""Permutation test for node-order leakage in the bank machinery.

The GOU/EB model is permutation-equivariant: relabeling nodes permutes
scores. Any AUC change under a random node permutation is an order-dependent
artifact (stable-sort tie-breaking on degenerate p-values), not signal.

Suspects: YelpChi dp_ebmix_posterior (Table 1/2 cell 95.3) and Facebook
tg_component_entropy (91.4), because index-AUC is 94.36 / 97.92 there and
kappa-saturated model p-values collapse to constants.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

np.random.seed(0)

from data_utils import load_data
from test_dynamic_score_bank import (
    horizon_score_bank,
    prepare_model_entries,
    q_for_fit,
    safe_auc,
    sanitize_pvalues,
)
from test_dynamic_hypotheses import (
    dynamic_profile_selector_score_bank,
    latent_time_mixture_score_bank,
    path_dissipation_pvalue_bank,
    path_dissipation_score_bank,
    two_groups_path_score_bank,
)

FINITE_HORIZONS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]


def permute_data(data, perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    data = data.clone()
    data.x = data.x[torch.from_numpy(perm)]
    data.y = data.y[torch.from_numpy(perm)]
    ei = data.edge_index.numpy()
    data.edge_index = torch.from_numpy(inv[ei])
    return data


def run(ds, cfg, permute):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    n = data.x.shape[0]
    if permute:
        rng = np.random.RandomState(123)
        perm = rng.permutation(n)
        data = permute_data(data, perm)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    y_eval = y[mask]

    entries = prepare_model_entries(
        data, graph_types=["original"], template_gammas=[cfg["gamma"]],
        pca_dim=cfg["pca"], truncated_k=cfg["tk"],
        modeled_subspace=cfg["sub"], template_type="low", device="cpu")
    entry = entries[0]
    q = q_for_fit(entry, {"rho": cfg["rho"], "kappa": cfg["kappa"]})

    path_scores, _ = path_dissipation_score_bank(entry, q, FINITE_HORIZONS)
    path_pvalues = path_dissipation_pvalue_bank(
        entry, q, FINITE_HORIZONS, path_scores)

    n_pv = len(path_pvalues)
    n_degen = sum(1 for v in path_pvalues.values()
                  if float(np.nanmax(sanitize_pvalues(v))
                           - np.nanmin(sanitize_pvalues(v))) < 1e-12)
    print("  path p-value vectors: %d, degenerate(constant): %d"
          % (n_pv, n_degen), flush=True)

    out = {}
    tg_scores, _ = two_groups_path_score_bank(
        path_pvalues, pi_init=0.05, pi_max=0.20, weight_alpha=1.05)
    for name in ("tg_component_entropy", "tg_posterior"):
        s = np.nan_to_num(np.asarray(tg_scores[name], np.float64))
        if np.nanmax(s) > np.nanmin(s):
            out[name] = 100 * safe_auc(y_eval, s[mask])

    if cfg.get("profile"):
        stationary_scores = horizon_score_bank(entry, q, [np.inf])
        mixture_scores, _ = latent_time_mixture_score_bank(
            entry, q, FINITE_HORIZONS + [np.inf], dirichlet_alpha=1.05)
        profile_scores, _, _ = dynamic_profile_selector_score_bank(
            path_scores, path_pvalues, mixture_scores, tg_scores,
            stationary_scores)
        for name in ("dp_ebmix_posterior", "dp_profile_mean_logp",
                     "dp_eb_acat"):
            if name in profile_scores:
                s = np.nan_to_num(np.asarray(profile_scores[name], np.float64))
                if np.nanmax(s) > np.nanmin(s):
                    out[name] = 100 * safe_auc(y_eval, s[mask])

    # controls: equilibrium anchors, which involve no tie-breaking
    stat = horizon_score_bank(entry, q, [np.inf])
    for name, s in stat.items():
        s = np.nan_to_num(np.asarray(s, np.float64))
        if np.nanmax(s) > np.nanmin(s):
            out[name] = 100 * safe_auc(y_eval, s[mask])
    return out


CFGS = {
    "yelpchi": dict(gamma=1.0, tk=500, pca=None, sub="drop_nullspace",
                    rho=1.0, kappa=20.0, profile=True),
    "facebook": dict(gamma=0.7, tk=None, pca=64, sub="drop_nullspace",
                     rho=0.7267152364486923, kappa=3.0, profile=False),
}


def main():
    for ds, cfg in CFGS.items():
        print("=== %s ===" % ds, flush=True)
        base = run(ds, cfg, permute=False)
        print("  original order:", {k: round(v, 2) for k, v in base.items()},
              flush=True)
        perm = run(ds, cfg, permute=True)
        print("  permuted order:", {k: round(v, 2) for k, v in perm.items()},
              flush=True)
        for k in sorted(set(base) & set(perm)):
            d = perm[k] - base[k]
            flag = "  <-- ORDER-DEPENDENT" if abs(d) > 1.0 else ""
            print("    %-24s %7.2f -> %7.2f  (%+.2f)%s"
                  % (k, base[k], perm[k], d, flag), flush=True)


if __name__ == "__main__":
    main()
