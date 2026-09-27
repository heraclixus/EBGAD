"""E2 isolation ablation for DGraph using the cached labeled-subgraph eigenpairs.

Mirrors test_e2_isolation_banks.py (five spectral precision families under the
identical label-free selector) but builds the frozen residual pipeline from
``cache/dgraph_eigen`` like test_dynamic_hypotheses_dgraph_cached.py, instead
of recomputing eigenpairs on the 3.7M-node graph.

Paper-era DGraph configuration: k=128 cached eigenspace, template gamma=1.0,
zscore normalization, stationary fit at Gamma=inf over the era kappa grid.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data_utils import load_data
from test_dynamic_score_bank import (
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    parse_grid,
)
from test_dynamic_hypotheses_dgraph_cached import build_entries, normalize_features
from test_e2_isolation_banks import ERA_KAPPAS, family_banks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/dgraph_eigen")
    parser.add_argument("--cache-file", default="labeled_original_k128.pt")
    parser.add_argument("--normalize", default="zscore", choices=["zscore", "minmax", "raw"])
    parser.add_argument("--template-gammas", default="1.0")
    parser.add_argument("--kappas", default=",".join("%g" % k for k in ERA_KAPPAS))
    parser.add_argument("--out", default="results/e2/e2_dgraph.json")
    args = parser.parse_args()

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
    print("  V=%s lam_range=[%.3g, %.3g]" % (
        tuple(V.shape), float(np.min(lam_L)), float(np.max(lam_L))), flush=True)
    if V.shape[0] != x_sub.shape[0]:
        raise ValueError("Cache rows %d != labeled nodes %d" % (V.shape[0], x_sub.shape[0]))

    gammas = parse_grid(args.template_gammas)
    kappas = parse_grid(args.kappas)
    cache_label = os.path.splitext(os.path.basename(args.cache_file))[0]
    entries = build_entries(x_sub, V, lam_L, gammas, cache_label)

    print("Fitting stationary EB geometry", flush=True)
    fit = fit_best_entry(entries, [np.inf], kappas, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    print("  fit: %s rho=%.3f kappa=%g ml=%.0f" % (
        fit["entry_key"], fit["rho"], fit["kappa"], fit["ml"]), flush=True)

    out = {"dataset": "dgraph",
           "fit": {k: fit[k] for k in ("entry_key", "rho", "kappa", "ml")},
           "graph": {"nodes_labeled": int(len(idx)), "k": int(V.shape[1]),
                     "gammas": gammas},
           "families": {}}
    for family, scores in family_banks(entry, fit).items():
        summary = evaluate_score_bank(y_sub, mask, scores, references=None)
        sel = summary.get("selectors", {})
        row = {
            "n_scores": summary.get("n_scores"),
            "auc_ks_selected": summary.get("auc_ks_selected"),
            "score_ks": summary.get("score_ks"),
            "auc_oracle": summary.get("auc_oracle"),
            "oracle_score": summary.get("oracle_score"),
            "auc_graph_tail": (sel.get("graph_tail") or {}).get("auc"),
            "auc_stability": (sel.get("stability") or {}).get("auc"),
        }
        out["families"][family] = row
        print("  %-10s sel(KS)=%5.1f%% (%s)  oracle=%5.1f%% (%s)" % (
            family,
            100.0 * (row["auc_ks_selected"] or 0),
            row["score_ks"],
            100.0 * (row["auc_oracle"] or 0),
            row["oracle_score"],
        ), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
