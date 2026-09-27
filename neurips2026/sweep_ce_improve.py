"""Targeted CE-only sweeps for 5 datasets where control_energy underperforms.

Goal: find CE prior configs that match or beat magnitude/spectral_ratio/directional.

Diagnosis:
- Elliptic/Elliptic++: need sharper spectral weighting (small κ, large ν)
- DGraph: need template/normalization to capture location-shift anomalies
- Disney: tiny graph, may need encoder or different α
- Enron: sparse anomalies, need different prior config
"""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse, itertools, json, os, sys, time
import numpy as np
import torch

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data

# ---------- Sweep grids per dataset ----------

# NOTE: epochs=1 because CE doesn't use the score network.
# Training still runs preprocessing (eigen, template, normalization).

ELLIPTIC_GRID = {
    "rho":            [0.0, 0.5, 1.0],
    "kappa":          [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    "gamma":          [0.1, 0.2, 0.5, 1.0, 2.0],
    "template_type":  ["zero", "low", "high"],
    "graph_type":     ["original"],
    "alpha":          [1.0, 2.0, 3.0],
    "lam_penalty":    [10, 50, 200],
    # fixed
    "d_hidden": [64], "n_hidden_layers": [2], "lr": [0.001],
    "epochs": [1], "patience": [100], "parameterization": ["score"],
    "time_weighting": ["importance"], "laplacian_variant": ["sym"],
    "score_method": ["control_energy"], "score_K": [1],
}
# 3×6×5×3×1×3×3 = 2430

ELLIPTIC_PP_GRID = {
    **ELLIPTIC_GRID,
    "alpha":          [1.0, 2.0, 3.0],
    "lam_penalty":    [1.0, 10, 50, 200],
}
# 3×6×5×3×1×3×4 = 3240

DGRAPH_GRID = {
    "rho":            [0.0, 0.25, 0.5, 1.0],
    "kappa":          [0.1, 0.5, 1.0, 5.0],
    "gamma":          [0.5, 1.0, 2.0, 5.0],
    "template_type":  ["zero", "low", "high"],
    "graph_type":     ["original"],
    "alpha":          [1.0, 2.0, 3.0],
    "lam_penalty":    [10, 100, 500],
    "labeled_only":   [True],
    # fixed
    "d_hidden": [256], "n_hidden_layers": [3], "lr": [0.0005],
    "epochs": [1], "patience": [100], "parameterization": ["score"],
    "time_weighting": ["importance"], "laplacian_variant": ["sym"],
    "score_network_type": ["mlp"], "truncated_k": [500],
    "score_method": ["control_energy"], "score_K": [1],
}
# 4×4×4×3×1×3×3 = 1728

DISNEY_GRID = {
    "rho":            [0.0, 0.5, 1.0],
    "kappa":          [0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
    "gamma":          [0.2, 0.5, 1.0, 2.0, 5.0],
    "template_type":  ["zero", "low", "high", "affinity"],
    "graph_type":     ["original", "affinity"],
    "alpha":          [1.0, 2.0, 3.0],
    "lam_penalty":    [5, 50, 100, 500],
    # fixed
    "d_hidden": [64], "n_hidden_layers": [2], "lr": [0.001],
    "epochs": [1], "patience": [30], "parameterization": ["score"],
    "score_method": ["control_energy"], "score_K": [1],
}
# 3×6×5×4×2×3×4 = 8640

ENRON_GRID = {
    "rho":            [0.0, 0.5, 1.0],
    "kappa":          [0.1, 0.5, 1.0, 5.0, 10.0],
    "gamma":          [0.2, 0.5, 1.0, 2.0],
    "template_type":  ["zero", "low", "high", "affinity"],
    "graph_type":     ["original", "affinity"],
    "alpha":          [1.0, 2.0, 3.0],
    "lam_penalty":    [10, 100, 500, 2000],
    # fixed
    "d_hidden": [64], "n_hidden_layers": [2], "lr": [0.001],
    "epochs": [1], "patience": [50], "parameterization": ["score"],
    "time_weighting": ["importance"],
    "score_method": ["control_energy"], "score_K": [1],
}

GRIDS = {
    "elliptic": ELLIPTIC_GRID,
    "elliptic_plus_plus": ELLIPTIC_PP_GRID,
    "dgraph": DGRAPH_GRID,
    "disney": DISNEY_GRID,
    "enron": ENRON_GRID,
}


def grid_to_configs(grid: dict) -> list[dict]:
    """Cartesian product of grid values."""
    keys = sorted(grid.keys())
    vals = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*vals):
        configs.append(dict(zip(keys, combo)))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--chunk", type=int, default=0,
                        help="Which chunk of configs to run (0-indexed)")
    parser.add_argument("--total-chunks", type=int, default=1,
                        help="Total number of chunks to split into")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ds = args.dataset.lower()
    if ds not in GRIDS:
        print("Unknown dataset: %s. Available: %s" % (ds, list(GRIDS.keys())))
        sys.exit(1)

    all_configs = grid_to_configs(GRIDS[ds])
    total = len(all_configs)

    # Chunk
    chunk_size = (total + args.total_chunks - 1) // args.total_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    configs = all_configs[start:end]

    print("Dataset: %s  |  Total configs: %d  |  Chunk %d/%d: configs %d-%d (%d)" %
          (ds, total, args.chunk, args.total_chunks, start, end - 1, len(configs)))

    ds_load = ds
    if ds == "yelpchi":
        ds_load = "YelpChi"
    data = load_data(ds_load)

    out_path = args.output or "results/ce_improve_%s_chunk%03d.jsonl" % (ds, args.chunk)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "a")

    best_auc = 0
    for i, params in enumerate(configs):
        aucs = []
        t0 = time.time()
        for trial in range(args.trials):
            torch.manual_seed(trial)
            np.random.seed(trial)
            try:
                cfg = SOCGADConfig(dataset=ds, **params)
                r = evaluate_single_trial(data, cfg, device=args.device)
                aucs.append(r.auc)
            except Exception as e:
                print("  ERROR config %d trial %d: %s" % (i, trial, e))
                aucs.append(0.0)
        elapsed = time.time() - t0
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        rec = {
            "dataset": ds, "auc_mean": mean_auc, "auc_std": std_auc,
            "trials": aucs, "elapsed_s": elapsed, "params": params,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if mean_auc > best_auc:
            best_auc = mean_auc
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print("[%d/%d] best=%.4f  last=%.4f±%.4f  (%.0fs)" %
                  (i + 1, len(configs), best_auc, mean_auc, std_auc, elapsed))

    fout.close()
    print("Done. Best AUC=%.4f  Saved to %s" % (best_auc, out_path))


if __name__ == "__main__":
    main()
