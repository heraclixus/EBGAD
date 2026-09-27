"""Fast DGraph training-free sweep using lightweight eigendecomposition.

Strategy 1: labeled_only=True (1.2M nodes instead of 3.7M)
Strategy 2: truncated_k=32 or 64 (fewer eigenvectors)
Strategy 3: Both combined

Each is much faster than full 3.7M × 128 eigenvectors.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data


def run(device="cpu", chunk=0, n_chunks=1):
    ds = "dgraph"
    out_file = f"results/eb_gad_dgraph_fast_{chunk}.json"

    print("=== Loading DGraph ===", flush=True)
    data = load_data("dgraph")
    n, d = data.x.shape
    print(f"  {n} nodes, {d} features, {data.edge_index.shape[1]} edges",
          flush=True)

    # PCA configs (training-free)
    encoder_configs = [
        None,
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 8},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 12},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 16},
    ]

    templates = ["zero", "low", "affinity"]
    graphs = ["original"]  # skip affinity graph (slow on 3.7M)
    gammas = [0.1, 0.5, 1.0, 2.0]
    norms = ["zscore"]
    prior_modes = ["stationary"]
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]

    # Lightweight strategies
    strategies = [
        {"labeled_only": True,  "truncated_k": 64,  "label": "labeled_k64"},
        {"labeled_only": True,  "truncated_k": 128, "label": "labeled_k128"},
        {"labeled_only": False, "truncated_k": 32,  "label": "full_k32"},
        {"labeled_only": False, "truncated_k": 64,  "label": "full_k64"},
    ]

    # Build flat config list
    all_configs = []
    for strat in strategies:
        for enc in encoder_configs:
            enc_name = "none" if enc is None else f"pca_{enc['encoder_hid_dim']}"
            for t in templates:
                for g in graphs:
                    for gamma in gammas:
                        for norm in norms:
                            for pm in prior_modes:
                                all_configs.append(
                                    (strat, enc, enc_name, pm, t, g,
                                     gamma, norm))

    total = len(all_configs)
    my_configs = [c for i, c in enumerate(all_configs)
                  if i % n_chunks == chunk]
    print(f"  Total configs: {total}, chunk {chunk}/{n_chunks}: "
          f"{len(my_configs)} configs", flush=True)

    best_auc = 0
    best_config = {}
    all_results = []
    count = 0

    for strat, enc, enc_name, pm, t, g, gamma, norm in my_configs:
        count += 1
        cfg_kwargs = {
            "kappa": 1.0, "nu": 1.0, "rho": 0.5,
            "lam_penalty": 50.0, "alpha": 2.0, "T": 1.0,
            "template_type": t, "graph_type": g,
            "gamma": gamma, "normalize_mode": norm,
            "prior_mean_mode": pm,
            "laplacian_variant": "sym",
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 0,
            "normalize_features": True,
            "score_K": 1,
            "labeled_only": strat["labeled_only"],
            "truncated_k": strat["truncated_k"],
        }
        if enc is not None:
            cfg_kwargs["use_encoder"] = enc["use_encoder"]
            cfg_kwargs["encoder_type"] = enc["encoder_type"]
            cfg_kwargs["encoder_hid_dim"] = enc["encoder_hid_dim"]
            cfg_kwargs["encoder_num_layers"] = 1
            cfg_kwargs["encoder_dropout"] = 0.0
            cfg_kwargs["encoder_lr"] = 0.0
            cfg_kwargs["encoder_epochs"] = 0
            cfg_kwargs["encoder_alpha"] = 1.0
            cfg_kwargs["encoder_weight_decay"] = 0.0

        try:
            cfg = SOCGADConfig(dataset=ds, **cfg_kwargs)
            r = evaluate_single_trial(data, cfg, device=device)

            for sm in score_methods:
                auc = getattr(r, {
                    "precision_energy": "ce_auc",
                    "precision_ratio": "cr_auc",
                    "control_energy": "auc",
                }.get(sm, "auc"), 0.0)

                if auc > best_auc:
                    best_auc = auc
                    best_config = {
                        **cfg_kwargs,
                        "score_method": sm,
                        "encoder_name": enc_name,
                        "strategy": strat["label"],
                    }
                    print(f"  [{count}/{len(my_configs)}] NEW BEST "
                          f"{sm} {auc:.4f} strat={strat['label']} "
                          f"enc={enc_name} t={t} gamma={gamma}",
                          flush=True)

                all_results.append({
                    "score_method": sm, "auc": float(auc),
                    "encoder": enc_name, "template": t,
                    "gamma": gamma, "strategy": strat["label"],
                })

        except Exception as e:
            print(f"  [{count}/{len(my_configs)}] ERROR "
                  f"strat={strat['label']}: {str(e)[:80]}", flush=True)

        if count % 10 == 0:
            with open(out_file, "w") as f:
                json.dump({
                    "best_auc": float(best_auc),
                    "best_config": best_config,
                    "n_evaluated": count,
                    "n_total": len(my_configs),
                    "all_results": all_results,
                }, f, indent=2, default=str)

    with open(out_file, "w") as f:
        json.dump({
            "best_auc": float(best_auc),
            "best_config": best_config,
            "n_evaluated": count,
            "n_total": len(my_configs),
            "all_results": all_results,
        }, f, indent=2, default=str)

    print(f"\n=== DONE: best AUC = {best_auc:.4f} ===")
    print(f"  Config: {best_config}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    run(args.device, chunk=args.chunk, n_chunks=args.n_chunks)
