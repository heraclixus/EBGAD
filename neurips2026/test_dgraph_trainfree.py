"""DGraph training-free sweep (PCA + no encoder only).

DGraph: 3.7M nodes, 4.3M edges, 17 features.
Sweeps over all closed-form scoring modes, templates, gammas,
prior modes, and PCA dimensions. No neural network training.
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
    out_file = f"results/eb_gad_dgraph_trainfree_{chunk}.json"

    print("=== Loading DGraph ===", flush=True)
    data = load_data("dgraph")
    n, d = data.x.shape
    print(f"  {n} nodes, {d} features, {data.edge_index.shape[1]} edges",
          flush=True)

    # Training-free encoder configs: none + PCA at various dims
    encoder_configs = [
        None,  # no encoder
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 8,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 12,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 16,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
    ]

    templates = ["zero", "low", "affinity"]
    graphs = ["original", "affinity"]
    gammas = [0.1, 0.2, 0.5, 1.0, 2.0]
    norms = ["zscore", "minmax"]
    prior_modes = ["zero", "stationary"]
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]

    best_auc = 0
    best_config = {}
    all_results = []

    # Build flat config list for chunking
    all_configs = []
    for enc in encoder_configs:
        enc_name = "none" if enc is None else f"pca_{enc['encoder_hid_dim']}"
        for pm in prior_modes:
            for t in templates:
                for g in graphs:
                    for gamma in gammas:
                        for norm in norms:
                            all_configs.append((enc, enc_name, pm, t, g,
                                                gamma, norm))

    total = len(all_configs)
    # Select this chunk's configs
    my_configs = [c for i, c in enumerate(all_configs)
                  if i % n_chunks == chunk]
    print(f"  Total configs: {total}, chunk {chunk}/{n_chunks}: "
          f"{len(my_configs)} configs", flush=True)

    count = 0
    for enc, enc_name, pm, t, g, gamma, norm in my_configs:
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
                            }
                            if enc is not None:
                                cfg_kwargs.update(enc)

                            try:
                                cfg = SOCGADConfig(dataset=ds, **cfg_kwargs)
                                r = evaluate_single_trial(data, cfg,
                                                          device=device)

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
                                        }
                                        print(f"  [{count}/{total}] NEW BEST "
                                              f"{sm} {auc:.4f} enc={enc_name} "
                                              f"pm={pm} t={t} g={g} "
                                              f"gamma={gamma} norm={norm}",
                                              flush=True)

                                    all_results.append({
                                        "score_method": sm,
                                        "auc": float(auc),
                                        "encoder": enc_name,
                                        "prior_mode": pm,
                                        "template": t,
                                        "graph": g,
                                        "gamma": gamma,
                                        "norm": norm,
                                    })

                            except Exception as e:
                                print(f"  [{count}/{total}] ERROR: "
                                      f"{str(e)[:80]}", flush=True)

                            # Save incrementally
                            if count % 20 == 0:
                                with open(out_file, "w") as f:
                                    json.dump({
                                        "best_auc": float(best_auc),
                                        "best_config": best_config,
                                        "n_evaluated": count,
                                        "n_total": total,
                                        "all_results": all_results,
                                    }, f, indent=2, default=str)

    # Final save
    with open(out_file, "w") as f:
        json.dump({
            "best_auc": float(best_auc),
            "best_config": best_config,
            "n_evaluated": count,
            "n_total": total,
            "all_results": all_results,
        }, f, indent=2, default=str)

    print(f"\n=== DONE: best AUC = {best_auc:.4f} ===")
    print(f"  Config: {best_config}")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Chunk index for parallel execution")
    parser.add_argument("--n-chunks", type=int, default=1,
                        help="Total number of parallel chunks")
    args = parser.parse_args()
    run(args.device, chunk=args.chunk, n_chunks=args.n_chunks)
