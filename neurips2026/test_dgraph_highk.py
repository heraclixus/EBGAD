"""DGraph sweep with high-k eigendecomposition.

The default k=128 captures only 0.003% of the spectrum on 3.7M nodes.
This sweep tests k=256, 500, 1000 on full graph and k=500, 1000, 2000
on labeled subgraph (1.2M nodes).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data


def run(device="cuda", chunk=0, n_chunks=1):
    ds = "dgraph"
    out_file = f"results/eb_gad_dgraph_highk_{chunk}.json"

    print("=== Loading DGraph ===", flush=True)
    data = load_data("dgraph")
    n, d = data.x.shape
    print(f"  {n} nodes, {d} features, {data.edge_index.shape[1]} edges",
          flush=True)

    # Configs focused on what worked best + higher k
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]

    configs = []
    # Strategy 1: Full graph with higher k
    for k in [256, 500, 1000]:
        for enc in [None,
                    {"use_encoder": True, "encoder_type": "pca",
                     "encoder_hid_dim": 8},
                    {"use_encoder": True, "encoder_type": "pca",
                     "encoder_hid_dim": 16}]:
            for tmpl in ["zero", "low", "affinity"]:
                for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    for graph in ["original", "affinity"]:
                        for pm in ["zero", "stationary"]:
                            for norm in ["zscore", "minmax"]:
                                configs.append({
                                    "truncated_k": k,
                                    "labeled_only": False,
                                    "enc": enc,
                                    "template_type": tmpl,
                                    "gamma": gamma,
                                    "graph_type": graph,
                                    "prior_mean_mode": pm,
                                    "normalize_mode": norm,
                                    "label": f"full_k{k}",
                                })

    # Strategy 2: Labeled subgraph with even higher k
    for k in [500, 1000, 2000]:
        for enc in [None,
                    {"use_encoder": True, "encoder_type": "pca",
                     "encoder_hid_dim": 8},
                    {"use_encoder": True, "encoder_type": "pca",
                     "encoder_hid_dim": 16}]:
            for tmpl in ["zero", "low", "affinity"]:
                for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    for graph in ["original", "affinity"]:
                        for pm in ["zero", "stationary"]:
                            for norm in ["zscore", "minmax"]:
                                configs.append({
                                    "truncated_k": k,
                                    "labeled_only": True,
                                    "enc": enc,
                                    "template_type": tmpl,
                                    "gamma": gamma,
                                    "graph_type": graph,
                                    "prior_mean_mode": pm,
                                    "normalize_mode": norm,
                                    "label": f"labeled_k{k}",
                                })

    total = len(configs)
    my_configs = [c for i, c in enumerate(configs) if i % n_chunks == chunk]
    print(f"  Total configs: {total}, chunk {chunk}/{n_chunks}: "
          f"{len(my_configs)} configs", flush=True)

    best_auc = 0
    best_config = {}
    all_results = []
    count = 0

    for cfg_dict in my_configs:
        count += 1
        enc = cfg_dict["enc"]
        enc_name = "none" if enc is None else f"pca_{enc['encoder_hid_dim']}"

        cfg_kwargs = {
            "kappa": 1.0, "nu": 1.0, "rho": 0.5,
            "lam_penalty": 50.0, "alpha": 2.0, "T": 1.0,
            "template_type": cfg_dict["template_type"],
            "graph_type": cfg_dict["graph_type"],
            "gamma": cfg_dict["gamma"],
            "normalize_mode": cfg_dict["normalize_mode"],
            "prior_mean_mode": cfg_dict["prior_mean_mode"],
            "laplacian_variant": "sym",
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 0, "normalize_features": True,
            "score_K": 1,
            "labeled_only": cfg_dict["labeled_only"],
            "truncated_k": cfg_dict["truncated_k"],
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
                        **cfg_kwargs, "score_method": sm,
                        "encoder_name": enc_name,
                        "strategy": cfg_dict["label"],
                    }
                    print(f"  [{count}/{len(my_configs)}] NEW BEST "
                          f"{sm} {auc:.4f} k={cfg_dict['truncated_k']} "
                          f"enc={enc_name} t={cfg_dict['template_type']} "
                          f"g={cfg_dict['graph_type']} "
                          f"gamma={cfg_dict['gamma']} "
                          f"pm={cfg_dict['prior_mean_mode']} "
                          f"norm={cfg_dict['normalize_mode']} "
                          f"labeled={cfg_dict['labeled_only']}",
                          flush=True)

                all_results.append({
                    "score_method": sm, "auc": float(auc),
                    "strategy": cfg_dict["label"],
                    "encoder": enc_name,
                    "template": cfg_dict["template_type"],
                    "gamma": cfg_dict["gamma"],
                    "graph": cfg_dict["graph_type"],
                    "pm": cfg_dict["prior_mean_mode"],
                    "norm": cfg_dict["normalize_mode"],
                })

        except Exception as e:
            print(f"  [{count}/{len(my_configs)}] ERROR "
                  f"k={cfg_dict['truncated_k']}: {str(e)[:80]}",
                  flush=True)

        if count % 10 == 0 or count == len(my_configs):
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    run(args.device, chunk=args.chunk, n_chunks=args.n_chunks)
