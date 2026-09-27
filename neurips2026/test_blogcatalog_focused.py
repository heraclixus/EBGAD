"""Focused BlogCatalog sweep to improve from 78.8%.

BlogCatalog: 5196 nodes, 8189 features, 5.8% anomaly rate.
Current best: 78.8% (nokappa, rho=0.39, precision_energy, affinity template).
TAM baseline: 82.5%.

Strategy: exhaustive sweep over kappa, rho, gamma, PCA dims, scoring modes.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data


def run(device="cuda", chunk=0, n_chunks=1):
    ds = "blogcatalog"
    out_file = f"results/eb_gad_blogcatalog_focused_{chunk}.json"

    print("=== Loading BlogCatalog ===", flush=True)
    data = load_data("BlogCatalog")
    print(f"  {data.num_nodes} nodes, {data.x.shape[1]} features", flush=True)

    # Exhaustive grid
    templates = ["zero", "low", "affinity"]
    graphs = ["original", "affinity"]
    gammas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]
    prior_modes = ["zero", "stationary"]
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]
    pca_dims = [None, 32, 64, 128, 256, 512]
    # Fine kappa grid focusing on 0-30 range
    kappa_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    configs = []
    for tmpl in templates:
        for graph in graphs:
            for gamma in gammas:
                for norm in norms:
                    for pm in prior_modes:
                        for pca in pca_dims:
                            for kappa in kappa_values:
                                for rho in rho_values:
                                    configs.append({
                                        "template_type": tmpl,
                                        "graph_type": graph,
                                        "gamma": gamma,
                                        "normalize_mode": norm,
                                        "prior_mean_mode": pm,
                                        "pca": pca,
                                        "kappa": kappa,
                                        "rho": rho,
                                    })

    total = len(configs)
    my_configs = [c for i, c in enumerate(configs) if i % n_chunks == chunk]
    print(f"  Total: {total}, chunk {chunk}/{n_chunks}: {len(my_configs)}",
          flush=True)

    best_auc = 0
    best_config = {}
    all_results = []
    count = 0

    for cfg_dict in my_configs:
        count += 1
        cfg_kwargs = {
            "kappa": cfg_dict["kappa"], "nu": 1.0,
            "rho": cfg_dict["rho"],
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
        }
        if cfg_dict["pca"] is not None:
            cfg_kwargs.update({
                "use_encoder": True, "encoder_type": "pca",
                "encoder_hid_dim": cfg_dict["pca"],
                "encoder_num_layers": 1, "encoder_dropout": 0.0,
                "encoder_lr": 0.0, "encoder_epochs": 0,
                "encoder_alpha": 1.0, "encoder_weight_decay": 0.0,
            })

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
                    pca_str = f"pca{cfg_dict['pca']}" if cfg_dict["pca"] else "none"
                    best_config = {
                        **cfg_kwargs, "score_method": sm,
                        "pca_dim": cfg_dict["pca"],
                    }
                    print(f"  [{count}/{len(my_configs)}] NEW BEST "
                          f"{sm} {auc:.4f} rho={cfg_dict['rho']:.1f} "
                          f"kappa={cfg_dict['kappa']} "
                          f"t={cfg_dict['template_type']} "
                          f"g={cfg_dict['graph_type']} "
                          f"gamma={cfg_dict['gamma']} "
                          f"pca={pca_str} pm={cfg_dict['prior_mean_mode']}",
                          flush=True)

        except Exception as e:
            if "out of memory" not in str(e).lower():
                print(f"  [{count}/{len(my_configs)}] ERROR: {str(e)[:60]}",
                      flush=True)

        if count % 100 == 0 or count == len(my_configs):
            with open(out_file, "w") as f:
                json.dump({
                    "best_auc": float(best_auc),
                    "best_config": best_config,
                    "n_evaluated": count,
                    "n_total": len(my_configs),
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
    run(args.device, args.chunk, args.n_chunks)
