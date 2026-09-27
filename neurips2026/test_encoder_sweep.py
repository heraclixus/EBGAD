"""Encoder sweep for datasets missing from the encoder comparison table.

Tests MLP and GraphSAGE encoders on: Amazon, YelpChi, ACM,
Elliptic, Elliptic++, T-Finance.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data


def run(device="cuda", chunk=0, n_chunks=1):
    datasets = ["amazon", "yelpchi", "acm", "elliptic",
                "elliptic_plus_plus", "t_finance"]

    encoder_configs = [
        {"encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1,
         "encoder_lr": 0.005, "encoder_epochs": 100,
         "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"encoder_type": "mlp", "encoder_hid_dim": 64,
         "encoder_num_layers": 2, "encoder_dropout": 0.1,
         "encoder_lr": 0.005, "encoder_epochs": 100,
         "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"encoder_type": "sage", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1,
         "encoder_lr": 0.005, "encoder_epochs": 100,
         "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"encoder_type": "sage", "encoder_hid_dim": 64,
         "encoder_num_layers": 2, "encoder_dropout": 0.1,
         "encoder_lr": 0.005, "encoder_epochs": 100,
         "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ]

    templates = ["zero", "low", "affinity"]
    gammas = [0.1, 0.5, 1.0, 2.0]
    graphs = ["original"]
    prior_modes = ["stationary"]
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]

    configs = []
    for ds in datasets:
        for enc in encoder_configs:
            for tmpl in templates:
                for gamma in gammas:
                    for graph in graphs:
                        for pm in prior_modes:
                            configs.append({
                                "ds": ds, "enc": enc,
                                "template_type": tmpl, "gamma": gamma,
                                "graph_type": graph,
                                "prior_mean_mode": pm,
                            })

    total = len(configs)
    my_configs = [c for i, c in enumerate(configs) if i % n_chunks == chunk]
    print(f"Total: {total}, chunk {chunk}/{n_chunks}: {len(my_configs)}",
          flush=True)

    results_by_ds = {}
    count = 0

    for cfg_dict in my_configs:
        count += 1
        ds = cfg_dict["ds"]
        enc = cfg_dict["enc"]

        if ds not in results_by_ds:
            results_by_ds[ds] = {"best_auc": 0, "best_config": {}}

        cfg_kwargs = {
            "kappa": 1.0, "nu": 1.0, "rho": 0.5,
            "lam_penalty": 50.0, "alpha": 2.0, "T": 1.0,
            "template_type": cfg_dict["template_type"],
            "graph_type": cfg_dict["graph_type"],
            "gamma": cfg_dict["gamma"],
            "normalize_mode": "zscore",
            "prior_mean_mode": cfg_dict["prior_mean_mode"],
            "laplacian_variant": "sym",
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 0, "normalize_features": True,
            "score_K": 1,
            "use_encoder": True,
            **enc,
        }

        try:
            load_name = ("YelpChi" if ds == "yelpchi" else
                         "Facebook" if ds == "facebook" else ds)
            data = load_data(load_name)
            cfg = SOCGADConfig(dataset=ds, **cfg_kwargs)
            r = evaluate_single_trial(data, cfg, device=device)

            for sm in score_methods:
                auc = getattr(r, {
                    "precision_energy": "ce_auc",
                    "precision_ratio": "cr_auc",
                    "control_energy": "auc",
                }.get(sm, "auc"), 0.0)

                if auc > results_by_ds[ds]["best_auc"]:
                    results_by_ds[ds]["best_auc"] = float(auc)
                    results_by_ds[ds]["best_config"] = {
                        **cfg_kwargs, "score_method": sm,
                    }
                    print(f"  [{count}/{len(my_configs)}] {ds} NEW BEST "
                          f"{sm} {auc:.4f} "
                          f"enc={enc['encoder_type']}_{enc['encoder_hid_dim']} "
                          f"t={cfg_dict['template_type']} "
                          f"gamma={cfg_dict['gamma']}",
                          flush=True)

        except Exception as e:
            if "out of memory" not in str(e).lower():
                print(f"  [{count}/{len(my_configs)}] {ds} ERROR: "
                      f"{str(e)[:60]}", flush=True)

        if count % 20 == 0 or count == len(my_configs):
            out_file = f"results/eb_gad_encoder_sweep_{chunk}.json"
            with open(out_file, "w") as f:
                json.dump(results_by_ds, f, indent=2, default=str)

    print("\n=== RESULTS ===")
    for ds, res in results_by_ds.items():
        enc_type = res["best_config"].get("encoder_type", "?")
        enc_dim = res["best_config"].get("encoder_hid_dim", "?")
        print(f"  {ds}: {res['best_auc']:.4f} "
              f"({enc_type}_{enc_dim})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    run(args.device, args.chunk, args.n_chunks)
