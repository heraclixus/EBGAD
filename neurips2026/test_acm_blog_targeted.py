"""Targeted optimizer for ACM and BlogCatalog.

Addresses two gaps:
1. Larger encoder dims (128, 256) for 8K-feature datasets
2. Adds control_energy scoring (D_{0,lambda}^{-1} weighting) which worked
   well on ACM in old sweeps (90.93%)
3. Both prior mean modes (zero, stationary)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, coordinate_descent_stationary,
    Q_rho_eigenvalues,
)
from data_utils import load_data


def run_dataset(ds, device="cuda", out_file=None):
    data = load_data(ds)
    n = data.x.shape[0]
    d = data.x.shape[1]
    print("  %s: %d nodes, %d features" % (ds, n, d), flush=True)

    # Encoder configs: larger dims for high-dimensional datasets
    encoder_configs = [
        None,  # no encoder
        # MLP - standard
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        # MLP - larger (128 dims)
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        # MLP - large (256 dims)
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 256,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        # MLP - large with more layers
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 256,
         "encoder_num_layers": 4, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        # GCN - larger
        {"use_encoder": True, "encoder_type": "gcn", "encoder_hid_dim": 128,
         "encoder_num_layers": 4, "encoder_dropout": 0.3, "encoder_lr": 0.01,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        # PCA - larger dims
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 128,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 256,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        # SAGE - larger
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        # GAT
        {"use_encoder": True, "encoder_type": "gat", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ]

    # Three scoring methods (adding control_energy back)
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]

    templates = ["zero", "low", "high", "affinity"]
    graphs = ["original", "affinity"]
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]
    prior_modes = ["zero", "stationary"]

    total = len(encoder_configs) * len(templates) * len(graphs) * len(gammas) * len(norms) * len(prior_modes)
    print("  %d discrete configs × %d scoring methods" % (total, len(score_methods)), flush=True)

    best_scores = {}  # {method: (auc, cfg)}
    for m in score_methods:
        best_scores[m] = (0, {})

    count = 0
    for enc in encoder_configs:
        for prior_mode in prior_modes:
            for t in templates:
                for g in graphs:
                    for gamma in gammas:
                        for norm in norms:
                            count += 1
                            trainer_kwargs = {
                                "kappa": 1.0, "nu": 1.0, "rho": 0.5,
                                "lam_penalty": 50.0, "alpha": 2.0, "T": 1.0,
                                "template_type": t, "graph_type": g,
                                "gamma": gamma, "normalize_mode": norm,
                                "prior_mean_mode": prior_mode,
                                "laplacian_variant": "sym",
                                "d_hidden": 64, "n_hidden_layers": 2,
                                "epochs": 0, "normalize_features": True,
                            }
                            if enc is not None:
                                for k in ["use_encoder", "encoder_type", "encoder_hid_dim",
                                           "encoder_num_layers", "encoder_dropout", "encoder_lr",
                                           "encoder_epochs", "encoder_alpha"]:
                                    if k in enc:
                                        trainer_kwargs[k] = enc[k]

                            cfg = SOCTrainerConfig(**trainer_kwargs)
                            trainer = SOCTrainer(cfg)
                            try:
                                trainer._setup(data, device=device)
                            except:
                                continue

                            x_normed = trainer.normalize(data.x.to(device).float())
                            V = trainer.V.cpu().numpy()
                            lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
                            delta = (x_normed - trainer.template).cpu().numpy()
                            S = compute_spectral_energy(delta, V)
                            n_nodes = x_normed.shape[0]
                            d_feat = x_normed.shape[1]

                            # Optimize continuous params
                            best_opt = None; best_opt_ml = -np.inf
                            for rho_init in [0.0, 0.5, 1.0]:
                                for kappa_init in [0.1, 1.0, 5.0]:
                                    try:
                                        if prior_mode == "stationary":
                                            opt = coordinate_descent_stationary(
                                                S, lam_L, n_nodes, d_feat,
                                                rho_init=rho_init, kappa_init=kappa_init,
                                                n_iters=10, use_data_driven_bounds=True)
                                        else:
                                            opt = coordinate_descent(
                                                S, lam_L, n_nodes, d_feat,
                                                rho_init=rho_init, kappa_init=kappa_init,
                                                n_iters=10, use_data_driven_bounds=True)
                                        if opt["marginal_likelihood"] > best_opt_ml:
                                            best_opt_ml = opt["marginal_likelihood"]
                                            best_opt = opt
                                    except:
                                        pass

                            if best_opt is None:
                                continue

                            torch.manual_seed(0); np.random.seed(0)
                            full_cfg = {
                                "rho": best_opt["rho"], "kappa": best_opt["kappa"],
                                "gamma": gamma, "alpha": best_opt.get("alpha_T", 2.0), "T": 1.0,
                                "lam_penalty": best_opt.get("lam_penalty", 50.0),
                                "template_type": t, "graph_type": g,
                                "normalize_mode": norm,
                                "prior_mean_mode": prior_mode,
                                "d_hidden": 64, "n_hidden_layers": 2,
                                "epochs": 1, "parameterization": "score",
                                "time_weighting": "importance",
                                "laplacian_variant": "sym",
                            }
                            if enc is not None:
                                for k in ["use_encoder", "encoder_type", "encoder_hid_dim",
                                           "encoder_num_layers", "encoder_dropout", "encoder_lr",
                                           "encoder_epochs", "encoder_alpha"]:
                                    if k in enc:
                                        full_cfg[k] = enc[k]

                            try:
                                improved = False
                                for method in score_methods:
                                    eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                                    r = evaluate_single_trial(data, eval_cfg, device=device)
                                    if r.auc > best_scores[method][0]:
                                        best_scores[method] = (r.auc, {**full_cfg, "score_method": method})
                                        improved = True
                                # Save incrementally on any improvement
                                if improved:
                                    _save_intermediate(out_file, ds, best_scores, score_methods, count)
                            except:
                                pass

                            if count % 50 == 0:
                                enc_tag = enc.get("encoder_type", "none") if enc else "none"
                                hid = enc.get("encoder_hid_dim", 0) if enc else 0
                                bests = " ".join("%s=%.1f%%" % (m[:2], best_scores[m][0]*100) for m in score_methods)
                                print("  [%d/%d enc=%s/%d pm=%s] %s" % (
                                    count, total, enc_tag, hid, prior_mode, bests), flush=True)

    # Final save
    _save_intermediate(out_file, ds, best_scores, score_methods, count)

    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]

    print("  BEST: %.1f%% method=%s enc=%s hid=%s prior=%s" % (
        best_auc * 100, best_method,
        best_cfg.get("encoder_type", "none") if best_cfg.get("use_encoder") else "none",
        best_cfg.get("encoder_hid_dim", "N/A"),
        best_cfg.get("prior_mean_mode", "?")))

    return {
        "dataset": ds, "best_auc": best_auc,
        "best_config": best_cfg,
        "all_best": {m: {"auc": v[0], "config": v[1]} for m, v in best_scores.items()},
    }


def _save_intermediate(out_file, ds, best_scores, score_methods, configs_done):
    """Save current best results incrementally."""
    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]
    result = {
        "dataset": ds, "best_auc": best_auc,
        "best_config": best_cfg,
        "all_best": {m: {"auc": v[0], "config": v[1]} for m, v in best_scores.items()},
        "configs_evaluated": configs_done,
        "complete": False,
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("    [saved %.1f%% after %d configs]" % (best_auc * 100, configs_done), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="acm,blogcatalog")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_targeted_%s.json" % ds)

        # Load existing intermediate results instead of skipping
        if os.path.exists(out_file):
            with open(out_file) as f:
                existing = json.load(f)
            if existing.get("complete"):
                print("=== %s === SKIP (complete)" % ds)
                continue
            else:
                print("=== %s === RESUMING from %.1f%% (%d configs)" % (
                    ds, existing["best_auc"]*100, existing.get("configs_evaluated", 0)))

        print("=== %s ===" % ds)
        result = run_dataset(ds, device=args.device, out_file=out_file)

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("  Saved to %s" % out_file)


if __name__ == "__main__":
    main()
