"""DGraph encoder-specific optimizer. Run one encoder type per job.

Usage:
    python test_dgraph_encoder.py --encoder mlp --device cuda
    python test_dgraph_encoder.py --encoder sage --device cuda
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent_stationary,
    optimize_joint_lbfgsb_stationary, coordinate_descent,
)
from data_utils import load_data


ENCODER_GRIDS = {
    "mlp": [
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 100, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 64,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 100, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 64,
         "encoder_num_layers": 4, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 200, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 100, "encoder_alpha": 0.5, "encoder_weight_decay": 0.01},
    ],
    "sage": [
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 100, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 64,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 100, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 64,
         "encoder_num_layers": 4, "encoder_dropout": 0.3, "encoder_lr": 0.005,
         "encoder_epochs": 100, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ],
}


def run(encoder_type, device="cuda"):
    ds = "dgraph"
    out_file = "results/eb_gad_dgraph_%s.json" % encoder_type

    print("=== DGraph %s encoder ===" % encoder_type.upper(), flush=True)
    data = load_data("dgraph")
    n, d = data.x.shape
    print("  %d nodes, %d features, %d edges" % (n, d, data.edge_index.shape[1]), flush=True)

    encoder_configs = ENCODER_GRIDS[encoder_type]
    templates = ["zero", "low", "high", "affinity"]
    graphs = ["original", "affinity"]
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]
    prior_modes = ["zero", "stationary"]
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]

    best_scores = {}
    for m in score_methods:
        best_scores[m] = (0, {})

    count = 0
    for enc in encoder_configs:
        for pm in prior_modes:
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
                                "prior_mean_mode": pm,
                                "laplacian_variant": "sym",
                                "d_hidden": 64, "n_hidden_layers": 2,
                                "epochs": 0, "normalize_features": True,
                            }
                            for k in ["use_encoder", "encoder_type", "encoder_hid_dim",
                                       "encoder_num_layers", "encoder_dropout", "encoder_lr",
                                       "encoder_epochs", "encoder_alpha"]:
                                if k in enc:
                                    trainer_kwargs[k] = enc[k]

                            cfg = SOCTrainerConfig(**trainer_kwargs)
                            trainer = SOCTrainer(cfg)
                            try:
                                trainer._setup(data, device=device)
                            except Exception as e:
                                if count % 50 == 0:
                                    print("  [%d] setup failed: %s" % (count, str(e)[:60]), flush=True)
                                continue

                            x_normed = trainer.normalize(data.x.to(device).float())
                            V_np = trainer.V.cpu().numpy()
                            lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V_np.shape[1])
                            delta = (x_normed - trainer.template).cpu().numpy()
                            S = compute_spectral_energy(delta, V_np)

                            # Both optimizers
                            best_opt = None; best_ml = -np.inf
                            try:
                                opt = optimize_joint_lbfgsb_stationary(
                                    S, lam_L, n, d, n_restarts=5)
                                if opt["marginal_likelihood"] > best_ml:
                                    best_ml = opt["marginal_likelihood"]
                                    best_opt = opt
                            except: pass

                            for ri in [0.0, 0.5, 1.0]:
                                for ki in [0.1, 1.0, 5.0]:
                                    try:
                                        if pm == "stationary":
                                            opt = coordinate_descent_stationary(
                                                S, lam_L, n, d, rho_init=ri, kappa_init=ki,
                                                n_iters=10, use_data_driven_bounds=True)
                                        else:
                                            opt = coordinate_descent(
                                                S, lam_L, n, d, rho_init=ri, kappa_init=ki,
                                                n_iters=10, use_data_driven_bounds=True)
                                        if opt["marginal_likelihood"] > best_ml:
                                            best_ml = opt["marginal_likelihood"]
                                            best_opt = opt
                                    except: pass

                            if best_opt is None:
                                continue

                            torch.manual_seed(0); np.random.seed(0)
                            full_cfg = {
                                "rho": best_opt["rho"], "kappa": best_opt["kappa"],
                                "gamma": gamma, "alpha": best_opt.get("alpha_T", 2.0), "T": 1.0,
                                "lam_penalty": best_opt.get("lam_penalty", 50.0),
                                "template_type": t, "graph_type": g,
                                "normalize_mode": norm, "prior_mean_mode": pm,
                                "d_hidden": 64, "n_hidden_layers": 2,
                                "epochs": 1, "parameterization": "score",
                                "time_weighting": "importance", "laplacian_variant": "sym",
                            }
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
                                if improved:
                                    _save(out_file, best_scores, score_methods, count)
                            except: pass

                            if count % 20 == 0:
                                hid = enc.get("encoder_hid_dim", 0)
                                bests = " ".join("%s=%.1f%%" % (m[:2], best_scores[m][0]*100) for m in score_methods)
                                print("  [%d enc=%s/%d t=%s pm=%s] %s" % (
                                    count, encoder_type, hid, t, pm, bests), flush=True)

    _save(out_file, best_scores, score_methods, count, complete=True)
    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    print("  FINAL: %.1f%% method=%s" % (best_scores[best_method][0]*100, best_method))


def _save(path, best_scores, score_methods, configs_done, complete=False):
    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]
    result = {
        "dataset": "dgraph", "best_auc": best_auc, "best_config": best_cfg,
        "all_best": {m: {"auc": v[0], "config": v[1]} for m, v in best_scores.items()},
        "configs_evaluated": configs_done, "complete": complete,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("    [saved %.1f%% after %d configs]" % (best_auc*100, configs_done), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True, choices=["mlp", "sage"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run(args.encoder, args.device)
