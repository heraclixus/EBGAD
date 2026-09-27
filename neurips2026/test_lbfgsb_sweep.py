"""Optimizer comparison: L-BFGS-B vs coordinate descent on stationary prior.

For each discrete config, runs BOTH optimizers and picks the better one.
This tests whether L-BFGS-B finds better optima than coordinate descent.

Includes all templates (with bandpass/mix), all encoders, both prior modes.
Saves incrementally.
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
    optimize_joint_lbfgsb_stationary, compute_kappa_bounds,
)
from data_utils import load_data


def run_dataset(ds, device="cuda", out_file=None):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    use_large = ds.lower() in ("dgraph", "t_finance")

    encoder_configs = [
        None,
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 64,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 128,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 64,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ]
    if use_large:
        encoder_configs = [e for e in encoder_configs
                           if e is None or e.get("encoder_type") in ("pca", "mlp", "sage")]

    templates = ["zero", "low", "high", "affinity", "affinity_high"]
    graphs = ["original", "affinity"]
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]
    score_methods = ["precision_energy", "precision_ratio"]

    total = len(encoder_configs) * len(templates) * len(graphs) * len(gammas) * len(norms)
    print("  %d discrete configs, comparing CD vs L-BFGS-B" % total, flush=True)

    best_ce = 0; best_cr = 0
    best_ce_cfg = {}; best_cr_cfg = {}
    cd_wins = 0; lbfgsb_wins = 0

    count = 0
    for enc in encoder_configs:
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
                            "prior_mean_mode": "stationary",
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
                        V_np = trainer.V.cpu().numpy()
                        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V_np.shape[1])
                        delta = (x_normed - trainer.template).cpu().numpy()
                        S = compute_spectral_energy(delta, V_np)

                        # Data-driven kappa bounds
                        kr = compute_kappa_bounds(S, lam_L, 0.5, n, d)

                        # === Run BOTH optimizers ===

                        # 1. Coordinate descent (multi-start)
                        cd_best = None; cd_ml = -np.inf
                        for rho_init in [0.0, 0.5, 1.0]:
                            for kappa_init in [0.1, 1.0, 5.0]:
                                try:
                                    opt = coordinate_descent_stationary(
                                        S, lam_L, n, d,
                                        rho_init=rho_init, kappa_init=kappa_init,
                                        n_iters=10, use_data_driven_bounds=True)
                                    if opt["marginal_likelihood"] > cd_ml:
                                        cd_ml = opt["marginal_likelihood"]
                                        cd_best = opt
                                except:
                                    pass

                        # 2. L-BFGS-B (multi-start with analytical gradients)
                        try:
                            lbfgsb_opt = optimize_joint_lbfgsb_stationary(
                                S, lam_L, n, d, kappa_range=kr, n_restarts=10)
                            lbfgsb_ml = lbfgsb_opt["marginal_likelihood"]
                        except:
                            lbfgsb_opt = None; lbfgsb_ml = -np.inf

                        # Pick the winner
                        if cd_best is not None and cd_ml >= lbfgsb_ml:
                            best_opt = cd_best
                            cd_wins += 1
                        elif lbfgsb_opt is not None:
                            best_opt = lbfgsb_opt
                            lbfgsb_wins += 1
                        else:
                            continue

                        torch.manual_seed(0); np.random.seed(0)
                        full_cfg = {
                            "rho": best_opt["rho"], "kappa": best_opt["kappa"],
                            "gamma": gamma, "alpha": 2.0, "T": 1.0,
                            "lam_penalty": 50.0,
                            "template_type": t, "graph_type": g,
                            "normalize_mode": norm,
                            "prior_mean_mode": "stationary",
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
                                if method == "precision_energy" and r.auc > best_ce:
                                    best_ce = r.auc
                                    best_ce_cfg = {**full_cfg, "score_method": method,
                                                   "optimizer": best_opt.get("optimizer", "cd")}
                                    improved = True
                                if method == "precision_ratio" and r.auc > best_cr:
                                    best_cr = r.auc
                                    best_cr_cfg = {**full_cfg, "score_method": method,
                                                   "optimizer": best_opt.get("optimizer", "cd")}
                                    improved = True
                            if improved and out_file:
                                _save_incremental(out_file, ds, best_ce, best_cr,
                                                  best_ce_cfg, best_cr_cfg, count,
                                                  cd_wins, lbfgsb_wins)
                        except:
                            pass

                        if count % 50 == 0:
                            enc_tag = enc.get("encoder_type", "none") if enc else "none"
                            print("  [%d/%d enc=%s] CE=%.1f%% CR=%.1f%% (CD=%d L-BFGS-B=%d)" % (
                                count, total, enc_tag, best_ce*100, best_cr*100,
                                cd_wins, lbfgsb_wins), flush=True)

    # Final save
    if out_file:
        _save_incremental(out_file, ds, best_ce, best_cr,
                          best_ce_cfg, best_cr_cfg, count,
                          cd_wins, lbfgsb_wins, complete=True)

    best_auc = max(best_ce, best_cr)
    best_cfg = best_ce_cfg if best_ce >= best_cr else best_cr_cfg
    print("  BEST: %.1f%% (CD wins=%d, L-BFGS-B wins=%d)" % (
        best_auc*100, cd_wins, lbfgsb_wins))


def _save_incremental(path, ds, best_ce, best_cr, ce_cfg, cr_cfg,
                       configs_done, cd_wins, lbfgsb_wins, complete=False):
    best_auc = max(best_ce, best_cr)
    best_cfg = ce_cfg if best_ce >= best_cr else cr_cfg
    result = {
        "dataset": ds, "best_auc": best_auc,
        "best_ce": best_ce, "best_cr": best_cr,
        "best_config": best_cfg,
        "configs_evaluated": configs_done,
        "cd_wins": cd_wins, "lbfgsb_wins": lbfgsb_wins,
        "complete": complete,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="enron,weibo,reddit,yelpchi,facebook,acm,amazon,blogcatalog,t_finance,elliptic,elliptic_plus_plus,dgraph")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_lbfgsb_%s.json" % ds)
        if os.path.exists(out_file):
            with open(out_file) as f:
                r = json.load(f)
            if r.get("complete"):
                print("=== %s === SKIP (complete)" % ds)
                continue

        print("=== %s ===" % ds)
        run_dataset(ds, device=args.device, out_file=out_file)


if __name__ == "__main__":
    main()
