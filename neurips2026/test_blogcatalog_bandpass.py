"""BlogCatalog bandpass template sweep.

Adds mix/affinity_mix templates with pi blending to the optimizer.
Tests all encoder types + both prior modes + bandpass templates.
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
    compute_spectral_energy, coordinate_descent, coordinate_descent_stationary,
)
from data_utils import load_data


def run_dataset(ds, device="cuda", out_file=None):
    data = load_data(ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    encoder_configs = [
        None,
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 64,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 128,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 256,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "gcn", "encoder_hid_dim": 128,
         "encoder_num_layers": 4, "encoder_dropout": 0.3, "encoder_lr": 0.01,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "gat", "encoder_hid_dim": 128,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ]

    # All templates including bandpass
    templates = ["zero", "low", "high", "mix", "affinity", "affinity_high", "affinity_mix"]
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
        for prior_mode in prior_modes:
            for t in templates:
                for g in graphs:
                    for gamma in gammas:
                        for norm in norms:
                            count += 1
                            # Setup trainer with pi=0.5 as starting point
                            trainer_kwargs = {
                                "kappa": 1.0, "nu": 1.0, "rho": 0.5,
                                "lam_penalty": 50.0, "alpha": 2.0, "T": 1.0,
                                "template_type": t, "graph_type": g,
                                "gamma": gamma, "pi": 0.5,
                                "normalize_mode": norm,
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
                            V = trainer.V
                            V_np = V.cpu().numpy()
                            lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V_np.shape[1])

                            # For mix templates, optimize pi continuously
                            if "mix" in t:
                                best_pi = 0.5; best_pi_ml = -np.inf
                                from scipy.optimize import minimize_scalar
                                from soc.soc_bridge import compute_template
                                lam_L_t = trainer.graph_eigen.lam_L if hasattr(trainer.graph_eigen, 'lam_L') else torch.ones(V.shape[1], device=V.device)
                                node_w = trainer.node_weights if hasattr(trainer, 'node_weights') else None

                                def _eval_pi(pi_val):
                                    tmpl = compute_template(
                                        x_normed, V, lam_L_t,
                                        template_type=t, gamma=gamma, nu=1.0, pi=pi_val,
                                        node_weights=node_w)
                                    d_vec = (x_normed - tmpl).cpu().numpy()
                                    s_vec = compute_spectral_energy(d_vec, V_np)
                                    best_ml = -np.inf
                                    for ri in [0.0, 0.5, 1.0]:
                                        for ki in [0.1, 1.0, 5.0]:
                                            try:
                                                if prior_mode == "stationary":
                                                    o = coordinate_descent_stationary(
                                                        s_vec, lam_L, n, d,
                                                        rho_init=ri, kappa_init=ki,
                                                        n_iters=5, use_data_driven_bounds=True)
                                                else:
                                                    o = coordinate_descent(
                                                        s_vec, lam_L, n, d,
                                                        rho_init=ri, kappa_init=ki,
                                                        n_iters=5, use_data_driven_bounds=True)
                                                if o["marginal_likelihood"] > best_ml:
                                                    best_ml = o["marginal_likelihood"]
                                            except: pass
                                    return -best_ml

                                res = minimize_scalar(_eval_pi, bounds=(0.01, 0.99), method='bounded')
                                best_pi = res.x
                                # Recompute template with optimal pi
                                trainer.cfg.pi = best_pi
                                trainer.template = compute_template(
                                    x_normed, V, lam_L_t,
                                    template_type=t, gamma=gamma, nu=1.0, pi=best_pi,
                                    node_weights=node_w)
                                pi = best_pi
                            else:
                                pi = 0.5

                            delta = (x_normed - trainer.template).cpu().numpy()
                            S = compute_spectral_energy(delta, V_np)

                            best_opt = None; best_opt_ml = -np.inf
                            for rho_init in [0.0, 0.5, 1.0]:
                                for kappa_init in [0.1, 1.0, 5.0]:
                                    try:
                                        if prior_mode == "stationary":
                                            opt = coordinate_descent_stationary(
                                                S, lam_L, n, d,
                                                rho_init=rho_init, kappa_init=kappa_init,
                                                n_iters=10, use_data_driven_bounds=True)
                                        else:
                                            opt = coordinate_descent(
                                                S, lam_L, n, d,
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
                                "gamma": gamma, "pi": pi,
                                "alpha": best_opt.get("alpha_T", 2.0), "T": 1.0,
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
                                if improved and out_file:
                                    _save(out_file, ds, best_scores, score_methods, count)
                            except:
                                pass

                            if count % 50 == 0:
                                enc_tag = enc.get("encoder_type", "none") if enc else "none"
                                hid = enc.get("encoder_hid_dim", 0) if enc else 0
                                bests = " ".join("%s=%.1f%%" % (m[:2], best_scores[m][0]*100) for m in score_methods)
                                pi_str = " pi=%.2f" % pi if "mix" in t else ""
                                print("  [%d enc=%s/%d t=%s pm=%s%s] %s" % (
                                    count, enc_tag, hid, t, prior_mode, pi_str, bests), flush=True)

    # Final save
    if out_file:
        _save(out_file, ds, best_scores, score_methods, count, complete=True)

    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]
    enc = best_cfg.get("encoder_type", "none") if best_cfg.get("use_encoder") else "none"
    print("  BEST: %.1f%% method=%s enc=%s template=%s pi=%.1f" % (
        best_auc * 100, best_method, enc,
        best_cfg.get("template_type", "?"), best_cfg.get("pi", 0.5)))


def _save(out_file, ds, best_scores, score_methods, configs_done, complete=False):
    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]
    result = {
        "dataset": ds, "best_auc": best_auc,
        "best_config": best_cfg,
        "all_best": {m: {"auc": v[0], "config": v[1]} for m, v in best_scores.items()},
        "configs_evaluated": configs_done,
        "complete": complete,
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    if complete:
        print("  [FINAL saved %.1f%% after %d configs]" % (best_auc * 100, configs_done), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="blogcatalog,acm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_bandpass_%s.json" % ds)
        print("=== %s ===" % ds)
        run_dataset(ds, device=args.device, out_file=out_file)


if __name__ == "__main__":
    main()
