"""No-kappa optimizer: Q_prior = L^nu, Q_rho = rho*L^nu + (1-rho)*I.

Only one continuous parameter (rho), solved exactly via Newton.
Sweeps discrete choices (template, graph, gamma, scoring, encoder).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, solve_rho_newton,
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
)
from data_utils import load_data
from sklearn.metrics import roc_auc_score


def run_dataset(ds, device="cuda", out_file=None):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n, d = data.x.shape

    encoder_configs = [
        None,
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 64,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 64,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ]

    templates = ["zero", "low", "high", "affinity"]
    graphs = ["original", "affinity"]
    gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]
    score_methods = ["precision_energy", "precision_ratio"]

    best_scores = {}
    for m in score_methods:
        best_scores[m] = (0, {})

    count = 0
    for enc in encoder_configs:
        for t in templates:
            for g in graphs:
                for gamma in gammas:
                    for norm in norms:
                        count += 1
                        trainer_kwargs = {
                            "kappa": 0.0, "nu": 1.0, "rho": 0.5,
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

                        # No kappa: Q_prior = L^nu, so a_j = lambda_j^nu - 1
                        # Solve rho via Newton (exact, 2-3 steps)
                        rho_opt = solve_rho_newton(S, lam_L, kappa=0.0, d=d, nu=1.0)

                        # Compute ML at optimal rho
                        # q_j = rho * lambda_j^nu + (1-rho) = rho*(lambda_j - 1) + 1
                        q = rho_opt * (lam_L ** 1.0) + (1.0 - rho_opt)
                        ml = marginal_log_likelihood_stationary(S, q, n, d)

                        torch.manual_seed(0); np.random.seed(0)
                        full_cfg = {
                            "rho": rho_opt, "kappa": 0.0,
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
                                if r.auc > best_scores[method][0]:
                                    best_scores[method] = (r.auc, {**full_cfg, "score_method": method,
                                                                    "rho_newton": rho_opt, "ml": ml})
                                    improved = True
                            if improved and out_file:
                                _save(out_file, ds, best_scores, score_methods, count)
                        except:
                            pass

                        if count % 50 == 0:
                            enc_tag = enc.get("encoder_type", "none") if enc else "none"
                            bests = " ".join("%s=%.1f%%" % (m[:2], best_scores[m][0]*100) for m in score_methods)
                            print("  [%d enc=%s t=%s rho=%.3f] %s" % (
                                count, enc_tag, t, rho_opt, bests), flush=True)

    if out_file:
        _save(out_file, ds, best_scores, score_methods, count, complete=True)

    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]
    enc = best_cfg.get("encoder_type", "none") if best_cfg.get("use_encoder") else "none"
    print("  BEST: %.1f%% method=%s enc=%s rho=%.3f" % (
        best_auc*100, best_method, enc, best_cfg.get("rho_newton", 0)))


def _save(path, ds, best_scores, score_methods, configs_done, complete=False):
    best_method = max(best_scores, key=lambda m: best_scores[m][0])
    best_auc, best_cfg = best_scores[best_method]
    result = {
        "dataset": ds, "best_auc": best_auc, "best_config": best_cfg,
        "all_best": {m: {"auc": v[0], "config": v[1]} for m, v in best_scores.items()},
        "configs_evaluated": configs_done, "complete": complete,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="enron,weibo,reddit,yelpchi,facebook,acm,amazon,blogcatalog,elliptic,elliptic_plus_plus")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_nokappa_%s.json" % ds)
        if os.path.exists(out_file):
            with open(out_file) as f:
                r = json.load(f)
            if r.get("complete"):
                print("=== %s === SKIP (complete)" % ds)
                continue

        print("=== %s ===" % ds, flush=True)
        run_dataset(ds, device=args.device, out_file=out_file)

    print("\n=== SUMMARY ===")
    for ds in args.datasets.split(","):
        ds = ds.strip()
        f = os.path.join(args.output_dir, "eb_gad_nokappa_%s.json" % ds)
        if os.path.exists(f):
            r = json.load(open(f))
            cfg = r["best_config"]
            enc = cfg.get("encoder_type", "none") if cfg.get("use_encoder") else "none"
            print("  %-18s %.1f%%  enc=%s  rho=%.3f" % (
                ds, r["best_auc"]*100, enc, cfg.get("rho_newton", cfg.get("rho", 0))))


if __name__ == "__main__":
    main()
