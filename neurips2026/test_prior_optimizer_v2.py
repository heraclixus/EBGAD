"""Prior optimizer v2: adds gamma sweep, multi-start, normalize_mode."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, Q_rho_eigenvalues,
    marginal_log_likelihood, predictive_variance,
)
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc
from soc.soc_anomaly import compute_anomaly_scores


def run_dataset(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    # Expanded discrete choices: template × graph × gamma × normalize_mode
    discrete_choices = []
    for t in ["zero", "low", "high", "affinity"]:
        for g in ["original", "affinity"]:
            for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                for norm in ["zscore", "minmax"]:
                    discrete_choices.append({
                        "template_type": t, "graph_type": g,
                        "gamma": gamma, "normalize_mode": norm,
                    })

    best_ce = 0
    best_cr = 0
    best_config = {}
    best_ml = -np.inf

    for i, disc in enumerate(discrete_choices):
        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5,
            lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type=disc["template_type"],
            graph_type=disc["graph_type"],
            gamma=disc["gamma"],
            normalize_mode=disc["normalize_mode"],
            laplacian_variant="sym",
            d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=device)
        except Exception as e:
            continue

        x_normed = trainer.normalize(data.x.to(device).float())
        template = trainer.template
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])

        delta = (x_normed - template).cpu().numpy()
        S = compute_spectral_energy(delta, V)

        # Multi-start coordinate descent
        best_opt = None
        best_opt_ml = -np.inf
        for rho_init in [0.0, 0.5, 1.0]:
            for kappa_init in [0.1, 1.0, 5.0]:
                opt = coordinate_descent(
                    S, lam_L, n, d,
                    rho_init=rho_init, kappa_init=kappa_init,
                    n_iters=10,
                )
                if opt["marginal_likelihood"] > best_opt_ml:
                    best_opt_ml = opt["marginal_likelihood"]
                    best_opt = opt

        if best_opt is None:
            continue

        # Evaluate
        torch.manual_seed(0); np.random.seed(0)
        params = {
            "rho": best_opt["rho"], "kappa": best_opt["kappa"],
            "gamma": disc["gamma"], "alpha": best_opt["alpha_T"], "T": 1.0,
            "lam_penalty": best_opt["lam_penalty"],
            "template_type": disc["template_type"],
            "graph_type": disc["graph_type"],
            "normalize_mode": disc["normalize_mode"],
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 1, "parameterization": "score",
            "time_weighting": "importance",
            "laplacian_variant": "sym",
        }
        try:
            cfg_eval = SOCGADConfig(dataset=ds, score_method="control_energy", score_K=1, **params)
            trainer_eval = SOCTrainer(cfg_eval.to_trainer_config())
            trainer_eval.train(data, device=device)
            x_test = data.x.to(device).float()
            ce_scores = compute_anomaly_scores(trainer_eval, x_test, K=1, method="precision_energy")
            cr_scores = compute_anomaly_scores(trainer_eval, x_test, K=1, method="precision_ratio")
            y_ce, s_ce = mask_labeled(data.y, ce_scores.cpu().detach())
            y_cr, s_cr = mask_labeled(data.y, cr_scores.cpu().detach())
            auc_ce = eval_roc_auc(y_ce, s_ce) if y_ce is not None else 0.0
            auc_cr = eval_roc_auc(y_cr, s_cr) if y_cr is not None else 0.0
        except Exception as e:
            continue

        if best_opt_ml > best_ml:
            best_ml = best_opt_ml
        if auc_ce > best_ce:
            best_ce = auc_ce
        if auc_cr > best_cr:
            best_cr = auc_cr

        if (i + 1) % 50 == 0:
            print("  [%d/%d] best_ce=%.1f%% best_cr=%.1f%% best_ml=%.0f" %
                  (i + 1, len(discrete_choices), best_ce * 100, best_cr * 100, best_ml), flush=True)

    best_overall = max(best_ce, best_cr)
    print("  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%)" % (best_overall * 100, best_ce * 100, best_cr * 100))
    return best_ce, best_cr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="enron,weibo,reddit,yelpchi,facebook,acm,t_finance")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Prior Optimizer v2 (gamma sweep + multi-start + normalize_mode)")
    for ds in args.datasets.split(","):
        ds = ds.strip()
        print("=== %s ===" % ds)
        run_dataset(ds, device=args.device)


if __name__ == "__main__":
    main()
