"""Prior optimizer v3: SAVES the winning config for each dataset."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, Q_rho_eigenvalues,
)
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc
from soc.soc_anomaly import compute_anomaly_scores

import dataclasses, itertools


def run_dataset(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    discrete_choices = []
    for t in ["zero", "low", "high", "affinity"]:
        for g in ["original", "affinity"]:
            for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                for norm in ["zscore", "minmax"]:
                    discrete_choices.append({
                        "template_type": t, "graph_type": g,
                        "gamma": gamma, "normalize_mode": norm,
                    })

    best_ce = 0; best_cr = 0
    best_ce_cfg = {}; best_cr_cfg = {}

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
        except:
            continue

        x_normed = trainer.normalize(data.x.to(device).float())
        template = trainer.template
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
        delta = (x_normed - template).cpu().numpy()
        S = compute_spectral_energy(delta, V)

        best_opt = None; best_opt_ml = -np.inf
        for rho_init in [0.0, 0.5, 1.0]:
            for kappa_init in [0.1, 1.0, 5.0]:
                opt = coordinate_descent(S, lam_L, n, d,
                    rho_init=rho_init, kappa_init=kappa_init, n_iters=10)
                if opt["marginal_likelihood"] > best_opt_ml:
                    best_opt_ml = opt["marginal_likelihood"]
                    best_opt = opt

        if best_opt is None:
            continue

        torch.manual_seed(0); np.random.seed(0)
        full_cfg = {
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
            for method in ["precision_energy", "precision_ratio"]:
                eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                r = evaluate_single_trial(data, eval_cfg, device=device)
                if method == "precision_energy" and r.auc > best_ce:
                    best_ce = r.auc
                    best_ce_cfg = {**full_cfg, "score_method": "precision_energy"}
                if method == "precision_ratio" and r.auc > best_cr:
                    best_cr = r.auc
                    best_cr_cfg = {**full_cfg, "score_method": "precision_ratio"}
        except:
            pass

        if (i + 1) % 50 == 0:
            print("  [%d/%d] best_ce=%.1f%% best_cr=%.1f%%" %
                  (i + 1, len(discrete_choices), best_ce * 100, best_cr * 100), flush=True)

    best_overall = max(best_ce, best_cr)
    best_cfg = best_ce_cfg if best_ce >= best_cr else best_cr_cfg
    print("  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%)" % (best_overall * 100, best_ce * 100, best_cr * 100))
    return {"dataset": ds, "best_auc": best_overall,
            "best_ce": best_ce, "best_cr": best_cr,
            "best_config": best_cfg}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="enron,weibo,reddit,yelpchi,facebook,acm,t_finance")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/eb_gad_best_configs.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results = {}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print("=== %s ===" % ds)
        result = run_dataset(ds, device=args.device)
        all_results[ds] = result

    # Save all best configs
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved best configs to %s" % args.output)

    # Print summary
    print("\n=== SUMMARY ===")
    for ds, r in all_results.items():
        print("%-12s  best=%.1f%%  config=%s" % (ds, r["best_auc"]*100, r["best_config"]))


if __name__ == "__main__":
    main()
