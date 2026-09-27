"""Test the closed-form prior optimizer on all datasets.

For each dataset × discrete choice (template, graph_type):
1. Compute eigenvectors, template, spectral energies S_j
2. Run coordinate descent to optimize (λ, αT, ρ, κ)
3. Evaluate CE and CR with the optimized prior
4. Compare against grid search results
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import control_energy_anomaly, ce_ratio_anomaly
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, Q_rho_eigenvalues,
    predictive_variance, marginal_log_likelihood, per_mode_diagnostic,
)
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc


def run_dataset(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    discrete_choices = [
        {"template_type": t, "graph_type": g}
        for t in ["zero", "low", "high", "affinity"]
        for g in ["original", "affinity"]
    ]

    best_ce = 0
    best_cr = 0
    best_config = {}
    results = []

    for disc in discrete_choices:
        # Build a minimal trainer to get eigenvectors and template
        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5,
            lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type=disc["template_type"],
            graph_type=disc["graph_type"],
            gamma=0.5, laplacian_variant="sym",
            d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=device)
        except Exception as e:
            print("  skip %s: %s" % (disc, e))
            continue

        x_normed = trainer.normalize(data.x.to(device).float())
        template = trainer.template
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])

        delta = (x_normed - template).cpu().numpy()
        S = compute_spectral_energy(delta, V)
        k = len(S)

        # Run coordinate descent
        opt = coordinate_descent(S, lam_L, n, d, n_iters=10)

        # Evaluate CE and CR with optimized parameters
        q_opt = Q_rho_eigenvalues(lam_L, opt["rho"], opt["kappa"])
        alpha_opt = opt["alpha_T"]  # we use T=1, so alpha = alpha_T
        lam_opt = opt["lam_penalty"]

        torch.manual_seed(0)
        np.random.seed(0)
        cfg_eval = SOCGADConfig(
            dataset=ds,
            rho=opt["rho"], kappa=opt["kappa"], gamma=0.5,
            lam_penalty=lam_opt, alpha=alpha_opt, T=1.0,
            template_type=disc["template_type"],
            graph_type=disc["graph_type"],
            d_hidden=64, n_hidden_layers=2,
            epochs=1, score_method="control_energy", score_K=1,
            laplacian_variant="sym",
        )
        try:
            r_ce = evaluate_single_trial(data, cfg_eval, device=device)
            cfg_eval.score_method = "ce_ratio"
            r_cr = evaluate_single_trial(data, cfg_eval, device=device)
        except Exception as e:
            print("  eval error %s: %s" % (disc, e))
            continue

        result = {
            **disc,
            **opt,
            "auc_ce": float(r_ce.auc),
            "auc_cr": float(r_cr.auc),
        }
        results.append(result)

        if r_ce.auc > best_ce:
            best_ce = r_ce.auc
            best_config = {**result, "score": "CE"}
        if r_cr.auc > best_cr:
            best_cr = r_cr.auc
            best_config_cr = {**result, "score": "CR"}

        print("  %-10s %-10s  rho=%.2f kap=%.2f lam=%.0f aT=%.1f  CE=%.1f%% CR=%.1f%%  ML=%.0f" % (
            disc["template_type"], disc["graph_type"],
            opt["rho"], opt["kappa"], opt["lam_penalty"], opt["alpha_T"],
            r_ce.auc * 100, r_cr.auc * 100, opt["marginal_likelihood"]))

    best_overall = max(best_ce, best_cr)
    print("  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%)" % (best_overall * 100, best_ce * 100, best_cr * 100))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="enron,weibo,reddit,yelpchi,facebook,acm,t_finance")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/prior_optimizer.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fout = open(args.output, "w")

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print("=== %s ===" % ds)
        results = run_dataset(ds, device=args.device)
        for r in results:
            r["dataset"] = ds
            fout.write(json.dumps(r) + "\n")
        fout.flush()

    fout.close()
    print("Saved to %s" % args.output)


if __name__ == "__main__":
    main()
