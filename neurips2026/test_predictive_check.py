"""Compare prior selection strategies: ML, predictive check, and oracle.

For each dataset, evaluate a grid of priors and compare:
1. ML-selected (argmax marginal likelihood)
2. PPC-selected (argmin mean absolute log-ratio)
3. Oracle (argmax AUC)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, itertools
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, Q_rho_eigenvalues,
    marginal_log_likelihood,
)
from soc.predictive_check import predictive_check_statistics, predictive_check_score
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc


def run_dataset(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n, d_feat = data.x.shape

    grid = []
    for tpl in ["zero", "low", "high", "affinity"]:
        for gt in ["original", "affinity"]:
            for gamma in [0.1, 0.2, 0.5, 1.0, 2.0]:
                for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    for kappa in [0.1, 0.5, 1.0, 5.0, 10.0]:
                        grid.append(dict(
                            template_type=tpl, graph_type=gt,
                            gamma=gamma, rho=rho, kappa=kappa,
                        ))

    best_ml = -np.inf; best_ml_auc = 0; best_ml_cfg = {}
    best_ppc = -np.inf; best_ppc_auc = 0; best_ppc_cfg = {}
    best_auc = 0; best_auc_cfg = {}
    total = 0

    for i, prior in enumerate(grid):
        torch.manual_seed(0); np.random.seed(0)
        try:
            cfg = SOCTrainerConfig(
                kappa=prior["kappa"], nu=1.0, rho=prior["rho"],
                lam_penalty=50.0, alpha=2.0, T=1.0,
                template_type=prior["template_type"],
                graph_type=prior["graph_type"],
                gamma=prior["gamma"], laplacian_variant="sym",
                d_hidden=64, n_hidden_layers=2,
                epochs=0, normalize_features=True,
            )
            trainer = SOCTrainer(cfg)
            trainer._setup(data, device=device)

            x_normed = trainer.normalize(data.x.to(device).float())
            template = trainer.template
            V = trainer.V.cpu().numpy()
            lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])

            delta = (x_normed - template).cpu().numpy()
            S = compute_spectral_energy(delta, V)
            q = Q_rho_eigenvalues(lam_L, prior["rho"], prior["kappa"])

            # ML score
            ml = sum(-S[j]*q[j]/2 + (n*d_feat/2)*np.log(max(q[j], 1e-12)) for j in range(len(q)))

            # PPC score
            ppc = predictive_check_score(S, q, n, d_feat)

            # AUC (evaluate CE and CR)
            cfg_eval = SOCGADConfig(
                dataset=ds, score_method="control_energy", score_K=1,
                d_hidden=64, n_hidden_layers=2, lr=0.001, epochs=1,
                parameterization="score", time_weighting="importance",
                lam_penalty=50.0, alpha=2.0, laplacian_variant="sym", **prior,
            )
            r = evaluate_single_trial(data, cfg_eval, device=device)
            auc_ce = r.auc

            cfg_eval2 = SOCGADConfig(
                dataset=ds, score_method="ce_ratio", score_K=1,
                d_hidden=64, n_hidden_layers=2, lr=0.001, epochs=1,
                parameterization="score", time_weighting="importance",
                lam_penalty=50.0, alpha=2.0, laplacian_variant="sym", **prior,
            )
            r2 = evaluate_single_trial(data, cfg_eval2, device=device)
            auc_cr = r2.auc
            auc = max(auc_ce, auc_cr)

            total += 1

            if ml > best_ml:
                best_ml = ml; best_ml_auc = auc; best_ml_cfg = prior.copy()
            if ppc > best_ppc:
                best_ppc = ppc; best_ppc_auc = auc; best_ppc_cfg = prior.copy()
            if auc > best_auc:
                best_auc = auc; best_auc_cfg = prior.copy()

        except Exception as e:
            pass

        if (i + 1) % 500 == 0:
            print("  [%d/%d] ML=%.1f%% PPC=%.1f%% Oracle=%.1f%%" %
                  (i + 1, len(grid), best_ml_auc * 100, best_ppc_auc * 100, best_auc * 100), flush=True)

    print("%-12s  %d configs | ML=%.1f%% | PPC=%.1f%% | Oracle=%.1f%%" %
          (ds, total, best_ml_auc * 100, best_ppc_auc * 100, best_auc * 100))
    print("  ML  config: %s" % best_ml_cfg)
    print("  PPC config: %s" % best_ppc_cfg)
    print("  Oracle cfg: %s" % best_auc_cfg)
    return best_ml_auc, best_ppc_auc, best_auc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="enron,weibo,reddit,yelpchi,facebook,t_finance")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Prior Selection Comparison: ML vs PPC vs Oracle")
    for ds in args.datasets.split(","):
        ds = ds.strip()
        print("=== %s ===" % ds)
        run_dataset(ds, device=args.device)


if __name__ == "__main__":
    main()
