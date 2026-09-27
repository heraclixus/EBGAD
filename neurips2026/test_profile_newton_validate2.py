"""Validate profile-Newton vs L-BFGS-B vs CD with CONSISTENT data-driven bounds."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, profile_newton_optimize,
    optimize_joint_lbfgsb_stationary, coordinate_descent_stationary,
    compute_kappa_bounds, Q_rho_eigenvalues, marginal_log_likelihood_stationary,
)
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from sklearn.metrics import roc_auc_score


def main():
    device = "cuda"
    datasets = ["enron", "weibo", "reddit", "facebook", "acm", "yelpchi",
                 "amazon", "blogcatalog", "elliptic", "elliptic_plus_plus"]

    print("%-18s | %10s %10s %10s | %8s %8s %8s | %6s %6s %6s" % (
        "Dataset", "CD ML", "LB ML", "PN ML", "CD ρ", "LB ρ", "PN ρ",
        "CD AUC", "LB AUC", "PN AUC"))
    print("-" * 120)

    for ds in datasets:
        data = load_data("YelpChi" if ds == "yelpchi" else ds)
        n, d = data.x.shape
        y = data.y.cpu().numpy()
        mask = (y == 0) | (y == 1)

        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type="low", graph_type="original", gamma=1.0,
            normalize_mode="zscore", prior_mean_mode="stationary",
            laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=device)
        except:
            print("%-18s SETUP FAILED" % ds)
            continue

        x_normed = trainer.normalize(data.x.to(device).float())
        V_np = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        delta = (x_normed - trainer.template).cpu().numpy()
        S = compute_spectral_energy(delta, V_np)

        # Consistent data-driven bounds
        kr = compute_kappa_bounds(S, lam_L, 0.5, n, d)

        # 1. CD with data-driven bounds
        cd_best = None; cd_ml = -np.inf
        for ri in [0.0, 0.5, 1.0]:
            for ki in [0.1, 1.0, 5.0]:
                opt = coordinate_descent_stationary(S, lam_L, n, d,
                    rho_init=ri, kappa_init=ki, n_iters=10,
                    kappa_range=kr)
                if opt["marginal_likelihood"] > cd_ml:
                    cd_ml = opt["marginal_likelihood"]
                    cd_best = opt

        # 2. L-BFGS-B with SAME bounds
        lb = optimize_joint_lbfgsb_stationary(S, lam_L, n, d,
                                               kappa_range=kr, n_restarts=10)

        # 3. Profile-Newton with SAME bounds
        pn = profile_newton_optimize(S, lam_L, n, d, kappa_range=kr)

        # Evaluate AUC for each
        aucs = {}
        for name, opt in [("cd", cd_best), ("lb", lb), ("pn", pn)]:
            trainer.cfg.rho = opt["rho"]
            trainer.cfg.kappa = opt["kappa"]
            lam_Q = torch.tensor(
                Q_rho_eigenvalues(lam_L, opt["rho"], opt["kappa"]),
                dtype=torch.float32, device=device)
            from soc.soc_anomaly import precision_energy_anomaly
            scores = precision_energy_anomaly(
                x_normed, trainer.template, trainer.V, lam_Q,
                alpha=2.0, T=1.0, prior_mean_mode="stationary").cpu().numpy()
            aucs[name] = roc_auc_score(y[mask], scores[mask])

        print("%-18s | %10.1f %10.1f %10.1f | %8.4f %8.4f %8.4f | %6.1f %6.1f %6.1f" % (
            ds, cd_best["marginal_likelihood"], lb["marginal_likelihood"],
            pn["marginal_likelihood"],
            cd_best["rho"], lb["rho"], pn["rho"],
            aucs["cd"]*100, aucs["lb"]*100, aucs["pn"]*100))

    print("\nDone!")


if __name__ == "__main__":
    main()
