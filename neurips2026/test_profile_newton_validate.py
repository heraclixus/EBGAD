"""Validate profile-Newton matches L-BFGS-B on all datasets."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, profile_newton_optimize,
    optimize_joint_lbfgsb_stationary, coordinate_descent_stationary,
)
from data_utils import load_data


def main():
    device = "cuda"
    datasets = ["enron", "weibo", "reddit", "facebook", "acm", "yelpchi",
                 "amazon", "blogcatalog", "elliptic", "elliptic_plus_plus"]

    print("%-18s %10s %10s %10s | %8s %8s %8s" % (
        "Dataset", "CD ML", "LBFGSB ML", "Newton ML", "CD ρ", "LB ρ", "Newt ρ"))
    print("-" * 90)

    for ds in datasets:
        data = load_data("YelpChi" if ds == "yelpchi" else ds)
        n, d = data.x.shape

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

        # 1. Coordinate descent
        cd = coordinate_descent_stationary(S, lam_L, n, d, n_iters=10,
                                           use_data_driven_bounds=True)

        # 2. L-BFGS-B
        lb = optimize_joint_lbfgsb_stationary(S, lam_L, n, d, n_restarts=10)

        # 3. Profile-Newton
        pn = profile_newton_optimize(S, lam_L, n, d, use_data_driven_bounds=True)

        print("%-18s %10.1f %10.1f %10.1f | %8.4f %8.4f %8.4f" % (
            ds, cd["marginal_likelihood"], lb["marginal_likelihood"],
            pn["marginal_likelihood"], cd["rho"], lb["rho"], pn["rho"]))

    print("\nDone!")


if __name__ == "__main__":
    main()
