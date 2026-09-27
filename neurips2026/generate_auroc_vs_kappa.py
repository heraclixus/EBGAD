"""Plot AUROC vs kappa for multiple datasets and rho values.

This directly tests whether kappa matters for anomaly detection.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import compute_spectral_energy, Q_rho_eigenvalues, solve_rho_newton
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from data_utils import load_data
from sklearn.metrics import roc_auc_score


def compute_auroc_grid(data, trainer, kappa_grid, rho_grid, device="cuda"):
    """Compute AUROC over (kappa, rho) grid for precision_energy and precision_ratio."""
    x_normed = trainer.normalize(data.x.to(device).float())
    V = trainer.V
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    y = data.y.cpu().numpy()
    mask = (y == 0) | (y == 1)

    results = {}
    for score_method in ["precision_energy", "precision_ratio"]:
        auc_grid = np.zeros((len(rho_grid), len(kappa_grid)))
        for i, rho in enumerate(rho_grid):
            for j, kappa in enumerate(kappa_grid):
                q = Q_rho_eigenvalues(lam_L, rho, kappa)
                lam_Q = torch.tensor(q, dtype=torch.float32, device=device)
                if score_method == "precision_energy":
                    scores = precision_energy_anomaly(
                        x_normed, trainer.template, V, lam_Q,
                        alpha=2.0, T=1.0, prior_mean_mode="stationary")
                else:
                    scores = precision_ratio_anomaly(
                        x_normed, trainer.template, V, lam_Q,
                        alpha=2.0, T=1.0, prior_mean_mode="stationary")
                s = scores.cpu().numpy()
                try:
                    auc_grid[i, j] = roc_auc_score(y[mask], s[mask])
                except:
                    auc_grid[i, j] = 0.5
        results[score_method] = auc_grid

    # Also compute with Newton-optimal rho for each kappa
    for score_method in ["precision_energy", "precision_ratio"]:
        d = data.x.shape[1]
        delta = (x_normed - trainer.template).cpu().numpy()
        V_np = V.cpu().numpy()
        S = compute_spectral_energy(delta, V_np)

        auc_newton = np.zeros(len(kappa_grid))
        rho_newton = np.zeros(len(kappa_grid))
        for j, kappa in enumerate(kappa_grid):
            rho_opt = solve_rho_newton(S, lam_L, kappa, d)
            rho_newton[j] = rho_opt
            q = Q_rho_eigenvalues(lam_L, rho_opt, kappa)
            lam_Q = torch.tensor(q, dtype=torch.float32, device=device)
            if score_method == "precision_energy":
                scores = precision_energy_anomaly(
                    x_normed, trainer.template, V, lam_Q,
                    alpha=2.0, T=1.0, prior_mean_mode="stationary")
            else:
                scores = precision_ratio_anomaly(
                    x_normed, trainer.template, V, lam_Q,
                    alpha=2.0, T=1.0, prior_mean_mode="stationary")
            s = scores.cpu().numpy()
            try:
                auc_newton[j] = roc_auc_score(y[mask], s[mask])
            except:
                auc_newton[j] = 0.5
        results[score_method + "_newton"] = auc_newton
        results["rho_newton"] = rho_newton

    return results


def plot_auroc_vs_kappa(results_dict, output_dir):
    """Plot AUROC vs kappa for each dataset."""
    datasets = list(results_dict.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels = {"enron": "Enron", "weibo": "Weibo", "facebook": "Facebook",
              "elliptic": "Elliptic", "acm": "ACM", "reddit": "Reddit",
              "yelpchi": "YelpChi", "amazon": "Amazon", "blogcatalog": "BlogCatalog"}

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        res = results_dict[ds]
        kappa_grid = res["kappa_grid"]
        rho_grid = res["rho_grid"]

        # Plot fixed-rho curves
        for i, rho in enumerate(rho_grid):
            if rho in [0.0, 0.3, 0.5, 0.7, 1.0]:
                ax.plot(kappa_grid, res["precision_energy"][i, :],
                        '--', alpha=0.4, linewidth=1, label=r"$J^*$, $\rho$=%.1f" % rho)

        # Plot Newton-optimal rho curve (thick)
        ax.plot(kappa_grid, res["precision_energy_newton"],
                'b-', linewidth=2.5, label=r"$J^*$, $\rho^*(\kappa)$")
        ax.plot(kappa_grid, res["precision_ratio_newton"],
                'r-', linewidth=2.5, label=r"$R$, $\rho^*(\kappa)$")

        # Mark kappa=0 (no-kappa)
        ax.axvline(x=kappa_grid[0], color="green", linestyle=":", alpha=0.7,
                   linewidth=2, label=r"$\kappa=0$")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa$", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(labels.get(ds, ds), fontsize=13, fontweight="bold")
        ax.set_ylim([0.0, 1.0])
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "auroc_vs_kappa.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="tex/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["enron", "weibo", "facebook", "elliptic"]
    kappa_grid = np.concatenate([[0.001], np.logspace(-2, 2, 40)])
    rho_grid = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    all_results = {}
    for ds in datasets:
        print("=== %s ===" % ds, flush=True)
        data = load_data(ds)

        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type="low", graph_type="original", gamma=1.0,
            normalize_mode="zscore", prior_mean_mode="stationary",
            laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        trainer._setup(data, device=args.device)

        res = compute_auroc_grid(data, trainer, kappa_grid, rho_grid, args.device)
        res["kappa_grid"] = kappa_grid
        res["rho_grid"] = rho_grid
        all_results[ds] = res

        # Print summary
        newton_je = res["precision_energy_newton"]
        newton_jr = res["precision_ratio_newton"]
        print("  J* at kappa=0: %.1f%%, best: %.1f%% at kappa=%.2f" % (
            newton_je[0]*100, newton_je.max()*100, kappa_grid[newton_je.argmax()]))
        print("  R  at kappa=0: %.1f%%, best: %.1f%% at kappa=%.2f" % (
            newton_jr[0]*100, newton_jr.max()*100, kappa_grid[newton_jr.argmax()]))

    plot_auroc_vs_kappa(all_results, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
