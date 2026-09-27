"""Generate spectral analysis figures for §4.4.

1. Spectral energy S_j vs eigenvalue lambda_j (where anomalies live)
2. Per-mode calibration r_j = q_j * S_j / d (prior quality)
3. Optimized rho values across datasets
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
from soc.prior_optimizer import (
    compute_spectral_energy, Q_rho_eigenvalues, solve_rho_newton,
)
from data_utils import load_data


def setup_trainer(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    cfg = SOCTrainerConfig(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0, alpha=2.0, T=1.0,
        template_type="low", graph_type="original", gamma=1.0,
        normalize_mode="zscore", prior_mean_mode="stationary",
        laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
        epochs=0, normalize_features=True,
    )
    trainer = SOCTrainer(cfg)
    trainer._setup(data, device=device)
    return data, trainer


def plot_spectral_energy(datasets, output_dir, device="cuda"):
    """Plot S_j vs lambda_j for multiple datasets."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5*len(datasets), 3.5))
    if len(datasets) == 1:
        axes = [axes]

    labels = {"enron": "Enron", "weibo": "Weibo", "facebook": "Facebook",
              "elliptic": "Elliptic", "acm": "ACM", "reddit": "Reddit"}

    for idx, ds in enumerate(datasets):
        data, trainer = setup_trainer(ds, device)
        x_normed = trainer.normalize(data.x.to(device).float())
        V_np = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        delta = (x_normed - trainer.template).cpu().numpy()
        S = compute_spectral_energy(delta, V_np)
        d = data.x.shape[1]

        ax = axes[idx]
        ax.scatter(lam_L, S/d, s=8, alpha=0.6, c="tab:blue")
        ax.set_xlabel(r"$\lambda_j$ (Laplacian eigenvalue)", fontsize=10)
        if idx == 0:
            ax.set_ylabel(r"$S_j / d$ (spectral energy per feature)", fontsize=10)
        ax.set_title(labels.get(ds, ds), fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "spectral_energy.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def plot_calibration(datasets, output_dir, device="cuda"):
    """Plot per-mode calibration r_j = q_j * S_j / d."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5*len(datasets), 3.5))
    if len(datasets) == 1:
        axes = [axes]

    labels = {"enron": "Enron", "weibo": "Weibo", "facebook": "Facebook",
              "elliptic": "Elliptic", "acm": "ACM", "reddit": "Reddit"}

    for idx, ds in enumerate(datasets):
        data, trainer = setup_trainer(ds, device)
        n, d = data.x.shape
        x_normed = trainer.normalize(data.x.to(device).float())
        V_np = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        delta = (x_normed - trainer.template).cpu().numpy()
        S = compute_spectral_energy(delta, V_np)

        # Solve optimal rho for kappa=1
        rho_opt = solve_rho_newton(S, lam_L, kappa=1.0, d=d)
        q = Q_rho_eigenvalues(lam_L, rho_opt, kappa=1.0)
        r_j = q * S / d

        ax = axes[idx]
        ax.scatter(lam_L, r_j, s=8, alpha=0.6, c="tab:green")
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label=r"$r_j=1$ (calibrated)")
        ax.set_xlabel(r"$\lambda_j$", fontsize=10)
        if idx == 0:
            ax.set_ylabel(r"$r_j = q_j S_j / d$", fontsize=10)
        ax.set_title(r"%s ($\rho^*$=%.2f)" % (labels.get(ds, ds), rho_opt),
                     fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "calibration_diagnostic.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def plot_auroc_vs_rho(datasets, output_dir, device="cuda"):
    """Plot AUROC vs rho for fixed kappa values."""
    from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
    from sklearn.metrics import roc_auc_score

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5*len(datasets), 3.5))
    if len(datasets) == 1:
        axes = [axes]

    labels = {"enron": "Enron", "weibo": "Weibo", "facebook": "Facebook",
              "elliptic": "Elliptic"}

    rho_grid = np.linspace(0.0, 1.0, 30)

    for idx, ds in enumerate(datasets):
        data, trainer = setup_trainer(ds, device)
        x_normed = trainer.normalize(data.x.to(device).float())
        V = trainer.V
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        y = data.y.cpu().numpy()
        mask = (y == 0) | (y == 1)

        ax = axes[idx]
        for kappa in [0.0, 0.1, 1.0, 10.0]:
            aucs_je = []
            for rho in rho_grid:
                q = Q_rho_eigenvalues(lam_L, rho, kappa)
                lam_Q = torch.tensor(q, dtype=torch.float32, device=device)
                scores = precision_energy_anomaly(
                    x_normed, trainer.template, V, lam_Q,
                    alpha=2.0, T=1.0, prior_mean_mode="stationary")
                s = scores.cpu().numpy()
                try:
                    aucs_je.append(roc_auc_score(y[mask], s[mask]))
                except:
                    aucs_je.append(0.5)
            ax.plot(rho_grid, aucs_je, linewidth=1.5,
                    label=r"$\kappa$=%.1f" % kappa)

        ax.set_xlabel(r"$\rho$ (graph trust)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(labels.get(ds, ds), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(output_dir, "auroc_vs_rho.pdf")
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

    print("=== Spectral energy ===", flush=True)
    plot_spectral_energy(datasets, args.output_dir, args.device)

    print("=== Calibration diagnostic ===", flush=True)
    plot_calibration(datasets, args.output_dir, args.device)

    print("=== AUROC vs rho ===", flush=True)
    plot_auroc_vs_rho(datasets, args.output_dir, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
