"""Generate optimization analysis figures for the paper.

1. ML landscape heatmap over (rho, kappa) for selected datasets
2. Convergence: ML vs iteration for L-BFGS-B
3. Score dynamic range vs kappa (why bounds matter)
4. Multi-start analysis: best-of-k vs k
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, Q_rho_eigenvalues,
    marginal_log_likelihood_stationary, optimize_joint_lbfgsb_stationary,
)
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from data_utils import load_data


def compute_ml_landscape(S, lam_L, d, rho_grid, kappa_grid):
    """Compute ML over a grid of (rho, kappa)."""
    ml = np.zeros((len(rho_grid), len(kappa_grid)))
    n = len(lam_L)  # not used in stationary ML, but needed for dimensions
    for i, rho in enumerate(rho_grid):
        for j, kappa in enumerate(kappa_grid):
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml[i, j] = marginal_log_likelihood_stationary(S, q, n, d)
    return ml


def compute_score_range(data, trainer, kappa_grid, rho=0.5, device="cuda"):
    """Compute score dynamic range (max-min) vs kappa."""
    x_normed = trainer.normalize(data.x.to(device).float())
    V = trainer.V
    lam_L = trainer.graph_eigen.lam_L
    ranges = []
    for kappa in kappa_grid:
        lam_Q = Q_rho_eigenvalues(lam_L.cpu().numpy(), rho, kappa)
        lam_Q_t = torch.tensor(lam_Q, dtype=torch.float32, device=device)
        scores = precision_energy_anomaly(
            x_normed, trainer.template, V, lam_Q_t,
            alpha=2.0, T=1.0, prior_mean_mode="stationary")
        s = scores.cpu().numpy()
        ranges.append(np.percentile(s, 95) - np.percentile(s, 5))
    return np.array(ranges)


def plot_ml_landscape(S, lam_L, d, n, ds_name, output_dir):
    """Plot 2D ML heatmap over (rho, kappa)."""
    rho_grid = np.linspace(0.01, 0.99, 50)
    kappa_grid = np.logspace(-2, 1.5, 50)
    ml = compute_ml_landscape(S, lam_L, d, rho_grid, kappa_grid)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.pcolormesh(kappa_grid, rho_grid, ml, shading="auto", cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$ (frequency resolution)", fontsize=11)
    ax.set_ylabel(r"$\rho$ (graph trust)", fontsize=11)
    ax.set_title("ML landscape — %s" % ds_name, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="log p(D|φ)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ml_landscape_%s.pdf" % ds_name.lower()),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved ml_landscape_%s.pdf" % ds_name.lower())


def plot_convergence(S, lam_L, d, n, ds_name, output_dir):
    """Plot L-BFGS-B convergence: ML vs function evaluation."""
    from scipy.optimize import minimize

    traces = []
    for rho_init in [0.1, 0.5, 0.9]:
        for lk_init in [-1.0, 0.0, 1.0]:
            ml_trace = []

            def neg_ml(params):
                rho, log_kappa = params
                kappa = np.exp(log_kappa)
                q = Q_rho_eigenvalues(lam_L, rho, kappa)
                val = -marginal_log_likelihood_stationary(S, q, n, d)
                ml_trace.append(-val)
                return val

            def neg_ml_grad(params):
                rho, log_kappa = params
                kappa = np.exp(log_kappa)
                q = Q_rho_eigenvalues(lam_L, rho, kappa)
                q_safe = np.maximum(q, 1e-12)
                common = -0.5 * S + (d / 2.0) / q_safe
                q_prior = (kappa ** 2 + lam_L)
                dq_drho = q_prior - 1.0
                dq_dkappa = 2.0 * rho * kappa
                grad_rho = -np.sum(common * dq_drho)
                grad_log_kappa = -np.sum(common * dq_dkappa) * kappa
                return np.array([grad_rho, grad_log_kappa])

            try:
                minimize(neg_ml, [rho_init, lk_init], jac=neg_ml_grad,
                        method='L-BFGS-B',
                        bounds=[(0.01, 0.99), (-3, 3)],
                        options={'maxiter': 100})
            except:
                pass
            if ml_trace:
                traces.append(ml_trace)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for i, trace in enumerate(traces):
        ax.plot(trace, alpha=0.5, linewidth=1, color="tab:blue")
    ax.set_xlabel("Function evaluation", fontsize=11)
    ax.set_ylabel("Marginal likelihood", fontsize=11)
    ax.set_title("L-BFGS-B convergence — %s" % ds_name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_%s.pdf" % ds_name.lower()),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved convergence_%s.pdf" % ds_name.lower())


def plot_score_range(data, trainer, ds_name, output_dir, device="cuda"):
    """Plot score dynamic range vs kappa."""
    kappa_grid = np.logspace(-2, 2, 40)
    ranges = compute_score_range(data, trainer, kappa_grid, rho=0.5, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(kappa_grid, ranges, "o-", markersize=3, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$", fontsize=11)
    ax.set_ylabel("Score dynamic range (95th - 5th pctile)", fontsize=10)
    ax.set_title("Score range vs. κ — %s" % ds_name, fontsize=12, fontweight="bold")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label=r"$\kappa=1$")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_range_%s.pdf" % ds_name.lower()),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved score_range_%s.pdf" % ds_name.lower())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["enron", "weibo", "facebook", "elliptic"]

    for ds in datasets:
        print("=== %s ===" % ds, flush=True)
        data = load_data(ds)
        n, d = data.x.shape

        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5,
            lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type="low", graph_type="original",
            gamma=1.0, normalize_mode="zscore",
            prior_mean_mode="stationary",
            laplacian_variant="sym",
            d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        trainer._setup(data, device=args.device)

        x_normed = trainer.normalize(data.x.to(args.device).float())
        V_np = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        delta = (x_normed - trainer.template).cpu().numpy()
        S = compute_spectral_energy(delta, V_np)

        # 1. ML landscape
        plot_ml_landscape(S, lam_L, d, n, ds.capitalize(), args.output_dir)

        # 2. Convergence
        plot_convergence(S, lam_L, d, n, ds.capitalize(), args.output_dir)

        # 3. Score dynamic range
        plot_score_range(data, trainer, ds.capitalize(), args.output_dir, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
