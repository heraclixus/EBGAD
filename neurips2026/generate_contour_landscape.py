"""Generate 2D contour plots of the ML landscape over (rho, kappa).

Shows the marginal log-likelihood surface as filled contours,
with the profile-Newton optimization trajectory overlaid.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    marginal_log_likelihood_stationary,
    compute_spectral_energy,
    solve_rho_newton,
)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

DS_NAMES = {
    "weibo": "Weibo", "facebook": "Facebook", "amazon": "Amazon",
    "enron": "Enron", "elliptic": "Elliptic", "yelpchi": "YelpChi",
    "t_finance": "T-Finance", "acm": "ACM", "reddit": "Reddit",
}

# Best configs (template, gamma, graph fixed; sweep rho and kappa)
CONFIGS = {
    "weibo": dict(
        gamma=0.2, graph_type="original", template_type="low",
        prior_mean_mode="stationary", score_method="precision_energy",
        use_encoder=False,
    ),
    "facebook": dict(
        gamma=0.1, graph_type="original", template_type="affinity",
        prior_mean_mode="stationary", score_method="precision_ratio",
        use_encoder=False,
    ),
    "amazon": dict(
        gamma=5.0, graph_type="original", template_type="low",
        prior_mean_mode="stationary", score_method="precision_energy",
        use_encoder=False,
    ),
    "elliptic": dict(
        gamma=0.7, graph_type="original", template_type="affinity",
        prior_mean_mode="stationary", score_method="precision_ratio",
        use_encoder=False,
    ),
    "enron": dict(
        gamma=0.1, graph_type="affinity", template_type="high",
        prior_mean_mode="stationary", score_method="precision_ratio",
        use_encoder=False,
    ),
    "reddit": dict(
        gamma=1.0, graph_type="original", template_type="low",
        prior_mean_mode="stationary", score_method="precision_energy",
        use_encoder=False,
    ),
    "yelpchi": dict(
        gamma=1.0, graph_type="original", template_type="low",
        prior_mean_mode="stationary", score_method="precision_energy",
        use_encoder=True, encoder_type="pca", encoder_hid_dim=32,
    ),
    "acm": dict(
        gamma=0.5, graph_type="original", template_type="affinity",
        prior_mean_mode="stationary", score_method="control_energy",
        use_encoder=True, encoder_type="pca", encoder_hid_dim=128,
    ),
    "t_finance": dict(
        gamma=0.1, graph_type="original", template_type="affinity",
        prior_mean_mode="stationary", score_method="precision_ratio",
        use_encoder=False,
    ),
    "blogcatalog": dict(
        gamma=0.2, graph_type="original", template_type="low",
        prior_mean_mode="stationary", score_method="precision_energy",
        use_encoder=True, encoder_type="pca", encoder_hid_dim=128,
    ),
}

DATASETS = ["weibo", "amazon", "facebook", "elliptic",
            "enron", "reddit", "yelpchi", "acm",
            "t_finance", "blogcatalog"]


def build_trainer(ds_name, cfg_overrides, device="cpu"):
    base = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, template_type="low", graph_type="original",
        normalize_mode="zscore", d_hidden=64, n_hidden_layers=2,
        epochs=0, normalize_features=True, laplacian_variant="sym",
        prior_mean_mode="stationary", gamma=0.5,
    )
    base.update(cfg_overrides)
    for k in ["score_method"]:
        base.pop(k, None)
    trainer_cfg = SOCTrainerConfig(**base)
    trainer = SOCTrainer(trainer_cfg)
    data = load_data("YelpChi" if ds_name == "yelpchi" else
                     "Facebook" if ds_name == "facebook" else ds_name)
    trainer.train(data, device=device)
    return trainer, data


def compute_ml_grid(trainer, data, rho_grid, kappa_grid, device="cpu"):
    """Compute ML on a 2D grid of (rho, kappa)."""
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    template = trainer.template.cpu().numpy()

    x = data.x.float()
    if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
        x = trainer.normalizer(x)
    if hasattr(trainer, 'encoder') and trainer.encoder is not None:
        with torch.no_grad():
            x = trainer.encoder(x.to(device), data.edge_index.to(device)).cpu()
    x_np = x.numpy()
    n, d = x_np.shape
    delta = x_np - template
    S = compute_spectral_energy(delta, V)

    ML = np.zeros((len(rho_grid), len(kappa_grid)))
    for i, rho in enumerate(rho_grid):
        for j, kappa in enumerate(kappa_grid):
            q = Q_rho_eigenvalues(lam_L, rho, kappa, nu=1.0)
            ML[i, j] = marginal_log_likelihood_stationary(S, q, n, d)

    return ML, S, lam_L, n, d


def profile_newton_path(S, lam_L, d, kappa_grid, rho_init=0.5):
    """Compute the profile-Newton path: for each kappa, find rho*(kappa)."""
    path_kappa = []
    path_rho = []
    path_ml = []

    best_ml = -np.inf
    best_rho = rho_init
    best_kappa = kappa_grid[0]

    for kappa in kappa_grid:
        rho_star = solve_rho_newton(S, lam_L, kappa, d, rho_init=rho_init)
        q = Q_rho_eigenvalues(lam_L, rho_star, kappa)
        ml = marginal_log_likelihood_stationary(S, q, len(lam_L), d)
        path_kappa.append(kappa)
        path_rho.append(rho_star)
        path_ml.append(ml)
        if ml > best_ml:
            best_ml = ml
            best_rho = rho_star
            best_kappa = kappa

    return (np.array(path_rho), np.array(path_kappa), np.array(path_ml),
            best_rho, best_kappa)


def plot_single_contour(ax, ds, device="cpu"):
    """Plot a single contour panel. Returns (best_rho, best_kappa, cf)."""
    cfg = CONFIGS[ds].copy()

    trainer, data = build_trainer(ds, cfg, device)

    rho_grid = np.linspace(0.01, 0.99, 80)
    kappa_grid = np.logspace(-2, 1.5, 80)

    ML, S, lam_L, n, d = compute_ml_grid(
        trainer, data, rho_grid, kappa_grid, device)

    ml_max = ML.max()
    ml_range = max(ml_max - ML.min(), 1e-6)
    ML_norm = (ML - ml_max) / ml_range * 100

    K, R = np.meshgrid(kappa_grid, rho_grid)
    levels = np.linspace(-100, 0, 20)
    cf = ax.contourf(K, R, ML_norm, levels=levels,
                     cmap="RdYlBu_r", extend="min")
    ax.contour(K, R, ML_norm, levels=levels[::2],
               colors="k", linewidths=0.2, alpha=0.3)

    # Profile-Newton path
    path_kappa_fine = np.logspace(-2, 1.5, 50)
    path_rho, path_kappa, path_ml, best_rho, best_kappa = \
        profile_newton_path(S, lam_L, d, path_kappa_fine)

    ax.plot(path_kappa, path_rho, 'w-', lw=1.5, alpha=0.8,
            label="Profile path $\\rho^\\star(\\kappa)$")
    ax.plot(path_kappa, path_rho, 'k--', lw=0.8, alpha=0.5)

    ax.plot(best_kappa, best_rho, 'w*', markersize=10,
            markeredgecolor='k', markeredgewidth=0.8, zorder=5,
            label="Optimum")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{DS_NAMES.get(ds, ds)}\n"
        f"($\\rho^\\star$={best_rho:.2f}, $\\kappa^\\star$={best_kappa:.2f})",
        fontsize=8)

    return best_rho, best_kappa, cf


def generate_contour(device="cpu"):
    n_ds = len(DATASETS)
    ncols = min(5, n_ds)
    nrows = (n_ds + ncols - 1) // ncols

    # Generate all-datasets grid
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.0, nrows * 2.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    cf = None
    for idx, ds in enumerate(DATASETS):
        r, c = idx // ncols, idx % ncols
        print(f"  [{idx+1}/{n_ds}] {DS_NAMES.get(ds, ds)}...", flush=True)

        try:
            best_rho, best_kappa, cf = plot_single_contour(
                axes[r, c], ds, device)
            if r == nrows - 1:
                pass  # xlabel already set
            if c == 0:
                axes[r, c].set_ylabel(r"$\rho$")
        except Exception as e:
            print(f"    ERROR: {str(e)[:60]}", flush=True)
            axes[r, c].set_title(f"{DS_NAMES.get(ds, ds)}\n(error)")

    # Hide unused axes
    for idx in range(n_ds, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].set_visible(False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes, orientation="vertical",
                            fraction=0.012, pad=0.02, shrink=0.9)
        cbar.set_label("Relative ML (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    plt.tight_layout(w_pad=0.5, h_pad=0.8)
    out = os.path.join("figures",
                       "ml_contour_all.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print("=== Generating ML contour landscape ===")
    generate_contour(args.device)
    print("\nDone.")
