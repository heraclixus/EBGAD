"""Generate publication-quality contour + convergence figures.

Generates one figure per candidate, each with contour (top) + convergence (bottom).
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
import matplotlib.gridspec as gridspec

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

CANDIDATES = [
    {
        "tag": "facebook_affinity_affinity_g2.0_nopca",
        "ds": "facebook", "display": "Facebook",
        "template_type": "affinity", "graph_type": "affinity",
        "gamma": 2.0, "use_encoder": False,
    },
    {
        "tag": "reddit_zero_affinity_g2.0_nopca",
        "ds": "reddit", "display": "Reddit",
        "template_type": "zero", "graph_type": "affinity",
        "gamma": 2.0, "use_encoder": False,
    },
    {
        "tag": "blogcatalog_affinity_original_g2.0_nopca",
        "ds": "blogcatalog", "display": "BlogCatalog",
        "template_type": "affinity", "graph_type": "original",
        "gamma": 2.0, "use_encoder": False,
    },
    {
        "tag": "reddit_affinity_original_g2.0_pca128",
        "ds": "reddit", "display": "Reddit",
        "template_type": "affinity", "graph_type": "original",
        "gamma": 2.0,
        "use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 128,
    },
    {
        "tag": "yelpchi_zero_original_g0.5_pca32",
        "ds": "yelpchi", "display": "YelpChi",
        "template_type": "zero", "graph_type": "original",
        "gamma": 0.5,
        "use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 32,
    },
]


def build_trainer(panel, device):
    cfg = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        d_hidden=64, n_hidden_layers=2, epochs=0,
        normalize_features=True, laplacian_variant="sym",
        prior_mean_mode="stationary",
        template_type=panel["template_type"],
        graph_type=panel["graph_type"],
        gamma=panel["gamma"],
    )
    if panel.get("use_encoder"):
        cfg.update({
            "use_encoder": True, "encoder_type": panel["encoder_type"],
            "encoder_hid_dim": panel["encoder_hid_dim"],
            "encoder_num_layers": 1, "encoder_dropout": 0.0,
            "encoder_lr": 0.0, "encoder_epochs": 0,
            "encoder_alpha": 1.0, "encoder_weight_decay": 0.0,
        })
    trainer_cfg = SOCTrainerConfig(**cfg)
    trainer = SOCTrainer(trainer_cfg)
    load_name = ("YelpChi" if panel["ds"] == "yelpchi" else
                 "Facebook" if panel["ds"] == "facebook" else panel["ds"])
    data = load_data(load_name)
    trainer.train(data, device=device)
    return trainer, data


def compute_all(trainer, data, device):
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

    rho_grid = np.linspace(0.005, 0.995, 100)
    kappa_grid = np.logspace(-2, 1.6, 100)  # plot range wider than search
    ML = np.zeros((len(rho_grid), len(kappa_grid)))
    for i, rho in enumerate(rho_grid):
        for j, kappa in enumerate(kappa_grid):
            q = Q_rho_eigenvalues(lam_L, rho, kappa, nu=1.0)
            ML[i, j] = marginal_log_likelihood_stationary(S, q, n, d)

    path_kappa = np.logspace(-2, 1.6, 80)  # match plot range
    path_rho = []
    best_ml, best_rho, best_kappa = -np.inf, 0.5, 1.0
    for kp in path_kappa:
        rs = solve_rho_newton(S, lam_L, kp, d)
        path_rho.append(rs)
        q = Q_rho_eigenvalues(lam_L, rs, kp)
        ml = marginal_log_likelihood_stationary(S, q, n, d)
        if ml > best_ml:
            best_ml, best_rho, best_kappa = ml, rs, kp

    a = (best_kappa**2 + lam_L)**1.0 - 1.0
    rho = 0.1
    newton_iters = [float(rho)]
    for _ in range(15):
        q = rho * a + 1.0
        q_safe = np.maximum(q, 1e-12)
        grad = np.sum(a * (-S / 2.0 + d / (2.0 * q_safe)))
        hess = -np.sum(d * a**2 / (2.0 * q_safe**2))
        if abs(hess) < 1e-15:
            break
        rho_new = float(np.clip(rho - grad / hess, 0.0, 1.0))
        newton_iters.append(rho_new)
        if abs(rho_new - rho) < 1e-12:
            break
        rho = rho_new

    return {
        "rho_grid": rho_grid, "kappa_grid": kappa_grid, "ML": ML,
        "path_rho": np.array(path_rho), "path_kappa": path_kappa,
        "best_rho": best_rho, "best_kappa": best_kappa,
        "newton_iters": newton_iters,
    }


def render_single(panel, res, out_path):
    """Render a single-panel figure (contour + convergence stacked)."""
    fig = plt.figure(figsize=(3.5, 4.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.45)

    # ── Contour ──
    ax_top = fig.add_subplot(gs[0])
    ml_norm = (res["ML"] - res["ML"].max()) / \
        max(res["ML"].max() - res["ML"].min(), 1e-6) * 100
    K, R = np.meshgrid(res["kappa_grid"], res["rho_grid"])
    levels = np.linspace(-100, 0, 25)
    cf = ax_top.contourf(K, R, ml_norm, levels=levels,
                         cmap="RdYlBu_r", extend="min")
    ax_top.contour(K, R, ml_norm, levels=levels[::3],
                   colors="k", linewidths=0.3, alpha=0.3)
    ax_top.plot(res["path_kappa"], res["path_rho"],
                '-', color="black", lw=2.5, alpha=0.7)
    ax_top.plot(res["path_kappa"], res["path_rho"],
                '--', color="white", lw=1.5, alpha=0.9,
                label="Profile path")
    ax_top.plot(res["best_kappa"], res["best_rho"],
                '*', color="black", ms=10,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=5, label="Optimum")
    ax_top.set_xscale("log")
    ax_top.set_xlabel(r"$\kappa$")
    ax_top.set_ylabel(r"$\rho$")
    ax_top.set_ylim(0, 1)
    ax_top.set_title(panel['display'])
    ax_top.legend(fontsize=7, loc="lower left",
                  framealpha=0.85, edgecolor='none')
    cbar = fig.colorbar(cf, ax=ax_top, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log p(\mathcal{D}\mid\varphi)$", fontsize=8)
    cbar.set_ticks([-100, -50, 0])
    cbar.set_ticklabels(["min", "", "max"])
    cbar.ax.tick_params(labelsize=7)

    # ── Newton convergence ──
    ax_bot = fig.add_subplot(gs[1])
    iters = res["newton_iters"]
    rho_star = iters[-1]
    errors = [abs(r - rho_star) + 1e-16 for r in iters]
    ax_bot.semilogy(range(len(errors)), errors, 'o-',
                    color='#d62728', ms=5, lw=1.5,
                    markeredgecolor='k', markeredgewidth=0.4)
    ax_bot.set_xlabel("Newton iteration")
    ax_bot.set_ylabel(r"$|\rho_k - \rho^\star|$")
    ax_bot.set_title("Newton convergence")
    ax_bot.set_xlim(-0.3, max(len(iters), 6) + 0.3)

    fig.savefig(out_path)
    print(f"    Saved {out_path}")
    plt.close()


def render_pair(panels, results, out_path):
    """Render a 2-panel figure (2 contours + 2 convergence)."""
    fig = plt.figure(figsize=(7, 4.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.4, 1],
                           hspace=0.5, wspace=0.3)

    cf_last = None
    for col, (panel, res) in enumerate(zip(panels, results)):
        # ── Contour ──
        ax_top = fig.add_subplot(gs[0, col])
        ml_norm = (res["ML"] - res["ML"].max()) / \
            max(res["ML"].max() - res["ML"].min(), 1e-6) * 100
        K, R = np.meshgrid(res["kappa_grid"], res["rho_grid"])
        levels = np.linspace(-100, 0, 25)
        cf = ax_top.contourf(K, R, ml_norm, levels=levels,
                             cmap="RdYlBu_r", extend="min")
        ax_top.contour(K, R, ml_norm, levels=levels[::3],
                       colors="k", linewidths=0.3, alpha=0.3)
        ax_top.plot(res["path_kappa"], res["path_rho"],
                    '-', color="black", lw=2.5, alpha=0.7)
        ax_top.plot(res["path_kappa"], res["path_rho"],
                    '--', color="white", lw=1.5, alpha=0.9,
                    label="Profile path")
        ax_top.plot(res["best_kappa"], res["best_rho"],
                    '*', color="black", ms=10,
                    markeredgecolor='white', markeredgewidth=1.2,
                    zorder=5, label="Optimum")
        ax_top.set_xscale("log")
        ax_top.set_xlabel(r"$\kappa$")
        ax_top.set_ylabel(r"$\rho$")
        ax_top.set_ylim(0, 1)
        ax_top.set_title(panel['display'])
        if col == 0:
            ax_top.legend(fontsize=7, loc="lower left",
                          framealpha=0.85, edgecolor='none')
        cf_last = cf

        # ── Newton convergence ──
        ax_bot = fig.add_subplot(gs[1, col])
        iters = res["newton_iters"]
        rho_star = iters[-1]
        errors = [abs(r - rho_star) + 1e-16 for r in iters]
        ax_bot.semilogy(range(len(errors)), errors, 'o-',
                        color='#d62728', ms=5, lw=1.5,
                        markeredgecolor='k', markeredgewidth=0.4)
        ax_bot.set_xlabel("Newton iteration")
        ax_bot.set_ylabel(r"$|\rho_k - \rho^\star|$")
        ax_bot.set_title("Newton convergence")
        ax_bot.set_xlim(-0.3, max(len(iters), 6) + 0.3)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.45, 0.015, 0.45])
    cbar = fig.colorbar(cf_last, cax=cbar_ax)
    cbar.set_label(r"$\log p(\mathcal{D}\mid\varphi)$ (normalized)", fontsize=8)
    cbar.set_ticks([-100, -50, 0])
    cbar.set_ticklabels(["min", "", "max"])
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path)
    print(f"    Saved {out_path}")
    plt.close()


def main(device="cpu"):
    out_dir = os.path.join("figures")
    os.makedirs(out_dir, exist_ok=True)

    # Compute all candidates
    all_results = []
    for i, panel in enumerate(CANDIDATES):
        print(f"  [{i+1}/{len(CANDIDATES)}] {panel['display']} "
              f"({panel['tag']})...", flush=True)
        trainer, data = build_trainer(panel, device)
        res = compute_all(trainer, data, device)
        all_results.append(res)

        # Individual figure
        render_single(panel, res,
                      os.path.join(out_dir, f"contour_{panel['tag']}.pdf"))

    # All 10 pairwise combinations
    from itertools import combinations
    for i, j in combinations(range(len(CANDIDATES)), 2):
        p1, p2 = CANDIDATES[i], CANDIDATES[j]
        pair_name = f"{p1['ds']}_{p2['ds']}"
        render_pair([p1, p2], [all_results[i], all_results[j]],
                    os.path.join(out_dir, f"contour_pair_{pair_name}.pdf"))

    print("\nDone. Generated individual + pairwise figures.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print("=== Generating all candidate contour figures ===")
    main(args.device)
