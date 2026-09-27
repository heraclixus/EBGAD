"""Generate optimization analysis figures for Section 4.3.

Figure 1 (ml_landscape.pdf): 1x4 ML vs rho showing concavity + Newton.
Figure 2 (ml_vs_auroc.pdf): 1x4 dual-axis ML(rho) and AUROC(rho),
  showing they peak near the same rho.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, warnings
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from data_utils import load_data
from eval_utils import mask_labeled
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
    "t_finance": "T-Finance", "elliptic": "Elliptic",
    "yelpchi": "YelpChi", "weibo": "Weibo", "facebook": "Facebook",
    "amazon": "Amazon", "enron": "Enron", "acm": "ACM",
    "blogcatalog": "BlogCatalog", "reddit": "Reddit",
    "elliptic_plus_plus": "Elliptic++",
}

# ── Actual best configs from sweeps ──
BEST_CONFIGS = {
    "weibo": dict(
        rho=1.0, kappa=0.0, gamma=0.2, graph_type="original",
        template_type="low", prior_mean_mode="stationary",
        score_method="precision_energy", use_encoder=False,
    ),
    "facebook": dict(
        rho=1.0, kappa=1.1, gamma=0.1, graph_type="original",
        template_type="affinity", prior_mean_mode="stationary",
        score_method="precision_ratio", use_encoder=False,
    ),
    "enron": dict(
        rho=1.0, kappa=0.0, gamma=0.1, graph_type="affinity",
        template_type="high", prior_mean_mode="stationary",
        score_method="precision_ratio", use_encoder=False,
    ),
    "amazon": dict(
        rho=0.2, kappa=0.0, gamma=5.0, graph_type="original",
        template_type="low", prior_mean_mode="stationary",
        score_method="precision_energy", use_encoder=False,
    ),
    "elliptic": dict(
        rho=0.99, kappa=0.2, gamma=0.7, graph_type="original",
        template_type="affinity", prior_mean_mode="stationary",
        score_method="precision_ratio", use_encoder=False,
    ),
    "t_finance": dict(
        rho=1.0, kappa=0.0, gamma=0.1, graph_type="original",
        template_type="affinity", prior_mean_mode="stationary",
        score_method="precision_ratio", use_encoder=False,
    ),
    "acm": dict(
        rho=1.0, kappa=0.1, gamma=0.5, graph_type="original",
        template_type="affinity", prior_mean_mode="stationary",
        score_method="control_energy", use_encoder=True,
        encoder_type="pca", encoder_hid_dim=128,
    ),
    "yelpchi": dict(
        rho=0.0, kappa=0.0, gamma=1.0, graph_type="original",
        template_type="low", prior_mean_mode="stationary",
        score_method="precision_energy", use_encoder=True,
        encoder_type="pca", encoder_hid_dim=32,
    ),
    "reddit": dict(
        rho=0.85, kappa=1.0, gamma=1.0, graph_type="original",
        template_type="low", prior_mean_mode="stationary",
        score_method="precision_energy", use_encoder=False,
    ),
}

# Choose datasets that show diverse rho* and clear concavity
LANDSCAPE_DATASETS = ["weibo", "amazon", "facebook", "yelpchi"]
DUAL_AXIS_DATASETS = ["weibo", "amazon", "facebook", "elliptic"]


def build_trainer(ds_name, cfg_overrides, device="cpu"):
    """Build SOCTrainer with given config, run train (eigendecomp only)."""
    base = dict(
        kappa=0.0, nu=1.0, rho=0.5, lam_penalty=50.0,
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


def get_trainer_eigen(trainer):
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    template = trainer.template.cpu().numpy()
    return lam_L, V, template


def get_processed_features(trainer, data, device="cpu"):
    """Get features after normalization and optional PCA encoder."""
    x = data.x.float()
    if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
        x = trainer.normalizer(x)
    if hasattr(trainer, 'encoder') and trainer.encoder is not None:
        with torch.no_grad():
            x = trainer.encoder(x.to(device), data.edge_index.to(device)).cpu()
    return x.numpy()


def sweep_rho(trainer, data, rho_grid, kappa, score_method, device="cpu"):
    """Sweep rho: compute ML and AUROC at each value."""
    lam_L, V, template = get_trainer_eigen(trainer)
    x_np = get_processed_features(trainer, data, device)
    n, d = x_np.shape
    delta = x_np - template
    S = compute_spectral_energy(delta, V)
    y_labels = data.y.numpy()
    mask = (y_labels >= 0) & (y_labels <= 1)

    mls, aurocs = [], []
    delta_hat = V.T @ delta  # (k, d)

    for rho in rho_grid:
        q = Q_rho_eigenvalues(lam_L, rho, kappa, nu=1.0)
        q_safe = np.maximum(q, 1e-12)

        # ML
        ml = marginal_log_likelihood_stationary(S, q, n, d)
        mls.append(ml)

        # AUROC
        if score_method == "precision_energy":
            w = np.sqrt(q_safe)[:, None] * delta_hat
            scores = np.sum((V @ w)**2, axis=1)
        elif score_method == "precision_ratio":
            w = np.sqrt(q_safe)[:, None] * delta_hat
            ce = np.sum((V @ w)**2, axis=1)
            tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
            scores = ce / tot
        elif score_method == "control_energy":
            lam_p, aT = 50.0, 2.0
            sig = (1.0 - np.exp(-2.0 * aT * q_safe)) / q_safe
            D = 1.0 / lam_p + sig
            D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-12))
            w = D_inv_sqrt[:, None] * delta_hat
            scores = np.sum((V @ w)**2, axis=1)
        else:
            scores = np.sum((V @ (np.sqrt(q_safe)[:, None] * delta_hat))**2,
                           axis=1)

        try:
            auc = roc_auc_score(y_labels[mask], scores[mask])
        except ValueError:
            auc = 0.5
        aurocs.append(auc * 100)

    return np.array(mls), np.array(aurocs), S, lam_L, n, d


def newton_trajectory(S, lam_L, kappa, d, rho_init=0.5, n_steps=6):
    a = (kappa**2 + lam_L)**1.0 - 1.0
    rho = np.clip(rho_init, 1e-6, 1.0 - 1e-6)
    trajectory = [float(rho)]
    for _ in range(n_steps):
        q = rho * a + 1.0
        q_safe = np.maximum(q, 1e-12)
        grad = np.sum(a * (-S / 2.0 + d / (2.0 * q_safe)))
        hess = -np.sum(d * a**2 / (2.0 * q_safe**2))
        if abs(hess) < 1e-15:
            break
        rho_new = float(np.clip(rho - grad / hess, 0.0, 1.0))
        trajectory.append(rho_new)
        if abs(rho_new - rho) < 1e-10:
            break
        rho = rho_new
    return trajectory


# ══════════════════════════════════════════════════════════════════
# Figure 1: ML landscape (1x4)
# ══════════════════════════════════════════════════════════════════

def generate_ml_landscape(device="cpu"):
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.0))
    rho_grid = np.linspace(0.001, 0.999, 300)
    colors = {"curve": "#1f77b4", "newton": "#d62728"}

    for idx, ds in enumerate(LANDSCAPE_DATASETS):
        cfg = BEST_CONFIGS[ds].copy()
        kappa = cfg.get("kappa", 0.0)
        sm = cfg.get("score_method", "precision_energy")
        print(f"  [{idx+1}/4] {DS_NAMES.get(ds, ds)} (rho={cfg['rho']}, "
              f"kappa={kappa})...", flush=True)

        trainer, data = build_trainer(ds, cfg, device)
        mls, aurocs, S, lam_L, n, d = sweep_rho(
            trainer, data, rho_grid, kappa, sm, device)

        ml_max = mls.max()
        ml_range = max(ml_max - mls.min(), 1e-6)
        ml_norm = (mls - ml_max) / ml_range * 100

        axes[idx].plot(rho_grid, ml_norm, color=colors["curve"], lw=1.5)

        # Newton trajectory
        rho_expected = cfg["rho"]
        rho_init = 0.9 if rho_expected < 0.5 else 0.1
        traj = newton_trajectory(S, lam_L, kappa, d, rho_init=rho_init)
        traj_ml = []
        for r in traj:
            q = Q_rho_eigenvalues(lam_L, r, kappa)
            traj_ml.append(
                (marginal_log_likelihood_stationary(S, q, n, d) - ml_max)
                / ml_range * 100)

        for i in range(len(traj) - 1):
            axes[idx].annotate(
                "", xy=(traj[i+1], traj_ml[i+1]),
                xytext=(traj[i], traj_ml[i]),
                arrowprops=dict(arrowstyle="->", color=colors["newton"],
                                lw=1.0, shrinkA=2, shrinkB=2))
        axes[idx].scatter(traj, traj_ml, color=colors["newton"],
                         s=18, zorder=5, edgecolors="k", linewidths=0.3)

        rho_star = traj[-1]
        axes[idx].axvline(rho_star, color="gray", ls="--", lw=0.6, alpha=0.6)

        axes[idx].set_title(
            f"{DS_NAMES.get(ds, ds)}\n($\\rho^\\star$={rho_star:.2f})",
            fontsize=8)
        axes[idx].set_xlabel(r"$\rho$")
        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(bottom=max(ml_norm.min(), -105), top=5)
        if idx == 0:
            axes[idx].set_ylabel("Relative ML (%)")
        axes[idx].yaxis.set_major_locator(MaxNLocator(5))
        if idx == 0:
            axes[idx].scatter([], [], color=colors["newton"], s=18,
                             edgecolors="k", linewidths=0.3,
                             label="Newton iterates")
            axes[idx].legend(loc="lower right")

    plt.tight_layout(w_pad=0.8)
    out = os.path.join("figures",
                       "ml_landscape.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 2: Dual-axis ML(rho) and AUROC(rho)
# ══════════════════════════════════════════════════════════════════

def generate_dual_axis(device="cpu"):
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.2))
    rho_grid = np.linspace(0.001, 0.999, 100)
    c_ml = "#1f77b4"
    c_auc = "#d62728"

    for idx, ds in enumerate(DUAL_AXIS_DATASETS):
        cfg = BEST_CONFIGS[ds].copy()
        kappa = cfg.get("kappa", 0.0)
        sm = cfg.get("score_method", "precision_energy")
        print(f"  [{idx+1}/4] {DS_NAMES.get(ds, ds)}...", flush=True)

        trainer, data = build_trainer(ds, cfg, device)
        mls, aurocs, S, lam_L, n, d = sweep_rho(
            trainer, data, rho_grid, kappa, sm, device)

        # Normalize ML to percentage of range
        ml_max = mls.max()
        ml_range = max(ml_max - mls.min(), 1e-6)
        ml_norm = (mls - ml_max) / ml_range * 100

        # Find ML-optimal and AUROC-optimal rho
        rho_ml = rho_grid[np.argmax(mls)]
        rho_auc = rho_grid[np.argmax(aurocs)]
        auc_at_ml_rho = aurocs[np.argmax(mls)]
        auc_best = aurocs.max()

        # Left y-axis: ML
        ax1 = axes[idx]
        l1, = ax1.plot(rho_grid, ml_norm, color=c_ml, lw=1.5, label="ML")
        ax1.set_xlabel(r"$\rho$")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(bottom=max(ml_norm.min(), -105), top=5)
        if idx == 0:
            ax1.set_ylabel("Relative ML (%)", color=c_ml)
        ax1.tick_params(axis='y', labelcolor=c_ml)
        ax1.yaxis.set_major_locator(MaxNLocator(5))

        # Right y-axis: AUROC
        ax2 = ax1.twinx()
        l2, = ax2.plot(rho_grid, aurocs, color=c_auc, lw=1.5, ls="--",
                       label="AUROC")
        ax2.tick_params(axis='y', labelcolor=c_auc)
        ax2.yaxis.set_major_locator(MaxNLocator(5))
        if idx == 3:
            ax2.set_ylabel("AUROC (%)", color=c_auc)

        # Mark ML-optimal rho
        ax1.axvline(rho_ml, color=c_ml, ls=":", lw=0.8, alpha=0.7)
        # Mark AUROC-optimal rho
        ax1.axvline(rho_auc, color=c_auc, ls=":", lw=0.8, alpha=0.7)

        ds_display = DS_NAMES.get(ds, ds)
        ax1.set_title(
            f"{ds_display}\n"
            f"$\\rho^\\star_{{ML}}$={rho_ml:.2f}, "
            f"AUC={auc_at_ml_rho:.1f}%",
            fontsize=7.5)

        if idx == 0:
            ax1.legend([l1, l2], ["ML", "AUROC"], loc="lower left",
                      fontsize=6)

    plt.tight_layout(w_pad=1.2)
    out = os.path.join("figures",
                       "ml_vs_auroc.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--landscape-only", action="store_true")
    parser.add_argument("--dual-only", action="store_true")
    args = parser.parse_args()

    if not args.dual_only:
        print("=== Generating ML landscape figure ===")
        generate_ml_landscape(args.device)

    if not args.landscape_only:
        print("\n=== Generating dual-axis ML vs AUROC figure ===")
        generate_dual_axis(args.device)

    print("\nDone.")
