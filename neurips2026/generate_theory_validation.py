"""Visualizations validating Corollary 1 and Remark 1.

Figure 1 (ppc_diagnostic.pdf): Posterior predictive check r_j = q_j S_j / d.
Figure 2 (variance_matching.pdf): q_j vs d/S_j scatter.
Figure 3 (kappa_divergence.pdf): ML vs kappa showing unbounded growth.
Figure 4 (gamma_landscape.pdf): ML vs gamma showing non-convexity.
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

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

DATASETS = {
    "weibo": dict(gamma=0.2, graph_type="original", template_type="low",
                  use_encoder=False),
    "facebook": dict(gamma=0.1, graph_type="original",
                     template_type="affinity", use_encoder=False),
    "enron": dict(gamma=0.1, graph_type="affinity", template_type="high",
                  use_encoder=False),
    "amazon": dict(gamma=5.0, graph_type="original", template_type="low",
                   use_encoder=False),
    "reddit": dict(gamma=1.0, graph_type="original", template_type="low",
                   use_encoder=False),
    "elliptic": dict(gamma=0.7, graph_type="original",
                     template_type="affinity", use_encoder=False),
}

DS_NAMES = {
    "weibo": "Weibo", "facebook": "Facebook", "amazon": "Amazon",
    "enron": "Enron", "elliptic": "Elliptic", "reddit": "Reddit",
}

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def build_and_extract(ds, cfg_ov, device="cpu"):
    """Build trainer, extract eigendata and spectral energies."""
    cfg = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        d_hidden=64, n_hidden_layers=2, epochs=0,
        normalize_features=True, laplacian_variant="sym",
        prior_mean_mode="stationary",
    )
    cfg.update(cfg_ov)

    trainer_cfg = SOCTrainerConfig(**cfg)
    trainer = SOCTrainer(trainer_cfg)
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    trainer.train(data, device=device)

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

    return lam_L, S, n, d, trainer, data


# ══════════════════════════════════════════════════════════════════
# Figure 1: PPC diagnostic r_j = q_j S_j / d
# ══════════════════════════════════════════════════════════════════

def generate_ppc(device="cpu"):
    """PPC diagnostic for multiple datasets."""
    fig, ax = plt.subplots(figsize=(5, 3))

    for idx, (ds, cfg_ov) in enumerate(DATASETS.items()):
        print(f"  PPC: {DS_NAMES[ds]}...", flush=True)
        lam_L, S, n, d, _, _ = build_and_extract(ds, cfg_ov, device)

        # Find optimal rho at kappa=0 (simplest case)
        kappa = 0.0
        rho_star = solve_rho_newton(S, lam_L, kappa, d)
        q = Q_rho_eigenvalues(lam_L, rho_star, kappa)

        # PPC ratio
        S_safe = np.maximum(S, 1e-12)
        r_j = q * S_safe / d

        # Sort by eigenvalue for plotting
        sort_idx = np.argsort(lam_L)
        r_sorted = r_j[sort_idx]

        ax.semilogy(range(len(r_sorted)), r_sorted, '-', color=COLORS[idx],
                    lw=1.2, alpha=0.8, label=DS_NAMES[ds])

    ax.axhline(1.0, color='k', ls='--', lw=1.0, alpha=0.5,
               label="Well-calibrated ($r_j=1$)")
    ax.set_xlabel("Eigenmode index $j$ (sorted by $\\lambda_j$)")
    ax.set_ylabel("Calibration ratio $r_j = q_j S_j / d$")
    ax.set_title("Posterior predictive check (Corollary 1)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(1e-2, 1e3)

    plt.tight_layout()
    out = os.path.join("figures",
                       "ppc_diagnostic.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 2: Variance matching q_j vs d/S_j
# ══════════════════════════════════════════════════════════════════

def generate_variance_matching(device="cpu"):
    """Scatter of optimized q_j vs variance-matching target d/S_j."""
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 2.2, 2.2))

    for idx, (ds, cfg_ov) in enumerate(DATASETS.items()):
        print(f"  Variance matching: {DS_NAMES[ds]}...", flush=True)
        lam_L, S, n, d, _, _ = build_and_extract(ds, cfg_ov, device)

        kappa = 0.0
        rho_star = solve_rho_newton(S, lam_L, kappa, d)
        q = Q_rho_eigenvalues(lam_L, rho_star, kappa)

        S_safe = np.maximum(S, 1e-12)
        q_target = d / S_safe  # Corollary 1 target

        ax = axes[idx]
        ax.loglog(q_target, q, '.', color=COLORS[idx], ms=3, alpha=0.6)

        # Diagonal
        lims = [min(q.min(), q_target.min()) * 0.5,
                max(q.max(), q_target.max()) * 2]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)

        ax.set_xlabel("$d/S_j$ (target)")
        if idx == 0:
            ax.set_ylabel("$q_j$ (optimized)")
        ax.set_title(DS_NAMES[ds], fontsize=9)
        ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join("figures",
                       "variance_matching.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 3: κ divergence (Remark 1.2)
# ══════════════════════════════════════════════════════════════════

def generate_kappa_divergence(device="cpu"):
    """ML vs kappa (unbounded) showing log-det drives kappa to infinity."""
    fig, ax = plt.subplots(figsize=(4, 2.8))

    kappa_range = np.logspace(-2, 3, 200)  # 0.01 to 1000

    for idx, (ds, cfg_ov) in enumerate(list(DATASETS.items())[:4]):
        print(f"  Kappa divergence: {DS_NAMES[ds]}...", flush=True)
        lam_L, S, n, d, _, _ = build_and_extract(ds, cfg_ov, device)

        mls = []
        for kappa in kappa_range:
            rho_star = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho_star, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            mls.append(ml)

        mls = np.array(mls)
        ml_norm = (mls - mls.min()) / max(mls.max() - mls.min(), 1e-6)
        ax.plot(kappa_range, ml_norm, '-', color=COLORS[idx],
                lw=1.5, label=DS_NAMES[ds])

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$ (unbounded)")
    ax.set_ylabel("Normalized ML")
    ax.set_title(r"$\kappa$ divergence (Remark 1): ML $\to \infty$ as $\kappa \to \infty$")
    ax.legend(fontsize=7)
    ax.axvline(20, color='gray', ls=':', lw=0.8, alpha=0.5,
               label=r"$\kappa_{\max}$ bound")

    plt.tight_layout()
    out = os.path.join("figures",
                       "kappa_divergence.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 4: ML vs gamma (Remark 1.3 — non-convexity)
# ══════════════════════════════════════════════════════════════════

def generate_gamma_landscape(device="cpu"):
    """ML vs gamma showing non-convex landscape."""
    fig, ax = plt.subplots(figsize=(4, 2.8))

    gamma_range = np.logspace(-2, 1, 100)  # 0.01 to 10

    for idx, (ds, cfg_ov) in enumerate(list(DATASETS.items())[:4]):
        print(f"  Gamma landscape: {DS_NAMES[ds]}...", flush=True)

        mls = []
        for gamma in gamma_range:
            cfg_local = cfg_ov.copy()
            cfg_local["gamma"] = gamma
            try:
                lam_L, S, n, d, _, _ = build_and_extract(
                    ds, cfg_local, device)
                kappa = 0.0
                rho_star = solve_rho_newton(S, lam_L, kappa, d)
                q = Q_rho_eigenvalues(lam_L, rho_star, kappa)
                ml = marginal_log_likelihood_stationary(S, q, n, d)
                mls.append(ml)
            except:
                mls.append(np.nan)

        mls = np.array(mls)
        valid = ~np.isnan(mls)
        if valid.any():
            ml_norm = np.full_like(mls, np.nan)
            ml_norm[valid] = ((mls[valid] - np.nanmin(mls[valid])) /
                              max(np.nanmax(mls[valid]) - np.nanmin(mls[valid]),
                                  1e-6))
            ax.plot(gamma_range, ml_norm, '-', color=COLORS[idx],
                    lw=1.5, label=DS_NAMES[ds])

    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$ (template smoothing)")
    ax.set_ylabel("Normalized ML")
    ax.set_title(r"$\gamma$ landscape (Remark 1): generally non-convex")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join("figures",
                       "gamma_landscape.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 5: Spectral fit diagnostic (Corollary 1 + κ bound)
# ══════════════════════════════════════════════════════════════════

def generate_spectral_fit(device="cpu"):
    """Show d/S_j targets vs parametric q_j(rho*,kappa*) curve.

    For each dataset:
    - Scatter: d/S_j vs lambda_j (variance matching targets from Corollary 1)
    - Curve: q_j(rho*,kappa*) = rho*(kappa*^2 + lambda_j) + (1-rho*) (parametric fit)
    - Vertical line: kappa*^2 showing the spectral cutoff
    - Shaded region: kappa bound from data-driven heuristic
    """
    ds_list = ["weibo", "reddit", "facebook", "amazon"]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(len(ds_list) * 2.5, 2.5))

    for idx, ds in enumerate(ds_list):
        cfg_ov = DATASETS[ds]
        print(f"  Spectral fit: {DS_NAMES[ds]}...", flush=True)
        lam_L, S, n, d, _, _ = build_and_extract(ds, cfg_ov, device)

        S_safe = np.maximum(S, 1e-12)
        q_target = d / S_safe  # Corollary 1 targets

        # Find optimal (rho, kappa) via profile search
        kappa_grid = np.logspace(-2, 1.5, 60)
        best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
        for kp in kappa_grid:
            rs = solve_rho_newton(S, lam_L, kp, d)
            q = Q_rho_eigenvalues(lam_L, rs, kp)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rs, kp

        # Parametric curve at optimal (rho*, kappa*)
        q_opt = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)

        # Also show q at kappa=0 for comparison
        rho_k0 = solve_rho_newton(S, lam_L, 0.0, d)
        q_k0 = Q_rho_eigenvalues(lam_L, rho_k0, 0.0)

        # Data-driven kappa bound
        q_med = np.median(q_target)
        lam_med = np.median(lam_L)
        kappa_bound_sq = max(0, (q_med - 1) / max(best_rho, 0.01) + 1 - lam_med)
        kappa_bound = np.sqrt(kappa_bound_sq)

        ax = axes[idx]
        sort_idx = np.argsort(lam_L)
        lam_sorted = lam_L[sort_idx]

        # Scatter: targets
        ax.semilogy(lam_sorted, q_target[sort_idx], '.',
                    color='#aaaaaa', ms=2, alpha=0.5, label="$d/S_j$ (target)")

        # Parametric curves
        ax.semilogy(lam_sorted, q_opt[sort_idx], '-',
                    color='#d62728', lw=1.5,
                    label=f"$q_j(\\rho^*,\\kappa^*)$")
        if best_kappa > 0.1:
            ax.semilogy(lam_sorted, q_k0[sort_idx], '--',
                        color='#1f77b4', lw=1.0, alpha=0.6,
                        label="$q_j(\\rho^*,\\kappa=0)$")

        # Mark kappa^2 as vertical line
        if best_kappa > 0.01:
            ax.axvline(best_kappa**2, color='#d62728', ls=':',
                       lw=0.8, alpha=0.7)
            ax.text(best_kappa**2, ax.get_ylim()[1] * 0.5,
                    f"$\\kappa^{{*2}}$", fontsize=6, color='#d62728',
                    ha='left', va='top')

        ax.set_xlabel("$\\lambda_j$ (Laplacian eigenvalue)")
        if idx == 0:
            ax.set_ylabel("Precision $q_j$")
        ax.set_title(DS_NAMES[ds], fontsize=9)
        if idx == 0:
            ax.legend(fontsize=5.5, loc="upper left")

    plt.tight_layout()
    out = os.path.join("figures",
                       "spectral_fit.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=== PPC diagnostic ===")
    generate_ppc(args.device)

    print("\n=== Variance matching ===")
    generate_variance_matching(args.device)

    print("\n=== Kappa divergence ===")
    generate_kappa_divergence(args.device)

    print("\n=== Gamma landscape ===")
    generate_gamma_landscape(args.device)

    print("\n=== Spectral fit ===")
    generate_spectral_fit(args.device)

    print("\nDone.")
