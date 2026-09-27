"""Generate individual components for Figure 1 (for PowerPoint assembly).

1. BlogCatalog contour (real data, no title, clean axes)
2. BlogCatalog Newton convergence (real data, no title)
3. Score separation KDE (matching paper density plot style)
4. Toy graph with anomaly nodes
5. Toy graph showing prior template (smoothed features)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "serif", "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.transparent": True,
})

OUT = os.path.join("figures", "fig1_parts")
os.makedirs(OUT, exist_ok=True)

C_NORMAL = "#4a90d9"
C_ANOMALY = "#e74c3c"


def make_contour_and_newton():
    """BlogCatalog contour + Newton using real data."""
    from data_utils import load_data
    from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
    from soc.prior_optimizer import (
        Q_rho_eigenvalues, marginal_log_likelihood_stationary,
        compute_spectral_energy, solve_rho_newton,
    )

    print("  Loading BlogCatalog (same config as contour_pair figure)...",
          flush=True)
    # Exact same config as in generate_final_contour.py for BlogCatalog
    cfg = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        d_hidden=64, n_hidden_layers=2, epochs=0,
        normalize_features=True, laplacian_variant="sym",
        prior_mean_mode="stationary",
        template_type="affinity", graph_type="original", gamma=2.0,
    )
    trainer_cfg = SOCTrainerConfig(**cfg)
    trainer = SOCTrainer(trainer_cfg)
    data = load_data("BlogCatalog")
    trainer.train(data, device="cpu")

    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    template = trainer.template.cpu().numpy()
    x = data.x.float()
    if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
        x = trainer.normalizer(x)
    elif hasattr(trainer, 'normalize') and trainer.normalize is not None:
        x = trainer.normalize(x)
    delta = x.numpy() - template
    S = compute_spectral_energy(delta, V)
    n, d = delta.shape

    print("  Computing ML grid...", flush=True)
    rho_grid = np.linspace(0.005, 0.995, 100)
    kappa_grid = np.logspace(-2, 1.6, 100)
    ML = np.zeros((len(rho_grid), len(kappa_grid)))
    for i, rho in enumerate(rho_grid):
        for j, kappa in enumerate(kappa_grid):
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ML[i, j] = marginal_log_likelihood_stationary(S, q, n, d)

    ml_norm = (ML - ML.max()) / max(ML.max() - ML.min(), 1e-6) * 100

    # Profile path
    path_kappa = np.logspace(-2, 1.6, 80)
    path_rho = []
    best_ml, best_rho, best_kappa = -np.inf, 0.5, 1.0
    for kp in path_kappa:
        rs = solve_rho_newton(S, lam_L, kp, d)
        path_rho.append(rs)
        q = Q_rho_eigenvalues(lam_L, rs, kp)
        ml = marginal_log_likelihood_stationary(S, q, n, d)
        if ml > best_ml:
            best_ml, best_rho, best_kappa = ml, rs, kp

    # --- Contour ---
    K, R = np.meshgrid(kappa_grid, rho_grid)
    levels = np.linspace(-100, 0, 25)

    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    cf = ax.contourf(K, R, ml_norm, levels=levels, cmap="RdYlBu_r",
                     extend="min")
    ax.contour(K, R, ml_norm, levels=levels[::3], colors="k",
               linewidths=0.2, alpha=0.3)
    ax.plot(path_kappa, path_rho, '-', color='k', lw=2.5, alpha=0.7)
    ax.plot(path_kappa, path_rho, '--', color='w', lw=1.5, alpha=0.9)
    ax.plot(best_kappa, best_rho, '*', color='k', ms=10,
            markeredgecolor='w', markeredgewidth=1.0, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\rho$")
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(OUT, "contour.pdf"))
    print("  Saved contour.pdf")
    plt.close()

    # --- Newton convergence ---
    a = (best_kappa**2 + lam_L) - 1.0
    rho = 0.1
    iters = [rho]
    for _ in range(15):
        q = rho * a + 1.0
        q_safe = np.maximum(q, 1e-12)
        grad = np.sum(a * (-S / 2.0 + d / (2.0 * q_safe)))
        hess = -np.sum(d * a**2 / (2.0 * q_safe**2))
        if abs(hess) < 1e-15:
            break
        rho_new = float(np.clip(rho - grad / hess, 0.0, 1.0))
        iters.append(rho_new)
        if abs(rho_new - rho) < 1e-12:
            break
        rho = rho_new

    rho_star = iters[-1]
    errors = [abs(r - rho_star) + 1e-16 for r in iters]

    fig, ax = plt.subplots(figsize=(2.8, 2.0))
    ax.semilogy(range(len(errors)), errors, 'o-', color='#d62728',
                ms=5, lw=1.5, markeredgecolor='k', markeredgewidth=0.4)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel(r"$|\rho_k - \rho^\star|$")
    ax.set_xlim(-0.3, len(errors) + 0.3)
    fig.savefig(os.path.join(OUT, "newton.pdf"))
    print("  Saved newton.pdf")
    plt.close()


def make_density():
    """Score separation KDE plot matching paper density style."""
    np.random.seed(42)
    normal = np.random.beta(2.5, 7, 5000) * 5
    anomaly = np.random.beta(5, 2, 400) * 4 + 2.5

    x_range = np.linspace(0, 7, 500)
    kde_n = gaussian_kde(normal, bw_method=0.12)
    kde_a = gaussian_kde(anomaly, bw_method=0.15)

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.fill_between(x_range, kde_n(x_range), alpha=0.5, color=C_NORMAL)
    ax.plot(x_range, kde_n(x_range), color=C_NORMAL, lw=1.8, label="Normal")
    ax.fill_between(x_range, kde_a(x_range), alpha=0.5, color=C_ANOMALY)
    ax.plot(x_range, kde_a(x_range), color=C_ANOMALY, lw=1.8, label="Anomaly")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend(fontsize=8)
    ax.set_xlim(0, 7)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT, "density.pdf"))
    print("  Saved density.pdf")
    plt.close()


def make_graphs():
    """Two toy graphs: (a) with anomalies, (b) showing template (smoothed)."""
    np.random.seed(42)

    # Build a small community graph
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G, seed=7, k=0.6)

    anomaly_nodes = {0, 33, 8}

    # Assign "features" (1D for visualization)
    features = {}
    for node in G.nodes():
        if node in anomaly_nodes:
            features[node] = np.random.uniform(0.8, 1.0)
        else:
            features[node] = np.random.uniform(0.1, 0.45)

    # Template = graph-smoothed features (simple 1-hop average)
    template = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            template[node] = np.mean([features[n] for n in neighbors])
        else:
            template[node] = features[node]

    def draw_graph(ax, node_values, cmap, vmin, vmax, title_text=None):
        # Draw edges
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], '-', color='#dddddd', lw=0.5,
                    alpha=0.6)
        # Draw nodes
        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]
        vals = [node_values[n] for n in G.nodes()]
        sc = ax.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                       s=60, zorder=3, edgecolors='white', linewidths=0.5)
        ax.axis("off")
        ax.set_aspect("equal")
        return sc

    # (a) Graph with anomalies (raw features)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    draw_graph(ax, features, "RdYlBu_r", 0, 1)
    fig.savefig(os.path.join(OUT, "graph_anomaly.pdf"))
    print("  Saved graph_anomaly.pdf")
    plt.close()

    # (b) Graph showing template (smoothed features)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    draw_graph(ax, template, "RdYlBu_r", 0, 1)
    fig.savefig(os.path.join(OUT, "graph_template.pdf"))
    print("  Saved graph_template.pdf")
    plt.close()


if __name__ == "__main__":
    print("=== Generating Figure 1 components ===")
    print("Graphs...")
    make_graphs()
    print("Density...")
    make_density()
    print("Contour + Newton (real BlogCatalog)...")
    make_contour_and_newton()
    print(f"Done. Components in {OUT}/")
